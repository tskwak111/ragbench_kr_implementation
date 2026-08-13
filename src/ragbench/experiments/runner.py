"""Immutable planning and cancellation-safe execution of development experiments."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ragbench.core.hashing import canonical_json_hash
from ragbench.core.money import BudgetExceededError
from ragbench.providers.upstage.client import ProviderHTTPError
from ragbench.providers.upstage.pricing import PriceBook, PricingRequest

MONEY_QUANTUM = Decimal("0.000001")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class SnapshotConfig(_FrozenModel):
    corpus: str = Field(min_length=1)
    parse: str = Field(min_length=1)
    chunks: str = Field(min_length=1)
    embeddings: str = Field(min_length=1)
    questions: str = Field(min_length=1)


class RetrievalBinding(_FrozenModel):
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k: int = Field(gt=0)


class GenerationBinding(_FrozenModel):
    provider: Literal["upstage"]
    model_alias: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    prompt_version: Literal["v1", "v2", "v3"]
    temperature: float = Field(ge=0, allow_inf_nan=False)
    max_output_tokens: int = Field(gt=0)
    worst_case_input_tokens: int = Field(gt=0)


class EvaluationBinding(_FrozenModel):
    deterministic_version: str = Field(min_length=1)
    judge_model_id: str = Field(min_length=1)
    judge_prompt_version: str = Field(min_length=1)


class ConcurrencyEvidence(_FrozenModel):
    sample_size: int = Field(ge=100)
    error_window: int = Field(ge=20)
    provider_error_rate: float = Field(ge=0, le=0.02, allow_inf_nan=False)
    rate_limit_429: float = Field(ge=0, le=0.01, allow_inf_nan=False)
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class RuntimeConfig(_FrozenModel):
    concurrency: Literal[5, 10]
    max_retries: int = Field(ge=0)
    seed: int = Field(ge=0)
    batch_cap: int = Field(gt=0)
    budget_cap_usd: Decimal = Field(gt=0)
    schema_error_rate_stop: float = Field(gt=0, le=1, allow_inf_nan=False)
    provider_error_rate_stop: float = Field(gt=0, le=1, allow_inf_nan=False)
    error_window: int = Field(gt=0)
    concurrency_evidence: ConcurrencyEvidence | None = None

    @field_validator("budget_cap_usd", mode="before")
    @classmethod
    def _parse_decimal(cls, value: object) -> Decimal:
        if isinstance(value, (str, int, Decimal)) and not isinstance(value, bool):
            return Decimal(value)
        raise ValueError("budget cap must be an exact decimal")

    @model_validator(mode="after")
    def _validate_concurrency_evidence(self) -> Self:
        if self.concurrency == 10 and self.concurrency_evidence is None:
            raise ValueError("concurrency evidence is required before increasing to 10")
        if self.concurrency == 5 and self.concurrency_evidence is not None:
            raise ValueError("concurrency evidence is only valid for concurrency 10")
        return self


class ExperimentConfig(_FrozenModel):
    """Resolved semantic configuration; aliases and exact provider IDs are both retained."""

    schema_version: Literal["generation-experiment-v1"]
    name: str = Field(min_length=1)
    snapshots: SnapshotConfig
    retrieval: RetrievalBinding
    generation: GenerationBinding
    evaluation: EvaluationBinding
    runtime: RuntimeConfig
    question_ids: tuple[str, ...] = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    code_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")

    @field_validator("name", "output_dir", "code_commit")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identity cannot be blank")
        return value.strip()

    @field_validator("question_ids")
    @classmethod
    def _questions_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("question IDs cannot be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("duplicate question IDs are not allowed")
        return normalized

    @field_validator("question_ids", mode="before")
    @classmethod
    def _parse_question_sequence(cls, value: object) -> tuple[object, ...]:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        raise ValueError("question IDs must be a sequence")

    @model_validator(mode="after")
    def _batch_cap_is_bounded(self) -> Self:
        if self.runtime.batch_cap > len(self.question_ids):
            raise ValueError("batch cap cannot exceed question count")
        return self

    @property
    def semantic_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"name", "output_dir"})
        return canonical_json_hash(payload)

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        if not path.is_file() or path.is_symlink():
            raise ValueError("experiment config must be a regular non-symlink file")
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        if not isinstance(loaded, dict):
            raise ValueError("experiment config YAML root must be a mapping")
        return cls.model_validate(loaded)


class UsageSnapshot(_FrozenModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: Decimal = Field(ge=0)


class ExperimentQuestionResult(_FrozenModel):
    question_id: str = Field(min_length=1)
    response: dict[str, Any]
    evidence: tuple[dict[str, Any], ...]
    usage: UsageSnapshot
    cached: bool
    correlation_id: str | None = None


class DryRunPlan(_FrozenModel):
    config_hash: str
    plan_hash: str
    question_count: int
    cached_calls: int | None
    new_calls: int | None
    cache_status: Literal["known", "unknown"]
    worst_case_input_tokens: int
    worst_case_output_tokens: int
    worst_case_tokens: int
    net_worst_cost_usd: Decimal
    vat_multiplier: Decimal
    gross_worst_cost_usd: Decimal
    experiment_budget_cap_usd: Decimal
    concurrency: int
    model_alias: str
    model_id: str
    prompt_version: str
    destinations: tuple[str, ...]


def build_dry_run_plan(
    config: ExperimentConfig,
    *,
    price_book: PriceBook,
    cached_question_ids: set[str] | None = None,
    vat_multiplier: Decimal = Decimal("1.10"),
) -> DryRunPlan:
    """Create the exact operator-review artifact without performing provider work."""
    if vat_multiplier < 1:
        raise ValueError("VAT multiplier must be at least one")
    if cached_question_ids is None:
        cached_calls = new_calls = None
        priced_calls = len(config.question_ids)
        cache_status: Literal["known", "unknown"] = "unknown"
    else:
        unknown_ids = cached_question_ids.difference(config.question_ids)
        if unknown_ids:
            raise ValueError("cache probe returned IDs outside the question cohort")
        cached_calls = len(cached_question_ids)
        new_calls = len(config.question_ids) - cached_calls
        priced_calls = new_calls
        cache_status = "known"
    per_call = price_book.estimate(
        PricingRequest(
            operation="generate",
            model_id=config.generation.model_id,
            input_tokens=config.generation.worst_case_input_tokens,
            output_tokens=config.generation.max_output_tokens,
        )
    )
    net = (per_call * priced_calls).quantize(MONEY_QUANTUM, rounding=ROUND_CEILING)
    gross = (net * vat_multiplier).quantize(MONEY_QUANTUM, rounding=ROUND_CEILING)
    destination = str(Path(config.output_dir) / config.semantic_hash)
    worst_input = len(config.question_ids) * config.generation.worst_case_input_tokens
    worst_output = len(config.question_ids) * config.generation.max_output_tokens
    identity = {
        "config_hash": config.semantic_hash,
        "question_count": len(config.question_ids),
        "cached_calls": cached_calls,
        "new_calls": new_calls,
        "cache_status": cache_status,
        "worst_case_input_tokens": worst_input,
        "worst_case_output_tokens": worst_output,
        "net_worst_cost_usd": net,
        "vat_multiplier": vat_multiplier,
        "gross_worst_cost_usd": gross,
        "experiment_budget_cap_usd": config.runtime.budget_cap_usd,
        "concurrency": config.runtime.concurrency,
        "model_alias": config.generation.model_alias,
        "model_id": config.generation.model_id,
        "prompt_version": config.generation.prompt_version,
        "destinations": (destination,),
    }
    return DryRunPlan(
        config_hash=config.semantic_hash,
        plan_hash=canonical_json_hash({"schema": "experiment-dry-run-v1", **identity}),
        question_count=len(config.question_ids),
        cached_calls=cached_calls,
        new_calls=new_calls,
        cache_status=cache_status,
        worst_case_input_tokens=worst_input,
        worst_case_output_tokens=worst_output,
        worst_case_tokens=worst_input + worst_output,
        net_worst_cost_usd=net,
        vat_multiplier=vat_multiplier,
        gross_worst_cost_usd=gross,
        experiment_budget_cap_usd=config.runtime.budget_cap_usd,
        concurrency=config.runtime.concurrency,
        model_alias=config.generation.model_alias,
        model_id=config.generation.model_id,
        prompt_version=config.generation.prompt_version,
        destinations=(destination,),
    )


def authorize_paid_execution(
    plan: DryRunPlan,
    *,
    execute: bool,
    live_enabled: bool,
    confirmed_plan_hash: str | None,
    price_book: PriceBook,
    available_budget_usd: Decimal,
    now: datetime,
) -> bool:
    """Fail closed unless every independently verifiable paid-execution gate passes."""
    if not execute:
        raise PermissionError("--execute is required")
    if not live_enabled:
        raise PermissionError("live gate RUN_LIVE_UPSTAGE_TESTS=1 is required")
    if confirmed_plan_hash != plan.plan_hash:
        raise PermissionError("exact plan hash confirmation is required")
    price_book.verify_paid_batch(now=now)
    if plan.gross_worst_cost_usd >= available_budget_usd:
        raise PermissionError("worst-case cost reaches available budget")
    if plan.gross_worst_cost_usd >= plan.experiment_budget_cap_usd:
        raise PermissionError("worst-case cost reaches experiment budget cap")
    return True


class RunStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STOPPED = "stopped"


class ExperimentRun(_FrozenModel):
    run_id: str
    config_hash: str
    status: RunStatus
    completed_question_ids: frozenset[str] = frozenset()
    failed_question_ids: frozenset[str] = frozenset()
    stop_reason: str | None = None


_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PLANNED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.RUNNING: frozenset(
        {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.STOPPED}
    ),
    RunStatus.FAILED: frozenset({RunStatus.RUNNING}),
    RunStatus.CANCELLED: frozenset({RunStatus.RUNNING}),
    RunStatus.STOPPED: frozenset({RunStatus.RUNNING}),
    RunStatus.COMPLETED: frozenset(),
}


class FileExperimentRepository:
    """Filesystem repository with immutable config, result, error, and history artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def create(self, config: ExperimentConfig, *, now: datetime | None = None) -> ExperimentRun:
        timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        run_id = f"{config.semantic_hash}-{timestamp}"
        hash_root = self.root / config.semantic_hash
        if hash_root.exists():
            raise FileExistsError("semantic config hash already has an experiment run")
        run_root = hash_root / run_id
        run_root.mkdir(parents=True)
        full_config = config.model_dump(mode="json")
        _write_exclusive_json(run_root / "config.json", full_config)
        (run_root / "config.sha256").write_text(
            canonical_json_hash(full_config) + "\n", encoding="ascii"
        )
        run = ExperimentRun(
            run_id=run_id,
            config_hash=config.semantic_hash,
            status=RunStatus.PLANNED,
        )
        self._write_state(run_root, run)
        self._write_history(run_root, run)
        return run

    def load_config(self, run_id: str) -> ExperimentConfig:
        run_root = self._find(run_id)
        try:
            config_text = (run_root / "config.json").read_text(encoding="utf-8")
            config = ExperimentConfig.model_validate_json(config_text)
            artifact_hash = (run_root / "config.sha256").read_text(encoding="ascii").strip()
        except (OSError, ValidationError) as error:
            raise ValueError("experiment config was mutated after planning") from error
        expected = run_root.parent.name
        if (
            config.semantic_hash != expected
            or not run_id.startswith(f"{expected}-")
            or canonical_json_hash(config.model_dump(mode="json")) != artifact_hash
        ):
            raise ValueError("experiment config was mutated after planning")
        return config

    def load(self, run_id: str) -> ExperimentRun:
        self.load_config(run_id)
        return ExperimentRun.model_validate_json(
            (self._find(run_id) / "state.json").read_text(encoding="utf-8")
        )

    def history(self, run_id: str) -> tuple[ExperimentRun, ...]:
        root = self._find(run_id) / "history"
        return tuple(
            ExperimentRun.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(root.glob("*.json"))
        )

    def transition(
        self,
        run_id: str,
        status: RunStatus,
        *,
        completed: frozenset[str] | None = None,
        failed: frozenset[str] | None = None,
        stop_reason: str | None = None,
    ) -> ExperimentRun:
        current = self.load(run_id)
        if status not in _TRANSITIONS[current.status]:
            raise ValueError(f"unsafe run transition: {current.status} -> {status}")
        updated = ExperimentRun(
            run_id=current.run_id,
            config_hash=current.config_hash,
            status=status,
            completed_question_ids=(
                current.completed_question_ids if completed is None else completed
            ),
            failed_question_ids=current.failed_question_ids if failed is None else failed,
            stop_reason=stop_reason,
        )
        root = self._find(run_id)
        self._write_state(root, updated)
        self._write_history(root, updated)
        return updated

    def result(self, run_id: str, question_id: str) -> ExperimentQuestionResult:
        path = self._find(run_id) / "results" / f"{_safe_id(question_id)}.json"
        return ExperimentQuestionResult.model_validate_json(path.read_text(encoding="utf-8"))

    def result_ids(self, run_id: str) -> frozenset[str]:
        config = self.load_config(run_id)
        return frozenset(
            question_id
            for question_id in config.question_ids
            if (self._find(run_id) / "results" / f"{_safe_id(question_id)}.json").is_file()
        )

    def save_result(self, run_id: str, result: ExperimentQuestionResult) -> None:
        if result.question_id not in self.load_config(run_id).question_ids:
            raise ValueError("result question is outside the immutable cohort")
        path = self._find(run_id) / "results" / f"{_safe_id(result.question_id)}.json"
        _write_exclusive_json(path, result.model_dump(mode="json"))

    def save_error(self, run_id: str, question_id: str, error: BaseException) -> None:
        root = self._find(run_id) / "errors" / _safe_id(question_id)
        root.mkdir(parents=True, exist_ok=True)
        attempt = len(tuple(root.glob("*.json"))) + 1
        _write_exclusive_json(
            root / f"{attempt:04d}.json",
            {
                "question_id": question_id,
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )

    def _find(self, run_id: str) -> Path:
        matches = tuple(self.root.glob(f"*/{run_id}"))
        if len(matches) != 1:
            raise FileNotFoundError(f"unknown experiment run: {run_id}")
        return matches[0]

    def _write_state(self, root: Path, run: ExperimentRun) -> None:
        _write_replace_json(root / "state.json", run.model_dump(mode="json"))

    def _write_history(self, root: Path, run: ExperimentRun) -> None:
        history = root / "history"
        history.mkdir(exist_ok=True)
        sequence = len(tuple(history.glob("*.json")))
        _write_exclusive_json(
            history / f"{sequence:04d}-{run.status.value}.json", run.model_dump(mode="json")
        )


QuestionExecutor = Callable[[str], Awaitable[ExperimentQuestionResult]]


class ExperimentRunner:
    """Bounded per-question worker pool; provider access is injected gateway-backed work."""

    def __init__(self, repository: FileExperimentRepository, execute: QuestionExecutor) -> None:
        self._repository = repository
        self._execute = execute

    async def run(
        self,
        run_id: str,
        *,
        resume: bool = False,
        diagnosis_acknowledgement: str | None = None,
    ) -> ExperimentRun:
        current = self._repository.load(run_id)
        if current.status is RunStatus.COMPLETED:
            raise ValueError("completed experiments cannot be resumed")
        if current.status is not RunStatus.PLANNED and not resume:
            raise ValueError("non-planned experiment requires --resume")
        threshold_stop = current.stop_reason in {
            "schema_error_threshold",
            "provider_error_threshold",
        }
        if current.status is RunStatus.STOPPED and threshold_stop and not diagnosis_acknowledgement:
            raise PermissionError("threshold-stopped experiment requires diagnosis acknowledgement")
        config = self._repository.load_config(run_id)
        completed = set(self._repository.result_ids(run_id))
        failed: set[str] = set()
        self._repository.transition(
            run_id,
            RunStatus.RUNNING,
            completed=frozenset(completed),
            failed=frozenset(),
        )
        pending = [item for item in config.question_ids if item not in completed]
        pending = pending[: config.runtime.batch_cap]
        semaphore = asyncio.Semaphore(config.runtime.concurrency)
        stop = asyncio.Event()
        stop_reason: str | None = None
        recent: list[str] = []
        lock = asyncio.Lock()

        async def one(question_id: str) -> None:
            nonlocal stop_reason
            async with semaphore:
                if stop.is_set():
                    return
                try:
                    result = await self._execute(question_id)
                    if result.question_id != question_id:
                        raise ValueError("executor returned a result for the wrong question")
                    self._repository.save_result(run_id, result)
                    async with lock:
                        completed.add(question_id)
                        recent.append("ok")
                except asyncio.CancelledError:
                    raise
                except BudgetExceededError as error:
                    self._repository.save_error(run_id, question_id, error)
                    async with lock:
                        failed.add(question_id)
                        stop_reason = "budget_exhausted"
                        stop.set()
                except Exception as error:
                    self._repository.save_error(run_id, question_id, error)
                    kind = (
                        "schema"
                        if isinstance(error, (TypeError, ValueError))
                        else ("provider" if isinstance(error, ProviderHTTPError) else "other")
                    )
                    async with lock:
                        failed.add(question_id)
                        recent.append(kind)
                        window = recent[-config.runtime.error_window :]
                        if len(window) == config.runtime.error_window:
                            schema_rate = window.count("schema") / len(window)
                            provider_rate = window.count("provider") / len(window)
                            if schema_rate >= config.runtime.schema_error_rate_stop:
                                stop_reason = "schema_error_threshold"
                                stop.set()
                            elif provider_rate >= config.runtime.provider_error_rate_stop:
                                stop_reason = "provider_error_threshold"
                                stop.set()

        tasks = [asyncio.create_task(one(question_id)) for question_id in pending]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._repository.transition(
                run_id,
                RunStatus.CANCELLED,
                completed=frozenset(completed),
                failed=frozenset(failed),
                stop_reason="operator_cancelled",
            )
            raise
        if stop_reason is not None:
            return self._repository.transition(
                run_id,
                RunStatus.STOPPED,
                completed=frozenset(completed),
                failed=frozenset(failed),
                stop_reason=stop_reason,
            )
        if failed:
            return self._repository.transition(
                run_id,
                RunStatus.FAILED,
                completed=frozenset(completed),
                failed=frozenset(failed),
            )
        if len(completed) != len(config.question_ids):
            return self._repository.transition(
                run_id,
                RunStatus.STOPPED,
                completed=frozenset(completed),
                stop_reason="batch_cap",
            )
        return self._repository.transition(
            run_id,
            RunStatus.COMPLETED,
            completed=frozenset(completed),
            failed=frozenset(),
        )


def build_development_campaign(
    *,
    retrieval_config_hashes: Sequence[str],
    prompt_versions: Sequence[Literal["v1", "v2", "v3"]],
    question_ids: Sequence[str],
    base_config: ExperimentConfig,
) -> tuple[ExperimentConfig, ...]:
    """Plan, but never execute, the fixed top-8 × 3-prompt × 500-question campaign."""
    if len(retrieval_config_hashes) != 8 or len(set(retrieval_config_hashes)) != 8:
        raise ValueError("development campaign requires exactly eight retrieval configs")
    if tuple(prompt_versions) != ("v1", "v2", "v3"):
        raise ValueError("development campaign requires prompt versions v1, v2, and v3")
    if len(question_ids) != 500 or len(set(question_ids)) != 500:
        raise ValueError("development campaign requires exactly 500 unique questions")
    configs: list[ExperimentConfig] = []
    for retrieval_hash in retrieval_config_hashes:
        for prompt in prompt_versions:
            payload = base_config.model_dump(mode="json")
            payload["name"] = f"dev-{retrieval_hash[:12]}-{prompt}"
            payload["retrieval"] = {**payload["retrieval"], "config_hash": retrieval_hash}
            payload["generation"] = {**payload["generation"], "prompt_version": prompt}
            payload["question_ids"] = list(question_ids)
            payload["runtime"] = {**payload["runtime"], "batch_cap": 500}
            configs.append(ExperimentConfig.model_validate(payload))
    if len({item.semantic_hash for item in configs}) != 24:
        raise ValueError("development campaign contains duplicate semantic configurations")
    return tuple(configs)


def _safe_id(value: str) -> str:
    return canonical_json_hash(value)


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
        stream.write("\n")


def _write_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    output: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in output:
            raise ValueError(f"duplicate YAML key: {key}")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)
