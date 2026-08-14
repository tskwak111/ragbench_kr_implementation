"""Deterministic, aggregate-only analysis exports from immutable experiment evidence."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import html
import json
import math
import os
import secrets
import stat
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal, Self

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragbench.core.hashing import canonical_json_hash
from ragbench.evaluation.taxonomy import FailureCategory, FailureLabels

INSPECTION_ORDER = (
    "original_pdf",
    "standard_parse",
    "enhanced_parse",
    "chunks",
    "rankings",
    "answer",
    "citations",
)
MONEY_QUANTUM = Decimal("0.000001")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AnalysisConfig(_FrozenModel):
    experiment_id: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parse_mode: Literal["standard", "enhanced"]
    chunk_strategy: str = Field(min_length=1)
    retriever: Literal["dense", "bm25", "hybrid"]
    top_k: int = Field(gt=0)
    prompt_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _immutable_id_is_bound_to_config(self) -> Self:
        if not self.experiment_id.startswith(f"{self.config_hash}-"):
            raise ValueError("experiment ID is not bound to its immutable config hash")
        return self


class AnalysisResult(_FrozenModel):
    experiment_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    question_type: str = Field(min_length=1)
    correctness: Decimal = Field(ge=0, le=1)
    faithfulness: Decimal = Field(ge=0, le=1)
    citation: Decimal = Field(ge=0, le=1)
    abstention: Decimal = Field(ge=0, le=1)
    hit: Decimal = Field(ge=0, le=1)
    recall: Decimal = Field(ge=0, le=1)
    mrr: Decimal = Field(ge=0, le=1)
    latency_ms: int = Field(ge=0)

    @field_validator(
        "correctness", "faithfulness", "citation", "abstention", "hit", "recall", "mrr",
        mode="before",
    )
    @classmethod
    def _decimal_from_json(cls, value: object) -> Decimal:
        if isinstance(value, (str, int, Decimal)) and not isinstance(value, bool):
            return Decimal(value)
        raise ValueError("metric must be an exact decimal")


class AnalysisUsage(_FrozenModel):
    experiment_id: str = Field(min_length=1)
    question_id: str | None = None
    question_type: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    estimated_cost_usd: Decimal = Field(ge=0)
    cached: bool

    @field_validator("estimated_cost_usd", mode="before")
    @classmethod
    def _decimal_from_json(cls, value: object) -> Decimal:
        if isinstance(value, (str, int, Decimal)) and not isinstance(value, bool):
            return Decimal(value)
        raise ValueError("estimated cost must be an exact decimal")


class AnalysisFailure(_FrozenModel):
    experiment_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    question_type: str = Field(min_length=1)
    primary: FailureCategory
    secondary: FailureCategory | None = None

    @field_validator("primary", "secondary", mode="before")
    @classmethod
    def _category_from_json(cls, value: object) -> FailureCategory | None:
        if value is None or isinstance(value, FailureCategory):
            return value
        if isinstance(value, str):
            return FailureCategory(value)
        raise ValueError("failure category must use the frozen taxonomy")

    @model_validator(mode="after")
    def _taxonomy_contract(self) -> Self:
        FailureLabels(primary=self.primary, secondary=self.secondary)
        return self


class AnalysisBundle(_FrozenModel):
    """Clean-snapshot adapter contract; DB implementations only need to emit this shape."""

    schema_version: Literal["analysis-input-v1"]
    cohort_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_version: str = Field(min_length=1)
    code_version: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    configs: tuple[AnalysisConfig, ...]
    results: tuple[AnalysisResult, ...]
    usage: tuple[AnalysisUsage, ...]
    failures: tuple[AnalysisFailure, ...]

    @field_validator("configs", "results", "usage", "failures", mode="before")
    @classmethod
    def _sequence_from_json(cls, value: object) -> tuple[object, ...]:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        raise ValueError("analysis bundle collections must be sequences")

    @model_validator(mode="after")
    def _validate_relations(self) -> Self:
        experiment_ids = [row.experiment_id for row in self.configs]
        if not experiment_ids or len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("bundle requires unique immutable experiments")
        known = set(experiment_ids)
        referenced_experiments = (
            *(row.experiment_id for row in self.results),
            *(row.experiment_id for row in self.usage),
            *(row.experiment_id for row in self.failures),
        )
        for experiment_id in referenced_experiments:
            if experiment_id not in known:
                raise ValueError("row references an unknown immutable experiment")
        result_keys = {(row.experiment_id, row.question_id) for row in self.results}
        if len(result_keys) != len(self.results):
            raise ValueError("duplicate experiment-question results are not allowed")
        question_cohorts = {
            experiment_id: {
                row.question_id for row in self.results if row.experiment_id == experiment_id
            }
            for experiment_id in experiment_ids
        }
        if any(
            cohort != question_cohorts[experiment_ids[0]]
            for cohort in question_cohorts.values()
        ):
            raise ValueError("experiments do not share the exact question cohort")
        question_types: dict[str, str] = {}
        for result in self.results:
            existing_type = question_types.setdefault(result.question_id, result.question_type)
            if existing_type != result.question_type:
                raise ValueError("question type changed within the exact question cohort")
        for usage in self.usage:
            if usage.question_id is not None and (
                usage.experiment_id,
                usage.question_id,
            ) not in result_keys:
                raise ValueError("usage is outside the exact question cohort")
            if usage.question_id is not None and usage.question_type != question_types[
                usage.question_id
            ]:
                raise ValueError("usage question type does not match its bound result")
        for failure in self.failures:
            if (failure.experiment_id, failure.question_id) not in result_keys:
                raise ValueError("failure is not bound to a result in the immutable cohort")
            if failure.question_type != question_types[failure.question_id]:
                raise ValueError("failure question type does not match its bound result")
        return self


class ExportRequest(_FrozenModel):
    bundles: tuple[AnalysisBundle, ...] = Field(min_length=1)
    public_salt: str = Field(min_length=1)
    failure_sample_size: int = Field(default=50, ge=50, le=100)
    vat_multiplier: Decimal = Field(default=Decimal("1.10"), ge=1)
    console_gross_usd: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _same_lineage(self) -> Self:
        identities = {
            (bundle.cohort_hash, bundle.data_version, bundle.code_version)
            for bundle in self.bundles
        }
        if len(identities) != 1:
            raise ValueError("all exports must use the same cohort and lineage")
        experiment_ids = [
            config.experiment_id for bundle in self.bundles for config in bundle.configs
        ]
        if len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("immutable experiment IDs cannot be repeated across bundles")
        if not self.public_salt.strip():
            raise ValueError("public salt cannot be blank")
        return self


class FailureSampleItem(_FrozenModel):
    public_failure_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_config_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_type: str
    primary: FailureCategory
    secondary: FailureCategory | None
    inspection_order: tuple[str, ...]


class CostRow(_FrozenModel):
    public_config_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation: str
    model_id: str
    question_type: str
    cache_status: Literal["cached", "new"]
    call_count: int = Field(ge=0)
    net_cost_usd: Decimal = Field(ge=0)
    gross_cost_usd: Decimal = Field(ge=0)
    parse_amortized_cost_usd: Decimal = Field(ge=0)


class ReconciliationRow(_FrozenModel):
    status: Literal["not_reconciled", "reconciled"]
    local_gross_usd: Decimal
    console_gross_usd: Decimal | None
    delta_usd: Decimal | None


class ExportManifest(_FrozenModel):
    schema_version: Literal["analysis-export-manifest-v1"] = "analysis-export-manifest-v1"
    cohort_hash: str
    data_version: str
    code_version: str
    experiment_ids: tuple[str, ...]
    config_hashes: tuple[str, ...]
    tables: tuple[str, ...]
    figures: tuple[str, ...]
    files: dict[str, str]
    claim_status: Literal["PENDING_EVIDENCE"] = "PENDING_EVIDENCE"


def _public_config_id(salt: str, config_hash: str) -> str:
    return canonical_json_hash({"public_salt": salt, "config_hash": config_hash})


def plan_failure_sample(
    bundle: AnalysisBundle,
    *,
    sample_size: int,
    public_salt: str,
) -> tuple[FailureSampleItem, ...]:
    """Allocate a stable round-robin sample across system/type strata."""
    if not 50 <= sample_size <= 100:
        raise ValueError("failure sample size must be 50 through 100")
    if len(bundle.failures) < sample_size:
        raise ValueError("not enough failures for the requested stratified sample")
    configs = {config.experiment_id: config for config in bundle.configs}
    strata: dict[tuple[str, str], list[AnalysisFailure]] = defaultdict(list)
    for failure in bundle.failures:
        strata[(failure.experiment_id, failure.question_type)].append(failure)
    for rows in strata.values():
        rows.sort(
            key=lambda row: canonical_json_hash(
                {"salt": public_salt, "experiment": row.experiment_id, "question": row.question_id}
            )
        )
    selected: list[AnalysisFailure] = []
    ordered_strata = sorted(strata)
    while len(selected) < sample_size:
        progressed = False
        for stratum in ordered_strata:
            if strata[stratum] and len(selected) < sample_size:
                selected.append(strata[stratum].pop(0))
                progressed = True
        if not progressed:
            raise ValueError("not enough failures for the requested stratified sample")
    return tuple(
        FailureSampleItem(
            public_failure_id=canonical_json_hash(
                {
                    "salt": public_salt,
                    "experiment": row.experiment_id,
                    "question": row.question_id,
                }
            ),
            public_config_id=_public_config_id(
                public_salt, configs[row.experiment_id].config_hash
            ),
            question_type=row.question_type,
            primary=row.primary,
            secondary=row.secondary,
            inspection_order=INSPECTION_ORDER,
        )
        for row in selected
    )


def build_cost_rows(
    bundle: AnalysisBundle,
    *,
    vat_multiplier: Decimal,
    console_gross_usd: Decimal | None = None,
    public_salt: str = "analysis-cost-v1",
) -> tuple[tuple[CostRow, ...], ReconciliationRow]:
    """Aggregate provider estimates without treating them as reconciled billing truth."""
    if vat_multiplier < 1:
        raise ValueError("VAT multiplier must be at least one")
    configs = {config.experiment_id: config for config in bundle.configs}
    grouped: dict[tuple[str, str, str, str, bool], list[AnalysisUsage]] = defaultdict(list)
    parse_totals: dict[str, Decimal] = defaultdict(Decimal)
    question_counts = Counter(row.experiment_id for row in bundle.results)
    for usage in bundle.usage:
        grouped[
            (
                usage.experiment_id,
                usage.operation,
                usage.model_id,
                usage.question_type,
                usage.cached,
            )
        ].append(usage)
        if usage.operation == "parse" and not usage.cached:
            parse_totals[usage.experiment_id] += usage.estimated_cost_usd
    rows: list[CostRow] = []
    for key, usage_rows in sorted(grouped.items()):
        experiment_id, operation, model_id, question_type, cached = key
        net = sum((row.estimated_cost_usd for row in usage_rows), Decimal())
        net = net.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
        denominator = question_counts[experiment_id]
        amortized = Decimal()
        if operation == "generate" and denominator:
            covered_questions = {row.question_id for row in usage_rows if row.question_id}
            amortized = (
                parse_totals[experiment_id] * len(covered_questions) / denominator
            ).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
        rows.append(
            CostRow(
                public_config_id=_public_config_id(
                    public_salt, configs[experiment_id].config_hash
                ),
                operation=operation,
                model_id=model_id,
                question_type=question_type,
                cache_status="cached" if cached else "new",
                call_count=len(usage_rows),
                net_cost_usd=net,
                gross_cost_usd=(net * vat_multiplier).quantize(
                    MONEY_QUANTUM, rounding=ROUND_HALF_UP
                ),
                parse_amortized_cost_usd=amortized,
            )
        )
    local_gross = sum((row.gross_cost_usd for row in rows), Decimal()).quantize(
        MONEY_QUANTUM, rounding=ROUND_HALF_UP
    )
    reconciliation = ReconciliationRow(
        status="not_reconciled" if console_gross_usd is None else "reconciled",
        local_gross_usd=local_gross,
        console_gross_usd=console_gross_usd,
        delta_usd=(
            None
            if console_gross_usd is None
            else (console_gross_usd - local_gross).quantize(
                MONEY_QUANTUM, rounding=ROUND_HALF_UP
            )
        ),
    )
    return tuple(rows), reconciliation


def export_analysis(request: ExportRequest, output: Path) -> ExportManifest:
    """Generate a deterministic aggregate export and publish its manifest last."""
    output = Path(os.path.abspath(output))
    with _secure_parent(output) as parent_fd, _export_lock(parent_fd):
        _require_absent(parent_fd, output.name)
        staging_name = f".{output.name}-{secrets.token_hex(16)}.partial"
        os.mkdir(staging_name, 0o700, dir_fd=parent_fd)
        staging = output.parent / staging_name
        staging_fd = os.open(
            staging_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
        )
        staging_identity = os.fstat(staging_fd)
        try:
            os.mkdir("tables", 0o700, dir_fd=staging_fd)
            os.mkdir("figures", 0o700, dir_fd=staging_fd)
            tables_fd = os.open(
                "tables", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=staging_fd
            )
            figures_fd = os.open(
                "figures", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=staging_fd
            )
            bundle = _merge_bundles(request.bundles)
            tables = _build_tables(bundle, request)
            try:
                for name, rows in tables.items():
                    _write_csv_at(tables_fd, f"{name}.csv", rows)
                    _write_parquet_at(tables_fd, f"{name}.parquet", rows)
                figure_names = tuple(
                    name
                    for name in tables
                    if name
                    in {
                        "leaderboard",
                        "parse_paired_difference",
                        "chunk_heatmap",
                        "retriever_by_type",
                        "top_k_tradeoff",
                        "prompt_abstention",
                        "pareto_frontier",
                        "latency_distribution",
                        "failure_taxonomy",
                    }
                )
                for name in figure_names:
                    _write_figure_at(figures_fd, f"{name}.svg", name, tables[name])
            finally:
                os.close(tables_fd)
                os.close(figures_fd)
            _assert_directory_identity(staging, staging_identity)
            file_hashes = _hash_files(staging)
            configs = sorted(bundle.configs, key=lambda row: row.experiment_id)
            manifest = ExportManifest(
                cohort_hash=bundle.cohort_hash,
                data_version=bundle.data_version,
                code_version=bundle.code_version,
                experiment_ids=tuple(row.experiment_id for row in configs),
                config_hashes=tuple(row.config_hash for row in configs),
                tables=tuple(tables),
                figures=figure_names,
                files=file_hashes,
            )
            _write_json_at(staging_fd, "manifest.json", manifest.model_dump(mode="json"))
            _assert_directory_identity(staging, staging_identity)
            _require_absent(parent_fd, output.name)
            os.rename(
                staging_name,
                output.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            try:
                published_fd = os.open(
                    output.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
            except OSError as error:
                _unlink_unsafe_publication(parent_fd, output.name)
                raise ValueError("published analysis is not the staged directory") from error
            try:
                published = os.fstat(published_fd)
                if (
                    published.st_dev != staging_identity.st_dev
                    or published.st_ino != staging_identity.st_ino
                ):
                    _unlink_unsafe_publication(parent_fd, output.name)
                    raise ValueError("published analysis identity differs from staging")
            finally:
                os.close(published_fd)
            os.fsync(parent_fd)
            return manifest
        finally:
            os.close(staging_fd)
            if staging.exists() and not staging.is_symlink():
                for path in sorted(staging.rglob("*"), reverse=True):
                    if path.is_file() and not path.is_symlink():
                        path.unlink()
                    elif path.is_dir() and not path.is_symlink():
                        path.rmdir()
                staging.rmdir()


def _merge_bundles(bundles: Sequence[AnalysisBundle]) -> AnalysisBundle:
    first = bundles[0]
    return AnalysisBundle(
        schema_version="analysis-input-v1",
        cohort_hash=first.cohort_hash,
        data_version=first.data_version,
        code_version=first.code_version,
        configs=tuple(config for bundle in bundles for config in bundle.configs),
        results=tuple(result for bundle in bundles for result in bundle.results),
        usage=tuple(usage for bundle in bundles for usage in bundle.usage),
        failures=tuple(failure for bundle in bundles for failure in bundle.failures),
    )


def _build_tables(
    bundle: AnalysisBundle, request: ExportRequest
) -> dict[str, list[dict[str, Any]]]:
    result_groups: dict[str, list[AnalysisResult]] = defaultdict(list)
    for result in bundle.results:
        result_groups[result.experiment_id].append(result)
    leaderboard: list[dict[str, Any]] = []
    for config in sorted(bundle.configs, key=lambda row: row.config_hash):
        rows = result_groups[config.experiment_id]
        leaderboard.append(
            {
                "public_config_id": _public_config_id(request.public_salt, config.config_hash),
                "parse_mode": config.parse_mode,
                "chunk_strategy": config.chunk_strategy,
                "retriever": config.retriever,
                "top_k": config.top_k,
                "prompt_version": config.prompt_version,
                "model_id": config.model_id,
                "question_count": len(rows),
                **{
                    f"mean_{metric}": _mean(getattr(row, metric) for row in rows)
                    for metric in (
                        "correctness",
                        "faithfulness",
                        "citation",
                        "abstention",
                        "hit",
                        "recall",
                        "mrr",
                    )
                },
                "mean_latency_ms": _mean(Decimal(row.latency_ms) for row in rows),
            }
        )
    parse_pairs = _paired_parse_rows(bundle, request.public_salt)
    chunk_heatmap = _aggregate_axes(bundle, request.public_salt, ("chunk_strategy", "parse_mode"))
    retriever_by_type = _aggregate_axes(
        bundle, request.public_salt, ("retriever", "question_type")
    )
    top_k = _aggregate_axes(bundle, request.public_salt, ("top_k",))
    prompt = _aggregate_axes(bundle, request.public_salt, ("prompt_version",))
    latency = _latency_rows(bundle, request.public_salt)
    failure_taxonomy = _failure_taxonomy_rows(bundle, request.public_salt)
    sample = plan_failure_sample(
        bundle, sample_size=request.failure_sample_size, public_salt=request.public_salt
    )
    cost_rows, reconciliation = build_cost_rows(
        bundle,
        vat_multiplier=request.vat_multiplier,
        console_gross_usd=request.console_gross_usd,
        public_salt=request.public_salt,
    )
    cost_by_config: dict[str, Decimal] = defaultdict(Decimal)
    for cost_row in cost_rows:
        cost_by_config[cost_row.public_config_id] += cost_row.gross_cost_usd
    pareto: list[dict[str, Any]] = []
    for leaderboard_row in leaderboard:
        public_id = str(leaderboard_row["public_config_id"])
        cost = cost_by_config[public_id]
        quality = Decimal(str(leaderboard_row["mean_correctness"]))
        pareto.append(
            {
                "public_config_id": public_id,
                "quality": quality,
                "gross_cost_usd": cost,
                "pareto": _is_pareto(public_id, quality, cost, leaderboard, cost_by_config),
            }
        )
    marginal = _marginal_rows(pareto)
    return {
        "leaderboard": leaderboard,
        "parse_paired_difference": parse_pairs,
        "chunk_heatmap": chunk_heatmap,
        "retriever_by_type": retriever_by_type,
        "top_k_tradeoff": top_k,
        "prompt_abstention": prompt,
        "pareto_frontier": pareto,
        "latency_distribution": latency,
        "failure_taxonomy": failure_taxonomy,
        "failure_sample_plan": [row.model_dump(mode="json") for row in sample],
        "cost_breakdown": [row.model_dump(mode="json") for row in cost_rows],
        "marginal_cost_quality": marginal,
        "reconciliation": [reconciliation.model_dump(mode="json")],
    }


def _paired_parse_rows(bundle: AnalysisBundle, salt: str) -> list[dict[str, Any]]:
    configs = {row.experiment_id: row for row in bundle.configs}
    grouped: dict[tuple[str, str, int, str, str], dict[str, str]] = defaultdict(dict)
    for config in bundle.configs:
        binding_key = (
            config.chunk_strategy,
            config.retriever,
            config.top_k,
            config.prompt_version,
            config.model_id,
        )
        grouped[binding_key][config.parse_mode] = config.experiment_id
    result_lookup = {
        (row.experiment_id, row.question_id): row for row in bundle.results
    }
    output: list[dict[str, Any]] = []
    for pair_key, pair in sorted(grouped.items()):
        if set(pair) != {"standard", "enhanced"}:
            continue
        standard_ids = {
            row.question_id for row in bundle.results if row.experiment_id == pair["standard"]
        }
        enhanced_ids = {
            row.question_id for row in bundle.results if row.experiment_id == pair["enhanced"]
        }
        if standard_ids != enhanced_ids:
            raise ValueError("Standard and Enhanced paired cohorts differ")
        differences = [
            result_lookup[(pair["enhanced"], question)].correctness
            - result_lookup[(pair["standard"], question)].correctness
            for question in sorted(standard_ids)
        ]
        mean = _mean(differences)
        low, high = _normal_interval(differences)
        output.append(
            {
                "pair_id": canonical_json_hash(
                    {
                        "salt": salt,
                        "standard": configs[pair["standard"]].config_hash,
                        "enhanced": configs[pair["enhanced"]].config_hash,
                    }
                ),
                "chunk_strategy": pair_key[0],
                "retriever": pair_key[1],
                "top_k": pair_key[2],
                "prompt_version": pair_key[3],
                "model_id": pair_key[4],
                "sample_count": len(differences),
                "correctness_effect": mean,
                "ci_low": low,
                "ci_high": high,
                "confidence": Decimal("0.95"),
            }
        )
    return output or [{"status": "PENDING_EVIDENCE", "reason": "no matched parse pair"}]


def _aggregate_axes(
    bundle: AnalysisBundle, salt: str, axes: tuple[str, ...]
) -> list[dict[str, Any]]:
    configs = {row.experiment_id: row for row in bundle.configs}
    grouped: dict[tuple[Any, ...], list[AnalysisResult]] = defaultdict(list)
    for row in bundle.results:
        config = configs[row.experiment_id]
        values = tuple(
            row.question_type if axis == "question_type" else getattr(config, axis)
            for axis in axes
        )
        grouped[values].append(row)
    return [
        {
            **dict(zip(axes, key, strict=True)),
            "question_count": len(rows),
            "mean_correctness": _mean(row.correctness for row in rows),
            "mean_abstention": _mean(row.abstention for row in rows),
            "mean_recall": _mean(row.recall for row in rows),
            "mean_latency_ms": _mean(Decimal(row.latency_ms) for row in rows),
            "cohort_public_id": canonical_json_hash({"salt": salt, "cohort": bundle.cohort_hash}),
        }
        for key, rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0])))
    ]


def _latency_rows(bundle: AnalysisBundle, salt: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    configs = {row.experiment_id: row for row in bundle.configs}
    for row in bundle.results:
        grouped[row.experiment_id].append(row.latency_ms)
    return [
        {
            "public_config_id": _public_config_id(salt, configs[experiment].config_hash),
            "count": len(values),
            "p50_ms": _percentile(values, Decimal("0.50")),
            "p95_ms": _percentile(values, Decimal("0.95")),
            "p99_ms": _percentile(values, Decimal("0.99")),
        }
        for experiment, values in sorted(grouped.items())
    ]


def _failure_taxonomy_rows(bundle: AnalysisBundle, salt: str) -> list[dict[str, Any]]:
    configs = {row.experiment_id: row for row in bundle.configs}
    counts = Counter(
        (
            _public_config_id(salt, configs[row.experiment_id].config_hash),
            row.question_type,
            row.primary.value,
            row.secondary.value if row.secondary else "",
        )
        for row in bundle.failures
    )
    return [
        {
            "public_config_id": key[0],
            "question_type": key[1],
            "primary": key[2],
            "secondary": key[3],
            "count": count,
        }
        for key, count in sorted(counts.items())
    ]


def _is_pareto(
    public_id: str,
    quality: Decimal,
    cost: Decimal,
    leaderboard: Sequence[Mapping[str, Any]],
    costs: Mapping[str, Decimal],
) -> bool:
    for candidate in leaderboard:
        candidate_id = str(candidate["public_config_id"])
        if candidate_id == public_id:
            continue
        candidate_quality = Decimal(str(candidate["mean_correctness"]))
        candidate_cost = costs[candidate_id]
        if (
            candidate_quality >= quality
            and candidate_cost <= cost
            and (candidate_quality > quality or candidate_cost < cost)
        ):
            return False
    return True


def _marginal_rows(pareto_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frontier = sorted(
        (row for row in pareto_rows if row["pareto"] is True),
        key=lambda row: (Decimal(str(row["quality"])), Decimal(str(row["gross_cost_usd"]))),
    )
    output: list[dict[str, Any]] = []
    baseline: Mapping[str, Any] | None = None
    for row in frontier:
        quality = Decimal(str(row["quality"]))
        cost = Decimal(str(row["gross_cost_usd"]))
        delta_quality = (
            None if baseline is None else quality - Decimal(str(baseline["quality"]))
        )
        delta_cost = (
            None if baseline is None else cost - Decimal(str(baseline["gross_cost_usd"]))
        )
        output.append(
            {
                "public_config_id": row["public_config_id"],
                "baseline_public_config_id": (
                    None if baseline is None else baseline["public_config_id"]
                ),
                "gross_cost_usd": cost,
                "quality": quality,
                "delta_cost_usd": delta_cost,
                "delta_quality_points": (
                    None if delta_quality is None else delta_quality * 100
                ),
                "marginal_cost_per_quality_point_usd": (
                    None
                    if delta_quality is None or delta_cost is None or delta_quality <= 0
                    else (delta_cost / (delta_quality * 100)).quantize(
                        MONEY_QUANTUM, rounding=ROUND_HALF_UP
                    )
                ),
            }
        )
        baseline = row
    return output


def _mean(values: Iterable[Decimal]) -> Decimal:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot aggregate an empty cohort")
    return (sum(materialized, Decimal()) / len(materialized)).quantize(
        MONEY_QUANTUM, rounding=ROUND_HALF_UP
    )


def _normal_interval(values: Sequence[Decimal]) -> tuple[Decimal, Decimal]:
    if len(values) < 2:
        return values[0], values[0]
    floats = [float(value) for value in values]
    mean = sum(floats) / len(floats)
    variance = sum((value - mean) ** 2 for value in floats) / (len(floats) - 1)
    margin = Decimal(str(1.96 * math.sqrt(variance / len(floats))))
    center = Decimal(str(mean))
    return (
        (center - margin).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP),
        (center + margin).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP),
    )


def _percentile(values: Sequence[int], quantile: Decimal) -> Decimal:
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    value = ordered[lower] if lower == upper else ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * (position - lower)
    return Decimal(str(value)).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def _normalize_cell(value: Any) -> str | int | bool | None:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, FailureCategory):
        return value.value
    if isinstance(value, (tuple, list)):
        return json.dumps(list(value), ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _exclusive_descriptor(directory_fd: int, name: str) -> int:
    if name in {"", ".", ".."} or os.sep in name:
        raise ValueError("artifact name must be a direct child")
    return os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )


def _write_csv_at(
    directory_fd: int, name: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    if not rows:
        raise ValueError(f"table {name} cannot be empty")
    fields = tuple(rows[0])
    descriptor = _exclusive_descriptor(directory_fd, name)
    with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _normalize_cell(row.get(key)) for key in fields})
        stream.flush()
        os.fsync(stream.fileno())


def _write_parquet_at(
    directory_fd: int, name: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    normalized = [
        {key: _normalize_cell(value) for key, value in row.items()} for row in rows
    ]
    table = pa.Table.from_pylist(normalized)
    descriptor = _exclusive_descriptor(directory_fd, name)
    with os.fdopen(descriptor, "wb") as stream:
        pq.write_table(table, stream, compression="zstd", write_statistics=True)
        stream.flush()
        os.fsync(stream.fileno())


_FIGURE_SPEC: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    "leaderboard": (("public_config_id",), ("mean_correctness",), "Correctness"),
    "parse_paired_difference": (
        ("parse_mode", "pair_id"),
        ("correctness_effect",),
        "Enhanced - Standard correctness",
    ),
    "chunk_heatmap": (
        ("chunk_strategy", "parse_mode"),
        ("mean_correctness",),
        "Mean correctness",
    ),
    "retriever_by_type": (
        ("retriever", "question_type"),
        ("mean_correctness",),
        "Mean correctness",
    ),
    "top_k_tradeoff": (("top_k",), ("mean_correctness", "mean_latency_ms"), "Quality / latency"),
    "prompt_abstention": (
        ("prompt_version",),
        ("mean_correctness", "mean_abstention"),
        "Correctness / abstention accuracy",
    ),
    "pareto_frontier": (
        ("public_config_id",),
        ("quality", "gross_cost_usd"),
        "Quality / gross cost (VAT included)",
    ),
    "latency_distribution": (
        ("public_config_id",),
        ("p50_ms", "p95_ms", "p99_ms"),
        "Latency percentiles (ms)",
    ),
    "failure_taxonomy": (
        ("primary", "question_type"),
        ("count",),
        "Failure count",
    ),
}


def _write_figure_at(
    directory_fd: int,
    name: str,
    title: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    label_fields, measures, axis_label = _FIGURE_SPEC[title]
    geometry = {
        "chunk_heatmap": "heatmap",
        "pareto_frontier": "scatter",
        "top_k_tradeoff": "dual-axis",
        "parse_paired_difference": "signed-bars",
    }.get(title, "grouped-bars")
    selected = rows
    if geometry == "heatmap":
        payload = _render_heatmap(title, selected, label_fields, measures[0], axis_label)
        _write_text_at(directory_fd, name, payload)
        return
    if geometry == "scatter":
        payload = _render_scatter(title, selected, measures, axis_label)
        _write_text_at(directory_fd, name, payload)
        return
    if geometry == "dual-axis":
        payload = _render_dual_axis(title, selected, label_fields, measures, axis_label)
        _write_text_at(directory_fd, name, payload)
        return
    series: list[tuple[str, str, float]] = []
    for row in selected:
        label = " / ".join(
            str(row.get(field, "all"))[:20] for field in label_fields if row.get(field) is not None
        )
        for measure in measures:
            value = row.get(measure)
            if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
                series.append((label or "all", measure, float(value)))
    absolute_maximum = max((abs(value) for _, _, value in series), default=1.0) or 1.0
    signed = geometry == "signed-bars"
    origin = 410 if signed else 210
    maximum_width = 320 if signed else 430
    bars = []
    colors = ("#355c7d", "#c06c84", "#6c8e68")
    for index, (label, measure, value) in enumerate(series):
        y = 72 + index * 24
        width = maximum_width * abs(value) / absolute_maximum
        x = origin - width if signed and value < 0 else origin
        bars.append(
            f'<text x="10" y="{y + 13}" font-size="10">{html.escape(label)}</text>'
            f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="16" '
            f'fill="{colors[measures.index(measure) % len(colors)]}"/>'
            f'<text x="{origin + width + 5:.2f}" y="{y + 12}" font-size="9">'
            f'{html.escape(measure)} {value:.4g}</text>'
        )
    height = 108 + 24 * (len(bars) + int(signed))
    if signed:
        bars.insert(
            0,
            f'<line x1="{origin}" y1="67" x2="{origin}" y2="{height}" stroke="#111"/>',
        )
    payload = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="820" height="{height}" '
        f'viewBox="0 0 820 {height}" data-measures="{html.escape(",".join(measures))}" '
        f'data-geometry="{geometry}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="10" y="28" font-size="20" font-family="sans-serif">{html.escape(title)}</text>'
        f'<text x="10" y="47" font-size="11" fill="#444">{html.escape(axis_label)}</text>'
        '<text x="10" y="62" font-size="9" fill="#666">aggregate public-safe data</text>'
        + "".join(bars)
        + "</svg>\n"
    )
    _write_text_at(directory_fd, name, payload)


def _render_heatmap(
    title: str,
    rows: Sequence[Mapping[str, Any]],
    label_fields: tuple[str, ...],
    measure: str,
    axis_label: str,
) -> str:
    x_values = sorted({str(row[label_fields[0]]) for row in rows})
    y_values = sorted({str(row[label_fields[1]]) for row in rows})
    values = [float(row[measure]) for row in rows]
    minimum, maximum = min(values), max(values)
    cells = []
    for row in rows:
        x_index = x_values.index(str(row[label_fields[0]]))
        y_index = y_values.index(str(row[label_fields[1]]))
        value = float(row[measure])
        ratio = 0.5 if maximum == minimum else (value - minimum) / (maximum - minimum)
        blue = int(245 - 145 * ratio)
        cells.append(
            f'<rect x="{180 + x_index * 180}" y="{80 + y_index * 55}" width="170" '
            f'height="45" fill="rgb({blue},{blue},245)"/>'
            f'<text x="{188 + x_index * 180}" y="{108 + y_index * 55}" font-size="11">'
            f'{value:.4g}</text>'
        )
    labels = [
        f'<text x="{188 + index * 180}" y="70" font-size="10">{html.escape(value)}</text>'
        for index, value in enumerate(x_values)
    ] + [
        f'<text x="10" y="{108 + index * 55}" font-size="10">{html.escape(value)}</text>'
        for index, value in enumerate(y_values)
    ]
    return _render_svg_document(
        title,
        measure,
        "heatmap",
        axis_label,
        cells + labels,
        760,
        150 + 55 * len(y_values),
    )


def _render_scatter(
    title: str,
    rows: Sequence[Mapping[str, Any]],
    measures: tuple[str, ...],
    axis_label: str,
) -> str:
    xs = [float(row[measures[1]]) for row in rows]
    ys = [float(row[measures[0]]) for row in rows]
    max_x = max(xs, default=1) or 1
    min_y, max_y = min(ys, default=0), max(ys, default=1)
    span_y = max_y - min_y or 1
    ordered = sorted(zip(xs, ys, strict=True))
    points = [
        f'<circle cx="{80 + 620 * x / max_x:.2f}" cy="{340 - 260 * (y - min_y) / span_y:.2f}" '
        'r="6" fill="#c06c84"/>'
        for x, y in ordered
    ]
    line_points = " ".join(
        f'{80 + 620 * x / max_x:.2f},{340 - 260 * (y - min_y) / span_y:.2f}'
        for x, y in ordered
    )
    points.append(f'<polyline points="{line_points}" fill="none" stroke="#355c7d"/>')
    return _render_svg_document(
        title, ",".join(measures), "scatter", axis_label, points, 780, 390
    )


def _render_dual_axis(
    title: str,
    rows: Sequence[Mapping[str, Any]],
    label_fields: tuple[str, ...],
    measures: tuple[str, ...],
    axis_label: str,
) -> str:
    ordered = sorted(rows, key=lambda row: float(row[label_fields[0]]))
    paths = []
    colors = ("#355c7d", "#c06c84")
    for measure_index, measure in enumerate(measures):
        values = [float(row[measure]) for row in ordered]
        minimum, maximum = min(values), max(values)
        span = maximum - minimum or 1
        points = " ".join(
            f'{80 + index * (600 / max(1, len(values) - 1)):.2f},'
            f'{330 - 240 * (value - minimum) / span:.2f}'
            for index, value in enumerate(values)
        )
        paths.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[measure_index]}" '
            f'stroke-width="3"/><text x="610" y="{35 + measure_index * 16}" '
            f'fill="{colors[measure_index]}" font-size="10">{html.escape(measure)}</text>'
        )
    return _render_svg_document(
        title, ",".join(measures), "dual-axis", axis_label, paths, 780, 380
    )


def _render_svg_document(
    title: str,
    measures: str,
    geometry: str,
    subtitle: str,
    shapes: Sequence[str],
    width: int,
    height: int,
) -> str:
    payload = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" data-measures="{html.escape(measures)}" '
        f'data-geometry="{geometry}"><rect width="100%" height="100%" fill="white"/>'
        f'<text x="10" y="28" font-size="20">{html.escape(title)}</text>'
        f'<text x="10" y="47" font-size="11">{html.escape(subtitle)}</text>'
        + "".join(shapes)
        + "</svg>\n"
    )
    return payload


def _hash_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _write_text_at(directory_fd: int, name: str, payload: str) -> None:
    descriptor = _exclusive_descriptor(directory_fd, name)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_at(directory_fd: int, name: str, payload: object) -> None:
    _write_text_at(
        directory_fd,
        name,
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
    )


@contextmanager
def _secure_parent(path: Path) -> Iterator[int]:
    """Open the existing parent component-by-component without following links."""
    if not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY")):
        raise RuntimeError("secure POSIX no-follow primitives are unavailable")
    descriptor = os.open(os.sep, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            for component in path.parent.parts[1:]:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                os.close(descriptor)
                descriptor = next_descriptor
        except OSError as error:
            raise ValueError("analysis output parent contains an unsafe path component") from error
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise PermissionError("analysis output parent must be an EUID-owned directory")
        if stat.S_IMODE(metadata.st_mode) & 0o022:
            raise PermissionError("analysis output parent cannot be group/world writable")
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _export_lock(parent_fd: int) -> Iterator[None]:
    descriptor = os.open(
        ".ragbench-analysis-export.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_fd,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _require_absent(parent_fd: int, name: str) -> None:
    if name in {"", ".", ".."} or os.sep in name:
        raise ValueError("analysis output must be a direct child path")
    try:
        metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("analysis output cannot be a symlink")
    raise FileExistsError(f"immutable analysis output already exists: {name}")


def _assert_directory_identity(path: Path, expected: os.stat_result) -> None:
    actual = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(actual.st_mode)
        or actual.st_dev != expected.st_dev
        or actual.st_ino != expected.st_ino
    ):
        raise ValueError("analysis staging directory identity changed during export")


def _unlink_unsafe_publication(parent_fd: int, name: str) -> None:
    """Remove only a non-directory entry introduced by a detected publication race."""
    metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(metadata.st_mode) or stat.S_ISREG(metadata.st_mode):
        os.unlink(name, dir_fd=parent_fd)
