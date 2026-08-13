"""Deterministic planning and gated, resumable synthetic benchmark generation."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from ragbench.chunking.tokenizer import encoding
from ragbench.core.hashing import canonical_json_hash
from ragbench.providers.base import GenerateRequest, ProviderGateway
from ragbench.providers.upstage.pricing import PriceBook, PriceBookError, PricingRequest


class QuestionType(StrEnum):
    FACT = "fact"
    NUMERIC_TABLE = "numeric_table"
    COMPARISON = "comparison"
    MULTIHOP = "multihop"
    UNANSWERABLE = "unanswerable"
    COMPLEX_SUMMARY = "complex_summary"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ValidationDecision(StrEnum):
    UNVALIDATED = "unvalidated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


DEFAULT_QUOTAS: dict[QuestionType, int] = {
    QuestionType.FACT: 300,
    QuestionType.NUMERIC_TABLE: 300,
    QuestionType.COMPARISON: 250,
    QuestionType.MULTIHOP: 250,
    QuestionType.UNANSWERABLE: 200,
    QuestionType.COMPLEX_SUMMARY: 200,
}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class EvidenceSpan(_FrozenModel):
    text: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    page: int = Field(gt=0)
    chunk_id: str = Field(min_length=1)

    @field_validator("text", "document_id", "chunk_id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence provenance cannot be blank")
        return value


class GeneratorMetadata(_FrozenModel):
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    batch_id: str = Field(min_length=1)
    source_window_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reasoning_kind: str = Field(min_length=1)
    correlation_id: str | None = None
    cache_hit: bool | None = None


class UnanswerableTransform(_FrozenModel):
    target_document_id: str = Field(min_length=1)
    original_fact: str = Field(min_length=1)
    transformed_fact: str = Field(min_length=1)
    operation: str = Field(default="controlled_substitution", min_length=1)

    @model_validator(mode="after")
    def _facts_must_differ(self) -> Self:
        if _search_text(self.original_fact) == _search_text(self.transformed_fact):
            raise ValueError("unanswerable transform must change the source fact")
        if not _fact_anchors(self.original_fact).intersection(
            _fact_anchors(self.transformed_fact)
        ):
            raise ValueError("unanswerable transform must preserve a source-fact anchor")
        return self


class ValidationStatus(_FrozenModel):
    decision: ValidationDecision
    rule_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _decision_matches_rules(self) -> Self:
        if self.decision is ValidationDecision.ACCEPTED and self.rule_codes:
            raise ValueError("accepted validation cannot contain rejection rules")
        if self.decision is ValidationDecision.REJECTED and not self.rule_codes:
            raise ValueError("rejected validation requires at least one rule")
        return self


class QuestionCandidate(_FrozenModel):
    """One immutable provider candidate with complete evidence lineage."""

    candidate_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold_answer: str | None
    evidence_spans: tuple[EvidenceSpan, ...]
    question_type: QuestionType
    difficulty: Difficulty
    answerable: bool
    asserted_absent_facts: tuple[str, ...] = ()
    unanswerable_transform: UnanswerableTransform | None = None
    generator: GeneratorMetadata
    validation: ValidationStatus

    @field_validator("candidate_id", "question")
    @classmethod
    def _identity_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("candidate identity and question cannot be blank")
        return value.strip()

    @field_validator("gold_answer")
    @classmethod
    def _normalize_answer(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(unicodedata.normalize("NFKC", value).split())
        return normalized or None

    @field_validator("asserted_absent_facts")
    @classmethod
    def _absence_facts_not_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("absence assertions cannot be blank")
        return value

    @model_validator(mode="after")
    def _answerability_shape(self) -> Self:
        if self.answerable:
            if self.question_type is QuestionType.UNANSWERABLE:
                raise ValueError("answerable candidate cannot use unanswerable type")
            if self.gold_answer is None or not self.evidence_spans:
                raise ValueError("answerable candidate requires gold answer and evidence")
            if self.asserted_absent_facts:
                raise ValueError("answerable candidate cannot assert absent facts")
            if self.unanswerable_transform is not None:
                raise ValueError("answerable candidate cannot contain a negative transform")
        else:
            if self.question_type is not QuestionType.UNANSWERABLE:
                raise ValueError("unanswerable candidate requires unanswerable type")
            if self.gold_answer is not None or self.evidence_spans:
                raise ValueError("unanswerable candidate cannot contain answer or evidence")
            if not self.asserted_absent_facts:
                raise ValueError("unanswerable candidate requires absence assertions")
            if self.unanswerable_transform is None:
                raise ValueError("unanswerable candidate requires controlled transformation")
            if self.asserted_absent_facts != (self.unanswerable_transform.transformed_fact,):
                raise ValueError("absence assertion must equal the transformed fact")
        return self


class CandidateBatchEnvelope(_FrozenModel):
    candidates: tuple[QuestionCandidate, ...]


class SourceUnit(_FrozenModel):
    page: int = Field(gt=0)
    chunk_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class SourceWindow(_FrozenModel):
    window_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    chunk_ids: tuple[str, ...] = Field(min_length=1)
    content: str = Field(min_length=1)
    source_units: tuple[SourceUnit, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_window(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("source window page range is invalid")
        if any(not value.strip() for value in (*self.chunk_ids, self.content)):
            raise ValueError("source window fields cannot be blank")
        unit_chunks = tuple(dict.fromkeys(unit.chunk_id for unit in self.source_units))
        if unit_chunks != self.chunk_ids:
            raise ValueError("source units must exactly cover window chunk IDs")
        if any(not self.page_start <= unit.page <= self.page_end for unit in self.source_units):
            raise ValueError("source unit page is outside the window range")
        unit_content = "\n".join(unit.content for unit in self.source_units)
        if _search_text(self.content) != _search_text(unit_content):
            raise ValueError("window content must be the ordered source-unit content")
        return self


class GenerationConfig(_FrozenModel):
    quotas: Mapping[QuestionType, int] = Field(
        default_factory=lambda: dict(DEFAULT_QUOTAS)
    )
    per_document_cap: int = Field(default=300, gt=0)
    per_page_cap: int = Field(default=30, gt=0)
    batch_size: int = Field(default=12, gt=0, le=100)
    source_window_max_chars: int = Field(default=20_000, gt=0)
    max_output_tokens: int = Field(default=8_000, gt=0)
    prompt_version: str = "benchmark-v1"
    normal_completion_floor: int = 1_000
    emergency_floor: int = 800

    @field_validator("quotas", mode="before")
    @classmethod
    def _freeze_quotas(cls, value: object) -> Mapping[QuestionType, int]:
        if not isinstance(value, Mapping):
            raise ValueError("quotas must be a mapping")
        quotas: dict[QuestionType, int] = {}
        for key, count in value.items():
            kind = key if isinstance(key, QuestionType) else QuestionType(str(key))
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("quota counts must be positive integers")
            quotas[kind] = count
        return quotas

    @model_validator(mode="after")
    def _completion_floors_are_ordered(self) -> Self:
        if not (0 < self.emergency_floor < self.normal_completion_floor):
            raise ValueError("completion floors must be ordered")
        return self


class GenerationJob(_FrozenModel):
    ordinal: int = Field(ge=0)
    question_type: QuestionType
    window: SourceWindow


class GenerationBatch(_FrozenModel):
    batch_id: str
    jobs: tuple[GenerationJob, ...]


class GenerationPlan(_FrozenModel):
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_snapshot_id: str
    model_id: str
    prompt_version: str
    jobs: tuple[GenerationJob, ...]
    batches: tuple[GenerationBatch, ...]


class GenerationPlanner:
    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()

    def plan(
        self,
        windows: Sequence[SourceWindow],
        *,
        corpus_snapshot_id: str,
        model_id: str,
    ) -> GenerationPlan:
        ordered = tuple(
            sorted(
                windows,
                key=lambda row: (
                    row.document_id,
                    row.page_start,
                    row.page_end,
                    row.window_id,
                ),
            )
        )
        if not ordered:
            raise ValueError("generation requires source windows")
        if any(len(window.content) > self.config.source_window_max_chars for window in ordered):
            raise ValueError("all source windows must be bounded before planning")
        target = sum(self.config.quotas.values())
        doc_capacity = len({item.document_id for item in ordered}) * self.config.per_document_cap
        source_pages = {
            (item.document_id, page)
            for item in ordered
            for page in range(item.page_start, item.page_end + 1)
        }
        page_capacity = len(source_pages) * self.config.per_page_cap
        if min(doc_capacity, page_capacity) < target:
            raise ValueError("source capacity cannot satisfy quotas and caps")

        expanded_types = tuple(
            kind for kind, count in self.config.quotas.items() for _ in range(count)
        )
        doc_counts: dict[str, int] = {}
        page_counts: dict[tuple[str, int], int] = {}
        jobs: list[GenerationJob] = []
        cursor = 0
        for ordinal, kind in enumerate(expanded_types):
            chosen: SourceWindow | None = None
            for offset in range(len(ordered)):
                index = (cursor + offset) % len(ordered)
                window = ordered[index]
                pages = tuple(range(window.page_start, window.page_end + 1))
                if doc_counts.get(window.document_id, 0) >= self.config.per_document_cap:
                    continue
                if any(
                    page_counts.get((window.document_id, page), 0) >= self.config.per_page_cap
                    for page in pages
                ):
                    continue
                chosen = window
                cursor = (index + 1) % len(ordered)
                break
            if chosen is None:
                raise ValueError("source capacity exhausted while applying quotas and caps")
            doc_counts[chosen.document_id] = doc_counts.get(chosen.document_id, 0) + 1
            for page in range(chosen.page_start, chosen.page_end + 1):
                key = (chosen.document_id, page)
                page_counts[key] = page_counts.get(key, 0) + 1
            jobs.append(GenerationJob(ordinal=ordinal, question_type=kind, window=chosen))

        batches = tuple(
            GenerationBatch(
                batch_id=f"batch-{start // self.config.batch_size:04d}",
                jobs=tuple(jobs[start : start + self.config.batch_size]),
            )
            for start in range(0, len(jobs), self.config.batch_size)
        )
        identity = {
            "schema": "benchmark-generation-plan-v1",
            "corpus_snapshot_id": corpus_snapshot_id,
            "model_id": model_id,
            "prompt_version": self.config.prompt_version,
            "config": self.config.model_dump(mode="json"),
            "jobs": [job.model_dump(mode="json") for job in jobs],
        }
        plan_hash = canonical_json_hash(identity)
        return GenerationPlan(
            plan_hash=plan_hash,
            corpus_snapshot_id=corpus_snapshot_id,
            model_id=model_id,
            prompt_version=self.config.prompt_version,
            jobs=tuple(jobs),
            batches=batches,
        )

    def plan_replacements(
        self,
        parent: GenerationPlan,
        *,
        accepted_counts: Mapping[QuestionType, int],
        attempt: int,
        target_quotas: Mapping[QuestionType, int] | None = None,
        prior_plans: Sequence[GenerationPlan] | None = None,
    ) -> GenerationPlan:
        """Plan only missing strata under a new immutable attempt identity."""
        if attempt <= 0:
            raise ValueError("replacement attempt must be positive")
        targets = target_quotas or self.config.quotas
        deficits = {
            kind: max(0, target - accepted_counts.get(kind, 0))
            for kind, target in targets.items()
        }
        deficits = {kind: count for kind, count in deficits.items() if count}
        if not deficits:
            raise ValueError("replacement plan requires at least one quota deficit")
        windows = tuple(dict.fromkeys(job.window for job in parent.jobs))
        prior = tuple(prior_plans or (parent,))
        doc_counts: dict[str, int] = {}
        page_counts: dict[tuple[str, int], int] = {}
        for prior_plan in prior:
            for job in prior_plan.jobs:
                doc = job.window.document_id
                doc_counts[doc] = doc_counts.get(doc, 0) + 1
                for page in range(job.window.page_start, job.window.page_end + 1):
                    key = (doc, page)
                    page_counts[key] = page_counts.get(key, 0) + 1
        jobs: list[GenerationJob] = []
        cursor = 0
        expanded_types = tuple(kind for kind, count in deficits.items() for _ in range(count))
        for local_ordinal, kind in enumerate(expanded_types):
            chosen: SourceWindow | None = None
            for offset in range(len(windows)):
                index = (cursor + offset) % len(windows)
                window = windows[index]
                pages = range(window.page_start, window.page_end + 1)
                if doc_counts.get(window.document_id, 0) >= self.config.per_document_cap:
                    continue
                if any(
                    page_counts.get((window.document_id, page), 0)
                    >= self.config.per_page_cap
                    for page in pages
                ):
                    continue
                chosen = window
                cursor = (index + 1) % len(windows)
                break
            if chosen is None:
                raise ValueError("replacement source capacity exhausted under campaign caps")
            doc_counts[chosen.document_id] = doc_counts.get(chosen.document_id, 0) + 1
            for page in range(chosen.page_start, chosen.page_end + 1):
                key = (chosen.document_id, page)
                page_counts[key] = page_counts.get(key, 0) + 1
            jobs.append(
                GenerationJob(
                    ordinal=attempt * 1_000_000 + local_ordinal,
                    question_type=kind,
                    window=chosen,
                )
            )
        batches = tuple(
            GenerationBatch(
                batch_id=f"replacement-{attempt:04d}-{start // self.config.batch_size:04d}",
                jobs=tuple(jobs[start : start + self.config.batch_size]),
            )
            for start in range(0, len(jobs), self.config.batch_size)
        )
        identity = {
            "schema": "benchmark-replacement-plan-v1",
            "parent_plan_hash": parent.plan_hash,
            "attempt": attempt,
            "deficits": {kind.value: count for kind, count in deficits.items()},
            "prior_plan_hashes": [item.plan_hash for item in prior],
            "jobs": [job.model_dump(mode="json") for job in jobs],
        }
        plan_hash = canonical_json_hash(identity)
        return GenerationPlan(
            plan_hash=plan_hash,
            corpus_snapshot_id=parent.corpus_snapshot_id,
            model_id=parent.model_id,
            prompt_version=parent.prompt_version,
            jobs=tuple(jobs),
            batches=batches,
        )


def generation_campaign_hash(
    plan: GenerationPlan,
    *,
    max_replacement_rounds: int,
    allow_reduced_scope: bool,
) -> str:
    """Commit operator confirmation to every bounded paid campaign dimension."""
    if max_replacement_rounds < 0:
        raise ValueError("max replacement rounds cannot be negative")
    return canonical_json_hash(
        {
            "schema": "benchmark-generation-campaign-v1",
            "initial_plan_hash": plan.plan_hash,
            "max_replacement_rounds": max_replacement_rounds,
            "allow_reduced_scope": allow_reduced_scope,
            "replacement_policy": "quota-deficits-with-cumulative-source-caps-v1",
        }
    )


@dataclass(frozen=True, slots=True)
class GenerationAuthorization:
    execute: bool
    confirm_paid: bool
    live_enabled: bool
    confirmed_plan_hash: str | None


def generation_execution_blockers(
    plan: GenerationPlan,
    *,
    authorization: GenerationAuthorization,
    price_book: PriceBook,
    projected_cost_usd: Decimal,
    remaining_budget_usd: Decimal,
    now: datetime,
    required_confirmation_hash: str | None = None,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if not authorization.execute:
        blockers.append("provider generation requires explicit execution")
    if not authorization.live_enabled:
        blockers.append("provider generation requires RUN_LIVE_UPSTAGE_TESTS=1")
    if not authorization.confirm_paid:
        blockers.append("paid generation requires explicit price confirmation")
    required_hash = required_confirmation_hash or plan.plan_hash
    if authorization.confirmed_plan_hash != required_hash:
        blockers.append("confirmed generation campaign hash does not match")
    try:
        price_book.verify_paid_batch(now=now)
    except PriceBookError as error:
        blockers.append(str(error))
    if projected_cost_usd < 0 or remaining_budget_usd <= 0:
        blockers.append("cost and remaining budget must be positive")
    elif projected_cost_usd >= remaining_budget_usd:
        blockers.append("projected cost reaches the remaining budget")
    return tuple(blockers)


@dataclass(frozen=True, slots=True)
class StoredBatch:
    plan_hash: str
    batch_id: str
    candidates: tuple[QuestionCandidate, ...]
    payload_hash: str


class BatchRepository(Protocol):
    async def get(self, plan_hash: str, batch_id: str) -> StoredBatch | None: ...

    async def put(self, batch: StoredBatch) -> None: ...


class MemoryBatchRepository:
    def __init__(self) -> None:
        self._batches: dict[tuple[str, str], StoredBatch] = {}

    async def get(self, plan_hash: str, batch_id: str) -> StoredBatch | None:
        stored = self._batches.get((plan_hash, batch_id))
        if stored is None:
            return None
        if stored.payload_hash != _candidate_payload_hash(stored.candidates):
            raise RuntimeError("stored generation batch failed integrity validation")
        return stored

    async def put(self, batch: StoredBatch) -> None:
        key = (batch.plan_hash, batch.batch_id)
        existing = self._batches.get(key)
        if existing is not None and existing != batch:
            raise RuntimeError("immutable generation batch already exists")
        if batch.payload_hash != _candidate_payload_hash(batch.candidates):
            raise RuntimeError("generation batch payload hash does not match")
        self._batches[key] = batch

    async def save_candidates(
        self,
        plan_hash: str,
        batch_id: str,
        candidates: tuple[QuestionCandidate, ...],
    ) -> None:
        await self.put(
            StoredBatch(
                plan_hash=plan_hash,
                batch_id=batch_id,
                candidates=candidates,
                payload_hash=_candidate_payload_hash(candidates),
            )
        )


class FileBatchRepository:
    """Canonical, integrity-checked checkpoints for cross-process batch resume."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    async def get(self, plan_hash: str, batch_id: str) -> StoredBatch | None:
        path = self._path(plan_hash, batch_id)
        if not path.exists():
            return None
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("generation checkpoint must be a regular file")
        raw = path.read_bytes()
        try:
            payload = json.loads(raw)
            candidates = tuple(
                TypeAdapter(list[QuestionCandidate]).validate_python(
                    payload["candidates"], strict=False
                )
            )
            stored = StoredBatch(
                plan_hash=str(payload["plan_hash"]),
                batch_id=str(payload["batch_id"]),
                candidates=candidates,
                payload_hash=str(payload["payload_hash"]),
            )
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise RuntimeError("generation checkpoint failed integrity validation") from error
        canonical = _stored_batch_bytes(stored)
        if (
            raw != canonical
            or stored.plan_hash != plan_hash
            or stored.batch_id != batch_id
            or stored.payload_hash != _candidate_payload_hash(stored.candidates)
        ):
            raise RuntimeError("generation checkpoint failed integrity validation")
        return stored

    async def put(self, batch: StoredBatch) -> None:
        existing = await self.get(batch.plan_hash, batch.batch_id)
        if existing is not None:
            if existing != batch:
                raise RuntimeError("immutable generation batch already exists")
            return
        if batch.payload_hash != _candidate_payload_hash(batch.candidates):
            raise RuntimeError("generation batch payload hash does not match")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(batch.plan_hash, batch.batch_id)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{batch.batch_id}-", suffix=".tmp", dir=self.directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_stored_batch_bytes(batch))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    async def save_candidates(
        self,
        plan_hash: str,
        batch_id: str,
        candidates: tuple[QuestionCandidate, ...],
    ) -> None:
        await self.put(
            StoredBatch(
                plan_hash=plan_hash,
                batch_id=batch_id,
                candidates=candidates,
                payload_hash=_candidate_payload_hash(candidates),
            )
        )

    def _path(self, plan_hash: str, batch_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", plan_hash):
            raise ValueError("plan hash must be a SHA-256 digest")
        if not re.fullmatch(r"(?:batch|replacement-\d{4})-\d{4,}", batch_id):
            raise ValueError("batch ID has an invalid shape")
        return self.directory / f"{plan_hash}-{batch_id}.json"


class BenchmarkGenerator:
    """Generate one planned batch only through the configured provider gateway."""

    def __init__(
        self,
        gateway: ProviderGateway,
        repository: BatchRepository,
        *,
        config: GenerationConfig | None = None,
    ) -> None:
        self._gateway = gateway
        self._repository = repository
        self.config = config or GenerationConfig()

    async def generate_batch(
        self, plan: GenerationPlan, batch: GenerationBatch
    ) -> tuple[QuestionCandidate, ...]:
        if batch not in plan.batches:
            raise ValueError("batch is not part of the immutable generation plan")
        stored = await self._repository.get(plan.plan_hash, batch.batch_id)
        if stored is not None:
            _validate_resumed_candidates(plan, batch, stored.candidates)
            return stored.candidates
        prompt = _generation_prompt(plan, batch)
        response = await self._gateway.generate(
            GenerateRequest(
                model_id=plan.model_id,
                prompt=prompt,
                context=(),
                provider_params={"response_format": {"type": "json_object"}},
                input_tokens=len(encoding().encode(prompt)) + 16,
                max_output_tokens=self.config.max_output_tokens,
            )
        )
        try:
            candidates = CandidateBatchEnvelope.model_validate_json(response.content).candidates
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as error:
            raise ValueError("provider returned invalid candidate batch JSON") from error
        expected_types = tuple(job.question_type for job in batch.jobs)
        if len(candidates) != len(batch.jobs) or tuple(
            item.question_type for item in candidates
        ) != expected_types:
            raise ValueError("provider batch does not match planned jobs")
        if len({item.candidate_id for item in candidates}) != len(candidates):
            raise ValueError("provider batch contains duplicate candidate IDs")
        authoritative: list[QuestionCandidate] = []
        for candidate, job in zip(candidates, batch.jobs, strict=True):
            _validate_candidate_window(candidate, job.window)
            metadata = candidate.generator.model_copy(
                update={
                    "model_id": plan.model_id,
                    "prompt_version": plan.prompt_version,
                    "plan_hash": plan.plan_hash,
                    "batch_id": batch.batch_id,
                    "source_window_hash": canonical_json_hash(
                        job.window.model_dump(mode="json")
                    ),
                    "correlation_id": response.correlation_id,
                    "cache_hit": response.cache_hit,
                }
            )
            authoritative.append(
                candidate.model_copy(
                    update={
                        "candidate_id": canonical_json_hash(
                            {
                                "plan_hash": plan.plan_hash,
                                "job_ordinal": job.ordinal,
                            }
                        ),
                        "generator": metadata,
                    }
                )
            )
        candidates = tuple(authoritative)
        stored_batch = StoredBatch(
            plan_hash=plan.plan_hash,
            batch_id=batch.batch_id,
            candidates=candidates,
            payload_hash=_candidate_payload_hash(candidates),
        )
        await self._repository.put(stored_batch)
        return candidates


def parse_candidate_json(payload: str) -> QuestionCandidate:
    try:
        return QuestionCandidate.model_validate_json(payload)
    except (ValidationError, ValueError) as error:
        raise ValueError("provider did not return valid candidate JSON") from error


def projected_generation_cost(
    plan: GenerationPlan,
    *,
    config: GenerationConfig,
    price_book: PriceBook,
    billing_multiplier: Decimal,
) -> Decimal:
    """Conservatively price every planned batch at its maximum response size."""
    if billing_multiplier < 1:
        raise ValueError("billing multiplier must be at least one")
    subtotal = Decimal("0")
    for batch in plan.batches:
        prompt = _generation_prompt(plan, batch)
        subtotal += price_book.estimate(
            PricingRequest(
                operation="generate",
                model_id=plan.model_id,
                input_tokens=len(encoding().encode(prompt)) + 16,
                output_tokens=config.max_output_tokens,
            )
        )
    return subtotal * billing_multiplier


def controlled_unanswerable(
    *,
    question: str,
    original_fact: str,
    asserted_absent_fact: str,
    document_windows: Sequence[SourceWindow],
    metadata: GeneratorMetadata,
) -> QuestionCandidate:
    target_documents = {window.document_id for window in document_windows}
    if len(target_documents) != 1:
        raise ValueError("controlled unanswerable requires exactly one target document")
    target_document_id = next(iter(target_documents))
    if not any(
        _search_text(original_fact) in _search_text(window.content)
        for window in document_windows
    ):
        raise ValueError("original fact is not present in the target document")
    normalized_fact = _search_text(asserted_absent_fact)
    if not normalized_fact:
        raise ValueError("controlled absent fact cannot be blank")
    if any(normalized_fact in _search_text(window.content) for window in document_windows):
        raise ValueError("asserted absent fact is present in the document snapshot")
    candidate_id = canonical_json_hash(
        {
            "question": question,
            "asserted_absent_fact": asserted_absent_fact,
            "plan_hash": metadata.plan_hash,
        }
    )
    return QuestionCandidate(
        candidate_id=candidate_id,
        question=question,
        gold_answer=None,
        evidence_spans=(),
        question_type=QuestionType.UNANSWERABLE,
        difficulty=Difficulty.HARD,
        answerable=False,
        asserted_absent_facts=(asserted_absent_fact,),
        unanswerable_transform=UnanswerableTransform(
            target_document_id=target_document_id,
            original_fact=original_fact,
            transformed_fact=asserted_absent_fact,
        ),
        generator=metadata,
        validation=ValidationStatus(decision=ValidationDecision.UNVALIDATED),
    )


def _generation_prompt(plan: GenerationPlan, batch: GenerationBatch) -> str:
    jobs = [
        {
            "ordinal": job.ordinal,
            "question_type": job.question_type,
            "source_window": job.window.model_dump(mode="json"),
            "source_window_hash": canonical_json_hash(job.window.model_dump(mode="json")),
        }
        for job in batch.jobs
    ]
    contract = {
        "plan_hash": plan.plan_hash,
        "batch_id": batch.batch_id,
        "requirements": [
            "Return a JSON object with a candidates array, one candidate per job in order.",
            "Copy every evidence span verbatim from its bounded source window.",
            "Use only supplied source text; include reasoning_kind metadata.",
            "For unanswerable jobs, transform a fact and assert the replacement is absent.",
        ],
        "jobs": jobs,
    }
    return json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _candidate_payload_hash(candidates: Sequence[QuestionCandidate]) -> str:
    return canonical_json_hash([item.model_dump(mode="json") for item in candidates])


def _validate_candidate_window(candidate: QuestionCandidate, window: SourceWindow) -> None:
    if candidate.unanswerable_transform is not None:
        transform = candidate.unanswerable_transform
        if (
            transform.target_document_id != window.document_id
            or _search_text(transform.original_fact) not in _search_text(window.content)
            or _search_text(transform.transformed_fact) in _search_text(window.content)
        ):
            raise ValueError("candidate transform is outside the assigned source window")
    for span in candidate.evidence_spans:
        units = tuple(
            unit
            for unit in window.source_units
            if unit.page == span.page and unit.chunk_id == span.chunk_id
        )
        if span.document_id != window.document_id or not any(
            _search_text(span.text) in _search_text(unit.content) for unit in units
        ):
            raise ValueError("candidate evidence is outside the assigned source window")


def _validate_resumed_candidates(
    plan: GenerationPlan,
    batch: GenerationBatch,
    candidates: tuple[QuestionCandidate, ...],
) -> None:
    if len(candidates) != len(batch.jobs):
        raise RuntimeError("checkpoint does not match current plan provenance")
    for candidate, job in zip(candidates, batch.jobs, strict=True):
        expected_id = canonical_json_hash(
            {"plan_hash": plan.plan_hash, "job_ordinal": job.ordinal}
        )
        expected_window_hash = canonical_json_hash(job.window.model_dump(mode="json"))
        if (
            candidate.candidate_id != expected_id
            or candidate.question_type is not job.question_type
            or candidate.generator.plan_hash != plan.plan_hash
            or candidate.generator.batch_id != batch.batch_id
            or candidate.generator.source_window_hash != expected_window_hash
            or candidate.generator.model_id != plan.model_id
            or candidate.generator.prompt_version != plan.prompt_version
        ):
            raise RuntimeError("checkpoint does not match current plan provenance")
        try:
            _validate_candidate_window(candidate, job.window)
        except ValueError as error:
            raise RuntimeError("checkpoint does not match current plan provenance") from error


def _stored_batch_bytes(batch: StoredBatch) -> bytes:
    payload = {
        "batch_id": batch.batch_id,
        "candidates": [item.model_dump(mode="json") for item in batch.candidates],
        "payload_hash": batch.payload_hash,
        "plan_hash": batch.plan_hash,
    }
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _search_text(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", unicodedata.normalize("NFKC", value).lower())


def _fact_anchors(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z가-힣]{2,}", unicodedata.normalize("NFKC", value).lower())
    }
