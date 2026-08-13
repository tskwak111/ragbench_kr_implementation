"""Immutable, resumable retrieval-only experiment screening."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol

from ragbench.benchmark.splits import SnapshotName, SplitSnapshot
from ragbench.core.hashing import canonical_json_hash
from ragbench.evaluation.retrieval import RetrievalCase, RetrievalEvaluation, aggregate_retrieval
from ragbench.experiments.config import RetrievalExperimentConfig, RetrieverName, RRFConfig
from ragbench.retrieval.base import Retriever, SearchFilter, SearchHit

_DEV_CAPABILITY = object()


@dataclass(frozen=True, slots=True)
class ScreeningQuestion:
    question_id: str
    prompt: str
    question_type: str
    evidence_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.question_id.strip()
            or not self.prompt.strip()
            or not self.question_type.strip()
        ):
            raise ValueError("screening question fields cannot be blank")
        if len(self.evidence_chunk_ids) != len(set(self.evidence_chunk_ids)):
            raise ValueError("question evidence chunk IDs must be unique")


@dataclass(frozen=True, slots=True)
class ScreeningQuestionSnapshot:
    snapshot_id: str
    split: str
    content_hash: str
    complete: bool
    questions: tuple[ScreeningQuestion, ...]

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip() or len(self.content_hash) != 64:
            raise ValueError("question snapshot identity is invalid")
        if len({question.question_id for question in self.questions}) != len(self.questions):
            raise ValueError("question IDs must be unique")
        if self.content_hash != canonical_json_hash(
            {"split": self.split, "questions": self.questions}
        ):
            raise ValueError("question snapshot content hash mismatch")


@dataclass(frozen=True, slots=True)
class DevelopmentAuthorization:
    snapshot_id: str
    membership_hash: str
    _capability: object


def authorize_development_snapshot(snapshot: SplitSnapshot) -> DevelopmentAuthorization:
    if snapshot.name is not SnapshotName.DEV_AUTO:
        raise PermissionError("retrieval screening requires a Task 12 dev_auto snapshot")
    return DevelopmentAuthorization(snapshot.snapshot_id, snapshot.membership_hash, _DEV_CAPABILITY)


@dataclass(frozen=True, slots=True)
class SnapshotBinding:
    snapshot_id: str
    parent_snapshot_id: str | None
    variant: str | None
    complete: bool


@dataclass(frozen=True, slots=True)
class SnapshotInventory:
    corpus: SnapshotBinding
    parse: SnapshotBinding
    chunk: SnapshotBinding
    embedding: SnapshotBinding


@dataclass(frozen=True, slots=True)
class BoundRetriever:
    """Runtime retriever with the exact semantic identity promised by its config."""

    name: RetrieverName
    retriever: Retriever
    rrf: RRFConfig | None

    def __post_init__(self) -> None:
        if (self.name == "hybrid") != (self.rrf is not None):
            raise ValueError("RRF parameters are required only for a hybrid binding")


@dataclass(frozen=True, slots=True)
class PersistedRankedHit:
    question_id: str
    chunk_id: str
    rank: int
    score: float
    retriever: str
    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float | None
    sparse_score: float | None
    fused_score: float


@dataclass(slots=True)
class ScreeningRunRecord:
    run_id: str
    config_hash: str
    question_snapshot_hash: str
    expected_question_ids: tuple[str, ...] = ()
    status: Literal["running", "interrupted", "complete"] = "running"
    completed_question_ids: set[str] = field(default_factory=set)
    hits: list[PersistedRankedHit] = field(default_factory=list)
    latencies_ms: dict[str, float] = field(default_factory=dict)
    evaluation: RetrievalEvaluation | None = None


class ScreeningStore(Protocol):
    def begin(self, record: ScreeningRunRecord) -> ScreeningRunRecord: ...

    def persist_question(
        self,
        run_id: str,
        question_id: str,
        hits: Sequence[PersistedRankedHit],
        latency_ms: float,
    ) -> None: ...

    def mark_interrupted(self, run_id: str) -> None: ...

    def finish(self, run_id: str, evaluation: RetrievalEvaluation) -> None: ...

    def get(self, run_id: str) -> ScreeningRunRecord: ...


class MemoryScreeningStore:
    """Deterministic reference store with per-question atomic commits."""

    def __init__(self) -> None:
        self._runs: dict[str, ScreeningRunRecord] = {}

    def begin(self, record: ScreeningRunRecord) -> ScreeningRunRecord:
        existing = self._runs.get(record.run_id)
        if existing is not None:
            if (
                existing.config_hash != record.config_hash
                or existing.question_snapshot_hash != record.question_snapshot_hash
                or existing.expected_question_ids != record.expected_question_ids
            ):
                raise ValueError("run identity collides with different immutable inputs")
            if existing.status != "complete":
                existing.status = "running"
            return existing
        self._runs[record.run_id] = record
        return record

    def persist_question(
        self,
        run_id: str,
        question_id: str,
        hits: Sequence[PersistedRankedHit],
        latency_ms: float,
    ) -> None:
        run = self.get(run_id)
        if question_id in run.completed_question_ids:
            return
        materialized = list(hits)
        run.hits.extend(materialized)
        run.latencies_ms[question_id] = latency_ms
        run.completed_question_ids.add(question_id)

    def mark_interrupted(self, run_id: str) -> None:
        self.get(run_id).status = "interrupted"

    def finish(self, run_id: str, evaluation: RetrievalEvaluation) -> None:
        run = self.get(run_id)
        run.evaluation = evaluation
        run.status = "complete"

    def get(self, run_id: str) -> ScreeningRunRecord:
        try:
            return self._runs[run_id]
        except KeyError:
            raise KeyError(f"unknown screening run: {run_id}") from None


class FileScreeningStore(MemoryScreeningStore):
    """Durable local checkpoint store using atomic per-run JSON replacement."""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self._root = root
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        if len(run_id) != 64 or any(character not in "0123456789abcdef" for character in run_id):
            raise ValueError("run ID must be a SHA-256 digest")
        return self._root / f"{run_id}.json"

    def _load(self, run_id: str) -> ScreeningRunRecord | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != "retrieval-checkpoint-v1":
            raise ValueError("checkpoint schema version mismatch")
        integrity_hash = raw.pop("integrity_hash", None)
        if integrity_hash != canonical_json_hash(raw):
            raise ValueError("checkpoint integrity hash mismatch")
        record = ScreeningRunRecord(
            run_id=raw["run_id"],
            config_hash=raw["config_hash"],
            question_snapshot_hash=raw["question_snapshot_hash"],
            expected_question_ids=tuple(raw["expected_question_ids"]),
            status=raw["status"],
            completed_question_ids=set(raw["completed_question_ids"]),
            hits=[PersistedRankedHit(**hit) for hit in raw["hits"]],
            latencies_ms={key: float(value) for key, value in raw["latencies_ms"].items()},
        )
        if record.run_id != run_id:
            raise ValueError("checkpoint run identity mismatch")
        expected = set(record.expected_question_ids)
        if len(expected) != len(record.expected_question_ids):
            raise ValueError("checkpoint question membership is invalid")
        if not record.completed_question_ids <= expected:
            raise ValueError("checkpoint completed question is outside expected membership")
        if set(record.latencies_ms) != record.completed_question_ids or any(
            not math.isfinite(value) or value < 0 for value in record.latencies_ms.values()
        ):
            raise ValueError("checkpoint latency evidence is invalid")
        hits_by_question: dict[str, list[PersistedRankedHit]] = {}
        for hit in record.hits:
            hits_by_question.setdefault(hit.question_id, []).append(hit)
            if (
                hit.question_id not in record.completed_question_ids
                or not math.isfinite(hit.score)
                or not math.isfinite(hit.fused_score)
            ):
                raise ValueError("checkpoint hit evidence is invalid")
        for question_id, hits in hits_by_question.items():
            if [hit.rank for hit in hits] != list(range(1, len(hits) + 1)) or len(
                {hit.chunk_id for hit in hits}
            ) != len(hits):
                raise ValueError(f"checkpoint ranking is invalid: {question_id}")
        return record

    def _flush(self, record: ScreeningRunRecord) -> None:
        payload = {
            "schema_version": "retrieval-checkpoint-v1",
            "run_id": record.run_id,
            "config_hash": record.config_hash,
            "question_snapshot_hash": record.question_snapshot_hash,
            "expected_question_ids": record.expected_question_ids,
            "status": record.status,
            "completed_question_ids": sorted(record.completed_question_ids),
            "hits": [asdict(hit) for hit in record.hits],
            "latencies_ms": record.latencies_ms,
        }
        payload["integrity_hash"] = canonical_json_hash(payload)
        path = self._path(record.run_id)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def begin(self, record: ScreeningRunRecord) -> ScreeningRunRecord:
        loaded = self._load(record.run_id)
        if loaded is not None:
            self._runs[record.run_id] = loaded
        resolved = super().begin(record)
        self._flush(resolved)
        return resolved

    def persist_question(
        self,
        run_id: str,
        question_id: str,
        hits: Sequence[PersistedRankedHit],
        latency_ms: float,
    ) -> None:
        super().persist_question(run_id, question_id, hits, latency_ms)
        self._flush(super().get(run_id))

    def mark_interrupted(self, run_id: str) -> None:
        super().mark_interrupted(run_id)
        self._flush(super().get(run_id))

    def finish(self, run_id: str, evaluation: RetrievalEvaluation) -> None:
        super().finish(run_id, evaluation)
        self._flush(super().get(run_id))

    def get(self, run_id: str) -> ScreeningRunRecord:
        loaded = self._load(run_id)
        if loaded is not None:
            self._runs[run_id] = loaded
        return super().get(run_id)


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    run_id: str
    evaluation: RetrievalEvaluation


class RetrievalScreenRunner:
    """Run a development retrieval screen with no generation/provider calls."""

    def __init__(
        self,
        *,
        store: ScreeningStore,
        inventory: SnapshotInventory,
        development_authorization: DevelopmentAuthorization,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._store = store
        self._inventory = inventory
        self._development_authorization = development_authorization
        self._clock = clock

    def run_identity(
        self, config: RetrievalExperimentConfig, questions: ScreeningQuestionSnapshot
    ) -> str:
        return canonical_json_hash(
            {
                "kind": "retrieval-screen-run-v1",
                "config_hash": config.semantic_hash,
                "question_snapshot_id": questions.snapshot_id,
                "question_content_hash": questions.content_hash,
            }
        )

    def _validate_inputs(
        self, config: RetrievalExperimentConfig, questions: ScreeningQuestionSnapshot
    ) -> None:
        if questions.split != "dev_auto":
            raise PermissionError("normal retrieval screening permits only the dev_auto split")
        authorization = self._development_authorization
        if authorization._capability is not _DEV_CAPABILITY:
            raise PermissionError("invalid Task 12 development snapshot authorization")
        if (
            authorization.snapshot_id != questions.snapshot_id
            or authorization.membership_hash
            != canonical_json_hash(tuple(sorted(q.question_id for q in questions.questions)))
        ):
            raise PermissionError("question content is not bound to the authorized dev snapshot")
        if not questions.complete:
            raise ValueError("question snapshot is incomplete")
        if questions.snapshot_id != config.question_snapshot_id:
            raise ValueError("question snapshot identity mismatch")
        expected = (
            (self._inventory.corpus, config.corpus_snapshot_id),
            (self._inventory.parse, config.parse_snapshot_id),
            (self._inventory.chunk, config.chunk_snapshot_id),
            (self._inventory.embedding, config.embedding_snapshot_id),
        )
        if any(not binding.complete for binding, _ in expected):
            raise ValueError("retrieval snapshot inventory contains an incomplete snapshot")
        if any(binding.snapshot_id != identifier for binding, identifier in expected):
            raise ValueError("retrieval snapshot identity mismatch")
        if (
            self._inventory.parse.parent_snapshot_id != config.corpus_snapshot_id
            or self._inventory.parse.variant != config.parse_mode
            or self._inventory.chunk.parent_snapshot_id != config.parse_snapshot_id
            or self._inventory.chunk.variant != config.chunk_strategy
            or self._inventory.embedding.parent_snapshot_id != config.chunk_snapshot_id
            or self._inventory.embedding.variant != config.chunk_strategy
        ):
            raise ValueError("retrieval snapshot lineage mismatch")

    @staticmethod
    def _persistable_hits(
        question_id: str,
        hits: Sequence[SearchHit],
        top_k: int,
        retriever_name: RetrieverName,
    ) -> tuple[PersistedRankedHit, ...]:
        if len(hits) > top_k:
            raise ValueError("retriever returned more than top_k hits")
        if [hit.rank for hit in hits] != list(range(1, len(hits) + 1)) or len(
            {hit.chunk_id for hit in hits}
        ) != len(hits):
            raise ValueError("retriever returned an invalid ranking")
        if any(not math.isfinite(hit.score) for hit in hits):
            raise ValueError("retriever returned a non-finite score")
        expected_label = "hybrid-rrf" if retriever_name == "hybrid" else retriever_name
        if any(hit.retriever != expected_label for hit in hits):
            raise ValueError("hit retriever identity does not match the bound retriever")
        if retriever_name == "hybrid":
            for hit in hits:
                evidence = hit.evidence
                ranks = (evidence.dense_rank, evidence.sparse_rank) if evidence else ()
                scores = (evidence.dense_score, evidence.sparse_score) if evidence else ()
                if (
                    evidence is None
                    or not math.isfinite(evidence.fused_score)
                    or evidence.fused_score != hit.score
                    or (evidence.dense_rank is None and evidence.sparse_rank is None)
                    or any(rank is not None and rank <= 0 for rank in ranks)
                    or any(
                        (rank is None) != (score is None)
                        for rank, score in zip(ranks, scores, strict=True)
                    )
                    or any(value is not None and not math.isfinite(value) for value in scores)
                ):
                    raise ValueError("hybrid hit component evidence is invalid")
        elif any(hit.evidence is not None for hit in hits):
            raise ValueError("non-hybrid hits cannot carry component evidence")
        output: list[PersistedRankedHit] = []
        for hit in hits:
            evidence = hit.evidence
            output.append(
                PersistedRankedHit(
                    question_id,
                    hit.chunk_id,
                    hit.rank,
                    hit.score,
                    hit.retriever,
                    None if evidence is None else evidence.dense_rank,
                    None if evidence is None else evidence.sparse_rank,
                    None if evidence is None else evidence.dense_score,
                    None if evidence is None else evidence.sparse_score,
                    hit.score if evidence is None else evidence.fused_score,
                )
            )
        return tuple(output)

    async def run(
        self,
        config: RetrievalExperimentConfig,
        questions: ScreeningQuestionSnapshot,
        binding: BoundRetriever,
    ) -> ScreeningResult:
        self._validate_inputs(config, questions)
        if binding.name != config.retriever:
            raise ValueError("retriever binding does not match experiment config")
        if binding.rrf != config.rrf:
            raise ValueError("RRF binding does not match experiment config")
        run_id = self.run_identity(config, questions)
        by_id = {question.question_id: question for question in questions.questions}
        run = self._store.begin(
            ScreeningRunRecord(
                run_id=run_id,
                config_hash=config.semantic_hash,
                question_snapshot_hash=questions.content_hash,
                expected_question_ids=tuple(sorted(by_id)),
            )
        )
        search_filter = SearchFilter(
            config.corpus_snapshot_id,
            config.parse_snapshot_id,
            config.chunk_strategy,
            config.embedding_snapshot_id,
        )
        try:
            retrieval_k = config.top_k
            for question in questions.questions:
                if question.question_id in run.completed_question_ids:
                    continue
                started = self._clock()
                hits = await binding.retriever.search(
                    question.prompt, top_k=retrieval_k, filter=search_filter
                )
                elapsed_ms = max(0.0, (self._clock() - started) * 1000)
                persisted = self._persistable_hits(
                    question.question_id, hits, retrieval_k, config.retriever
                )
                self._store.persist_question(run_id, question.question_id, persisted, elapsed_ms)
        except BaseException:
            self._store.mark_interrupted(run_id)
            raise

        run = self._store.get(run_id)
        rankings: dict[str, list[PersistedRankedHit]] = {question_id: [] for question_id in by_id}
        for hit in run.hits:
            rankings[hit.question_id].append(hit)
        cases = tuple(
            RetrievalCase(
                question.question_id,
                question.question_type,
                tuple(
                    hit.chunk_id
                    for hit in sorted(rankings[question.question_id], key=lambda row: row.rank)
                ),
                question.evidence_chunk_ids,
                run.latencies_ms[question.question_id],
            )
            for question in questions.questions
        )
        evaluation = aggregate_retrieval(
            cases,
            k=config.top_k,
        )
        self._store.finish(run_id, evaluation)
        return ScreeningResult(run_id, evaluation)
