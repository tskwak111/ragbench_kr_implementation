"""Immutable, resumable retrieval-only experiment screening."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Protocol

from ragbench.core.hashing import canonical_json_hash
from ragbench.evaluation.retrieval import RetrievalCase, RetrievalEvaluation, aggregate_retrieval
from ragbench.experiments.config import RetrievalExperimentConfig
from ragbench.retrieval.base import Retriever, SearchFilter, SearchHit


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
        if self.content_hash != canonical_json_hash(self.questions):
            raise ValueError("question snapshot content hash mismatch")


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
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._store = store
        self._inventory = inventory
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
        question_id: str, hits: Sequence[SearchHit], top_k: int
    ) -> tuple[PersistedRankedHit, ...]:
        if len(hits) > top_k:
            raise ValueError("retriever returned more than top_k hits")
        if [hit.rank for hit in hits] != list(range(1, len(hits) + 1)) or len(
            {hit.chunk_id for hit in hits}
        ) != len(hits):
            raise ValueError("retriever returned an invalid ranking")
        if any(not math.isfinite(hit.score) for hit in hits):
            raise ValueError("retriever returned a non-finite score")
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
        retriever: Retriever,
    ) -> ScreeningResult:
        self._validate_inputs(config, questions)
        run_id = self.run_identity(config, questions)
        run = self._store.begin(
            ScreeningRunRecord(run_id, config.semantic_hash, questions.content_hash)
        )
        search_filter = SearchFilter(
            config.corpus_snapshot_id,
            config.parse_snapshot_id,
            config.chunk_strategy,
            config.embedding_snapshot_id,
        )
        by_id = {question.question_id: question for question in questions.questions}
        try:
            for question in questions.questions:
                if question.question_id in run.completed_question_ids:
                    continue
                started = self._clock()
                hits = await retriever.search(
                    question.prompt, top_k=config.top_k, filter=search_filter
                )
                elapsed_ms = max(0.0, (self._clock() - started) * 1000)
                persisted = self._persistable_hits(question.question_id, hits, config.top_k)
                self._store.persist_question(
                    run_id, question.question_id, persisted, elapsed_ms
                )
        except BaseException:
            self._store.mark_interrupted(run_id)
            raise

        rankings: dict[str, list[PersistedRankedHit]] = {question_id: [] for question_id in by_id}
        for hit in run.hits:
            rankings[hit.question_id].append(hit)
        evaluation = aggregate_retrieval(
            tuple(
                RetrievalCase(
                    question.question_id,
                    question.question_type,
                    tuple(
                        hit.chunk_id
                        for hit in sorted(
                            rankings[question.question_id], key=lambda row: row.rank
                        )
                    ),
                    question.evidence_chunk_ids,
                    run.latencies_ms[question.question_id],
                )
                for question in questions.questions
            ),
            k=config.top_k,
        )
        self._store.finish(run_id, evaluation)
        return ScreeningResult(run_id, evaluation)
