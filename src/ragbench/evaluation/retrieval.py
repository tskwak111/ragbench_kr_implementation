"""Hand-checkable retrieval metrics and stable resampling inputs."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from ragbench.core.hashing import canonical_json_hash


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    question_id: str
    question_type: str
    ranked_chunk_ids: tuple[str, ...]
    evidence_chunk_ids: tuple[str, ...]
    latency_ms: float

    def __post_init__(self) -> None:
        if not self.question_id.strip() or not self.question_type.strip():
            raise ValueError("retrieval case identity cannot be blank")
        if not math.isfinite(self.latency_ms):
            raise ValueError("retrieval latency must be finite")
        if self.latency_ms < 0:
            raise ValueError("retrieval latency cannot be negative")
        if len(self.ranked_chunk_ids) != len(set(self.ranked_chunk_ids)):
            raise ValueError("ranked chunk IDs must be unique")
        if len(self.evidence_chunk_ids) != len(set(self.evidence_chunk_ids)):
            raise ValueError("evidence chunk IDs must be unique")


@dataclass(frozen=True, slots=True)
class RetrievalMetric:
    question_id: str
    question_type: str
    k: int
    evidence_count: int
    retrieved_evidence_count: int
    hit_at_k: float | None
    evidence_recall_at_k: float | None
    mrr: float | None
    latency_ms: float

    @property
    def is_scorable(self) -> bool:
        return self.evidence_count > 0


@dataclass(frozen=True, slots=True)
class BootstrapInput:
    """One independently ordered observation; CI policy is applied downstream."""

    question_id: str
    question_type: str
    hit_at_k: float
    evidence_recall_at_k: float
    mrr: float


@dataclass(frozen=True, slots=True)
class PairedBootstrapInput:
    """Aligned observations for a later paired resampling calculation."""

    question_id: str
    question_type: str
    left_hit_at_k: float
    right_hit_at_k: float
    left_evidence_recall_at_k: float
    right_evidence_recall_at_k: float
    left_mrr: float
    right_mrr: float


@dataclass(frozen=True, slots=True)
class RetrievalAggregate:
    question_count: int
    scorable_count: int
    no_evidence_count: int
    macro_hit_at_k: float | None
    macro_evidence_recall_at_k: float | None
    micro_evidence_recall_at_k: float | None
    macro_mrr: float | None
    mean_latency_ms: float | None


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    k: int
    overall: RetrievalAggregate
    by_question_type: dict[str, RetrievalAggregate]
    question_metrics: tuple[RetrievalMetric, ...]
    bootstrap_inputs: tuple[BootstrapInput, ...]

    @property
    def bootstrap_inputs_hash(self) -> str:
        return canonical_json_hash(self.bootstrap_inputs)


def evaluate_retrieval(case: RetrievalCase, *, k: int) -> RetrievalMetric:
    """Score one ranking; no-evidence questions are explicitly unscored."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    evidence = set(case.evidence_chunk_ids)
    if not evidence:
        return RetrievalMetric(
            case.question_id,
            case.question_type,
            k,
            0,
            0,
            None,
            None,
            None,
            case.latency_ms,
        )
    top = case.ranked_chunk_ids[:k]
    retrieved_count = sum(chunk_id in evidence for chunk_id in top)
    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, chunk_id in enumerate(case.ranked_chunk_ids, start=1)
            if chunk_id in evidence
        ),
        0.0,
    )
    return RetrievalMetric(
        case.question_id,
        case.question_type,
        k,
        len(evidence),
        retrieved_count,
        float(retrieved_count > 0),
        retrieved_count / len(evidence),
        reciprocal_rank,
        case.latency_ms,
    )


def _aggregate(rows: Sequence[RetrievalMetric]) -> RetrievalAggregate:
    scored = [row for row in rows if row.is_scorable]
    latency = sum(row.latency_ms for row in rows) / len(rows) if rows else None
    if not scored:
        return RetrievalAggregate(len(rows), 0, len(rows), None, None, None, None, latency)
    hit = sum(row.hit_at_k or 0.0 for row in scored) / len(scored)
    recall = sum(row.evidence_recall_at_k or 0.0 for row in scored) / len(scored)
    total_evidence = sum(row.evidence_count for row in scored)
    micro_recall = sum(row.retrieved_evidence_count for row in scored) / total_evidence
    mrr = sum(row.mrr or 0.0 for row in scored) / len(scored)
    return RetrievalAggregate(
        len(rows), len(scored), len(rows) - len(scored), hit, recall, micro_recall, mrr, latency
    )


def aggregate_retrieval(cases: Sequence[RetrievalCase], *, k: int) -> RetrievalEvaluation:
    """Return overall/per-type aggregates and deterministic raw bootstrap observations."""
    rows = tuple(evaluate_retrieval(case, k=k) for case in cases)
    grouped: dict[str, list[RetrievalMetric]] = defaultdict(list)
    for row in rows:
        grouped[row.question_type].append(row)
    ordered = tuple(sorted(rows, key=lambda row: row.question_id))
    bootstrap = tuple(
        BootstrapInput(
            row.question_id,
            row.question_type,
            row.hit_at_k or 0.0,
            row.evidence_recall_at_k or 0.0,
            row.mrr or 0.0,
        )
        for row in ordered
        if row.is_scorable
    )
    return RetrievalEvaluation(
        k,
        _aggregate(rows),
        {name: _aggregate(grouped[name]) for name in sorted(grouped)},
        ordered,
        bootstrap,
    )


def paired_bootstrap_inputs(
    left: RetrievalEvaluation, right: RetrievalEvaluation
) -> tuple[PairedBootstrapInput, ...]:
    """Align two systems' raw observations without computing or claiming an interval."""
    left_by_id = {row.question_id: row for row in left.bootstrap_inputs}
    right_by_id = {row.question_id: row for row in right.bootstrap_inputs}
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("paired bootstrap requires the same scorable questions")
    output: list[PairedBootstrapInput] = []
    for question_id in sorted(left_by_id):
        left_row, right_row = left_by_id[question_id], right_by_id[question_id]
        if left_row.question_type != right_row.question_type:
            raise ValueError("paired bootstrap question types do not match")
        output.append(
            PairedBootstrapInput(
                question_id,
                left_row.question_type,
                left_row.hit_at_k,
                right_row.hit_at_k,
                left_row.evidence_recall_at_k,
                right_row.evidence_recall_at_k,
                left_row.mrr,
                right_row.mrr,
            )
        )
    return tuple(output)
