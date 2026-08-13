"""Predeclared retrieval-screen ranking, diversity, and public leaderboard export."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType

from ragbench.core.hashing import canonical_json_hash
from ragbench.experiments.config import RetrievalExperimentConfig
from ragbench.experiments.planner import CHUNK_STRATEGIES, PARSE_MODES, RETRIEVERS, TOP_K_VALUES

SELECTION_RULE = {
    "version": "retrieval-shortlist-v1",
    "order": [
        "recall_at_5 descending",
        "mrr descending",
        "mean_latency_ms ascending",
        "semantic_hash ascending",
    ],
    "near_duplicate_family": ["parse_mode", "chunk_strategy", "retriever"],
    "diversity_caps": {
        "parse_mode": "at most ceil(shortlist_size / 2)",
        "retriever": "at most ceil(shortlist_size / 2)",
    },
}
SELECTION_RULE_HASH = canonical_json_hash(SELECTION_RULE)


@dataclass(frozen=True, slots=True)
class ScreeningOutcome:
    config: RetrievalExperimentConfig
    hit_at_5: float
    recall_at_5: float
    micro_recall_at_5: float
    mrr: float
    mean_latency_ms: float
    question_count: int
    scorable_count: int
    no_evidence_count: int
    per_type: Mapping[str, Mapping[str, float | int | None]]
    bootstrap_inputs_hash: str

    def __post_init__(self) -> None:
        values = (
            self.hit_at_5,
            self.recall_at_5,
            self.micro_recall_at_5,
            self.mrr,
            self.mean_latency_ms,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("screening outcome values must be finite")
        if any(not 0 <= value <= 1 for value in values[:4]):
            raise ValueError("screening quality metrics must be between zero and one")
        if self.mean_latency_ms < 0:
            raise ValueError("screening latency cannot be negative")
        if len(self.bootstrap_inputs_hash) != 64:
            raise ValueError("bootstrap input hash must be a SHA-256 digest")
        if (
            self.question_count <= 0
            or self.scorable_count < 0
            or self.no_evidence_count < 0
            or self.scorable_count + self.no_evidence_count != self.question_count
        ):
            raise ValueError("screening outcome counts are inconsistent")
        frozen_per_type: dict[str, Mapping[str, float | int | None]] = {}
        required = {
            "hit_at_5",
            "recall_at_5",
            "micro_recall_at_5",
            "mrr",
            "question_count",
            "scorable_count",
            "no_evidence_count",
        }
        for name, metrics in self.per_type.items():
            if not name.strip() or metrics.keys() != required:
                raise ValueError("per-type metrics must use the complete retrieval schema")
            frozen_per_type[name] = MappingProxyType(dict(metrics))
        object.__setattr__(self, "per_type", MappingProxyType(frozen_per_type))


def _quality_key(outcome: ScreeningOutcome) -> tuple[float, float, float, str]:
    return (
        -outcome.recall_at_5,
        -outcome.mrr,
        outcome.mean_latency_ms,
        outcome.config.semantic_hash,
    )


def select_retrieval_shortlist(
    outcomes: Sequence[ScreeningOutcome],
    *,
    size: int = 8,
    enforce_core_diversity: bool = True,
    require_complete_grid: bool = True,
) -> tuple[ScreeningOutcome, ...]:
    """Rank by the frozen rule, applying caps only after collapsing K-only variants."""
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("shortlist size must be a positive integer")
    hashes = [outcome.config.semantic_hash for outcome in outcomes]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate semantic screening outcomes are not allowed")
    if require_complete_grid and len(outcomes) != 126:
        raise ValueError("shortlist selection requires the complete 126-configuration grid")
    if require_complete_grid:
        axes = {
            (
                row.config.parse_mode,
                row.config.chunk_strategy,
                row.config.retriever,
                row.config.top_k,
            )
            for row in outcomes
        }
        expected_axes = {
            (mode, strategy, retriever, top_k)
            for mode in PARSE_MODES
            for strategy in CHUNK_STRATEGIES
            for retriever in RETRIEVERS
            for top_k in TOP_K_VALUES
        }
        if axes != expected_axes:
            raise ValueError("screening outcomes do not cover the exact core configuration grid")
        parse_bindings = {
            mode: {
                row.config.parse_snapshot_id for row in outcomes if row.config.parse_mode == mode
            }
            for mode in PARSE_MODES
        }
        artifact_bindings: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for row in outcomes:
            artifact_bindings.setdefault(
                (row.config.parse_mode, row.config.chunk_strategy), set()
            ).add((row.config.chunk_snapshot_id, row.config.embedding_snapshot_id))
        if (
            any(len(values) != 1 for values in parse_bindings.values())
            or len(artifact_bindings) != 14
            or any(len(values) != 1 for values in artifact_bindings.values())
        ):
            raise ValueError("core grid snapshot bindings are inconsistent")
    cohorts = {
        (
            row.config.corpus_snapshot_id,
            row.config.question_snapshot_id,
            row.config.code_commit,
            row.config.metric_version,
            row.config.random_seed,
        )
        for row in outcomes
    }
    if len(cohorts) != 1:
        raise ValueError("screening outcomes must belong to one immutable comparison cohort")

    best_by_family: dict[tuple[str, str, str], ScreeningOutcome] = {}
    selection_pool = (
        [outcome for outcome in outcomes if outcome.config.top_k == 5]
        if require_complete_grid
        else list(outcomes)
    )
    for outcome in sorted(selection_pool, key=_quality_key):
        family = (
            outcome.config.parse_mode,
            outcome.config.chunk_strategy,
            outcome.config.retriever,
        )
        best_by_family.setdefault(family, outcome)
    ranked = sorted(best_by_family.values(), key=_quality_key)
    if not enforce_core_diversity:
        if len(ranked) < size:
            raise ValueError("not enough distinct screening families for shortlist")
        return tuple(ranked[:size])

    cap = math.ceil(size / 2)
    parse_counts: Counter[str] = Counter()
    retriever_counts: Counter[str] = Counter()
    selected: list[ScreeningOutcome] = []
    for outcome in ranked:
        if parse_counts[outcome.config.parse_mode] >= cap:
            continue
        if retriever_counts[outcome.config.retriever] >= cap:
            continue
        selected.append(outcome)
        parse_counts[outcome.config.parse_mode] += 1
        retriever_counts[outcome.config.retriever] += 1
        if len(selected) == size:
            return tuple(selected)
    raise ValueError("outcomes cannot satisfy the predeclared shortlist diversity constraints")


def export_retrieval_leaderboard(outcomes: Sequence[ScreeningOutcome], path: Path) -> None:
    """Publish metrics and CI input identities without fabricating uncomputed intervals."""
    ordered = sorted(outcomes, key=_quality_key)
    payload = {
        "schema_version": "retrieval-leaderboard-v1",
        "selection_rule": SELECTION_RULE,
        "selection_rule_hash": SELECTION_RULE_HASH,
        "rows": [
            {
                "config_hash": outcome.config.semantic_hash,
                "config": outcome.config.model_dump(mode="json"),
                "hit_at_5": outcome.hit_at_5,
                "recall_at_5": outcome.recall_at_5,
                "micro_recall_at_5": outcome.micro_recall_at_5,
                "mrr": outcome.mrr,
                "mean_latency_ms": outcome.mean_latency_ms,
                "question_count": outcome.question_count,
                "scorable_count": outcome.scorable_count,
                "no_evidence_count": outcome.no_evidence_count,
                "per_type": {name: dict(metrics) for name, metrics in outcome.per_type.items()},
                "bootstrap_inputs_hash": outcome.bootstrap_inputs_hash,
            }
            for outcome in ordered
        ],
        "paired_ci_status": "not-computed; paired inputs are identified by hash",
    }
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
        stream.write("\n")


GENERATION_SELECTION_RULE = {
    "version": "generation-top-three-v1",
    "quality": "mean(correctness, faithfulness, citation_f1, abstention_accuracy)",
    "order": [
        "quality descending",
        "mean_latency_ms ascending",
        "total_cost_usd ascending",
        "config_hash ascending",
    ],
    "best_value": (
        "lowest cost whose quality_ci_high reaches the leader quality_ci_low; "
        "replace rank three when absent"
    ),
    "constraints": ["same cohort", "same question count", "calibrated judge only"],
}
GENERATION_SELECTION_RULE_HASH = canonical_json_hash(GENERATION_SELECTION_RULE)


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    """One immutable, calibrated development result on a shared question cohort."""

    config_hash: str
    cohort_hash: str
    question_count: int
    correctness: float
    faithfulness: float
    citation_f1: float
    abstention_accuracy: float
    mean_latency_ms: float
    total_cost_usd: Decimal
    quality_ci_low: float
    quality_ci_high: float
    judge_calibrated: bool

    def __post_init__(self) -> None:
        if len(self.config_hash) != 64 or len(self.cohort_hash) != 64:
            raise ValueError("generation outcome identities must be SHA-256 digests")
        quality = (
            self.correctness,
            self.faithfulness,
            self.citation_f1,
            self.abstention_accuracy,
            self.quality_ci_low,
            self.quality_ci_high,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in quality):
            raise ValueError("generation quality values must be finite and between zero and one")
        if self.quality_ci_low > self.quality_ci_high:
            raise ValueError("quality confidence interval is reversed")
        if self.question_count <= 0 or self.mean_latency_ms < 0 or self.total_cost_usd < 0:
            raise ValueError("generation counts, latency, and cost must be nonnegative")

    @property
    def quality(self) -> float:
        return (
            self.correctness + self.faithfulness + self.citation_f1 + self.abstention_accuracy
        ) / 4


def select_generation_top_three(
    outcomes: Sequence[GenerationOutcome],
) -> tuple[GenerationOutcome, ...]:
    """Apply the preregistered quality ranking and retain a competitive best-value row."""
    if len(outcomes) < 3:
        raise ValueError("at least three generation outcomes are required")
    hashes = [outcome.config_hash for outcome in outcomes]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate generation outcomes are not allowed")
    if (
        len({outcome.cohort_hash for outcome in outcomes}) != 1
        or len({outcome.question_count for outcome in outcomes}) != 1
    ):
        raise ValueError("generation outcomes must use the same cohort and question count")
    if not all(outcome.judge_calibrated for outcome in outcomes):
        raise ValueError("generation selection requires calibrated judge status")
    ranked = sorted(
        outcomes,
        key=lambda row: (
            -row.quality,
            row.mean_latency_ms,
            row.total_cost_usd,
            row.config_hash,
        ),
    )
    leader = ranked[0]
    competitive = [row for row in ranked if row.quality_ci_high >= leader.quality_ci_low]
    best_value = min(
        competitive,
        key=lambda row: (row.total_cost_usd, -row.quality, row.config_hash),
    )
    selected = list(ranked[:3])
    if best_value not in selected:
        selected[2] = best_value
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class BillingReconciliation:
    """Local/provider comparison that explicitly represents unavailable console billing."""

    local_gross_usd: Decimal
    provider_console_gross_usd: Decimal | None
    delta_usd: Decimal | None
    status: str


def reconcile_provider_billing(
    *, local_gross_usd: Decimal, provider_console_gross_usd: Decimal | None
) -> BillingReconciliation:
    if local_gross_usd < 0 or (
        provider_console_gross_usd is not None and provider_console_gross_usd < 0
    ):
        raise ValueError("billing totals cannot be negative")
    if provider_console_gross_usd is None:
        return BillingReconciliation(local_gross_usd, None, None, "console-unavailable")
    delta = provider_console_gross_usd - local_gross_usd
    return BillingReconciliation(local_gross_usd, provider_console_gross_usd, delta, "reconciled")
