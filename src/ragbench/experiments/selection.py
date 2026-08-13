"""Predeclared retrieval-screen ranking, diversity, and public leaderboard export."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ragbench.core.hashing import canonical_json_hash
from ragbench.experiments.config import RetrievalExperimentConfig

SELECTION_RULE = {
    "version": "retrieval-shortlist-v1",
    "order": [
        "recall_at_k descending",
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
    recall_at_k: float
    mrr: float
    mean_latency_ms: float
    per_type: Mapping[str, Mapping[str, float | int | None]]
    bootstrap_inputs_hash: str

    def __post_init__(self) -> None:
        values = (self.recall_at_k, self.mrr, self.mean_latency_ms)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("screening outcome values must be finite")
        if not 0 <= self.recall_at_k <= 1 or not 0 <= self.mrr <= 1:
            raise ValueError("screening quality metrics must be between zero and one")
        if self.mean_latency_ms < 0:
            raise ValueError("screening latency cannot be negative")
        if len(self.bootstrap_inputs_hash) != 64:
            raise ValueError("bootstrap input hash must be a SHA-256 digest")


def _quality_key(outcome: ScreeningOutcome) -> tuple[float, float, float, str]:
    return (
        -outcome.recall_at_k,
        -outcome.mrr,
        outcome.mean_latency_ms,
        outcome.config.semantic_hash,
    )


def select_retrieval_shortlist(
    outcomes: Sequence[ScreeningOutcome],
    *,
    size: int = 8,
    enforce_core_diversity: bool = True,
) -> tuple[ScreeningOutcome, ...]:
    """Rank by the frozen rule, applying caps only after collapsing K-only variants."""
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("shortlist size must be a positive integer")
    hashes = [outcome.config.semantic_hash for outcome in outcomes]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate semantic screening outcomes are not allowed")

    best_by_family: dict[tuple[str, str, str], ScreeningOutcome] = {}
    for outcome in sorted(outcomes, key=_quality_key):
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


def export_retrieval_leaderboard(
    outcomes: Sequence[ScreeningOutcome], path: Path
) -> None:
    """Publish metrics and CI input identities without fabricating uncomputed intervals."""
    ordered = sorted(outcomes, key=_quality_key)
    payload = {
        "schema_version": "retrieval-leaderboard-v1",
        "selection_rule": SELECTION_RULE,
        "selection_rule_hash": SELECTION_RULE_HASH,
        "rows": [
            {
                "config_hash": outcome.config.semantic_hash,
                "recall_at_k": outcome.recall_at_k,
                "mrr": outcome.mrr,
                "mean_latency_ms": outcome.mean_latency_ms,
                "per_type": outcome.per_type,
                "bootstrap_inputs_hash": outcome.bootstrap_inputs_hash,
            }
            for outcome in ordered
        ],
        "paired_ci_status": "not-computed; paired inputs are identified by hash",
    }
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
        stream.write("\n")
