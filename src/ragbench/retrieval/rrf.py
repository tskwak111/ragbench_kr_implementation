"""Deterministic reciprocal rank fusion."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ragbench.retrieval.base import SearchHit


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[SearchHit]:
    """Fuse rankings using ``weight / (k + rank)`` with ranks starting at one.

    Duplicate IDs inside one ranking are rejected because they make a component rank ambiguous.
    Equal fused scores are ordered by ascending stable chunk ID.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    resolved_weights = tuple(1.0 for _ in rankings) if weights is None else tuple(weights)
    if len(resolved_weights) != len(rankings):
        raise ValueError("weights length must match rankings length")
    if any(not math.isfinite(weight) or weight <= 0 for weight in resolved_weights):
        raise ValueError("weights must be finite and positive")

    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, resolved_weights, strict=True):
        seen: set[str] = set()
        for expected_rank, hit in enumerate(ranking, start=1):
            if hit.rank != expected_rank:
                raise ValueError("component ranks must be sequential and start at one")
            if hit.chunk_id in seen:
                raise ValueError("duplicate chunk IDs are not allowed within a ranking")
            seen.add(hit.chunk_id)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (k + hit.rank)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        SearchHit(chunk_id, score, rank, "rrf")
        for rank, (chunk_id, score) in enumerate(ranked, start=1)
    ]
