"""Validated paired bootstrap confidence intervals for benchmark comparisons."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PairedObservation:
    observation_id: str
    left: float
    right: float
    document_cluster_id: str | None

    def __post_init__(self) -> None:
        if not self.observation_id.strip():
            raise ValueError("observation ID cannot be blank")
        if not math.isfinite(self.left) or not math.isfinite(self.right):
            raise ValueError("paired values must be finite")
        if self.document_cluster_id is not None and not self.document_cluster_id.strip():
            raise ValueError("document cluster ID cannot be blank")


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    effect: float
    ci_low: float
    ci_high: float
    confidence: float
    resamples: int
    seed: int
    method: str
    sample_count: int
    cluster_count: int | None


def align_paired_observations(
    left: Sequence[tuple[str, float, str | None]],
    right: Sequence[tuple[str, float, str | None]],
) -> tuple[PairedObservation, ...]:
    """Align raw system observations by immutable ID and document cluster."""
    left_ids = [row[0] for row in left]
    right_ids = [row[0] for row in right]
    if len(left_ids) != len(set(left_ids)) or len(right_ids) != len(set(right_ids)):
        raise ValueError("duplicate observation IDs are not allowed")
    left_by_id = {row[0]: row for row in left}
    right_by_id = {row[0]: row for row in right}
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("paired systems must contain the same observation IDs")
    output: list[PairedObservation] = []
    for observation_id in sorted(left_by_id):
        left_row, right_row = left_by_id[observation_id], right_by_id[observation_id]
        if left_row[2] != right_row[2]:
            raise ValueError("paired observation document cluster IDs do not match")
        output.append(PairedObservation(observation_id, left_row[1], right_row[1], left_row[2]))
    return tuple(output)


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap(
    observations: Sequence[PairedObservation],
    *,
    seed: int | None,
    resamples: int,
    final: bool,
    cluster_by_document: bool = True,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Compute a deterministic percentile CI over paired effects (left minus right)."""
    if seed is None:
        raise ValueError("a fixed bootstrap seed is required")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    if final and resamples < 10_000:
        raise ValueError("final confidence intervals require at least 10,000 resamples")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if len(observations) < 2:
        raise ValueError("bootstrap requires at least two paired observations")
    identifiers = [row.observation_id for row in observations]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate observation IDs are not allowed")
    if final and not cluster_by_document:
        raise ValueError("final intervals require document-cluster bootstrap")

    effects = [row.left - row.right for row in observations]
    effect = sum(effects) / len(effects)
    rng = random.Random(seed)
    sampled_effects: list[float] = []
    cluster_count: int | None = None
    if cluster_by_document:
        if any(row.document_cluster_id is None for row in observations):
            raise ValueError("document cluster IDs are required for cluster bootstrap")
        clusters: dict[str, list[float]] = defaultdict(list)
        for row, difference in zip(observations, effects, strict=True):
            cluster_id = row.document_cluster_id
            if cluster_id is None:  # guarded above; narrows for mypy
                raise AssertionError("unreachable missing cluster ID")
            clusters[cluster_id].append(difference)
        cluster_ids = sorted(clusters)
        if len(cluster_ids) < 2:
            raise ValueError("cluster bootstrap requires at least two document clusters")
        cluster_count = len(cluster_ids)
        for _ in range(resamples):
            sampled = [clusters[rng.choice(cluster_ids)] for _ in cluster_ids]
            flattened = [value for cluster in sampled for value in cluster]
            sampled_effects.append(sum(flattened) / len(flattened))
        method = "document-cluster-paired-bootstrap"
    else:
        for _ in range(resamples):
            sampled_effects.append(
                sum(rng.choice(effects) for _ in effects) / len(effects)
            )
        method = "observation-paired-bootstrap-sensitivity"
    sampled_effects.sort()
    alpha = (1 - confidence) / 2
    return BootstrapInterval(
        effect,
        _percentile(sampled_effects, alpha),
        _percentile(sampled_effects, 1 - alpha),
        confidence,
        resamples,
        seed,
        method,
        len(observations),
        cluster_count,
    )
