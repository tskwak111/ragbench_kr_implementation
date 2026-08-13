from __future__ import annotations

import pytest

from ragbench.evaluation.bootstrap import (
    PairedObservation,
    align_paired_observations,
    paired_bootstrap,
)


def _observations() -> tuple[PairedObservation, ...]:
    return tuple(
        PairedObservation(
            observation_id=f"q{index}",
            left=0.8 + (index % 3) * 0.01,
            right=0.7 + (index % 2) * 0.01,
            document_cluster_id=f"d{index // 5}",
        )
        for index in range(30)
    )


def test_final_bootstrap_requires_fixed_seed_and_ten_thousand_resamples() -> None:
    with pytest.raises(ValueError, match="10,000"):
        paired_bootstrap(_observations(), seed=1, resamples=9_999, final=True)
    with pytest.raises(ValueError, match="seed"):
        paired_bootstrap(_observations(), seed=None, resamples=10_000, final=True)


def test_paired_document_cluster_bootstrap_is_deterministic() -> None:
    left = paired_bootstrap(_observations(), seed=20260813, resamples=10_000, final=True)
    right = paired_bootstrap(_observations(), seed=20260813, resamples=10_000, final=True)
    assert left == right
    assert left.method == "document-cluster-paired-bootstrap"
    assert left.effect == pytest.approx(0.105)
    assert left.ci_low <= left.effect <= left.ci_high
    assert left.sample_count == 30
    assert left.cluster_count == 6


def test_bootstrap_requires_cluster_ids_for_preferred_final_method() -> None:
    rows = (
        PairedObservation("q1", 1.0, 0.0, None),
        PairedObservation("q2", 0.0, 1.0, None),
    )
    with pytest.raises(ValueError, match="cluster"):
        paired_bootstrap(rows, seed=1, resamples=10_000, final=True)


def test_pair_alignment_rejects_missing_duplicate_or_changed_cluster_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        align_paired_observations(
            (("q1", 1.0, "d1"), ("q1", 0.0, "d1")),
            (("q1", 1.0, "d1"),),
        )
    with pytest.raises(ValueError, match="same observation"):
        align_paired_observations(
            (("q1", 1.0, "d1"),),
            (("q2", 1.0, "d1"),),
        )
    with pytest.raises(ValueError, match="cluster"):
        align_paired_observations(
            (("q1", 1.0, "d1"),),
            (("q1", 1.0, "d2"),),
        )


def test_nonfinal_observation_bootstrap_is_explicit_sensitivity_mode() -> None:
    result = paired_bootstrap(
        _observations(),
        seed=9,
        resamples=500,
        final=False,
        cluster_by_document=False,
    )
    assert result.method == "observation-paired-bootstrap-sensitivity"
    assert result.resamples == 500


def test_bootstrap_runtime_contract_rejects_bool_scores_and_noninteger_seed() -> None:
    with pytest.raises(TypeError, match="scores"):
        PairedObservation("q", True, False, "d")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="seed"):
        paired_bootstrap(_observations(), seed="a", resamples=500, final=False)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="final"):
        paired_bootstrap(_observations(), seed=1, resamples=500, final=1)  # type: ignore[arg-type]
