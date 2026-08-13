from dataclasses import replace
from decimal import Decimal

import pytest

from ragbench.experiments.selection import (
    GenerationOutcome,
    reconcile_provider_billing,
    select_generation_top_three,
)


def _outcome(
    hash_char: str, quality: float, cost: str, *, calibrated: bool = True
) -> GenerationOutcome:
    return GenerationOutcome(
        config_hash=hash_char * 64,
        cohort_hash="f" * 64,
        question_count=500,
        correctness=quality,
        faithfulness=quality,
        citation_f1=quality,
        abstention_accuracy=quality,
        mean_latency_ms=100,
        total_cost_usd=Decimal(cost),
        quality_ci_low=quality - 0.01,
        quality_ci_high=quality + 0.01,
        judge_calibrated=calibrated,
    )


def test_selects_quality_leaders_and_preserves_statistically_competitive_best_value() -> None:
    outcomes = (
        _outcome("a", 0.91, "10"),
        _outcome("b", 0.90, "9"),
        _outcome("c", 0.895, "2"),
        _outcome("d", 0.70, "1"),
    )

    selected = select_generation_top_three(outcomes)

    assert [row.config_hash for row in selected] == ["a" * 64, "b" * 64, "c" * 64]
    assert selected[-1].total_cost_usd == Decimal("2")


def test_selection_rejects_mixed_cohorts_uncalibrated_or_duplicate_results() -> None:
    one = _outcome("a", 0.9, "2")
    with pytest.raises(ValueError, match="cohort"):
        select_generation_top_three(
            (
                one,
                replace(one, config_hash="b" * 64, cohort_hash="e" * 64),
                _outcome("c", 0.7, "1"),
            )
        )
    with pytest.raises(ValueError, match="calibrated"):
        select_generation_top_three(
            (one, _outcome("b", 0.8, "1", calibrated=False), _outcome("c", 0.7, "1"))
        )
    with pytest.raises(ValueError, match="duplicate"):
        select_generation_top_three((one, one, _outcome("c", 0.7, "1")))


def test_billing_reconciliation_never_fabricates_console_totals() -> None:
    pending = reconcile_provider_billing(
        local_gross_usd=Decimal("1.10"), provider_console_gross_usd=None
    )
    reconciled = reconcile_provider_billing(
        local_gross_usd=Decimal("1.10"), provider_console_gross_usd=Decimal("1.12")
    )

    assert (pending.status, pending.delta_usd) == ("console-unavailable", None)
    assert reconciled.status == "reconciled"
    assert reconciled.delta_usd == Decimal("0.02")
