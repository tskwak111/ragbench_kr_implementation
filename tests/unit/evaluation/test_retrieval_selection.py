import json
from pathlib import Path

import pytest

from ragbench.experiments.config import RetrievalExperimentConfig
from ragbench.experiments.planner import generate_core_retrieval_configs
from ragbench.experiments.selection import (
    ScreeningOutcome,
    export_retrieval_leaderboard,
    select_retrieval_shortlist,
)


def _outcome(
    config: RetrievalExperimentConfig,
    *,
    recall: float,
    mrr: float,
    latency: float,
) -> ScreeningOutcome:
    return ScreeningOutcome(
        config=config,
        recall_at_k=recall,
        mrr=mrr,
        mean_latency_ms=latency,
        per_type={"fact": {"recall_at_k": recall, "mrr": mrr}},
        bootstrap_inputs_hash="b" * 64,
    )


def test_shortlist_uses_predeclared_quality_order_with_diversity_constraints() -> None:
    configs = generate_core_retrieval_configs(
        corpus_snapshot_id="corpus",
        question_snapshot_id="dev",
        code_commit="0bce46e",
        random_seed=1,
    )
    outcomes = tuple(
        _outcome(config, recall=1 - index / 1000, mrr=0.5, latency=10 + index)
        for index, config in enumerate(configs)
    )

    selected = select_retrieval_shortlist(outcomes, size=8)

    assert len(selected) == 8
    families = {
        (row.config.parse_mode, row.config.chunk_strategy, row.config.retriever)
        for row in selected
    }
    assert len(families) == 8
    assert max(
        sum(row.config.parse_mode == mode for row in selected)
        for mode in ("standard", "enhanced")
    ) <= 4
    assert max(
        sum(row.config.retriever == name for row in selected)
        for name in ("dense", "bm25", "hybrid")
    ) <= 4


def test_shortlist_ties_break_by_mrr_then_latency_then_semantic_hash() -> None:
    configs = generate_core_retrieval_configs(
        corpus_snapshot_id="corpus",
        question_snapshot_id="dev",
        code_commit="0bce46e",
        random_seed=1,
    )
    candidates = (
        _outcome(configs[0], recall=0.8, mrr=0.6, latency=20),
        _outcome(configs[9], recall=0.8, mrr=0.7, latency=30),
        _outcome(configs[18], recall=0.8, mrr=0.7, latency=10),
    )

    selected = select_retrieval_shortlist(candidates, size=2, enforce_core_diversity=False)

    assert selected == (candidates[2], candidates[1])


def test_leaderboard_export_contains_metrics_per_type_rule_and_no_fake_ci(tmp_path: Path) -> None:
    config = generate_core_retrieval_configs(
        corpus_snapshot_id="corpus",
        question_snapshot_id="dev",
        code_commit="0bce46e",
        random_seed=1,
    )[0]
    outcome = _outcome(config, recall=0.75, mrr=0.5, latency=12)
    path = tmp_path / "leaderboard.json"

    export_retrieval_leaderboard((outcome,), path)

    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["selection_rule"]["order"] == [
        "recall_at_k descending",
        "mrr descending",
        "mean_latency_ms ascending",
        "semantic_hash ascending",
    ]
    assert artifact["rows"][0]["per_type"]["fact"]["recall_at_k"] == 0.75
    assert artifact["rows"][0]["bootstrap_inputs_hash"] == "b" * 64
    assert "confidence_interval" not in artifact["rows"][0]


def test_shortlist_rejects_duplicate_or_nonfinite_outcomes() -> None:
    config = generate_core_retrieval_configs(
        corpus_snapshot_id="corpus",
        question_snapshot_id="dev",
        code_commit="0bce46e",
        random_seed=1,
    )[0]
    outcome = _outcome(config, recall=0.7, mrr=0.5, latency=12)
    with pytest.raises(ValueError, match="duplicate"):
        select_retrieval_shortlist((outcome, outcome), size=1)
    with pytest.raises(ValueError, match="finite"):
        ScreeningOutcome(
            config=config,
            recall_at_k=float("nan"),
            mrr=0.5,
            mean_latency_ms=12,
            per_type={},
            bootstrap_inputs_hash="b" * 64,
        )
