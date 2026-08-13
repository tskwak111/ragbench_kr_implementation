import json
from pathlib import Path

import pytest

from ragbench.experiments.config import RetrievalExperimentConfig
from ragbench.experiments.planner import (
    CHUNK_STRATEGIES,
    CoreSnapshotBinding,
    generate_core_retrieval_configs,
)
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
        hit_at_5=float(recall > 0),
        recall_at_5=recall,
        micro_recall_at_5=recall,
        mrr=mrr,
        mean_latency_ms=latency,
        question_count=10,
        scorable_count=9,
        no_evidence_count=1,
        per_type={
            "fact": {
                "hit_at_5": float(recall > 0),
                "recall_at_5": recall,
                "micro_recall_at_5": recall,
                "mrr": mrr,
                "question_count": 10,
                "scorable_count": 9,
                "no_evidence_count": 1,
            }
        },
        bootstrap_inputs_hash="b" * 64,
    )


def _bindings() -> tuple[CoreSnapshotBinding, ...]:
    return tuple(
        CoreSnapshotBinding(
            parse_mode=mode,
            parse_snapshot_id=f"parse-{mode}",
            chunk_strategy=strategy,
            chunk_snapshot_id=f"chunk-{mode}-{strategy}",
            embedding_snapshot_id=f"00000000-0000-0000-{mode_index:04d}-{index:012d}",
        )
        for mode_index, mode in enumerate(("standard", "enhanced"), start=1)
        for index, strategy in enumerate(CHUNK_STRATEGIES, start=1)
    )


def test_shortlist_uses_predeclared_quality_order_with_diversity_constraints() -> None:
    configs = generate_core_retrieval_configs(
        corpus_snapshot_id="corpus",
        question_snapshot_id="dev",
        code_commit="0bce46e",
        random_seed=1,
        snapshot_bindings=_bindings(),
    )
    outcomes = tuple(
        _outcome(config, recall=1 - index / 1000, mrr=0.5, latency=10 + index)
        for index, config in enumerate(configs)
    )

    selected = select_retrieval_shortlist(outcomes, size=8)

    assert len(selected) == 8
    families = {
        (row.config.parse_mode, row.config.chunk_strategy, row.config.retriever) for row in selected
    }
    assert len(families) == 8
    assert (
        max(
            sum(row.config.parse_mode == mode for row in selected)
            for mode in ("standard", "enhanced")
        )
        <= 4
    )
    assert (
        max(
            sum(row.config.retriever == name for row in selected)
            for name in ("dense", "bm25", "hybrid")
        )
        <= 4
    )


def test_shortlist_ties_break_by_mrr_then_latency_then_semantic_hash() -> None:
    configs = generate_core_retrieval_configs(
        corpus_snapshot_id="corpus",
        question_snapshot_id="dev",
        code_commit="0bce46e",
        random_seed=1,
        snapshot_bindings=_bindings(),
    )
    candidates = (
        _outcome(configs[0], recall=0.8, mrr=0.6, latency=20),
        _outcome(configs[9], recall=0.8, mrr=0.7, latency=30),
        _outcome(configs[18], recall=0.8, mrr=0.7, latency=10),
    )

    selected = select_retrieval_shortlist(
        candidates, size=2, enforce_core_diversity=False, require_complete_grid=False
    )

    assert selected == (candidates[2], candidates[1])


def test_leaderboard_export_contains_metrics_per_type_rule_and_no_fake_ci(tmp_path: Path) -> None:
    config = generate_core_retrieval_configs(
        corpus_snapshot_id="corpus",
        question_snapshot_id="dev",
        code_commit="0bce46e",
        random_seed=1,
        snapshot_bindings=_bindings(),
    )[0]
    outcome = _outcome(config, recall=0.75, mrr=0.5, latency=12)
    path = tmp_path / "leaderboard.json"

    export_retrieval_leaderboard((outcome,), path)

    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["selection_rule"]["order"] == [
        "recall_at_5 descending",
        "mrr descending",
        "mean_latency_ms ascending",
        "semantic_hash ascending",
    ]
    assert artifact["rows"][0]["per_type"]["fact"]["recall_at_5"] == 0.75
    assert artifact["rows"][0]["hit_at_5"] == 1.0
    assert artifact["rows"][0]["micro_recall_at_5"] == 0.75
    assert artifact["rows"][0]["question_count"] == 10
    assert artifact["rows"][0]["bootstrap_inputs_hash"] == "b" * 64
    assert "confidence_interval" not in artifact["rows"][0]


def test_shortlist_rejects_duplicate_or_nonfinite_outcomes() -> None:
    config = generate_core_retrieval_configs(
        corpus_snapshot_id="corpus",
        question_snapshot_id="dev",
        code_commit="0bce46e",
        random_seed=1,
        snapshot_bindings=_bindings(),
    )[0]
    outcome = _outcome(config, recall=0.7, mrr=0.5, latency=12)
    with pytest.raises(ValueError, match="duplicate"):
        select_retrieval_shortlist((outcome, outcome), size=1)
    with pytest.raises(ValueError, match="finite"):
        ScreeningOutcome(
            config=config,
            hit_at_5=1,
            recall_at_5=float("nan"),
            micro_recall_at_5=0.5,
            mrr=0.5,
            mean_latency_ms=12,
            question_count=1,
            scorable_count=1,
            no_evidence_count=0,
            per_type={},
            bootstrap_inputs_hash="b" * 64,
        )
