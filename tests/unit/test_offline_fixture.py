from __future__ import annotations

import json
from pathlib import Path

from ragbench.offline import run_offline_experiment, run_offline_screen

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_mini_retrieval_screen_executes_real_bm25_and_metrics_without_provider(
    tmp_path: Path,
) -> None:
    result = run_offline_screen(
        PROJECT_ROOT / "tests" / "fixtures" / "mini-screen.yaml", output_root=tmp_path
    )

    assert result["schema_version"] == "offline-screen-result-v1"
    assert result["provider_calls"] == 0
    assert result["question_count"] == 2
    assert result["metrics"]["hit_at_k"] == 1.0
    assert result["metrics"]["mrr"] == 1.0
    artifact = tmp_path / f"{result['run_id']}.json"
    assert json.loads(artifact.read_text(encoding="utf-8")) == result


def test_mini_experiment_generates_cited_answers_and_is_idempotent(tmp_path: Path) -> None:
    config = PROJECT_ROOT / "tests" / "fixtures" / "mini-experiment.yaml"

    first = run_offline_experiment(config, output_root=tmp_path)
    second = run_offline_experiment(config, output_root=tmp_path)

    assert first["schema_version"] == "offline-experiment-result-v1"
    assert first["provider_calls"] == 0
    assert first["new_responses"] == 2
    assert first["cache_reused"] is False
    assert second["new_responses"] == 0
    assert second["cache_reused"] is True
    assert second["response_count"] == 2
    assert first["responses"] == second["responses"]
    assert second["metrics"] == {
        "abstention_accuracy": 1.0,
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "deterministic_correctness": 1.0,
        "faithfulness": 1.0,
    }
    assert first["responses"][0]["citations"] == ["fixture-sales"]
    artifact = tmp_path / f"{first['run_id']}.json"
    assert len(tuple(tmp_path.glob("*.json"))) == 1
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert saved["new_responses"] == 2
    assert "cache_reused" not in saved


def test_offline_fixture_rejects_provider_and_gold_configuration(tmp_path: Path) -> None:
    config = tmp_path / "unsafe.yaml"
    config.write_text(
        """
schema_version: offline-fixture-v1
name: unsafe
snapshots:
  corpus: fixture-corpus-v1
  parse: fixture-parse-v1
  chunks: fixture-chunks-v1
  embeddings: none
  questions: sealed-gold
retrieval: {top_k: 1}
documents: [{chunk_id: c1, document_id: d1, content: text}]
questions:
  - question_id: q1
    question_type: fact
    prompt: text
    expected_answer: text
    answerable: true
    evidence_chunk_ids: [c1]
""".strip(),
        encoding="utf-8",
    )

    try:
        run_offline_experiment(config, output_root=tmp_path / "out")
    except ValueError as error:
        assert "gold" in str(error).lower() or "restricted" in str(error).lower()
    else:
        raise AssertionError("restricted fixture must be rejected")
