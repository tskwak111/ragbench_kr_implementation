from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from ragbench.experiments.runner import (
    ExperimentConfig,
    ExperimentQuestionResult,
    ExperimentRunner,
    FileExperimentRepository,
    RunStatus,
    UsageSnapshot,
)


def _config(root: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "schema_version": "generation-experiment-v1",
            "name": "resume",
            "snapshots": {
                "corpus": "c",
                "parse": "p",
                "chunks": "h",
                "embeddings": "e",
                "questions": "q",
            },
            "retrieval": {"config_hash": "a" * 64, "top_k": 5},
            "generation": {
                "provider": "upstage",
                "model_alias": "solar-pro3",
                "model_id": "solar-pro3-exact",
                "prompt_version": "v3",
                "temperature": 0.0,
                "max_output_tokens": 32,
                "worst_case_input_tokens": 128,
            },
            "evaluation": {
                "deterministic_version": "v1",
                "judge_model_id": "solar-pro4-exact",
                "judge_prompt_version": "v1",
            },
            "runtime": {
                "concurrency": 5,
                "max_retries": 1,
                "seed": 1,
                "batch_cap": 3,
                "budget_cap_usd": "1",
                "schema_error_rate_stop": 0.9,
                "provider_error_rate_stop": 0.9,
                "error_window": 3,
            },
            "question_ids": ["q1", "q2", "q3"],
            "output_dir": str(root),
            "code_commit": "d1cc890",
        }
    )


@pytest.mark.asyncio
async def test_partial_failure_resumes_idempotently_without_repeating_success(
    tmp_path: Path,
) -> None:
    repo = FileExperimentRepository(tmp_path / "runs")
    run = repo.create(_config(tmp_path), now=datetime(2026, 8, 14, tzinfo=UTC))
    calls: list[str] = []

    async def first(question_id: str) -> ExperimentQuestionResult:
        calls.append(question_id)
        if question_id == "q2":
            raise RuntimeError("temporary")
        return ExperimentQuestionResult(
            question_id=question_id,
            response={"answer": "ok"},
            evidence=(),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1, cost_usd=Decimal("0")),
            cached=False,
        )

    failed = await ExperimentRunner(repo, first).run(run.run_id)
    assert failed.status is RunStatus.FAILED
    assert failed.completed_question_ids == frozenset({"q1", "q3"})

    async def second(question_id: str) -> ExperimentQuestionResult:
        calls.append(question_id)
        return ExperimentQuestionResult(
            question_id=question_id,
            response={"answer": "ok"},
            evidence=(),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1, cost_usd=Decimal("0")),
            cached=True,
        )

    completed = await ExperimentRunner(repo, second).run(run.run_id, resume=True)
    assert completed.status is RunStatus.COMPLETED
    assert completed.failed_question_ids == frozenset()
    assert calls.count("q1") == calls.count("q3") == 1
    assert calls.count("q2") == 2
    assert repo.result(run.run_id, "q2").cached is True


def test_resume_rejects_config_mutation_and_threshold_stop_needs_diagnosis(tmp_path: Path) -> None:
    repo = FileExperimentRepository(tmp_path / "runs")
    run = repo.create(_config(tmp_path), now=datetime(2026, 8, 14, tzinfo=UTC))
    config_path = tmp_path / "runs" / run.config_hash / run.run_id / "config.json"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace('"name": "resume"', '"name": "changed"'),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mutated"):
        repo.load(run.run_id)


@pytest.mark.asyncio
async def test_threshold_stop_requires_diagnosis_acknowledgement(tmp_path: Path) -> None:
    repo = FileExperimentRepository(tmp_path / "runs")
    run = repo.create(_config(tmp_path), now=datetime(2026, 8, 14, tzinfo=UTC))

    async def malformed(question_id: str) -> ExperimentQuestionResult:
        raise ValueError(f"malformed {question_id}")

    stopped = await ExperimentRunner(repo, malformed).run(run.run_id)
    assert stopped.status is RunStatus.STOPPED
    assert stopped.stop_reason == "schema_error_threshold"

    with pytest.raises(PermissionError, match="diagnosis acknowledgement"):
        await ExperimentRunner(repo, malformed).run(run.run_id, resume=True)
