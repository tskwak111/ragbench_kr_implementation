from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from ragbench.core.money import BudgetExceededError
from ragbench.experiments.runner import (
    ExperimentConfig,
    ExperimentQuestionResult,
    ExperimentRunner,
    FileExperimentRepository,
    RunStatus,
    UsageSnapshot,
    authorize_paid_execution,
    build_development_campaign,
    build_dry_run_plan,
)
from ragbench.providers.upstage.pricing import PriceBook


def _config(tmp_path: Path, *, question_count: int = 3) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "schema_version": "generation-experiment-v1",
            "name": "dev-hybrid-v3",
            "snapshots": {
                "corpus": "corpus-a",
                "parse": "parse-a",
                "chunks": "chunks-a",
                "embeddings": "embeddings-a",
                "questions": "dev-a",
            },
            "retrieval": {"config_hash": "a" * 64, "top_k": 5},
            "generation": {
                "provider": "upstage",
                "model_alias": "solar-pro3",
                "model_id": "solar-pro3-2026-08-01",
                "prompt_version": "v3",
                "temperature": 0.0,
                "max_output_tokens": 512,
                "worst_case_input_tokens": 2048,
            },
            "evaluation": {
                "deterministic_version": "v1",
                "judge_model_id": "solar-pro4-2026-08-01",
                "judge_prompt_version": "v1",
            },
            "runtime": {
                "concurrency": 5,
                "max_retries": 5,
                "seed": 20260813,
                "batch_cap": question_count,
                "budget_cap_usd": "1.00",
                "schema_error_rate_stop": 0.25,
                "provider_error_rate_stop": 0.5,
                "error_window": 4,
            },
            "question_ids": [f"q-{index}" for index in range(question_count)],
            "output_dir": str(tmp_path / "runs"),
            "code_commit": "d1cc890",
        }
    )


def _result(question_id: str, *, cached: bool = False) -> ExperimentQuestionResult:
    return ExperimentQuestionResult(
        question_id=question_id,
        response={"answer": f"answer-{question_id}", "abstained": False},
        evidence=({"chunk_id": "chunk-1", "page": 1},),
        usage=UsageSnapshot(input_tokens=10, output_tokens=2, cost_usd=Decimal("0.0001")),
        cached=cached,
        correlation_id=f"correlation-{question_id}",
    )


def test_config_hash_is_semantic_and_concurrency_ten_requires_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    copy = config.model_copy()
    relocated = config.model_copy(
        update={"name": "operator-label", "output_dir": str(tmp_path / "elsewhere")}
    )

    assert copy.semantic_hash == config.semantic_hash
    assert relocated.semantic_hash == config.semantic_hash
    with pytest.raises(ValidationError, match="concurrency evidence"):
        ExperimentConfig.model_validate(
            {
                **config.model_dump(mode="json"),
                "runtime": {**config.runtime.model_dump(mode="json"), "concurrency": 10},
            }
        )
    with pytest.raises(ValidationError, match="duplicate"):
        ExperimentConfig.model_validate(
            {**config.model_dump(mode="json"), "question_ids": ["q-1", "q-1"]}
        )


def test_dry_run_discloses_unknown_cache_and_gross_worst_case(tmp_path: Path) -> None:
    config = _config(tmp_path)
    price_book = PriceBook(
        {
            "schema_version": "test",
            "verified_at": "2026-08-14T00:00:00Z",
            "vat_excluded": True,
            "models": {
                "solar-pro3-2026-08-01": {
                    "generation": {
                        "input_usd_per_million": "1",
                        "output_usd_per_million": "2",
                    }
                }
            },
        }
    )

    plan = build_dry_run_plan(config, price_book=price_book, cached_question_ids={"q-0"})

    assert plan.question_count == 3
    assert (plan.cached_calls, plan.new_calls, plan.cache_status) == (1, 2, "known")
    assert plan.worst_case_tokens == 3 * (2048 + 512)
    assert plan.net_worst_cost_usd == Decimal("0.006144")
    assert plan.gross_worst_cost_usd == Decimal("0.006759")
    assert plan.model_alias == "solar-pro3"
    assert plan.model_id == "solar-pro3-2026-08-01"
    assert plan.prompt_version == "v3"
    assert plan.destinations[0].endswith(config.semantic_hash)

    unknown = build_dry_run_plan(config, price_book=price_book)
    assert unknown.cached_calls is None and unknown.new_calls is None
    assert unknown.cache_status == "unknown"


def test_paid_authorization_requires_exact_fresh_plan_live_gate_and_budget(tmp_path: Path) -> None:
    config = _config(tmp_path)
    price_book = PriceBook(
        {
            "schema_version": "test",
            "verified_at": "2026-08-14T00:00:00Z",
            "vat_excluded": True,
            "models": {
                config.generation.model_id: {
                    "generation": {
                        "input_usd_per_million": "1",
                        "output_usd_per_million": "2",
                    }
                }
            },
        }
    )
    plan = build_dry_run_plan(config, price_book=price_book, cached_question_ids=set())

    with pytest.raises(PermissionError, match="live gate"):
        authorize_paid_execution(
            plan,
            execute=True,
            live_enabled=False,
            confirmed_plan_hash=plan.plan_hash,
            price_book=price_book,
            available_budget_usd=Decimal("1"),
            now=datetime(2026, 8, 14, 1, tzinfo=UTC),
        )
    with pytest.raises(PermissionError, match="plan hash"):
        authorize_paid_execution(
            plan,
            execute=True,
            live_enabled=True,
            confirmed_plan_hash="0" * 64,
            price_book=price_book,
            available_budget_usd=Decimal("1"),
            now=datetime(2026, 8, 14, 1, tzinfo=UTC),
        )
    assert authorize_paid_execution(
        plan,
        execute=True,
        live_enabled=True,
        confirmed_plan_hash=plan.plan_hash,
        price_book=price_book,
        available_budget_usd=Decimal("1"),
        now=datetime(2026, 8, 14, 1, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_state_machine_completes_and_duplicate_hash_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    repository = FileExperimentRepository(tmp_path / "runs")
    run = repository.create(config, now=datetime(2026, 8, 14, tzinfo=UTC))

    async def execute(question_id: str) -> ExperimentQuestionResult:
        return _result(question_id)

    summary = await ExperimentRunner(repository, execute).run(run.run_id)

    assert summary.status is RunStatus.COMPLETED
    assert summary.completed_question_ids == frozenset(config.question_ids)
    assert [item.status for item in repository.history(run.run_id)] == [
        RunStatus.PLANNED,
        RunStatus.RUNNING,
        RunStatus.COMPLETED,
    ]
    with pytest.raises(FileExistsError, match="semantic config"):
        repository.create(config, now=datetime(2026, 8, 15, tzinfo=UTC))


@pytest.mark.asyncio
async def test_runner_stops_on_budget_and_cancellation_is_persisted(tmp_path: Path) -> None:
    config = _config(tmp_path, question_count=1)
    repository = FileExperimentRepository(tmp_path / "runs")
    budget_run = repository.create(config, now=datetime(2026, 8, 14, tzinfo=UTC))

    async def over_budget(question_id: str) -> ExperimentQuestionResult:
        raise BudgetExceededError(question_id)

    stopped = await ExperimentRunner(repository, over_budget).run(budget_run.run_id)
    assert stopped.status is RunStatus.STOPPED
    assert stopped.stop_reason == "budget_exhausted"

    other = _config(tmp_path / "other", question_count=1)
    cancelled_run = FileExperimentRepository(tmp_path / "other-runs").create(
        other, now=datetime(2026, 8, 14, tzinfo=UTC)
    )
    other_repo = FileExperimentRepository(tmp_path / "other-runs")

    async def cancel(_: str) -> ExperimentQuestionResult:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await ExperimentRunner(other_repo, cancel).run(cancelled_run.run_id)
    assert other_repo.load(cancelled_run.run_id).status is RunStatus.CANCELLED


def test_campaign_is_exactly_top_eight_by_three_prompts_by_five_hundred(tmp_path: Path) -> None:
    campaign = build_development_campaign(
        retrieval_config_hashes=tuple(f"{index:064x}" for index in range(8)),
        prompt_versions=("v1", "v2", "v3"),
        question_ids=tuple(f"q-{index}" for index in range(500)),
        base_config=_config(tmp_path, question_count=3),
    )

    assert len(campaign) == 24
    assert all(len(config.question_ids) == 500 for config in campaign)
    assert len({config.semantic_hash for config in campaign}) == 24
