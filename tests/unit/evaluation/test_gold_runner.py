from __future__ import annotations

import asyncio
import json
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ragbench.benchmark.splits import GoldItem
from ragbench.core.hashing import canonical_json_hash
from ragbench.evaluation.gold import (
    BootstrapSpec,
    ExecutorSpec,
    GoldCohort,
    GoldEvaluationResult,
    GoldMetric,
    GoldPreregistration,
    GoldResultRepository,
    GoldRunner,
    PreregistrationEnvelope,
    PrimaryComparison,
    aggregate_gold_results,
    append_invalidation,
    build_gold_dry_run,
)
from ragbench.experiments.runner import ExperimentConfig


def _prereg(configs: tuple[ExperimentConfig, ...] | None = None) -> PreregistrationEnvelope:
    hashes = (
        tuple(config.semantic_hash for config in configs)
        if configs is not None
        else ("a" * 64, "b" * 64, "c" * 64)
    )
    registration = GoldPreregistration(
        schema_version="sealed-gold-preregistration-v1",
        config_hashes=hashes,
        metrics=tuple(GoldMetric),
        primary_comparison=PrimaryComparison(
            left_config_hash=hashes[0],
            right_config_hash=hashes[1],
            metric=GoldMetric.CORRECTNESS,
        ),
        bootstrap=BootstrapSpec(
            method="document-cluster-paired-bootstrap",
            resamples=10_000,
            seed=7,
            confidence_level=0.95,
        ),
        exclusions=("benchmark_defect",),
        stopping_rule="Run all once.",
        cohort=GoldCohort(
            snapshot_id="d" * 64,
            content_sha256="e" * 64,
            item_count=150,
            ordered_membership_hash=canonical_json_hash(
                tuple(f"restricted-{index:03d}" for index in range(150))
            ),
        ),
        code_commit="5421abd",
        protected_output_path_hash=canonical_json_hash("/protected/test-gold-run"),
        executor=ExecutorSpec(
            entrypoint="ragbench.evaluation.test_adapter:execute",
            source_sha256="4" * 64,
        ),
    )
    return PreregistrationEnvelope.sign(
        registration,
        signed_by="owner",
        signing_key=b"owner-test-signing-key",
        signed_at=datetime(2026, 8, 14, tzinfo=UTC),
    )


def _items() -> tuple[GoldItem, ...]:
    return tuple(
        GoldItem(
            item_id=f"restricted-{index:03d}",
            natural_question=f"비공개 질문 {index}",
            expected_answer=f"답 {index}",
            evidence=(f"근거 {index}",),
            question_type="fact" if index % 2 == 0 else "table",
            difficulty="medium",
            answerable=True,
            document_cluster_id=f"private-document-{index // 10}",
        )
        for index in reversed(range(150))
    )


def _result(config_hash: str, item: GoldItem) -> GoldEvaluationResult:
    index = int(item.item_id.rsplit("-", 1)[1])
    score = 1.0 if config_hash[0] == "a" else 0.5
    return GoldEvaluationResult(
        config_hash=config_hash,
        item_id=item.item_id,
        question_type=item.question_type,
        document_cluster_id=f"private-document-{index // 10}",
        correctness=score,
        faithfulness=score,
        citation=score,
        abstention=1.0,
        latency=10.0 + index,
        cost=0.001,
        hit=1.0,
        recall=score,
        mrr=score,
        output_hash=f"{index:064x}",
    )


def test_dry_run_uses_only_preregistration_metadata_and_exposes_no_restricted_identity() -> None:
    plan = build_gold_dry_run(_prereg())
    serialized = json.dumps(plan.model_dump(mode="json"), sort_keys=True)
    assert plan.executed is False
    assert plan.item_count == 150
    assert plan.config_count == 3
    assert "restricted" not in serialized
    assert "snapshot_id" not in serialized


def test_runner_uses_same_sorted_cohort_for_all_configs_and_resumes_idempotently(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []

    async def execute(config: ExperimentConfig, item: GoldItem) -> GoldEvaluationResult:
        config_hash = config.semantic_hash
        calls.append((config_hash, item.item_id))
        return _result(config_hash, item)

    repository = GoldResultRepository(tmp_path / "protected")
    runner = GoldRunner(repository, execute)
    configs = _runner_configs(tmp_path)
    envelope = _prereg(configs)
    first = asyncio.run(runner.run(envelope, configs, _items()))
    assert first.completed == 450
    expected_ids = tuple(f"restricted-{index:03d}" for index in range(150))
    for config_hash in envelope.preregistration.config_hashes:
        assert tuple(item_id for config, item_id in calls if config == config_hash) == expected_ids

    calls.clear()
    resumed = asyncio.run(runner.run(envelope, configs, _items(), resume=True))
    assert resumed.completed == 450
    assert calls == []
    assert stat.S_IMODE((tmp_path / "protected").stat().st_mode) == 0o700


def test_runner_rejects_wrong_cohort_or_result_identity_without_saving(tmp_path: Path) -> None:
    async def wrong(config: ExperimentConfig, item: GoldItem) -> GoldEvaluationResult:
        return _result(config.semantic_hash, item).model_copy(update={"item_id": "other"})

    runner = GoldRunner(GoldResultRepository(tmp_path / "protected"), wrong)
    with pytest.raises(ValueError, match="ordered cohort"):
        configs = _runner_configs(tmp_path)
        asyncio.run(runner.run(_prereg(configs), configs, _items()[:-1]))
    with pytest.raises(ValueError, match="wrong identity"):
        configs = _runner_configs(tmp_path)
        asyncio.run(runner.run(_prereg(configs), configs, _items()))


def test_runner_rejects_executor_controlled_type_or_cluster(tmp_path: Path) -> None:
    async def wrong_type(config: ExperimentConfig, item: GoldItem) -> GoldEvaluationResult:
        return _result(config.semantic_hash, item).model_copy(update={"question_type": "injected"})

    with pytest.raises(ValueError, match="stratum"):
        configs = _runner_configs(tmp_path)
        asyncio.run(
            GoldRunner(GoldResultRepository(tmp_path / "protected"), wrong_type).run(
                _prereg(configs), configs, _items()
            )
        )


def test_aggregate_is_paired_overall_and_by_type_and_public_safe(tmp_path: Path) -> None:
    repository = GoldResultRepository(tmp_path / "protected")
    envelope = _prereg()
    repository.initialize(envelope, tuple(sorted(_items(), key=lambda item: item.item_id)))
    for config_hash in envelope.preregistration.config_hashes:
        for item in _items():
            repository.save(_result(config_hash, item))

    report = aggregate_gold_results(repository, envelope, public_export_salt="publish-v1")
    serialized = json.dumps(report.model_dump(mode="json"), sort_keys=True)
    assert report.comparisons[0].metric is GoldMetric.CORRECTNESS
    assert report.comparisons[0].resamples == 10_000
    assert len(report.comparisons) == len(GoldMetric) * 3
    assert {row.metric for row in report.comparisons} == set(GoldMetric)
    assert {row.question_type for row in report.by_type} == {"fact", "table"}
    assert "restricted-" not in serialized
    assert "private-document" not in serialized
    assert "aaaaaaaa" not in serialized
    assert envelope.preregistration.cohort.snapshot_id not in serialized


def test_output_change_invalidation_is_append_only_and_blocks_aggregation(tmp_path: Path) -> None:
    repository = GoldResultRepository(tmp_path / "protected")
    envelope = _prereg()
    repository.initialize(envelope, tuple(sorted(_items(), key=lambda item: item.item_id)))
    record = append_invalidation(
        repository,
        reason="citation scoring bug changed outputs",
        replacement_code_commit="1234abc",
        invalidated_at=datetime(2026, 8, 15, tzinfo=UTC),
    )
    assert record.sequence == 1
    with pytest.raises(FileExistsError):
        repository.write_invalidation(record)
    with pytest.raises(ValueError, match="invalidated"):
        aggregate_gold_results(repository, envelope, public_export_salt="publish-v1")


def test_checkpoint_tampering_and_preregistration_rebinding_are_rejected(tmp_path: Path) -> None:
    repository = GoldResultRepository(tmp_path / "protected")
    envelope = _prereg()
    items = tuple(sorted(_items(), key=lambda item: item.item_id))
    repository.initialize(envelope, items)
    result = _result(envelope.preregistration.config_hashes[0], items[0])
    repository.save(result)
    result_path = next((tmp_path / "protected" / "results").glob("*/*.json"))
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    (payload.get("result") or payload)["correctness"] = 0.125
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    result_path.chmod(0o600)
    with pytest.raises(ValueError, match="artifact hash"):
        repository.load(result.config_hash, result.item_id)

    rebound = envelope.model_copy(update={"artifact_sha256": "9" * 64})
    with pytest.raises(ValueError, match="hash mismatch"):
        repository.assert_bound(rebound)


def test_aggregation_rejects_symlinked_or_misnamed_checkpoint(tmp_path: Path) -> None:
    repository = GoldResultRepository(tmp_path / "protected")
    envelope = _prereg()
    items = tuple(sorted(_items(), key=lambda item: item.item_id))
    repository.initialize(envelope, items)
    result = _result(envelope.preregistration.config_hashes[0], items[0])
    repository.save(result)
    result_path = next((tmp_path / "protected" / "results").glob("*/*.json"))
    moved = tmp_path / "moved.json"
    result_path.rename(moved)
    result_path.symlink_to(moved)
    with pytest.raises(ValueError, match="unsafe"):
        repository.all_results()


def _runner_configs(tmp_path: Path) -> tuple[ExperimentConfig, ...]:
    def config(suffix: str) -> ExperimentConfig:
        return ExperimentConfig.model_validate(
            {
                "schema_version": "generation-experiment-v1",
                "name": f"gold-{suffix}",
                "snapshots": {
                    "corpus": "corpus-a",
                    "parse": "parse-a",
                    "chunks": "chunks-a",
                    "embeddings": "embeddings-a",
                    "questions": "dev-a",
                },
                "retrieval": {"config_hash": suffix * 64, "top_k": 5},
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
                    "batch_cap": 150,
                    "budget_cap_usd": "10.00",
                    "schema_error_rate_stop": 0.25,
                    "provider_error_rate_stop": 0.5,
                    "error_window": 4,
                },
                "question_ids": [f"placeholder-{index}" for index in range(150)],
                "output_dir": str(tmp_path / f"runs-{suffix}"),
                "code_commit": "5421abd",
            }
        )

    return tuple(config(value) for value in "abc")
