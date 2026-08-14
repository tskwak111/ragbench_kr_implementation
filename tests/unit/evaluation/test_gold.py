from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from ragbench.benchmark.splits import GoldMetadata
from ragbench.evaluation.gold import (
    BootstrapSpec,
    ExecutorSpec,
    GoldCohort,
    GoldMetric,
    GoldPreregistration,
    PreregistrationEnvelope,
    PrimaryComparison,
    SolarExploratorySpec,
    load_preregistration,
    verify_frozen_inputs,
    write_preregistration,
)
from ragbench.experiments.runner import ExperimentConfig


def _config(tmp_path: Path, *, suffix: str) -> ExperimentConfig:
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


def _metadata() -> GoldMetadata:
    return GoldMetadata(
        snapshot_id="1" * 64,
        version="gold-v1",
        file_name="gold-v1.jsonl",
        content_sha256="2" * 64,
        item_count=150,
        scope_status="reduced",
        sealed_at=datetime(2026, 8, 14, tzinfo=UTC),
    )


def _preregistration(configs: tuple[ExperimentConfig, ...]) -> GoldPreregistration:
    return GoldPreregistration(
        schema_version="sealed-gold-preregistration-v1",
        config_hashes=tuple(config.semantic_hash for config in configs),
        metrics=tuple(GoldMetric),
        primary_comparison=PrimaryComparison(
            left_config_hash=configs[0].semantic_hash,
            right_config_hash=configs[1].semantic_hash,
            metric=GoldMetric.CORRECTNESS,
        ),
        bootstrap=BootstrapSpec(
            method="document-cluster-paired-bootstrap",
            resamples=10_000,
            seed=20260814,
            confidence_level=0.95,
        ),
        exclusions=("benchmark_defect",),
        stopping_rule="Run all preregistered items once; stop only on safety threshold.",
        cohort=GoldCohort(
            snapshot_id=_metadata().snapshot_id,
            content_sha256=_metadata().content_sha256,
            item_count=150,
            ordered_membership_hash="3" * 64,
        ),
        code_commit="5421abd",
        protected_output_path_hash="5" * 64,
        executor=ExecutorSpec(
            entrypoint="ragbench.evaluation.test_adapter:execute",
            source_sha256="4" * 64,
        ),
    )


def test_preregistration_requires_exactly_three_unique_hashes_all_metrics_and_final_bootstrap(
    tmp_path: Path,
) -> None:
    configs = tuple(_config(tmp_path, suffix=value) for value in "abc")
    prereg = _preregistration(configs)
    assert prereg.config_hashes == tuple(config.semantic_hash for config in configs)

    payload = prereg.model_dump()
    with pytest.raises(ValidationError, match="exactly three"):
        GoldPreregistration.model_validate(
            {**payload, "config_hashes": payload["config_hashes"][:2]}
        )
    with pytest.raises(ValidationError, match="unique"):
        GoldPreregistration.model_validate(
            {**payload, "config_hashes": (payload["config_hashes"][0],) * 3}
        )
    with pytest.raises(ValidationError, match="complete frozen metric set"):
        GoldPreregistration.model_validate({**payload, "metrics": payload["metrics"][:-1]})
    with pytest.raises(ValidationError, match="10000"):
        BootstrapSpec(
            method="document-cluster-paired-bootstrap",
            resamples=9_999,
            seed=20260814,
            confidence_level=0.95,
        )


def test_signed_preregistration_is_exclusive_hash_bound_and_detects_tampering(
    tmp_path: Path,
) -> None:
    configs = tuple(_config(tmp_path, suffix=value) for value in "abc")
    path = tmp_path / "preregistration.json"
    envelope = write_preregistration(
        path,
        _preregistration(configs),
        signed_by="benchmark-owner",
        signing_key=b"owner-test-signing-key",
        signed_at=datetime(2026, 8, 14, tzinfo=UTC),
    )

    assert load_preregistration(path, signing_key=b"owner-test-signing-key") == envelope
    assert len(envelope.artifact_sha256) == 64
    with pytest.raises(FileExistsError):
        write_preregistration(
            path,
            _preregistration(configs),
            signed_by="benchmark-owner",
            signing_key=b"owner-test-signing-key",
            signed_at=datetime(2026, 8, 14, tzinfo=UTC),
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["preregistration"]["stopping_rule"] = "changed after signing"
    from ragbench.core.hashing import canonical_json_hash

    payload["artifact_sha256"] = canonical_json_hash(payload["preregistration"])
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="signature"):
        load_preregistration(path, signing_key=b"owner-test-signing-key")


def test_frozen_inputs_reject_config_order_substitution_mutation_and_cohort_change(
    tmp_path: Path,
) -> None:
    configs = tuple(_config(tmp_path, suffix=value) for value in "abc")
    prereg = _preregistration(configs)
    envelope = PreregistrationEnvelope.sign(
        prereg,
        signed_by="owner",
        signing_key=b"owner-test-signing-key",
        signed_at=datetime(2026, 8, 14, tzinfo=UTC),
    )
    verified = verify_frozen_inputs(envelope, configs=configs, metadata=_metadata())
    assert verified.config_hashes == prereg.config_hashes

    with pytest.raises(ValueError, match="order"):
        verify_frozen_inputs(
            envelope, configs=(configs[1], configs[0], configs[2]), metadata=_metadata()
        )
    substitute = _config(tmp_path, suffix="d")
    with pytest.raises(ValueError, match="exact frozen"):
        verify_frozen_inputs(
            envelope, configs=(configs[0], configs[1], substitute), metadata=_metadata()
        )
    changed_metadata = _metadata().model_copy(update={"content_sha256": "9" * 64})
    with pytest.raises(ValueError, match="cohort"):
        verify_frozen_inputs(envelope, configs=configs, metadata=changed_metadata)

    changed_code = tuple(config.model_copy(update={"code_commit": "fffffff"}) for config in configs)
    with pytest.raises(ValueError, match="code commit"):
        verify_frozen_inputs(envelope, configs=changed_code, metadata=_metadata())


def test_optional_solar_comparison_is_fixed_budgeted_and_after_core_only() -> None:
    spec = SolarExploratorySpec(
        model_ids=("solar-pro3-2026-08-01", "solar-pro4-2026-08-01"),
        subset_membership_hash="a" * 64,
        subset_size=50,
        budget_cap_usd=Decimal("5.00"),
        after_core_results=True,
        preregistered=False,
    )
    assert spec.result_label == "exploratory"
    with pytest.raises(ValidationError, match="after core"):
        SolarExploratorySpec.model_validate({**spec.model_dump(), "after_core_results": False})
    with pytest.raises(ValidationError, match="Solar Pro 3"):
        SolarExploratorySpec.model_validate(
            {**spec.model_dump(), "model_ids": ("solar-pro4", "solar-pro3")}
        )


def test_signature_authenticates_signer_and_timestamp(tmp_path: Path) -> None:
    configs = tuple(_config(tmp_path, suffix=value) for value in "abc")
    envelope = PreregistrationEnvelope.sign(
        _preregistration(configs),
        signed_by="owner",
        signing_key=b"owner-test-signing-key",
        signed_at=datetime(2026, 8, 14, tzinfo=UTC),
    )
    path = tmp_path / "prereg.json"
    path.write_text(envelope.model_dump_json(), encoding="utf-8")
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["signed_by"] = "attacker"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="signature"):
        load_preregistration(path, signing_key=b"owner-test-signing-key")
