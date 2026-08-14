from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest

from ragbench.analysis import (
    INSPECTION_ORDER,
    AnalysisBundle,
    ExportRequest,
    build_cost_rows,
    export_analysis,
    plan_failure_sample,
)


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "export_results.py"
    spec = importlib.util.spec_from_file_location("export_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _bundle(*, cohort: str = "cohort-a", data_version: str = "data-a") -> AnalysisBundle:
    configs = []
    results = []
    failures = []
    usage = []
    for config_index, (parse_mode, retriever, prompt) in enumerate(
        (
            ("standard", "dense", "v1"),
            ("enhanced", "dense", "v1"),
            ("enhanced", "hybrid", "v3"),
        )
    ):
        config_hash = _sha(f"config-{config_index}")
        experiment_id = f"{config_hash}-20260814T12000{config_index}.000000Z"
        configs.append(
            {
                "experiment_id": experiment_id,
                "config_hash": config_hash,
                "parse_mode": parse_mode,
                "chunk_strategy": "fixed-512-64",
                "retriever": retriever,
                "top_k": 5,
                "prompt_version": prompt,
                "model_id": "solar-pro3-2026-08",
            }
        )
        for question_index in range(40):
            question_id = f"private-question-{question_index:03d}"
            score = Decimal("0.55") + Decimal(config_index) * Decimal("0.10")
            results.append(
                {
                    "experiment_id": experiment_id,
                    "question_id": question_id,
                    "question_type": "table" if question_index % 2 else "fact",
                    "correctness": str(score),
                    "faithfulness": str(score + Decimal("0.05")),
                    "citation": str(score),
                    "abstention": "1" if prompt == "v3" else "0.8",
                    "hit": "1",
                    "recall": "0.8",
                    "mrr": "0.75",
                    "latency_ms": 500 + config_index * 100 + question_index,
                }
            )
            usage.append(
                {
                    "experiment_id": experiment_id,
                    "question_id": question_id,
                    "question_type": "table" if question_index % 2 else "fact",
                    "operation": "generate",
                    "model_id": "solar-pro3-2026-08",
                    "estimated_cost_usd": "0.001000",
                    "cached": question_index % 3 == 0,
                }
            )
            if question_index < 20:
                failures.append(
                    {
                        "experiment_id": experiment_id,
                        "question_id": question_id,
                        "question_type": "table" if question_index % 2 else "fact",
                        "primary": "RETRIEVAL_MISS" if question_index % 2 else "PARSER_ERROR",
                        "secondary": "CHUNK_BOUNDARY" if question_index % 3 == 0 else None,
                    }
                )
    return AnalysisBundle.model_validate(
        {
            "schema_version": "analysis-input-v1",
            "cohort_hash": _sha(cohort),
            "data_version": data_version,
            "code_version": "fc6e5aa",
            "configs": configs,
            "results": results,
            "usage": usage,
            "failures": failures,
        }
    )


def test_bundle_rejects_cross_cohort_and_unknown_experiment_rows() -> None:
    payload = _bundle().model_dump(mode="json")
    payload["results"][0]["experiment_id"] = f"{_sha('unknown')}-20260814T000000.000000Z"
    with pytest.raises(ValueError, match="unknown immutable experiment"):
        AnalysisBundle.model_validate(payload)

    second = _bundle(cohort="other")
    with pytest.raises(ValueError, match="same cohort and lineage"):
        ExportRequest(bundles=(_bundle(), second), public_salt="public-v1")

    mismatched = _bundle().model_dump(mode="json")
    mismatched["results"] = [
        row
        for row in mismatched["results"]
        if not (
            row["experiment_id"] == mismatched["configs"][1]["experiment_id"]
            and row["question_id"] == "private-question-039"
        )
    ]
    mismatched["usage"] = [
        row
        for row in mismatched["usage"]
        if not (
            row["experiment_id"] == mismatched["configs"][1]["experiment_id"]
            and row["question_id"] == "private-question-039"
        )
    ]
    with pytest.raises(ValueError, match="exact question cohort"):
        AnalysisBundle.model_validate(mismatched)

    wrong_usage_type = _bundle().model_dump(mode="json")
    wrong_usage_type["usage"][0]["question_type"] = "WRONG_TYPE"
    with pytest.raises(ValueError, match="usage question type"):
        AnalysisBundle.model_validate(wrong_usage_type)
    wrong_failure_type = _bundle().model_dump(mode="json")
    wrong_failure_type["failures"][0]["question_type"] = "WRONG_TYPE"
    with pytest.raises(ValueError, match="failure question type"):
        AnalysisBundle.model_validate(wrong_failure_type)


def test_failure_plan_is_deterministic_stratified_and_carries_required_inspection_order() -> None:
    first = plan_failure_sample(_bundle(), sample_size=50, public_salt="public-v1")
    second = plan_failure_sample(_bundle(), sample_size=50, public_salt="public-v1")
    assert first == second
    assert len(first) == 50
    assert {row.inspection_order for row in first} == {INSPECTION_ORDER}
    assert len({(row.public_config_id, row.question_type) for row in first}) == 6
    serialized = json.dumps([row.model_dump(mode="json") for row in first])
    assert "private-question" not in serialized
    with pytest.raises(ValueError, match="50 through 100"):
        plan_failure_sample(_bundle(), sample_size=49, public_salt="public-v1")


def test_cost_rows_separate_cache_apply_vat_and_do_not_invent_reconciliation_delta() -> None:
    bundle = _bundle()
    rows, reconciliation = build_cost_rows(bundle, vat_multiplier=Decimal("1.10"))
    assert {row.cache_status for row in rows} == {"cached", "new"}
    assert sum(row.net_cost_usd for row in rows) == Decimal("0.120000")
    assert sum(row.gross_cost_usd for row in rows) == Decimal("0.132000")
    assert reconciliation.status == "not_reconciled"
    assert reconciliation.console_gross_usd is None
    assert reconciliation.delta_usd is None

    _, reconciled = build_cost_rows(
        bundle,
        vat_multiplier=Decimal("1.10"),
        console_gross_usd=Decimal("0.140000"),
    )
    assert reconciled.status == "reconciled"
    assert reconciled.delta_usd == Decimal("0.008000")


def test_parse_amortization_is_allocated_once_and_marginal_cost_is_incremental(
    tmp_path: Path,
) -> None:
    payload = _bundle().model_dump(mode="json")
    config_positions = {
        config["experiment_id"]: index for index, config in enumerate(payload["configs"])
    }
    for usage in payload["usage"]:
        usage["estimated_cost_usd"] = str(
            Decimal("0.001000") * (config_positions[usage["experiment_id"]] + 1)
        )
    for config in payload["configs"]:
        payload["usage"].append(
            {
                "experiment_id": config["experiment_id"],
                "question_id": None,
                "question_type": "all",
                "operation": "parse",
                "model_id": "document-parse-2026-08",
                "estimated_cost_usd": "0.400000",
                "cached": False,
            }
        )
    bundle = AnalysisBundle.model_validate(payload)
    rows, _ = build_cost_rows(bundle, vat_multiplier=Decimal("1.10"))
    for config_id in {row.public_config_id for row in rows}:
        config_rows = [row for row in rows if row.public_config_id == config_id]
        assert sum(row.parse_amortized_cost_usd for row in config_rows) == Decimal("0.400000")
        assert all(
            row.parse_amortized_cost_usd == 0 for row in config_rows if row.operation != "generate"
        )

    output = tmp_path / "analysis"
    export_analysis(
        ExportRequest(bundles=(bundle,), public_salt="public-v1", failure_sample_size=50),
        output,
    )
    with (output / "tables" / "marginal_cost_quality.csv").open(newline="") as stream:
        marginal = list(csv.DictReader(stream))
    nonbaseline = [row for row in marginal if row["baseline_public_config_id"]]
    assert nonbaseline
    assert nonbaseline[0]["marginal_cost_per_quality_point_usd"] != (
        Decimal(nonbaseline[0]["gross_cost_usd"]) / (Decimal(nonbaseline[0]["quality"]) * 100)
    ).quantize(Decimal("0.000001"))


def test_export_writes_core_csv_parquet_svg_and_hash_manifest_without_private_ids(
    tmp_path: Path,
) -> None:
    output = tmp_path / "analysis"
    manifest = export_analysis(
        ExportRequest(bundles=(_bundle(),), public_salt="public-v1", failure_sample_size=50),
        output,
    )
    expected_tables = {
        "leaderboard",
        "parse_paired_difference",
        "chunk_heatmap",
        "retriever_by_type",
        "top_k_tradeoff",
        "prompt_abstention",
        "pareto_frontier",
        "latency_distribution",
        "failure_taxonomy",
        "failure_sample_plan",
        "cost_breakdown",
        "marginal_cost_quality",
        "reconciliation",
    }
    assert expected_tables <= set(manifest.tables)
    assert manifest.schema_version == "analysis-export-manifest-v1"
    assert manifest.cohort_hash == _bundle().cohort_hash
    assert manifest.data_version == "data-a"
    assert manifest.code_version == "fc6e5aa"
    assert manifest.files
    for relative, digest in manifest.files.items():
        path = output / relative
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert (output / "tables" / "leaderboard.parquet").read_bytes()[:4] == b"PAR1"
    assert (
        'data-measures="mean_correctness"' in (output / "figures" / "leaderboard.svg").read_text()
    )
    assert (
        'data-measures="correctness_effect"'
        in (output / "figures" / "parse_paired_difference.svg").read_text()
    )
    assert (
        'data-measures="mean_correctness,mean_abstention"'
        in (output / "figures" / "prompt_abstention.svg").read_text()
    )
    assert (
        'data-measures="quality,gross_cost_usd"'
        in (output / "figures" / "pareto_frontier.svg").read_text()
    )
    assert 'data-geometry="heatmap"' in (output / "figures" / "chunk_heatmap.svg").read_text()
    assert 'data-geometry="scatter"' in (output / "figures" / "pareto_frontier.svg").read_text()
    assert 'data-geometry="dual-axis"' in (output / "figures" / "top_k_tradeoff.svg").read_text()
    assert (
        'data-geometry="signed-bars"'
        in (output / "figures" / "parse_paired_difference.svg").read_text()
    )
    all_csv = "".join(path.read_text() for path in (output / "tables").glob("*.csv"))
    assert "private-question" not in all_csv
    assert "question_id" not in all_csv

    with (output / "tables" / "reconciliation.csv").open(newline="") as stream:
        reconciliation = next(csv.DictReader(stream))
    assert reconciliation["status"] == "not_reconciled"
    assert reconciliation["delta_usd"] == ""

    with (output / "tables" / "prompt_abstention.csv").open(newline="") as stream:
        prompt = next(csv.DictReader(stream))
    assert "mean_abstention" in prompt

    with pytest.raises(FileExistsError, match="immutable"):
        export_analysis(
            ExportRequest(bundles=(_bundle(),), public_salt="public-v1"),
            output,
        )


def test_export_refuses_symlink_output_and_invalid_secondary_taxonomy(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        export_analysis(
            ExportRequest(bundles=(_bundle(),), public_salt="public-v1"),
            linked,
        )

    payload = _bundle().model_dump(mode="json")
    payload["failures"][0]["secondary"] = payload["failures"][0]["primary"]
    with pytest.raises(ValueError, match="distinct"):
        AnalysisBundle.model_validate(payload)


def test_cli_reads_clean_snapshot_adapter_json_and_exports(tmp_path: Path) -> None:
    source = tmp_path / "bundle.json"
    source.write_text(_bundle().model_dump_json(), encoding="utf-8")
    output = tmp_path / "published"
    assert (
        _load_script().main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--public-salt",
                "public-v1",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["claim_status"] == "PENDING_EVIDENCE"
