from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from ragbench.experiments.runner import ExperimentConfig, FileExperimentRepository

_PATH = Path(__file__).parents[3] / "scripts" / "run_experiment.py"
_SPEC = importlib.util.spec_from_file_location("run_experiment", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_run = _MODULE._run


def _args(*, execute: bool) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    return argparse.Namespace(
        config=root / "configs" / "experiments" / "dev-hybrid-v3.yaml",
        prices=root / "configs" / "prices.yaml",
        execute=execute,
        resume=None,
        confirm_plan=None,
        available_budget_usd=None,
        diagnosis_acknowledgement=None,
    )


def test_command_defaults_to_dry_run_without_creating_a_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(_args(execute=False)) == 0

    output = capsys.readouterr().out
    assert '"mode": "dry-run"' in output
    assert '"executed": false' in output
    assert '"cache_status": "unknown"' in output


def test_command_execution_fails_closed_without_explicit_budget() -> None:
    with pytest.raises(RuntimeError, match="available-budget"):
        _run(_args(execute=True))


def test_resume_rejects_a_run_from_another_semantic_config(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    raw = yaml.safe_load(
        (root / "configs" / "experiments" / "dev-hybrid-v3.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(raw, dict)
    raw["output_dir"] = str(tmp_path / "runs")
    old = ExperimentConfig.model_validate(raw)
    old = old.model_copy(
        update={"generation": old.generation.model_copy(update={"prompt_version": "v2"})}
    )
    run = FileExperimentRepository(tmp_path / "runs").create(
        old, now=datetime(2026, 8, 14, tzinfo=UTC)
    )
    config_path = tmp_path / "new.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
    args = _args(execute=False)
    args.config = config_path
    args.resume = run.run_id

    with pytest.raises(ValueError, match="different semantic config"):
        _run(args)


def test_relative_output_destination_is_resolved_against_config_location(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    raw = yaml.safe_load(
        (root / "configs" / "experiments" / "dev-hybrid-v3.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(raw, dict)
    raw["output_dir"] = "../artifacts"
    config_path = tmp_path / "configs" / "experiment.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")

    loaded = ExperimentConfig.from_yaml(config_path)

    assert loaded.output_dir == str((tmp_path / "artifacts").resolve())
