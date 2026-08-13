from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

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
