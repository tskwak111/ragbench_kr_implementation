from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "scripts" / "run_gold_test.py"
    spec = importlib.util.spec_from_file_location("run_gold_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_never_accepts_or_opens_a_gold_path(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()
    opened = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("gold loader must not run in dry-run")

    monkeypatch.setattr(module, "load_authorized_gold_cohort", forbidden)
    envelope = type("Envelope", (), {"artifact_sha256": "a" * 64})()
    plan = type(
        "Plan",
        (),
        {"model_dump": lambda self, **kwargs: {"mode": "dry-run", "executed": False}},
    )()
    monkeypatch.setattr(module, "_load_and_verify", lambda args: (envelope, (), object()))
    monkeypatch.setattr(module, "build_gold_dry_run", lambda value: plan)
    args = argparse.Namespace(
        preregistration=Path("prereg.json"),
        execute=False,
        gold=None,
        metadata=None,
        configs=(),
        output=Path("protected"),
        resume=False,
    )
    assert module._run(args) == 0
    assert opened is False


def test_execute_loads_only_the_source_hashed_preregistered_async_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    source = tmp_path / "bound_adapter.py"
    source.write_text(
        "async def execute(config, item):\n    raise RuntimeError('fixture only')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("bound_adapter", None)
    executor = type(
        "Executor",
        (),
        {
            "entrypoint": "bound_adapter:execute",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    )()
    registration = type("Registration", (), {"executor": executor})()
    envelope = type("Envelope", (), {"preregistration": registration})()
    assert module._load_executor(envelope).__name__ == "execute"

    executor.source_sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="source hash"):
        module._load_executor(envelope)
