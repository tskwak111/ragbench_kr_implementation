import importlib.util
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "run_retrieval_screen", PROJECT_ROOT / "scripts/run_retrieval_screen.py"
)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def test_dry_run_is_default_and_prints_public_safe_grid_identity(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = SCRIPT.main(
        [
            "--corpus-snapshot-id",
            "corpus-a",
            "--question-snapshot-id",
            "dev-a",
            "--code-commit",
            "0bce46e",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["mode"] == "dry-run"
    assert payload["config_count"] == 126
    assert len(payload["grid_hash"]) == 64
    assert "questions" not in payload


def test_real_execution_is_fail_closed_until_a_store_and_snapshot_loader_are_bound() -> None:
    with pytest.raises(SystemExit, match="not available"):
        SCRIPT.main(
            [
                "--execute",
                "--corpus-snapshot-id",
                "corpus-a",
                "--question-snapshot-id",
                "dev-a",
                "--code-commit",
                "0bce46e",
            ]
        )
