import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from ragbench.experiments.planner import CHUNK_STRATEGIES

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "run_retrieval_screen", PROJECT_ROOT / "scripts/run_retrieval_screen.py"
)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)


def _inventory(tmp_path: Path) -> Path:
    path = tmp_path / "inventory.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "bindings": [
                    {
                        "parse_mode": mode,
                        "parse_snapshot_id": f"parse-{mode}",
                        "chunk_strategy": strategy,
                        "chunk_snapshot_id": f"chunk-{mode}-{strategy}",
                        "embedding_snapshot_id": (
                            f"00000000-0000-0000-{mode_index:04d}-{index:012d}"
                        ),
                    }
                    for mode_index, mode in enumerate(("standard", "enhanced"), start=1)
                    for index, strategy in enumerate(CHUNK_STRATEGIES, start=1)
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_dry_run_is_default_and_prints_public_safe_grid_identity(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    exit_code = SCRIPT.main(
        [
            "--corpus-snapshot-id",
            "corpus-a",
            "--question-snapshot-id",
            "dev-a",
            "--code-commit",
            "0bce46e",
            "--snapshot-inventory",
            str(_inventory(tmp_path)),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["mode"] == "dry-run"
    assert payload["config_count"] == 126
    assert len(payload["grid_hash"]) == 64
    assert "questions" not in payload


def test_real_execution_is_fail_closed_until_a_store_and_snapshot_loader_are_bound(
    tmp_path: Path,
) -> None:
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
                "--snapshot-inventory",
                str(_inventory(tmp_path)),
            ]
        )


def test_offline_fixture_config_runs_real_no_cost_screen(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    config = PROJECT_ROOT / "tests" / "fixtures" / "mini-screen.yaml"

    exit_code = SCRIPT.main(["--config", str(config), "--output", str(tmp_path)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema_version"] == "offline-screen-result-v1"
    assert payload["provider_calls"] == 0
    assert payload["metrics"]["hit_at_k"] == 1.0
