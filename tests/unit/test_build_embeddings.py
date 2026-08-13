"""Offline planning and live-gate contracts for the embedding builder."""

import importlib.util
import json
import sys
from pathlib import Path
from uuid import UUID

import pytest

_PATH = Path(__file__).parents[2] / "scripts" / "build_embeddings.py"
_SPEC = importlib.util.spec_from_file_location("build_embeddings", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
BuildGateError = _MODULE.BuildGateError
build_plan = _MODULE.build_plan
require_live_gate = _MODULE.require_live_gate


def _write_dataset(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "chunk_id": "00000000-0000-0000-0000-000000000001",
                        "parse_snapshot_id": "parse-a",
                        "strategy": "fixed-300-overlap-0",
                        "content": "첫째",
                        "token_count": 2,
                    }
                ),
                json.dumps(
                    {
                        "chunk_id": "00000000-0000-0000-0000-000000000002",
                        "parse_snapshot_id": "parse-a",
                        "strategy": "fixed-300-overlap-0",
                        "content": "둘째",
                        "token_count": 3,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_plan_is_immutable_and_hashes_every_response_affecting_input(tmp_path: Path) -> None:
    """Catch an unversioned plan whose model or chunk bytes can drift unnoticed."""
    dataset = tmp_path / "chunks.jsonl"
    _write_dataset(dataset)

    first = build_plan(
        dataset,
        corpus_snapshot_id="corpus-a",
        model_id="embedding-passage",
        query_model_id="embedding-query",
        dimension=4096,
    )
    second = build_plan(
        dataset,
        corpus_snapshot_id="corpus-a",
        model_id="embedding-passage",
        query_model_id="embedding-query",
        dimension=4096,
    )
    changed = build_plan(
        dataset,
        corpus_snapshot_id="corpus-a",
        model_id="embedding-passage-v2",
        query_model_id="embedding-query",
        dimension=4096,
    )

    assert first == second
    assert str(UUID(first.snapshot.snapshot_id)) == first.snapshot.snapshot_id
    assert first.total_chunks == 2
    assert first.total_tokens == 5
    assert first.snapshot.parse_snapshot_id == "parse-a"
    assert first.plan_hash != changed.plan_hash
    assert first.snapshot.snapshot_id != changed.snapshot.snapshot_id


def test_build_plan_rejects_mixed_chunk_provenance(tmp_path: Path) -> None:
    """Catch one embedding snapshot silently spanning multiple parse snapshots."""
    dataset = tmp_path / "chunks.jsonl"
    _write_dataset(dataset)
    lines = dataset.read_text(encoding="utf-8").splitlines()
    second = json.loads(lines[1])
    second["parse_snapshot_id"] = "parse-b"
    dataset.write_text(lines[0] + "\n" + json.dumps(second) + "\n", encoding="utf-8")

    with pytest.raises(BuildGateError, match="one parse snapshot"):
        build_plan(
            dataset,
            corpus_snapshot_id="corpus-a",
            model_id="embedding-passage",
            query_model_id="embedding-query",
            dimension=4096,
        )


def test_live_execution_requires_two_explicit_flags() -> None:
    """Catch a paid provider run activated by a single easy-to-mistype switch."""
    with pytest.raises(BuildGateError, match="--live"):
        require_live_gate(live=False, confirm_paid=False)
    with pytest.raises(BuildGateError, match="--confirm-paid"):
        require_live_gate(live=True, confirm_paid=False)
    require_live_gate(live=True, confirm_paid=True)
