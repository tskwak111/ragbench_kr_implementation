import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[3] / "scripts" / "build_chunks.py"
_SPEC = importlib.util.spec_from_file_location("build_chunks", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
BuildIntegrityError = _MODULE.BuildIntegrityError
build_chunk_snapshots = _MODULE.build_chunk_snapshots


def _checkpoint(document, mode, snapshot="corpus-1", status="succeeded"):
    return {
        "snapshot_id": snapshot,
        "parse_snapshot_id": f"parse-{mode}",
        "document_id": document,
        "source_sha256": (document * 64)[:64],
        "expected_pages": 1,
        "mode": mode,
        "status": status,
        "elements": [{"page": 1, "category": "paragraph", "content": "대한민국 보고서"}],
        "page_mappings": [{"page": 1, "source_page": 1}],
    }


def test_build_exports_all_14_immutable_variants(tmp_path):
    output = tmp_path / "chunks"
    result = build_chunk_snapshots(
        [_checkpoint("a", "standard"), _checkpoint("a", "enhanced")], output
    )
    assert len(result.datasets) == 14
    assert len({item.snapshot_id for item in result.datasets}) == 14
    assert all(item.path.exists() and item.path.suffix == ".jsonl" for item in result.datasets)
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["tokenizer"] == {
        "library": "tiktoken",
        "version": "0.11.0",
        "encoding": "cl100k_base",
    }
    assert len(metadata["datasets"]) == 14


@pytest.mark.parametrize(
    "checkpoints",
    [
        [_checkpoint("a", "standard")],
        [_checkpoint("a", "standard"), _checkpoint("a", "enhanced", status="failed")],
        [_checkpoint("a", "standard"), _checkpoint("a", "enhanced", snapshot="corpus-2")],
    ],
)
def test_build_refuses_incomplete_or_mixed_parse_inputs(tmp_path, checkpoints):
    with pytest.raises(BuildIntegrityError):
        build_chunk_snapshots(checkpoints, tmp_path / "chunks")


def test_build_refuses_incomplete_page_checkpoint(tmp_path):
    standard = _checkpoint("a", "standard")
    enhanced = _checkpoint("a", "enhanced")
    standard["expected_pages"] = 2
    with pytest.raises(BuildIntegrityError, match="page mappings"):
        build_chunk_snapshots([standard, enhanced], tmp_path / "chunks")


def test_build_refuses_to_overwrite_tampered_snapshot(tmp_path):
    output = tmp_path / "chunks"
    checkpoints = [_checkpoint("a", "standard"), _checkpoint("a", "enhanced")]
    result = build_chunk_snapshots(checkpoints, output)
    result.datasets[0].path.write_text("tampered", encoding="utf-8")
    with pytest.raises(BuildIntegrityError, match="immutable"):
        build_chunk_snapshots(checkpoints, output)
