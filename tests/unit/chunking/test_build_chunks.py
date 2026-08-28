import copy
import importlib.util
import json
import sys
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

import pytest

from ragbench.ingestion.parser import ParseCheckpoint

_PATH = Path(__file__).parents[3] / "scripts" / "build_chunks.py"
_SPEC = importlib.util.spec_from_file_location("build_chunks", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
BuildIntegrityError = _MODULE.BuildIntegrityError
build_chunk_snapshots = _MODULE.build_chunk_snapshots


def _checkpoint(document, mode, snapshot="corpus-1", status="succeeded"):
    raw = {
        "model_version": "v1",
        "content": {"markdown": "대한민국 보고서", "html": "<p>대한민국 보고서</p>"},
        "elements": [{"page": 1, "category": "paragraph", "content": "대한민국 보고서"}],
        "pages": [{"page": 1, "source_page": 1}],
        "usage": {"pages": 1},
    }
    return {
        "snapshot_id": snapshot,
        "document_id": document,
        "source_sha256": (document * 64)[:64],
        "expected_pages": 1,
        "mode": mode,
        "status": status,
        "provider_model_id": "document-parse",
        "provider_model_version": "v1",
        "raw_response": raw,
        "raw_response_hash": _MODULE.canonical_json_hash(raw),
        "elements": raw["elements"],
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
        "asset_sha256": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    }
    assert len(metadata["datasets"]) == 14


def test_chunk_snapshot_id_changes_when_derived_records_change(tmp_path):
    checkpoints = [_checkpoint("a", "standard"), _checkpoint("a", "enhanced")]
    changed = copy.deepcopy(checkpoints)
    for checkpoint in changed:
        checkpoint["elements"][0]["category"] = "heading1"

    before = build_chunk_snapshots(checkpoints, tmp_path / "before")
    after = build_chunk_snapshots(changed, tmp_path / "after")

    assert {item.snapshot_id for item in before.datasets}.isdisjoint(
        item.snapshot_id for item in after.datasets
    )


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


def test_build_consumes_real_parse_checkpoint_schema_and_derives_parse_snapshot(tmp_path):
    def checkpoint(mode):
        raw = {
            "model_version": "v1",
            "content": {"markdown": "본문", "html": "<p>본문</p>"},
            "elements": [{"page": 1, "category": "paragraph", "content": "본문"}],
            "pages": [{"page": 1, "source_page": 1}],
            "usage": {"pages": 1},
        }
        return ParseCheckpoint(
            "corpus-1",
            "a",
            "a" * 64,
            1,
            "document-parse",
            "v1",
            mode,
            "succeeded",
            raw,
            _MODULE.canonical_json_hash(raw),
            "본문",
            "<p>본문</p>",
            tuple(raw["elements"]),
            ({"page": 1, "source_page": 1},),
            1,
            Decimal("0"),
        )

    result = build_chunk_snapshots(
        [asdict(checkpoint("standard")), asdict(checkpoint("enhanced"))], tmp_path / "chunks"
    )
    row = json.loads(result.datasets[0].path.read_text().splitlines()[0])
    assert row["parse_snapshot_id"]
    assert row["chunk_id"].startswith(row["parse_snapshot_id"] + ":")


def test_identical_existing_snapshot_permission_is_repaired_safely(tmp_path):
    output = tmp_path / "chunks"
    checkpoints = [_checkpoint("a", "standard"), _checkpoint("a", "enhanced")]
    result = build_chunk_snapshots(checkpoints, output)
    result.datasets[0].path.chmod(0o644)
    build_chunk_snapshots(checkpoints, output)
    assert result.datasets[0].path.stat().st_mode & 0o777 == 0o600


def test_existing_symlink_snapshot_is_rejected(tmp_path):
    output = tmp_path / "chunks"
    checkpoints = [_checkpoint("a", "standard"), _checkpoint("a", "enhanced")]
    result = build_chunk_snapshots(checkpoints, output)
    path = result.datasets[0].path
    target = tmp_path / "target"
    target.write_text(path.read_text(), encoding="utf-8")
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(BuildIntegrityError, match="symlink"):
        build_chunk_snapshots(checkpoints, output)
