import importlib.util
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from ragbench.ingestion.parser import ParseCheckpoint


def _load_export_module():
    path = Path(__file__).parents[3] / "scripts" / "export_parse_checkpoints.py"
    if not path.exists():
        pytest.fail("parse-checkpoint export CLI is missing")
    spec = importlib.util.spec_from_file_location("export_parse_checkpoints", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_export_writes_private_build_compatible_jsonl(tmp_path):
    module = _load_export_module()
    raw = {
        "model_version": "v1",
        "elements": [{"page": 1, "category": "paragraph", "content": "본문"}],
        "pages": [{"page": 1, "source_page": 1}],
    }
    checkpoint = ParseCheckpoint(
        "corpus-1",
        "doc-a",
        "a" * 64,
        1,
        "document-parse",
        "v1",
        "standard",
        "succeeded",
        raw,
        "b" * 64,
        "본문",
        "<p>본문</p>",
        tuple(raw["elements"]),
        ({"page": 1, "source_page": 1},),
        12,
        Decimal("0.011000"),
        "correlation-1",
    )
    output = tmp_path / "checkpoints.jsonl"

    assert module.write_checkpoints([checkpoint], output) == 1

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row == {
        "snapshot_id": "corpus-1",
        "document_id": "doc-a",
        "source_sha256": "a" * 64,
        "expected_pages": 1,
        "provider_model_id": "document-parse",
        "provider_model_version": "v1",
        "mode": "standard",
        "status": "succeeded",
        "raw_response": raw,
        "raw_response_hash": "b" * 64,
        "markdown": "본문",
        "html": "<p>본문</p>",
        "elements": raw["elements"],
        "page_mappings": [{"page": 1, "source_page": 1}],
        "latency_ms": 12,
        "cost_usd": "0.011000",
        "correlation_id": "correlation-1",
        "error": None,
    }
    assert output.stat().st_mode & 0o777 == 0o600
