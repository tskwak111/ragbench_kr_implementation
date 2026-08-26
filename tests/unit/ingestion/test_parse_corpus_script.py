import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).parents[3] / "scripts" / "parse_corpus.py"
_SPEC = importlib.util.spec_from_file_location("parse_corpus_script", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
select_documents = _MODULE._select_documents


def _documents():
    return tuple(SimpleNamespace(document_id=value) for value in ("a", "b", "c"))


def test_select_documents_keeps_manifest_order_and_defaults_to_all():
    documents = _documents()

    assert select_documents(documents, None) == documents
    assert [item.document_id for item in select_documents(documents, ["c", "a"])] == ["a", "c"]


@pytest.mark.parametrize("selected", [[], ["missing"], ["a", "a"]])
def test_select_documents_rejects_empty_unknown_or_duplicate_ids(selected):
    with pytest.raises(ValueError):
        select_documents(_documents(), selected)
