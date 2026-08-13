"""Offline safety tests for the local corpus collector."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _collector_module() -> object:
    script = Path(__file__).parents[3] / "scripts" / "collect_corpus.py"
    specification = importlib.util.spec_from_file_location("collect_corpus", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_collector_refuses_symlinks_outside_the_approved_root(tmp_path: Path) -> None:
    module = _collector_module()
    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-not-really")
    linked = approved / "linked.pdf"
    linked.symlink_to(outside)

    with pytest.raises(ValueError, match="non-symlink regular file"):
        module._resolve_approved_source(linked, approved)


def test_collector_refuses_overwrite_and_writes_an_atomic_copy(tmp_path: Path) -> None:
    module = _collector_module()
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-test")
    destination = tmp_path / "raw" / "document.pdf"

    module._atomic_copy(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module._atomic_copy(source, destination)
