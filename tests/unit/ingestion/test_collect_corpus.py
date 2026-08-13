"""Offline safety tests for the descriptor-based local corpus collector."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import yaml


def _collector_module() -> object:
    script = Path(__file__).parents[3] / "scripts" / "collect_corpus.py"
    specification = importlib.util.spec_from_file_location("collect_corpus", script)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_pdf(path: Path) -> None:
    path.write_bytes(
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n"
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n"
        b"0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n186\n%%EOF\n"
    )


def _arguments(approved: Path, source: Path, raw: Path, private: Path) -> list[str]:
    return [
        "collect_corpus.py",
        "--operator-approved",
        "--approved-root",
        str(approved),
        "--source",
        str(source),
        "--raw-dir",
        str(raw),
        "--private-output-dir",
        str(private),
        "--document-id",
        "sample",
        "--title",
        "Sample",
        "--organization",
        "Example",
        "--year",
        "2025",
        "--document-type",
        "report",
        "--language",
        "ko",
        "--sector",
        "corporate",
        "--content-stratum",
        "mixed",
        "--template-family",
        "example-template",
        "--source-url",
        "https://example.test/sample.pdf",
        "--downloaded-at",
        "2026-08-14",
        "--license",
        "unknown",
        "--redistribution-status",
        "unknown",
        "--inclusion-rationale",
        "test",
        "--report",
        str(private / "report.yaml"),
        "--fragment",
        str(private / "fragment.yaml"),
    ]


def test_collector_refuses_source_symlink_redirection(tmp_path: Path) -> None:
    module = _collector_module()
    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside.pdf"
    _write_pdf(outside)
    linked = approved / "linked.pdf"
    linked.symlink_to(outside)

    with pytest.raises(OSError):
        module._open_source_descriptor(linked, approved)


def test_collector_never_overwrites_destination_even_when_raced(tmp_path: Path) -> None:
    module = _collector_module()
    approved = tmp_path / "approved"
    raw = tmp_path / "raw"
    approved.mkdir()
    raw.mkdir()
    source = approved / "sample.pdf"
    _write_pdf(source)
    source_fd, _ = module._open_source_descriptor(source, approved)
    try:
        with module._open_secure_directory(raw) as raw_fd:
            module._copy_validate_and_publish(source_fd, raw_fd, "sample.pdf")
            os.lseek(source_fd, 0, os.SEEK_SET)
            with pytest.raises(FileExistsError, match="refusing to overwrite"):
                module._copy_validate_and_publish(source_fd, raw_fd, "sample.pdf")
    finally:
        os.close(source_fd)


def test_collector_removes_malformed_staging_file_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _collector_module()
    approved, raw, private = tmp_path / "approved", tmp_path / "raw", tmp_path / "private"
    approved.mkdir()
    raw.mkdir()
    private.mkdir()
    source = approved / "sample.pdf"
    source.write_bytes(b"%PDF-malformed")
    monkeypatch.setattr(sys, "argv", _arguments(approved, source, raw, private))

    assert module.main() == 1
    assert not (raw / "sample.pdf").exists()
    assert not list(raw.glob(".collect-*.partial"))
    assert "collection failed" in capsys.readouterr().err

    _write_pdf(source)
    assert module.main() == 0
    assert (raw / "sample.pdf").is_file()


def test_collector_cli_reports_unknown_license_and_rejects_output_redirection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _collector_module()
    approved, raw, private = tmp_path / "approved", tmp_path / "raw", tmp_path / "private"
    approved.mkdir()
    raw.mkdir()
    private.mkdir()
    source = approved / "sample.pdf"
    _write_pdf(source)
    monkeypatch.setattr(sys, "argv", _arguments(approved, source, raw, private))

    assert module.main() == 0
    report = yaml.safe_load((private / "report.yaml").read_text(encoding="utf-8"))
    assert report["requires_license_review"] is True

    redirected = tmp_path / "redirected.yaml"
    (private / "new-report.yaml").symlink_to(redirected)
    arguments = _arguments(approved, source, raw, private)
    arguments[arguments.index(str(private / "report.yaml"))] = str(private / "new-report.yaml")
    arguments[arguments.index("sample")] = "second"
    monkeypatch.setattr(sys, "argv", arguments)
    assert module.main() == 1
    assert not redirected.exists()
