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


def test_metadata_validation_failure_leaves_no_outputs_and_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _collector_module()
    approved, raw, private = tmp_path / "approved", tmp_path / "raw", tmp_path / "private"
    approved.mkdir()
    raw.mkdir()
    private.mkdir()
    source = approved / "sample.pdf"
    _write_pdf(source)
    arguments = _arguments(approved, source, raw, private)
    arguments[arguments.index("https://example.test/sample.pdf")] = "not-an-http-url"
    monkeypatch.setattr(sys, "argv", arguments)

    assert module.main() == 1
    assert not (raw / "sample.pdf").exists()
    assert not (private / "fragment.yaml").exists()
    assert not (private / "report.yaml").exists()
    assert not list(raw.glob(".collect-*.partial"))
    assert not list(private.glob(".collect-*.partial"))

    monkeypatch.setattr(sys, "argv", _arguments(approved, source, raw, private))
    assert module.main() == 0


@pytest.mark.parametrize("existing_name", ["fragment.yaml", "report.yaml"])
def test_preexisting_metadata_output_fails_before_pdf_and_preserves_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_name: str
) -> None:
    module = _collector_module()
    approved, raw, private = tmp_path / "approved", tmp_path / "raw", tmp_path / "private"
    approved.mkdir()
    raw.mkdir()
    private.mkdir()
    source = approved / "sample.pdf"
    _write_pdf(source)
    existing = private / existing_name
    existing.write_text("preserve-me", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _arguments(approved, source, raw, private))

    assert module.main() == 1
    assert existing.read_text(encoding="utf-8") == "preserve-me"
    assert not (raw / "sample.pdf").exists()
    if existing_name == "fragment.yaml":
        assert not (private / "report.yaml").exists()
    else:
        assert (private / "fragment.yaml").is_file()


@pytest.mark.parametrize("failing_link", [1, 2, 3])
def test_publish_failures_roll_back_only_this_invocations_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing_link: int
) -> None:
    module = _collector_module()
    approved, raw, private = tmp_path / "approved", tmp_path / "raw", tmp_path / "private"
    approved.mkdir()
    raw.mkdir()
    private.mkdir()
    source = approved / "sample.pdf"
    _write_pdf(source)
    original_link = module.os.link
    link_calls = 0

    def fail_after_link(*args: object, **kwargs: object) -> None:
        nonlocal link_calls
        link_calls += 1
        original_link(*args, **kwargs)
        if link_calls == failing_link:
            raise OSError("injected publish failure")

    monkeypatch.setattr(module.os, "link", fail_after_link)
    monkeypatch.setattr(sys, "argv", _arguments(approved, source, raw, private))

    assert module.main() == 1
    assert (raw / "sample.pdf").exists() is (failing_link == 3)
    assert (private / "fragment.yaml").exists() is (failing_link >= 1)
    assert (private / "report.yaml").exists() is (failing_link >= 2)
    assert not list(raw.glob(".collect-*.partial"))
    assert not list(private.glob(".collect-*.partial"))


def test_partial_metadata_is_idempotently_reused_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _collector_module()
    approved, raw, private = tmp_path / "approved", tmp_path / "raw", tmp_path / "private"
    approved.mkdir()
    raw.mkdir()
    private.mkdir()
    source = approved / "sample.pdf"
    _write_pdf(source)
    original_link = module.os.link
    calls = 0

    def fail_after_report(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        original_link(*args, **kwargs)
        if calls == 2:
            raise OSError("stop after report")

    monkeypatch.setattr(module.os, "link", fail_after_report)
    monkeypatch.setattr(sys, "argv", _arguments(approved, source, raw, private))
    assert module.main() == 1
    assert (private / "fragment.yaml").is_file()
    assert (private / "report.yaml").is_file()
    monkeypatch.setattr(module.os, "link", original_link)
    assert module.main() == 0
    assert (raw / "sample.pdf").is_file()


def test_raw_directory_fsync_occurs_after_pdf_commit_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _collector_module()
    approved, raw, private = tmp_path / "approved", tmp_path / "raw", tmp_path / "private"
    approved.mkdir()
    raw.mkdir()
    private.mkdir()
    source = approved / "sample.pdf"
    _write_pdf(source)
    raw_fd = os.open(raw, os.O_RDONLY)
    os.close(raw_fd)
    events: list[str] = []
    original_link, original_fsync = module.os.link, module.os.fsync

    def record_link(source_name: str, destination_name: str, **kwargs: object) -> None:
        original_link(source_name, destination_name, **kwargs)
        if destination_name == "sample.pdf":
            events.append("pdf-link")

    def record_fsync(descriptor: int) -> None:
        original_fsync(descriptor)
        events.append("fsync")

    monkeypatch.setattr(module.os, "link", record_link)
    monkeypatch.setattr(module.os, "fsync", record_fsync)
    monkeypatch.setattr(sys, "argv", _arguments(approved, source, raw, private))
    assert module.main() == 0
    assert events.index("pdf-link") < len(events) - 1


def test_temp_cleanup_leaves_replaced_name_untouched(tmp_path: Path) -> None:
    module = _collector_module()
    directory = tmp_path / "private"
    directory.mkdir()
    with module._open_secure_directory(directory) as directory_fd:
        staged = module._stage_bytes(directory_fd, b"owned")
        os.unlink(staged.name, dir_fd=directory_fd)
        replacement_fd = os.open(
            staged.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory_fd
        )
        try:
            os.write(replacement_fd, b"replacement")
        finally:
            os.close(replacement_fd)
        module._cleanup_staged(directory_fd, staged)
    assert (directory / staged.name).read_bytes() == b"replacement"
