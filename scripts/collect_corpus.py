#!/usr/bin/env python3
"""Safely collect one operator-approved local PDF into the private raw corpus.

This collector requires POSIX ``O_NOFOLLOW``, directory file descriptors, and
``link(..., dir_fd=...)``. Platforms without those primitives fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader

from ragbench.ingestion.manifest import DocumentRecord


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-approved", action="store_true")
    parser.add_argument("--approved-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--organization", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--document-type", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--sector", choices=("corporate", "public"), required=True)
    parser.add_argument(
        "--content-stratum", choices=("table_heavy", "text_heavy", "mixed"), required=True
    )
    parser.add_argument("--template-family", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--downloaded-at", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument(
        "--redistribution-status",
        choices=("redistributable", "nonredistributable", "unknown"),
        required=True,
    )
    parser.add_argument("--inclusion-rationale", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    return parser.parse_args()


def _require_safe_posix() -> None:
    required_os = ("O_NOFOLLOW", "O_DIRECTORY")
    if any(not hasattr(os, name) for name in required_os) or os.link not in os.supports_dir_fd:
        raise RuntimeError("safe POSIX no-follow directory-descriptor primitives are unavailable")


@contextmanager
def _open_secure_directory(path: Path) -> Iterator[int]:
    """Open every existing absolute directory component without following symlinks."""
    _require_safe_posix()
    absolute = Path(os.path.abspath(path))
    if not absolute.is_absolute():
        raise ValueError("directory path must be absolute")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute.parts[1:]:
            next_descriptor = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError("path is not a directory")
        yield descriptor
    finally:
        os.close(descriptor)


def _relative_regular_file(path: Path, root: Path) -> str:
    """Require a lexical child path without traversal; components are opened no-follow later."""
    try:
        relative = Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root)))
    except ValueError as error:
        raise ValueError("source is outside the approved root") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("source path is unsafe")
    return str(relative)


def _open_source_descriptor(source: Path, approved_root: Path) -> tuple[int, Path]:
    """Open a regular source file through no-follow descriptors anchored at approved_root."""
    resolved_root = Path(os.path.abspath(approved_root))
    relative = _relative_regular_file(source, resolved_root)
    with _open_secure_directory(resolved_root) as root_fd:
        components = Path(relative).parts
        directory_fd = os.dup(root_fd)
        try:
            for part in components[:-1]:
                next_descriptor = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                os.close(directory_fd)
                directory_fd = next_descriptor
            descriptor = os.open(components[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ValueError("source must be a non-symlink regular file")
    if Path(relative).suffix.lower() != ".pdf":
        os.close(descriptor)
        raise ValueError("source must have a .pdf suffix")
    return descriptor, resolved_root / relative


def _safe_output_name(path: Path, private_dir: Path) -> str:
    absolute = Path(os.path.abspath(path))
    root = Path(os.path.abspath(private_dir))
    if absolute.parent != root or absolute.name in {"", ".", ".."}:
        raise ValueError("report and fragment must be direct files inside private-output-dir")
    return absolute.name


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write every byte to an already-open descriptor or fail before publication."""
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("unable to write staged corpus bytes")
        view = view[written:]


def _copy_validate_and_publish(
    source_fd: int, raw_fd: int, destination_name: str
) -> tuple[str, int]:
    """Stage, hash, and parse a source descriptor before atomically linking it into raw_fd."""
    source_stat = os.fstat(source_fd)
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError("source must be a regular file")
    temporary_name = f".collect-{next(tempfile._get_candidate_names())}.partial"  # type: ignore[attr-defined]
    temporary_fd = -1
    try:
        # tempfile provides an unpredictable name; openat keeps it bound to raw_fd.
        temporary_fd = os.open(
            temporary_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=raw_fd,
        )
        digest = hashlib.sha256()
        os.lseek(source_fd, 0, os.SEEK_SET)
        first_bytes = os.read(source_fd, 5)
        if first_bytes != b"%PDF-":
            raise ValueError("source is not a PDF (missing PDF magic)")
        _write_all(temporary_fd, first_bytes)
        digest.update(first_bytes)
        while chunk := os.read(source_fd, 1024 * 1024):
            _write_all(temporary_fd, chunk)
            digest.update(chunk)
        os.fsync(temporary_fd)
        with os.fdopen(os.dup(temporary_fd), "rb") as staged:
            page_count = len(PdfReader(staged, strict=True).pages)
        os.link(
            temporary_name,
            destination_name,
            src_dir_fd=raw_fd,
            dst_dir_fd=raw_fd,
            follow_symlinks=False,
        )
        os.fsync(raw_fd)
        return digest.hexdigest(), page_count
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing destination: {destination_name}"
        ) from error
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=raw_fd)


def _write_private_yaml_no_replace(private_fd: int, name: str, content: dict[str, Any]) -> None:
    temporary_name = f".collect-{next(tempfile._get_candidate_names())}.partial"  # type: ignore[attr-defined]
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=private_fd,
        )
        payload = yaml.safe_dump(content, allow_unicode=True, sort_keys=False).encode()
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.link(
            temporary_name,
            name,
            src_dir_fd=private_fd,
            dst_dir_fd=private_fd,
            follow_symlinks=False,
        )
        os.fsync(private_fd)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite private output: {name}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=private_fd)


def main() -> int:
    args = _parse_args()
    if not args.operator_approved:
        print("refusing collection: pass --operator-approved after human review", file=sys.stderr)
        return 2
    try:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", args.document_id):
            raise ValueError("document-id must be a safe lowercase identifier")
        report_name = _safe_output_name(args.report, args.private_output_dir)
        fragment_name = _safe_output_name(args.fragment, args.private_output_dir)
        source_fd, source_path = _open_source_descriptor(args.source, args.approved_root)
        try:
            with (
                _open_secure_directory(args.raw_dir) as raw_fd,
                _open_secure_directory(args.private_output_dir) as private_fd,
            ):
                destination_name = f"{args.document_id}.pdf"
                sha256, page_count = _copy_validate_and_publish(source_fd, raw_fd, destination_name)
                destination = Path(os.path.abspath(args.raw_dir)) / destination_name
                record = DocumentRecord(
                    document_id=args.document_id,
                    title=args.title,
                    organization=args.organization,
                    year=args.year,
                    document_type=args.document_type,
                    language=args.language,
                    sector=args.sector,
                    content_stratum=args.content_stratum,
                    template_family=args.template_family,
                    source_url=args.source_url,
                    downloaded_at=args.downloaded_at,
                    license=args.license,
                    redistribution_status=args.redistribution_status,
                    local_path=destination,
                    sha256=sha256,
                    page_count=page_count,
                    inclusion_rationale=args.inclusion_rationale,
                )
                report = {
                    "document_id": record.document_id,
                    "source": str(source_path),
                    "destination": str(destination),
                    "sha256": record.sha256,
                    "page_count": record.page_count,
                    "license": record.license,
                    "redistribution_status": record.redistribution_status,
                    "requires_license_review": record.redistribution_status == "unknown",
                }
                _write_private_yaml_no_replace(
                    private_fd, fragment_name, {"documents": [record.model_dump(mode="json")]}
                )
                _write_private_yaml_no_replace(private_fd, report_name, report)
        finally:
            os.close(source_fd)
    except Exception as error:
        print(f"collection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
