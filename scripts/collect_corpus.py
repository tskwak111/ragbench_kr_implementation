#!/usr/bin/env python3
"""Safely collect one operator-approved local PDF into the private raw corpus.

This collector requires POSIX ``O_NOFOLLOW``, directory file descriptors, and
``link(..., dir_fd=...)``. Platforms without those primitives fail closed.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import re
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import yaml
from pypdf import PdfReader

from ragbench.ingestion.manifest import DocumentRecord

_LINK_SUPPORTS_DIR_FD = os.link in os.supports_dir_fd


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
    if any(not hasattr(os, name) for name in required_os) or not _LINK_SUPPORTS_DIR_FD:
        raise RuntimeError("safe POSIX no-follow directory-descriptor primitives are unavailable")
    if not hasattr(fcntl, "flock"):
        raise RuntimeError("safe advisory collector locking is unavailable")


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


class _StagedFile:
    """A fsynced temporary inode, identified so rollback cannot remove another writer's file."""

    def __init__(self, *, name: str, device: int, inode: int) -> None:
        self.name = name
        self.device = device
        self.inode = inode


def _new_temporary_name() -> str:
    return f".collect-{next(tempfile._get_candidate_names())}.partial"  # type: ignore[attr-defined]


def _stage_bytes(directory_fd: int, payload: bytes) -> _StagedFile:
    """Create and fsync an unpredictable temporary regular file in directory_fd."""
    name = _new_temporary_name()
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        return _StagedFile(name=name, device=metadata.st_dev, inode=metadata.st_ino)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(name, dir_fd=directory_fd)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _cleanup_staged(directory_fd: int, staged: _StagedFile) -> None:
    """Remove a still-owned staging inode; a replaced temp name is left untouched."""
    try:
        metadata = os.stat(staged.name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if metadata.st_dev == staged.device and metadata.st_ino == staged.inode:
        os.unlink(staged.name, dir_fd=directory_fd)


@contextmanager
def _collector_lock(private_fd: int) -> Iterator[None]:
    """Serialize cooperating collectors inside the trusted private output directory."""
    descriptor = os.open(
        ".collector.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=private_fd
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _preflight_absent(directory_fd: int, names: tuple[str, ...]) -> None:
    """Reject already-present link targets before publishing any transaction member."""
    for name in names:
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise FileExistsError(f"refusing to overwrite existing destination: {name}")


def _publish_staged(directory_fd: int, staged: _StagedFile, destination_name: str) -> None:
    """Link no-replace; accept only a byte-identical prior artifact under the collector lock."""
    try:
        os.link(
            staged.name,
            destination_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileExistsError as error:
        existing_fd = os.open(destination_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        staged_fd = os.open(staged.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        try:
            existing = b"".join(iter(lambda: os.read(existing_fd, 1024 * 1024), b""))
            expected = b"".join(iter(lambda: os.read(staged_fd, 1024 * 1024), b""))
        finally:
            os.close(existing_fd)
            os.close(staged_fd)
        if existing != expected:
            raise FileExistsError(
                f"refusing to overwrite conflicting destination: {destination_name}"
            ) from error


def _stage_pdf(source_fd: int, raw_fd: int) -> tuple[_StagedFile, str, int]:
    """Stage, hash, and parse source descriptor bytes before publishing any final link."""
    source_stat = os.fstat(source_fd)
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError("source must be a regular file")
    temporary_name = _new_temporary_name()
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
        metadata = os.fstat(temporary_fd)
        return (
            _StagedFile(name=temporary_name, device=metadata.st_dev, inode=metadata.st_ino),
            digest.hexdigest(),
            page_count,
        )
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=raw_fd)
        raise
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)


def _copy_validate_and_publish(
    source_fd: int, raw_fd: int, destination_name: str
) -> tuple[str, int]:
    """Compatibility helper for direct callers; transaction-aware ``main`` stages all artifacts."""
    staged, sha256, page_count = _stage_pdf(source_fd, raw_fd)
    try:
        _preflight_absent(raw_fd, (destination_name,))
        _publish_staged(raw_fd, staged, destination_name)
        return sha256, page_count
    finally:
        _cleanup_staged(raw_fd, staged)


def _build_record(
    args: argparse.Namespace, destination: Path, sha256: str, page_count: int
) -> DocumentRecord:
    """Validate and construct one record before any transaction member is published."""
    return DocumentRecord(
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
        destination = Path(os.path.abspath(args.raw_dir)) / f"{args.document_id}.pdf"
        # Catch all operator-supplied metadata errors before staging source bytes.
        _build_record(args, destination, "0" * 64, 1)
        source_fd, source_path = _open_source_descriptor(args.source, args.approved_root)
        try:
            with (
                _open_secure_directory(args.raw_dir) as raw_fd,
                _open_secure_directory(args.private_output_dir) as private_fd,
                _collector_lock(private_fd),
            ):
                destination_name = f"{args.document_id}.pdf"
                pdf_staged, sha256, page_count = _stage_pdf(source_fd, raw_fd)
                record = _build_record(args, destination, sha256, page_count)
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
                fragment_payload = yaml.safe_dump(
                    {"documents": [record.model_dump(mode="json")]},
                    allow_unicode=True,
                    sort_keys=False,
                ).encode()
                report_payload = yaml.safe_dump(
                    report, allow_unicode=True, sort_keys=False
                ).encode()
                fragment_staged: _StagedFile | None = None
                report_staged: _StagedFile | None = None
                try:
                    fragment_staged = _stage_bytes(private_fd, fragment_payload)
                    report_staged = _stage_bytes(private_fd, report_payload)
                    # Publish metadata first; the PDF link is the transaction commit marker.
                    for directory_fd, name, staged in (
                        (private_fd, fragment_name, fragment_staged),
                        (private_fd, report_name, report_staged),
                    ):
                        _publish_staged(directory_fd, staged, name)
                    os.fsync(private_fd)
                    _publish_staged(raw_fd, pdf_staged, destination_name)
                    os.fsync(raw_fd)
                finally:
                    _cleanup_staged(raw_fd, pdf_staged)
                    if fragment_staged is not None:
                        _cleanup_staged(private_fd, fragment_staged)
                    if report_staged is not None:
                        _cleanup_staged(private_fd, report_staged)
        finally:
            os.close(source_fd)
    except Exception as error:
        print(f"collection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
