#!/usr/bin/env python3
"""Safely collect one operator-approved local PDF into the private raw corpus."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from pypdf import PdfReader

from ragbench.ingestion.manifest import DocumentRecord


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator-approved", action="store_true", help="confirm human source/license review"
    )
    parser.add_argument("--approved-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--organization", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--document-type", required=True)
    parser.add_argument("--language", required=True)
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
    parser.add_argument(
        "--fragment", type=Path, required=True, help="private manifest fragment output"
    )
    return parser.parse_args()


def _resolve_approved_source(source: Path, approved_root: Path) -> tuple[Path, Path]:
    if source.is_symlink() or not source.is_file():
        raise ValueError("source must be a non-symlink regular file")
    root = approved_root.resolve(strict=True)
    resolved = source.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("source is outside the approved root")
    if resolved.suffix.lower() != ".pdf":
        raise ValueError("source must have a .pdf suffix")
    with resolved.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("source is not a PDF (missing PDF magic)")
    return resolved, root


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise ValueError("raw-dir must not be a symlink")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing destination: {destination}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary destination already exists: {temporary}")
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_yaml_atomically(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("output paths must not be symlinks")
    temporary.write_text(
        yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    args = _parse_args()
    if not args.operator_approved:
        print(
            "refusing collection: pass --operator-approved after human source/license review",
            file=sys.stderr,
        )
        return 2
    try:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", args.document_id):
            raise ValueError("document-id must be a safe lowercase identifier")
        source, _ = _resolve_approved_source(args.source, args.approved_root)
        destination = args.raw_dir / f"{args.document_id}.pdf"
        _atomic_copy(source, destination)
        record = DocumentRecord(
            document_id=args.document_id,
            title=args.title,
            organization=args.organization,
            year=args.year,
            document_type=args.document_type,
            language=args.language,
            source_url=args.source_url,
            downloaded_at=args.downloaded_at,
            license=args.license,
            redistribution_status=args.redistribution_status,
            local_path=destination.resolve(),
            sha256=_sha256(destination),
            page_count=len(PdfReader(destination, strict=True).pages),
            inclusion_rationale=args.inclusion_rationale,
        )
        report = {
            "document_id": record.document_id,
            "source": str(source),
            "destination": str(destination.resolve()),
            "sha256": record.sha256,
            "page_count": record.page_count,
            "license": record.license,
            "redistribution_status": record.redistribution_status,
            "requires_license_review": record.redistribution_status == "unknown",
        }
        _write_yaml_atomically(args.fragment, {"documents": [record.model_dump(mode="json")]})
        _write_yaml_atomically(args.report, report)
    except Exception as error:
        print(f"collection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
