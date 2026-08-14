"""Export aggregate benchmark analysis from clean immutable snapshot adapters."""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

from pydantic import ValidationError

from ragbench.analysis import AnalysisBundle, ExportRequest, export_analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create public-safe CSV/Parquet/figure exports from immutable experiment IDs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Clean-snapshot analysis-input-v1 JSON; repeat for additional immutable runs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-salt", required=True)
    parser.add_argument("--failure-sample-size", type=int, default=50)
    parser.add_argument("--vat-multiplier", default="1.10")
    parser.add_argument(
        "--console-gross-usd",
        help="Actual provider-console gross total; omit to preserve not_reconciled state.",
    )
    return parser


def _load_bundle(path: Path) -> AnalysisBundle:
    absolute = Path(os.path.abspath(path))
    parent_fd = os.open(os.sep, os.O_RDONLY | os.O_DIRECTORY)
    descriptor: int | None = None
    try:
        for component in absolute.parent.parts[1:]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
            os.close(parent_fd)
            parent_fd = next_fd
        descriptor = os.open(absolute.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("analysis input must be a regular non-symlink file")
        payload = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
        return AnalysisBundle.model_validate_json(bytes(payload))
    except (OSError, ValidationError) as error:
        raise ValueError(f"invalid clean-snapshot analysis input: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        vat = Decimal(args.vat_multiplier)
        console = (
            None if args.console_gross_usd is None else Decimal(args.console_gross_usd)
        )
    except InvalidOperation as error:
        raise ValueError("cost arguments must be exact decimal strings") from error
    request = ExportRequest(
        bundles=tuple(_load_bundle(path) for path in args.input),
        public_salt=args.public_salt,
        failure_sample_size=args.failure_sample_size,
        vat_multiplier=vat,
        console_gross_usd=console,
    )
    manifest = export_analysis(request, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cohort_hash": manifest.cohort_hash,
                "table_count": len(manifest.tables),
                "figure_count": len(manifest.figures),
                "claim_status": manifest.claim_status,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
