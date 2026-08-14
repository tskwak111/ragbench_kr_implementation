#!/usr/bin/env python3
"""Verify or execute the preregistered sealed-gold evaluation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import os
import subprocess
from pathlib import Path
from typing import cast

from ragbench.benchmark.splits import GoldMetadata
from ragbench.evaluation.gold import (
    GoldExecutor,
    GoldResultRepository,
    GoldRunner,
    PreregistrationEnvelope,
    aggregate_gold_results,
    build_gold_dry_run,
    load_authorized_gold_cohort,
    load_preregistration,
    verify_frozen_inputs,
    verify_protected_output_path,
    verify_runtime_code_commit,
    write_public_gold_report,
)
from ragbench.experiments.runner import ExperimentConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--config", dest="configs", type=Path, action="append", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-report", type=Path)
    parser.add_argument("--public-export-salt")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_and_verify(
    args: argparse.Namespace,
) -> tuple[PreregistrationEnvelope, tuple[ExperimentConfig, ...], GoldMetadata]:
    signing_key = os.environ.get("RAGBENCH_PREREGISTRATION_SIGNING_KEY", "").encode()
    envelope = load_preregistration(args.preregistration, signing_key=signing_key)
    configs = tuple(ExperimentConfig.from_yaml(path) for path in args.configs)
    try:
        metadata = GoldMetadata.model_validate_json(args.metadata.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError("gold metadata artifact is invalid") from error
    verify_frozen_inputs(envelope, configs=configs, metadata=metadata)
    return envelope, configs, metadata


def _run(args: argparse.Namespace, *, executor: GoldExecutor | None = None) -> int:
    envelope, configs, metadata = _load_and_verify(args)
    if not args.execute:
        plan = build_gold_dry_run(envelope)
        print(json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
        return 0
    if args.gold is None:
        raise ValueError("--gold is required only with --execute")
    verify_runtime_code_commit(PROJECT_ROOT, envelope.preregistration)
    verify_protected_output_path(args.output, envelope.preregistration)
    if executor is None:
        executor = _load_executor(envelope)
    items = load_authorized_gold_cohort(
        args.gold,
        metadata=metadata,
        envelope=envelope,
        configs=configs,
        explicit=True,
    )
    repository = GoldResultRepository(args.output)
    summary = asyncio.run(
        GoldRunner(repository, executor).run(envelope, configs, items, resume=args.resume)
    )
    if args.public_report is not None:
        if not args.public_export_salt:
            raise ValueError("--public-export-salt is required with --public-report")
        report = aggregate_gold_results(
            repository, envelope, public_export_salt=args.public_export_salt
        )
        write_public_gold_report(args.public_report, report)
    print(
        json.dumps(
            {"ok": True, "completed": summary.completed, "expected": summary.expected},
            sort_keys=True,
        )
    )
    return 0


def _load_executor(envelope: PreregistrationEnvelope) -> GoldExecutor:
    """Load only the exact source-hashed adapter frozen in preregistration."""
    module_name, function_name = envelope.preregistration.executor.entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function) or not inspect.iscoroutinefunction(function):
        raise RuntimeError("preregistered gold executor is not an async callable")
    source_path = inspect.getsourcefile(function)
    if source_path is None:
        raise RuntimeError("preregistered gold executor source is unavailable")
    resolved_source = Path(source_path).resolve()
    _verify_tracked_source(resolved_source)
    actual_hash = hashlib.sha256(resolved_source.read_bytes()).hexdigest()
    if actual_hash != envelope.preregistration.executor.source_sha256:
        raise RuntimeError("preregistered gold executor source hash mismatch")
    return cast(GoldExecutor, function)


def _verify_tracked_source(source_path: Path) -> None:
    try:
        relative = source_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise RuntimeError(
            "preregistered gold executor must be inside the project repository"
        ) from None
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(relative)],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "preregistered gold executor must be tracked at the frozen commit"
        ) from error


def main() -> None:
    try:
        raise SystemExit(_run(_parse_args()))
    except Exception as error:
        _ = error
        print(
            json.dumps(
                {"ok": False, "error": "gold command failed safely"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
