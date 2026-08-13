#!/usr/bin/env python3
"""Verify or execute the preregistered sealed-gold evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

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
    write_public_gold_report,
)
from ragbench.experiments.runner import ExperimentConfig


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
    envelope = load_preregistration(args.preregistration)
    configs = tuple(ExperimentConfig.from_yaml(path) for path in args.configs)
    try:
        metadata = GoldMetadata.model_validate_json(args.metadata.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError("gold metadata artifact is invalid") from error
    verify_frozen_inputs(envelope, configs=configs, metadata=metadata)
    return envelope, configs, metadata


def _run(args: argparse.Namespace, *, executor: GoldExecutor | None = None) -> int:
    if args.execute and executor is None:
        raise RuntimeError(
            "gold execution requires the application executor; this CLI does not construct "
            "provider calls directly"
        )
    envelope, configs, metadata = _load_and_verify(args)
    if not args.execute:
        plan = build_gold_dry_run(envelope)
        print(json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
        return 0
    if args.gold is None:
        raise ValueError("--gold is required only with --execute")
    if executor is None:  # guarded before loading any artifacts; narrows for mypy
        raise AssertionError("unreachable missing gold executor")
    items = load_authorized_gold_cohort(
        args.gold,
        metadata=metadata,
        envelope=envelope,
        configs=configs,
        explicit=True,
    )
    repository = GoldResultRepository(args.output)
    summary = asyncio.run(GoldRunner(repository, executor).run(envelope, items, resume=args.resume))
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
