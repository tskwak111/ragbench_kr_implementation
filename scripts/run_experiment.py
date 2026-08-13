#!/usr/bin/env python3
"""Plan immutable generation experiments; paid execution is always fail-closed."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from ragbench.experiments.runner import (
    DryRunPlan,
    ExperimentConfig,
    FileExperimentRepository,
    authorize_paid_execution,
    build_dry_run_plan,
)
from ragbench.providers.upstage.pricing import PriceBook

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentCommandError(RuntimeError):
    """Safe operator-facing failure without provider response leakage."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prices", type=Path, default=PROJECT_ROOT / "configs" / "prices.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--confirm-plan")
    parser.add_argument("--available-budget-usd", type=Decimal)
    parser.add_argument("--diagnosis-acknowledgement")
    return parser.parse_args()


def _plan_payload(plan: DryRunPlan) -> dict[str, object]:
    payload: dict[str, object] = plan.model_dump(mode="json")
    payload["mode"] = "dry-run"
    payload["executed"] = False
    return payload


def _run(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_yaml(args.config)
    prices = PriceBook.from_yaml(args.prices)
    repository = FileExperimentRepository(Path(config.output_dir))
    cached_ids = set(repository.result_ids(args.resume)) if args.resume is not None else None
    plan = build_dry_run_plan(config, price_book=prices, cached_question_ids=cached_ids)
    if not args.execute:
        print(json.dumps(_plan_payload(plan), ensure_ascii=False, sort_keys=True))
        return 0
    if args.available_budget_usd is None:
        raise ExperimentCommandError("--available-budget-usd is required for paid execution")
    authorize_paid_execution(
        plan,
        execute=True,
        live_enabled=os.environ.get("RUN_LIVE_UPSTAGE_TESTS") == "1",
        confirmed_plan_hash=args.confirm_plan,
        price_book=prices,
        available_budget_usd=args.available_budget_usd,
        now=datetime.now(UTC),
    )
    raise ExperimentCommandError(
        "paid execution requires the application gateway/RAG executor; this CLI environment "
        "does not construct direct HTTP requests"
    )


def main() -> None:
    try:
        raise SystemExit(_run(_parse_args()))
    except Exception as error:
        print(
            json.dumps(
                {"ok": False, "error": str(error), "executed": False},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
