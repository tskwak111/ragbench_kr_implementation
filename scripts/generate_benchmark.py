#!/usr/bin/env python3
"""Plan synthetic benchmark generation offline; execute only behind every paid gate."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from ragbench.benchmark.generation import (
    BenchmarkGenerator,
    FileBatchRepository,
    GenerationAuthorization,
    GenerationConfig,
    GenerationPlanner,
    QuestionCandidate,
    SourceWindow,
    generation_execution_blockers,
    projected_generation_cost,
)
from ragbench.benchmark.validation import ValidationConfig, report_payload, validate_candidates
from ragbench.core.config import Settings
from ragbench.core.money import BudgetGuard, SqlAlchemyBudgetRepository
from ragbench.db.models import ApiUsage, BudgetReservation
from ragbench.db.session import create_lock_session_factory, create_session_factory
from ragbench.providers.upstage.client import SqlAlchemyProviderStore, UpstageGateway
from ragbench.providers.upstage.pricing import PriceBook

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkGenerationError(RuntimeError):
    """A safe, user-facing benchmark generation preflight failure."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("windows", type=Path)
    parser.add_argument("--corpus-snapshot-id", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/benchmark.yaml")
    parser.add_argument("--prices", type=Path, default=PROJECT_ROOT / "configs/prices.yaml")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data/benchmarks/generated"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid", action="store_true")
    parser.add_argument("--confirm-plan")
    return parser.parse_args()


def _load_windows(path: Path) -> tuple[SourceWindow, ...]:
    if not path.is_file() or path.is_symlink():
        raise BenchmarkGenerationError("source windows must be a regular non-symlink file")
    windows: list[SourceWindow] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            windows.append(SourceWindow.model_validate_json(line))
        except ValueError as error:
            raise BenchmarkGenerationError(
                f"invalid source window JSON at line {line_number}"
            ) from error
    if not windows:
        raise BenchmarkGenerationError("source window dataset cannot be empty")
    return tuple(windows)


def _load_config(path: Path) -> tuple[GenerationConfig, ValidationConfig]:
    if not path.is_file() or path.is_symlink():
        raise BenchmarkGenerationError("benchmark config must be a regular non-symlink file")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or loaded.get("schema_version") != "benchmark-generation-v1":
        raise BenchmarkGenerationError("unsupported benchmark config schema")
    if set(loaded) != {"schema_version", "generation", "validation"}:
        raise BenchmarkGenerationError("benchmark config contains unknown sections")
    try:
        generation = GenerationConfig.model_validate(loaded["generation"])
        validation_raw = loaded["validation"]
        if not isinstance(validation_raw, dict):
            raise ValueError("validation config must be a mapping")
        validation = ValidationConfig(
            quotas=generation.quotas,
            **validation_raw,
        )
    except (TypeError, ValueError) as error:
        raise BenchmarkGenerationError("benchmark config is invalid") from error
    return generation, validation


async def _remaining_budget(
    session_factory: async_sessionmaker[Any], hard_limit: Decimal
) -> Decimal:
    async with session_factory() as session:
        settled = await session.scalar(
            select(func.coalesce(func.sum(ApiUsage.estimated_cost_usd), Decimal("0")))
        )
        reserved = await session.scalar(
            select(
                func.coalesce(func.sum(BudgetReservation.reserved_cost_usd), Decimal("0"))
            ).where(BudgetReservation.status == "open")
        )
    return hard_limit - Decimal(settled or 0) - Decimal(reserved or 0)


def _write_results(
    output_dir: Path,
    plan_hash: str,
    candidates: tuple[QuestionCandidate, ...],
    report: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / f"{plan_hash}-candidates.jsonl"
    report_path = output_dir / f"{plan_hash}-validation-report.json"
    candidate_payload = "".join(
        item.model_dump_json() + "\n" for item in candidates
    )
    candidate_path.write_text(candidate_payload, encoding="utf-8")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


async def _run(args: argparse.Namespace) -> int:
    windows = _load_windows(args.windows)
    generation_config, validation_config = _load_config(args.config)
    settings = Settings()
    model_id = args.model_id or settings.upstage_solar_pro3_model_id
    plan = GenerationPlanner(generation_config).plan(
        windows,
        corpus_snapshot_id=args.corpus_snapshot_id,
        model_id=model_id,
    )
    if not args.execute:
        print(
            json.dumps(
                {
                    "batch_count": len(plan.batches),
                    "candidate_target": len(plan.jobs),
                    "live_executed": False,
                    "mode": "dry-run",
                    "plan_hash": plan.plan_hash,
                },
                sort_keys=True,
            )
        )
        return 0

    if not settings.upstage_api_key:
        raise BenchmarkGenerationError("UPSTAGE_API_KEY is required for execution")
    price_book = PriceBook.from_yaml(args.prices)
    projected_cost = projected_generation_cost(
        plan,
        config=generation_config,
        price_book=price_book,
        billing_multiplier=settings.billing_cost_multiplier,
    )
    session_factory = create_session_factory(settings)
    lock_factory = create_lock_session_factory(settings)
    gateway: UpstageGateway | None = None
    try:
        remaining = await _remaining_budget(session_factory, settings.max_project_budget_usd)
        blockers = generation_execution_blockers(
            plan,
            authorization=GenerationAuthorization(
                execute=args.execute,
                confirm_paid=args.confirm_paid,
                live_enabled=settings.run_live_upstage_tests,
                confirmed_plan_hash=args.confirm_plan,
            ),
            price_book=price_book,
            projected_cost_usd=projected_cost,
            remaining_budget_usd=remaining,
            now=datetime.now(UTC),
        )
        if blockers:
            print(json.dumps({"executed": False, "blockers": blockers}, ensure_ascii=False))
            return 2
        gateway = UpstageGateway(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
            price_book=price_book,
            budget_guard=BudgetGuard(
                SqlAlchemyBudgetRepository(session_factory),
                hard_limit=settings.max_project_budget_usd,
            ),
            store=SqlAlchemyProviderStore(
                session_factory,
                lock_session_factory=lock_factory,
                max_lock_connections=settings.max_lock_connections,
            ),
            max_concurrency=settings.max_concurrency,
            max_retries=settings.max_retries,
            billing_cost_multiplier=settings.billing_cost_multiplier,
        )
        generator = BenchmarkGenerator(
            gateway,
            FileBatchRepository(args.output_dir / "checkpoints"),
            config=generation_config,
        )
        generated: list[QuestionCandidate] = []
        for batch in plan.batches:
            generated.extend(await generator.generate_batch(plan, batch))
        report = validate_candidates(
            tuple(generated), windows, config=validation_config
        )
        summary = report_payload(report)
        _write_results(args.output_dir, plan.plan_hash, report.items, summary)
        print(
            json.dumps(
                {
                    "accepted_count": report.accepted_count,
                    "completion_level": summary["completion_level"],
                    "executed": True,
                    "plan_hash": plan.plan_hash,
                    "projected_cost_usd": str(projected_cost),
                },
                sort_keys=True,
            )
        )
        return 0 if report.accepted_count >= generation_config.normal_completion_floor else 3
    finally:
        if gateway is not None:
            await gateway.aclose()
        await session_factory.kw["bind"].dispose()
        await lock_factory.kw["bind"].dispose()


def main() -> None:
    try:
        raise SystemExit(asyncio.run(_run(_parse_args())))
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
