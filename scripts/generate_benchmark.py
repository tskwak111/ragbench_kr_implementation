#!/usr/bin/env python3
"""Plan synthetic benchmark generation offline; execute only behind every paid gate."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from collections import Counter
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
    QuestionType,
    SourceWindow,
    generation_execution_blockers,
    projected_generation_cost,
)
from ragbench.benchmark.validation import (
    NORMAL_SCOPE_QUOTAS,
    CompletionLevel,
    ValidationConfig,
    completion_level,
    report_payload,
    validate_candidates,
)
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
    parser.add_argument("--allow-reduced-scope", action="store_true")
    parser.add_argument("--max-replacement-rounds", type=int, default=10)
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
    validation_run_hash = report.get("validation_run_hash")
    if not isinstance(validation_run_hash, str):
        raise BenchmarkGenerationError("validation report is missing its immutable identity")
    prefix = f"{plan_hash}-{validation_run_hash}"
    candidate_path = output_dir / f"{prefix}-candidates.jsonl"
    report_path = output_dir / f"{prefix}-validation-report.json"
    candidate_payload = "".join(
        item.model_dump_json() + "\n" for item in candidates
    ).encode()
    report_bytes = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    _write_immutable(candidate_path, candidate_payload)
    _write_immutable(report_path, report_bytes)


def _write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise BenchmarkGenerationError("immutable benchmark artifact already conflicts")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                raise BenchmarkGenerationError(
                    "immutable benchmark artifact already conflicts"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


async def _run(args: argparse.Namespace) -> int:
    if args.max_replacement_rounds < 0:
        raise BenchmarkGenerationError("--max-replacement-rounds cannot be negative")
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
        current_plan = plan
        report = None
        for attempt in range(args.max_replacement_rounds + 1):
            for batch in current_plan.batches:
                generated.extend(await generator.generate_batch(current_plan, batch))
            report = validate_candidates(tuple(generated), windows, config=validation_config)
            level = completion_level(report.type_distribution)
            if level is CompletionLevel.TARGET or (
                args.allow_reduced_scope and level is CompletionLevel.NORMAL_FLOOR
            ):
                break
            if attempt == args.max_replacement_rounds:
                break
            accepted_counts = Counter(
                {
                    QuestionType(kind): count
                    for kind, count in report.type_distribution.items()
                }
            )
            current_plan = GenerationPlanner(generation_config).plan_replacements(
                plan,
                accepted_counts=accepted_counts,
                attempt=attempt + 1,
                target_quotas=(NORMAL_SCOPE_QUOTAS if args.allow_reduced_scope else None),
            )
        assert report is not None
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
        achieved = completion_level(report.type_distribution)
        if achieved is CompletionLevel.TARGET:
            return 0
        if args.allow_reduced_scope and achieved is CompletionLevel.NORMAL_FLOOR:
            return 0
        return 3
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
