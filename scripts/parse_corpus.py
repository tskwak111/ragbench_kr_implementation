#!/usr/bin/env python3
"""Plan a corpus parse offline; execute only an exactly confirmed fresh plan."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from ragbench.core.config import Settings
from ragbench.core.money import BudgetGuard, SqlAlchemyBudgetRepository
from ragbench.db.models import ApiUsage
from ragbench.db.session import create_lock_session_factory, create_session_factory
from ragbench.ingestion.manifest import CorpusManifest
from ragbench.ingestion.parser import (
    CorpusSnapshot,
    ParseAuthorization,
    ParserPipeline,
    SqlAlchemyParseRepository,
    execution_blockers,
)
from ragbench.providers.base import ParseRequest
from ragbench.providers.upstage.client import SqlAlchemyProviderStore, UpstageGateway
from ragbench.providers.upstage.pricing import PriceBook

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _DryRunGateway:
    async def parse(self, request: ParseRequest) -> Any:
        del request
        raise RuntimeError("dry-run gateway cannot execute provider calls")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs" / "corpus.yaml")
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--mode", choices=("standard", "enhanced"), required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--vat-buffer", type=Decimal, default=Decimal("0.10"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-plan")
    return parser.parse_args()


async def _settled_cost(factory: Any) -> Decimal:
    async with factory() as session:
        total = func.coalesce(func.sum(ApiUsage.estimated_cost_usd), 0)
        value = await session.scalar(select(total))
    return Decimal(value or 0)


async def _run(args: argparse.Namespace) -> int:
    settings = Settings()
    manifest = CorpusManifest.load(args.manifest)
    validation = manifest.validate()
    if validation.corpus_snapshot_id != args.snapshot_id:
        raise RuntimeError("--snapshot-id does not match the manifest content")
    snapshot = CorpusSnapshot(args.snapshot_id, manifest.documents)
    price_book = PriceBook.from_yaml(PROJECT_ROOT / "configs" / "prices.yaml")
    session_factory = create_session_factory(settings)
    lock_factory = create_lock_session_factory(settings)
    gateway: Any = _DryRunGateway()
    try:
        settled = await _settled_cost(session_factory)
        repository = SqlAlchemyParseRepository(session_factory)
        pipeline = ParserPipeline(
            snapshots={snapshot.snapshot_id: snapshot},
            gateway=gateway,
            repository=repository,
            price_book=price_book,
            model_id=settings.upstage_document_parse_model_id,
            model_version=args.model_version,
            hard_budget_usd=settings.max_project_budget_usd,
            settled_cost_usd=lambda: settled,
            vat_buffer=args.vat_buffer,
            authorization=(
                ParseAuthorization(args.confirm_plan) if args.confirm_plan is not None else None
            ),
        )
        resume = not args.no_resume
        plan = await pipeline.plan_corpus(args.snapshot_id, args.mode, resume)
        payload: dict[str, Any] = {"dry_run": not args.execute, "plan": asdict(plan)}
        if not args.execute:
            print(json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True))
            return 0
        blockers = execution_blockers(
            execute=True,
            confirm_plan=args.confirm_plan,
            live_enabled=os.environ.get("RUN_LIVE_UPSTAGE_TESTS") == "1",
            api_key_present=bool(settings.upstage_api_key),
        )
        if blockers:
            payload["executed"] = False
            payload["blockers"] = blockers
            print(json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True))
            return 2
        # Fresh pricing and exact plan confirmation are checked again by parse_corpus.
        assert settings.upstage_api_key is not None
        store = SqlAlchemyProviderStore(
            session_factory,
            lock_session_factory=lock_factory,
            max_lock_connections=settings.max_lock_connections,
        )
        gateway = UpstageGateway(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
            price_book=price_book,
            budget_guard=BudgetGuard(
                SqlAlchemyBudgetRepository(session_factory),
                hard_limit=settings.max_project_budget_usd,
            ),
            store=store,
            max_concurrency=settings.max_concurrency,
            max_retries=settings.max_retries,
        )
        pipeline = ParserPipeline(
            snapshots={snapshot.snapshot_id: snapshot},
            gateway=gateway,
            repository=repository,
            price_book=price_book,
            model_id=settings.upstage_document_parse_model_id,
            model_version=args.model_version,
            hard_budget_usd=settings.max_project_budget_usd,
            settled_cost_usd=lambda: settled,
            vat_buffer=args.vat_buffer,
            authorization=ParseAuthorization(args.confirm_plan),
        )
        summary = await pipeline.parse_corpus(args.snapshot_id, args.mode, resume)
        payload.update({"dry_run": False, "executed": True, "summary": asdict(summary)})
        print(json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True))
        return 0 if summary.failed_documents == 0 else 1
    finally:
        closer = getattr(gateway, "aclose", None)
        if closer is not None:
            await closer()
        await session_factory.kw["bind"].dispose()
        await lock_factory.kw["bind"].dispose()


def main() -> None:
    args = _parse_args()
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
