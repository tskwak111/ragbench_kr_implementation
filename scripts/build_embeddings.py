#!/usr/bin/env python3
"""Plan or explicitly execute a resumable immutable embedding snapshot."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragbench.core.config import Settings
from ragbench.core.hashing import canonical_json_hash
from ragbench.core.ids import deterministic_uuid
from ragbench.core.money import BudgetGuard, SqlAlchemyBudgetRepository
from ragbench.db.session import create_lock_session_factory, create_session_factory
from ragbench.embeddings.repository import (
    ChunkEmbeddingInput,
    EmbeddingSnapshot,
    SqlAlchemyEmbeddingRepository,
    chunk_manifest_hash,
    embedding_index_plan,
    frozen_source_metadata,
)
from ragbench.embeddings.service import EmbeddingService
from ragbench.providers.upstage.client import SqlAlchemyProviderStore, UpstageGateway
from ragbench.providers.upstage.pricing import PriceBook


class BuildGateError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EmbeddingBuildPlan:
    plan_hash: str
    snapshot: EmbeddingSnapshot
    chunks: tuple[ChunkEmbeddingInput, ...]
    total_chunks: int
    total_tokens: int
    dataset_sha256: str


def build_plan(
    dataset: Path,
    *,
    corpus_snapshot_id: str,
    model_id: str,
    query_model_id: str,
    dimension: int,
) -> EmbeddingBuildPlan:
    """Validate one chunk dataset and derive its immutable embedding identity."""
    if not dataset.is_file() or dataset.is_symlink():
        raise BuildGateError("chunk dataset must be a regular non-symlink file")
    payload = dataset.read_bytes()
    rows: list[dict[str, Any]] = []
    for line in payload.decode("utf-8").splitlines():
        if line:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise BuildGateError("every chunk row must be a JSON object")
            rows.append(item)
    if not rows:
        raise BuildGateError("chunk dataset cannot be empty")
    parse_ids = {str(item.get("parse_snapshot_id") or "") for item in rows}
    strategies = {str(item.get("strategy") or "") for item in rows}
    if len(parse_ids) != 1 or "" in parse_ids:
        raise BuildGateError("embedding input must use exactly one parse snapshot")
    if len(strategies) != 1 or "" in strategies:
        raise BuildGateError("embedding input must use exactly one chunk strategy")
    index_plan = embedding_index_plan(dimension)
    if any(not item.get("document_id") for item in rows):
        raise BuildGateError("every chunk row requires document_id")
    try:
        chunks = tuple(
            ChunkEmbeddingInput(
                str(item["chunk_id"]),
                str(item["document_id"]),
                str(item["content"]),
                int(item["token_count"]),
                source_metadata=frozen_source_metadata(
                    {
                        key: item[key]
                        for key in (
                            "page_start",
                            "page_end",
                            "section_path",
                            "source_block_ids",
                            "strategy_hash",
                            "token_start",
                            "token_end",
                        )
                        if key in item
                    }
                ),
            )
            for item in rows
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BuildGateError("chunk rows have invalid embedding fields") from error
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise BuildGateError("chunk IDs must be unique")
    dataset_hash = hashlib.sha256(payload).hexdigest()
    identity = {
        "schema": "embedding-plan-v1",
        "dataset_sha256": dataset_hash,
        "corpus_snapshot_id": corpus_snapshot_id,
        "parse_snapshot_id": next(iter(parse_ids)),
        "chunk_strategy": next(iter(strategies)),
        "model_id": model_id,
        "query_model_id": query_model_id,
        "dimension": dimension,
        "normalization": "l2",
        "artifact_manifest_hash": chunk_manifest_hash(chunks),
        "index_strategy": index_plan.strategy,
        "candidate_factor": index_plan.candidate_factor,
    }
    plan_hash = canonical_json_hash(identity)
    created_at = datetime.fromtimestamp(dataset.stat().st_mtime, tz=UTC)
    snapshot = EmbeddingSnapshot(
        snapshot_id=str(deterministic_uuid(identity)),
        corpus_snapshot_id=corpus_snapshot_id,
        parse_snapshot_id=next(iter(parse_ids)),
        chunk_strategy=next(iter(strategies)),
        model_id=model_id,
        query_model_id=query_model_id,
        dimension=dimension,
        normalization="l2",
        expected_chunk_count=len(chunks),
        artifact_manifest_hash=chunk_manifest_hash(chunks),
        index_strategy=index_plan.strategy,
        candidate_factor=index_plan.candidate_factor,
        created_at=created_at,
    )
    return EmbeddingBuildPlan(
        plan_hash,
        snapshot,
        chunks,
        len(chunks),
        sum(chunk.token_count for chunk in chunks),
        dataset_hash,
    )


def require_live_gate(*, live: bool, confirm_paid: bool) -> None:
    """Require both deliberate live switches before constructing a provider gateway."""
    if not live:
        raise BuildGateError("provider execution requires --live")
    if not confirm_paid:
        raise BuildGateError("provider execution requires --confirm-paid after price review")


async def execute_plan(
    plan: EmbeddingBuildPlan,
    *,
    settings: Settings,
    prices_path: Path,
    max_batch_items: int,
    max_batch_tokens: int,
    supports_input_type: bool,
) -> EmbeddingSnapshot:
    """Execute only after price/API gates; all provider calls pass through the gateway."""
    if not settings.upstage_api_key:
        raise BuildGateError("UPSTAGE_API_KEY is required for live embedding execution")
    price_book = PriceBook.from_yaml(prices_path)
    price_book.verify_paid_batch()
    session_factory = create_session_factory(settings)
    lock_factory = create_lock_session_factory(settings)
    budget_repository = SqlAlchemyBudgetRepository(session_factory)
    gateway = UpstageGateway(
        api_key=settings.upstage_api_key,
        base_url=settings.upstage_base_url,
        price_book=price_book,
        budget_guard=BudgetGuard(
            budget_repository,
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
    try:
        service = EmbeddingService(
            gateway,
            SqlAlchemyEmbeddingRepository(session_factory),
            max_batch_items=max_batch_items,
            max_batch_tokens=max_batch_tokens,
            supports_input_type=supports_input_type,
        )
        return await service.embed_chunks(plan.snapshot, plan.chunks)
    finally:
        await gateway.aclose()


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--corpus-snapshot-id", required=True)
    parser.add_argument("--document-model-id")
    parser.add_argument("--query-model-id")
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--max-batch-items", type=int, default=100)
    parser.add_argument("--max-batch-tokens", type=int, default=100_000)
    parser.add_argument("--supports-input-type", action="store_true")
    parser.add_argument("--prices", type=Path, default=Path("configs/prices.yaml"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--confirm-paid", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    plan = build_plan(
        args.dataset,
        corpus_snapshot_id=args.corpus_snapshot_id,
        model_id=args.document_model_id or settings.upstage_document_embedding_model_id,
        query_model_id=args.query_model_id or settings.upstage_query_embedding_model_id,
        dimension=args.dimension,
    )
    if not args.live:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "plan_hash": plan.plan_hash,
                    "snapshot": asdict(plan.snapshot),
                    "total_chunks": plan.total_chunks,
                    "total_tokens": plan.total_tokens,
                    "live_executed": False,
                },
                default=str,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    require_live_gate(live=args.live, confirm_paid=args.confirm_paid)
    completed = await execute_plan(
        plan,
        settings=settings,
        prices_path=args.prices,
        max_batch_items=args.max_batch_items,
        max_batch_tokens=args.max_batch_tokens,
        supports_input_type=args.supports_input_type,
    )
    print(json.dumps({"snapshot_id": completed.snapshot_id, "complete": completed.complete}))


if __name__ == "__main__":
    asyncio.run(_main())
