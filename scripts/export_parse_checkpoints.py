#!/usr/bin/env python3
"""Export successful parse checkpoints from PostgreSQL as private JSONL."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragbench.core.config import Settings
from ragbench.db.models import Document, ParseRun
from ragbench.db.session import create_session_factory
from ragbench.ingestion.parser import ParseCheckpoint, ParseMode


def write_checkpoints(checkpoints: list[ParseCheckpoint], output: Path) -> int:
    payload = "".join(
        json.dumps(asdict(checkpoint), default=str, ensure_ascii=False, sort_keys=True) + "\n"
        for checkpoint in checkpoints
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(output, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
        destination.write(payload)
    return len(checkpoints)


async def load_checkpoints(
    factory: async_sessionmaker[AsyncSession], snapshot_id: str
) -> list[ParseCheckpoint]:
    async with factory() as session:
        rows = (
            await session.execute(
                select(ParseRun, Document)
                .join(Document, Document.id == ParseRun.document_id)
                .where(
                    ParseRun.corpus_snapshot_id == snapshot_id,
                    ParseRun.status == "succeeded",
                )
                .order_by(ParseRun.config_snapshot["document_id"].astext, ParseRun.mode)
            )
        ).all()
    checkpoints = []
    for run, document in rows:
        if run.mode not in ("standard", "enhanced"):
            raise ValueError(f"unexpected parse mode: {run.mode}")
        mode: ParseMode = run.mode
        checkpoints.append(
            ParseCheckpoint(
                run.corpus_snapshot_id,
                str(run.config_snapshot["document_id"]),
                document.sha256,
                int(run.config_snapshot["expected_pages"]),
                run.provider_model_id,
                run.provider_model_version,
                mode,
                run.status,
                dict(run.raw_response) if run.raw_response is not None else None,
                run.raw_response_hash,
                run.markdown,
                run.html,
                tuple(dict(item) for item in run.elements),
                tuple(dict(item) for item in run.page_mappings),
                run.latency_ms,
                run.cost_usd,
                run.config_snapshot.get("correlation_id"),
                run.error,
            )
        )
    if not checkpoints:
        raise RuntimeError("no successful parse checkpoints found")
    return checkpoints


async def _run(snapshot_id: str, output: Path) -> None:
    factory = create_session_factory(Settings())
    try:
        count = write_checkpoints(await load_checkpoints(factory, snapshot_id), output)
        print(json.dumps({"checkpoints": count, "output": str(output)}, sort_keys=True))
    finally:
        await factory.kw["bind"].dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--snapshot-id", required=True)
    args = parser.parse_args()
    asyncio.run(_run(args.snapshot_id, args.output))


if __name__ == "__main__":
    main()
