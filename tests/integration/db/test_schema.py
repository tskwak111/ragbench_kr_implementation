"""PostgreSQL migration contract tests."""

import asyncio
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragbench.core.hashing import canonical_json_hash
from ragbench.core.money import SqlAlchemyBudgetRepository, Usage
from ragbench.ingestion.parser import ParseCheckpoint, SqlAlchemyParseRepository

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _alembic_config(database_url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


async def _reset_schema(database_url: str) -> None:
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            await connection.execute(text("CREATE SCHEMA public"))
    finally:
        await engine.dispose()


def _test_database_url() -> str | None:
    return os.getenv("RAGBENCH_TEST_DATABASE_URL")


def test_database_url_must_be_explicitly_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catch accidental skipping of an integration test with an intended database URL."""
    monkeypatch.delenv("RAGBENCH_TEST_DATABASE_URL", raising=False)
    assert _test_database_url() is None


@pytest.mark.asyncio
async def test_configured_database_connection_failures_are_not_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch swallowing a CI database outage after its URL has been supplied."""

    class FailingConnection:
        async def __aenter__(self) -> None:
            raise OSError("connection refused")

        async def __aexit__(self, *args: object) -> None:
            return None

    class FailingEngine:
        def begin(self) -> FailingConnection:
            return FailingConnection()

        async def dispose(self) -> None:
            return None

    monkeypatch.setattr(sys.modules[__name__], "create_async_engine", lambda _: FailingEngine())

    with pytest.raises(OSError, match="connection refused"):
        await _reset_schema("postgresql+asyncpg://configured.test/ragbench")


def test_offline_downgrade_preserves_shared_vector_extension() -> None:
    """Catch a downgrade that removes an extension it did not necessarily create."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "downgrade",
            "20260813_0001:base",
            "--sql",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "DROP EXTENSION" not in completed.stdout


@pytest.mark.integration
@pytest.mark.asyncio
async def test_initial_migration_creates_experiment_evidence_schema() -> None:
    """Catch a missing migration or a migration that omits evidence constraints."""
    database_url = _test_database_url()
    if database_url is None:
        pytest.skip("RAGBENCH_TEST_DATABASE_URL is not configured")

    await _reset_schema(database_url)
    await asyncio.to_thread(command.upgrade, _alembic_config(database_url), "head")

    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            extension = await connection.scalar(
                text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            )
            assert extension is True

            document_id = await connection.scalar(
                text(
                    """
                    INSERT INTO document (title, sha256, source_uri, metadata_snapshot)
                    VALUES (
                        'Korean annual report',
                        repeat('a', 64),
                        'https://example.test/report.pdf',
                        '{}'
                    )
                    RETURNING id
                    """
                )
            )
            experiment_id = await connection.scalar(
                text(
                    """
                    INSERT INTO experiment (name, status, config_snapshot)
                    VALUES ('baseline', 'planned', '{"top_k": 5}')
                    RETURNING id
                    """
                )
            )
            assert document_id is not None
            assert experiment_id is not None

            with pytest.raises(IntegrityError), connection.begin_nested():
                await connection.execute(
                    text(
                        """
                        INSERT INTO document (title, sha256, source_uri, metadata_snapshot)
                        VALUES (
                            'Duplicate report',
                            repeat('a', 64),
                            'https://example.test/duplicate.pdf',
                            '{}'
                        )
                        """
                    )
                )

        repository = SqlAlchemyBudgetRepository(async_sessionmaker(engine, expire_on_commit=False))
        reservation = await repository.reserve_atomic(
            correlation_id=uuid4(),
            projected_cost=Decimal("0.100000"),
            hard_limit=Decimal("135.000000"),
        )
        await repository.settle(
            reservation.id,
            operation="generate",
            model_id="solar-pro4",
            usage=Usage(10, 4, 0, Decimal("0.000008")),
            cache_hit=False,
        )
        async with engine.connect() as connection:
            transition = (
                await connection.execute(
                    text(
                        """
                        SELECT br.status, br.settled_cost_usd, au.estimated_cost_usd
                        FROM budget_reservation br
                        JOIN api_usage au ON au.correlation_id = br.correlation_id
                        WHERE br.id = :reservation_id
                        """
                    ),
                    {"reservation_id": reservation.id},
                )
            ).one()
        assert transition.status == "settled"
        assert transition.settled_cost_usd == Decimal("0.000008")
        assert transition.estimated_cost_usd == Decimal("0.000008")
    finally:
        await engine.dispose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parse_checkpoint_repository_round_trips_complete_evidence() -> None:
    """Catch SQL persistence losing provider output or breaking checkpoint idempotency."""
    database_url = _test_database_url()
    if database_url is None:
        pytest.skip("RAGBENCH_TEST_DATABASE_URL is not configured")

    await _reset_schema(database_url)
    await asyncio.to_thread(command.upgrade, _alembic_config(database_url), "head")
    engine = create_async_engine(database_url)
    try:
        repository = SqlAlchemyParseRepository(async_sessionmaker(engine, expire_on_commit=False))
        raw = {
            "model_version": "v1",
            "content": {"markdown": "first", "html": "<p>first</p>"},
            "elements": [{"category": "paragraph", "page": 1}],
            "pages": [{"page": 1, "source_page": 1}],
            "usage": {"pages": 1},
        }
        checkpoint = ParseCheckpoint(
            "a" * 64,
            "doc-a",
            "b" * 64,
            1,
            "document-parse",
            "v1",
            "standard",
            "succeeded",
            raw,
            canonical_json_hash(raw),
            "first",
            "<p>first</p>",
            ({"category": "paragraph", "page": 1},),
            ({"page": 1, "source_page": 1},),
            12,
            Decimal("0.010000"),
            "correlation-a",
        )
        await repository.put(checkpoint)
        await repository.put(checkpoint)

        restored = await repository.get(
            "a" * 64, "b" * 64, "standard", "document-parse", "v1"
        )
        wrong_version = await repository.get(
            "a" * 64, "b" * 64, "standard", "document-parse", "v2"
        )
        async with engine.connect() as connection:
            count = await connection.scalar(text("SELECT count(*) FROM parse_run"))
        assert count == 1
        assert restored == checkpoint
        assert wrong_version is None
    finally:
        await engine.dispose()
