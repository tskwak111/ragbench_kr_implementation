"""PostgreSQL migration contract tests."""

import asyncio
import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import create_async_engine

TEST_DATABASE_URL = os.getenv(
    "RAGBENCH_TEST_DATABASE_URL",
    "postgresql+asyncpg://ragbench:ragbench@localhost:5433/ragbench_test",
)
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
    except (OSError, OperationalError) as error:
        pytest.skip(f"PostgreSQL integration database is unavailable: {error}")
    finally:
        await engine.dispose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_initial_migration_creates_experiment_evidence_schema() -> None:
    """Catch a missing migration or a migration that omits evidence constraints."""
    await _reset_schema(TEST_DATABASE_URL)
    await asyncio.to_thread(command.upgrade, _alembic_config(TEST_DATABASE_URL), "head")

    engine = create_async_engine(TEST_DATABASE_URL)
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
    finally:
        await engine.dispose()
