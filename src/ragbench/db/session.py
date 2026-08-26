"""Async SQLAlchemy session factory configured from ``Settings``."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from ragbench.core.config import Settings


def require_distinct_database(test_url: str, runtime_url: str) -> None:
    """Refuse destructive tests against the configured runtime database."""

    def identity(value: str) -> tuple[str | None, int, str | None]:
        url = make_url(value)
        host = "loopback" if url.host in {"localhost", "127.0.0.1", "::1"} else url.host
        return host, url.port or 5432, url.database

    if identity(test_url) == identity(runtime_url):
        raise RuntimeError("refusing to reset the runtime database")


def create_session_factory(settings: Settings) -> async_sessionmaker[AsyncSession]:
    """Build a non-expiring async session factory for the configured PostgreSQL database."""
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    return async_sessionmaker(engine, expire_on_commit=False)


def create_lock_session_factory(settings: Settings) -> async_sessionmaker[AsyncSession]:
    """Build the dedicated unpooled factory for transaction advisory locks."""
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def session_scope(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """Yield one transactional session, committing only if the caller succeeds."""
    async with factory() as session, session.begin():
        yield session
