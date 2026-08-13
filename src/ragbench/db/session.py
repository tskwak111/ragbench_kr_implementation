"""Async SQLAlchemy session factory configured from ``Settings``."""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ragbench.core.config import Settings


def create_session_factory(settings: Settings) -> async_sessionmaker[AsyncSession]:
    """Build a non-expiring async session factory for the configured PostgreSQL database."""
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    return async_sessionmaker(engine, expire_on_commit=False)


async def session_scope(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """Yield one transactional session, committing only if the caller succeeds."""
    async with factory() as session, session.begin():
        yield session
