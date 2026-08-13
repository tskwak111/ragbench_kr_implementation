"""Tests for database session lifecycle helpers."""

import pytest
from sqlalchemy.pool import NullPool

from ragbench.core.config import Settings
from ragbench.db.session import create_lock_session_factory, create_session_factory, session_scope


class _Transaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> None:
        return None


class _Session:
    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def begin(self) -> _Transaction:
        return _Transaction()


class _SessionFactory:
    def __init__(self, session: _Session) -> None:
        self._session = session

    def __call__(self) -> _Session:
        return self._session


@pytest.mark.asyncio
async def test_session_scope_supports_async_with() -> None:
    """Catch exposing a bare async generator instead of an async context manager."""
    session = _Session()
    async with session_scope(_SessionFactory(session)) as yielded:
        assert yielded is session


@pytest.mark.asyncio
async def test_lock_session_factory_uses_a_distinct_nullpool_engine() -> None:
    """Catch Task 4 wiring lock waiters onto the operational connection pool."""
    settings = Settings(database_url="postgresql+asyncpg://unused:unused@localhost/unused")
    operational_factory = create_session_factory(settings)
    lock_factory = create_lock_session_factory(settings)
    operational_engine = operational_factory.kw["bind"]
    lock_engine = lock_factory.kw["bind"]
    try:
        assert lock_engine is not operational_engine
        assert isinstance(lock_engine.sync_engine.pool, NullPool)
        assert not isinstance(operational_engine.sync_engine.pool, NullPool)
    finally:
        await lock_engine.dispose()
        await operational_engine.dispose()
