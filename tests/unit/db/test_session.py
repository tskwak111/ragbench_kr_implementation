"""Tests for database session lifecycle helpers."""

import pytest

from ragbench.db.session import session_scope


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
