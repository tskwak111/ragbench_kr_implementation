"""Deterministic provider request hashing contracts."""

import asyncio
from typing import Any

import pytest
from sqlalchemy.dialects.postgresql import dialect

from ragbench.core.hashing import canonical_json_hash
from ragbench.providers.upstage.client import CacheKeyParts, SqlAlchemyProviderStore


def test_canonical_hash_is_stable_across_mapping_key_order() -> None:
    """Catch serializing mappings in insertion order instead of canonical order."""
    left = {"model": "solar-pro4", "params": {"temperature": 0, "top_p": 0.9}}
    right = {"params": {"top_p": 0.9, "temperature": 0}, "model": "solar-pro4"}

    assert canonical_json_hash(left) == canonical_json_hash(right)


def test_cache_key_covers_every_billable_request_dimension_without_secret() -> None:
    """Catch cache aliases and accidental inclusion of provider credentials."""
    parts = CacheKeyParts(
        operation="generate",
        model_id="solar-pro4",
        provider_params={"temperature": 0},
        prompt_hash=canonical_json_hash("질문"),
        context_hash=canonical_json_hash(["근거"]),
        document_sha256=None,
        schema_version="provider-cache-v1",
    )
    key = parts.digest()

    assert len(key) == 64
    assert (
        key
        != CacheKeyParts(
            operation="generate",
            model_id="solar-pro3",
            provider_params={"temperature": 0},
            prompt_hash=parts.prompt_hash,
            context_hash=parts.context_hash,
            document_sha256=None,
            schema_version="provider-cache-v1",
        ).digest()
    )
    assert "secret-api-key" not in key


class _Transaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> None:
        return None


class _StatementSession:
    def __init__(self) -> None:
        self.statement: Any = None
        self.events: list[tuple[str, int]] = []

    async def __aenter__(self) -> "_StatementSession":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def begin(self) -> _Transaction:
        return _Transaction()

    async def execute(self, statement: Any, params: dict[str, int] | None = None) -> None:
        self.statement = statement
        rendered = str(statement)
        if "pg_advisory_lock" in rendered:
            assert params is not None
            self.events.append(("lock", params["lock_id"]))
        if "pg_advisory_unlock" in rendered:
            assert params is not None
            self.events.append(("unlock", params["lock_id"]))


class _StatementFactory:
    def __init__(self, session: _StatementSession) -> None:
        self.session = session

    def __call__(self) -> _StatementSession:
        return self.session


@pytest.mark.asyncio
async def test_sql_cache_put_replaces_expired_conflict() -> None:
    """Catch leaving an expired unique cache row permanently unreplaceable."""
    session = _StatementSession()
    store = SqlAlchemyProviderStore(_StatementFactory(session))

    await store.put(
        "a" * 64,
        operation="generate",
        model_id="solar-pro4",
        response={"choices": []},
    )

    rendered = str(session.statement.compile(dialect=dialect()))
    assert "ON CONFLICT" in rendered
    assert "DO UPDATE" in rendered
    assert "expires_at" in rendered


@pytest.mark.asyncio
async def test_memory_singleflight_cleans_unique_key_locks() -> None:
    """Catch unbounded retention of completed singleflight locks."""
    from ragbench.providers.upstage.client import MemoryProviderStore

    store = MemoryProviderStore()
    for ordinal in range(200):
        async with store.singleflight(f"cache-key-{ordinal}"):
            pass

    assert store.singleflight_lock_count == 0


@pytest.mark.asyncio
async def test_memory_singleflight_cleans_cancelled_waiter() -> None:
    """Catch retaining a key lock when a waiting request is cancelled."""
    from ragbench.providers.upstage.client import MemoryProviderStore

    store = MemoryProviderStore()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def holder() -> None:
        async with store.singleflight("shared"):
            entered.set()
            await release.wait()

    async def waiter() -> None:
        async with store.singleflight("shared"):
            pass

    holder_task = asyncio.create_task(holder())
    await entered.wait()
    waiter_task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    waiter_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_task
    release.set()
    await holder_task

    assert store.singleflight_lock_count == 0


@pytest.mark.asyncio
async def test_sql_singleflight_uses_deterministic_signed_lock_and_unlocks() -> None:
    """Catch losing cross-process exclusion or leaking PostgreSQL session advisory locks."""
    first_session = _StatementSession()
    first_store = SqlAlchemyProviderStore(_StatementFactory(first_session))
    async with first_store.singleflight("a" * 64):
        assert first_session.events[0][0] == "lock"
    first_lock_id = first_session.events[0][1]

    second_session = _StatementSession()
    second_store = SqlAlchemyProviderStore(_StatementFactory(second_session))
    async with second_store.singleflight("b" * 64):
        pass

    assert first_session.events == [("lock", first_lock_id), ("unlock", first_lock_id)]
    assert -(2**63) <= first_lock_id < 2**63
    assert second_session.events[0][1] != first_lock_id
