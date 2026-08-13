"""Deterministic provider request hashing contracts."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.dialects.postgresql import dialect
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

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
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def __aenter__(self) -> None:
        self.events.append("transaction-enter")
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_value, traceback
        suffix = exc_type.__name__ if exc_type is not None else "none"
        self.events.append(f"transaction-exit:{suffix}")
        return None


class _StatementSession:
    def __init__(self) -> None:
        self.statement: Any = None
        self.events: list[tuple[str, int]] = []
        self.transaction_events: list[str] = []
        self.session_events: list[str] = []

    async def __aenter__(self) -> "_StatementSession":
        self.session_events.append("session-enter")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_value, traceback
        suffix = exc_type.__name__ if exc_type is not None else "none"
        self.session_events.append(f"session-exit:{suffix}")
        return None

    def begin(self) -> _Transaction:
        return _Transaction(self.transaction_events)

    async def scalar(self, statement: Any) -> None:
        self.statement = statement
        return None

    async def execute(self, statement: Any, params: dict[str, int] | None = None) -> None:
        self.statement = statement
        rendered = str(statement)
        if "pg_advisory_xact_lock" in rendered:
            assert params is not None
            self.events.append(("xact-lock", params["lock_id"]))


class _StatementFactory:
    def __init__(self, session: _StatementSession, *, null_pool: bool = False) -> None:
        self.session = session
        self.calls = 0
        pool: object = NullPool(lambda: None) if null_pool else object()
        self.kw = {"bind": SimpleNamespace(sync_engine=SimpleNamespace(pool=pool))}

    def __call__(self) -> _StatementSession:
        self.calls += 1
        return self.session


def _make_store(
    operational_session: _StatementSession | None = None,
    lock_session: _StatementSession | None = None,
) -> tuple[SqlAlchemyProviderStore, _StatementFactory, _StatementFactory]:
    operational_factory = _StatementFactory(operational_session or _StatementSession())
    lock_factory = _StatementFactory(lock_session or _StatementSession(), null_pool=True)
    store = SqlAlchemyProviderStore(
        operational_factory,  # type: ignore[arg-type]
        lock_session_factory=lock_factory,  # type: ignore[arg-type]
    )
    return store, operational_factory, lock_factory


@pytest.mark.asyncio
async def test_sql_cache_put_replaces_expired_conflict() -> None:
    """Catch leaving an expired unique cache row permanently unreplaceable."""
    session = _StatementSession()
    store, _, _ = _make_store(operational_session=session)

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
async def test_sql_singleflight_uses_deterministic_signed_transaction_lock() -> None:
    """Catch losing cross-process exclusion or reverting to a session advisory lock."""
    first_session = _StatementSession()
    first_store, first_operational, first_lock = _make_store(lock_session=first_session)
    async with first_store.singleflight("a" * 64):
        assert first_session.events[0][0] == "xact-lock"
    first_lock_id = first_session.events[0][1]

    second_session = _StatementSession()
    second_store, _, _ = _make_store(lock_session=second_session)
    async with second_store.singleflight("b" * 64):
        pass

    assert first_session.events == [("xact-lock", first_lock_id)]
    assert first_session.transaction_events == ["transaction-enter", "transaction-exit:none"]
    assert first_session.session_events == ["session-enter", "session-exit:none"]
    assert first_operational.calls == 0
    assert first_lock.calls == 1
    assert -(2**63) <= first_lock_id < 2**63
    assert second_session.events[0][1] != first_lock_id


def test_sql_store_requires_distinct_nullpool_lock_factory() -> None:
    """Catch silent reuse of the bounded operational pool for advisory-lock waiters."""
    session = _StatementSession()
    shared_factory = _StatementFactory(session, null_pool=True)
    with pytest.raises(ValueError, match="distinct"):
        SqlAlchemyProviderStore(
            shared_factory,  # type: ignore[arg-type]
            lock_session_factory=shared_factory,  # type: ignore[arg-type]
        )

    operational_factory = _StatementFactory(session)
    pooled_lock_factory = _StatementFactory(_StatementSession())
    with pytest.raises(ValueError, match="NullPool"):
        SqlAlchemyProviderStore(
            operational_factory,  # type: ignore[arg-type]
            lock_session_factory=pooled_lock_factory,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_sql_store_accepts_real_dedicated_nullpool_factory() -> None:
    """Prove the production SQLAlchemy factory shape uses an isolated NullPool engine."""
    database_url = "postgresql+asyncpg://unused:unused@localhost/unused"
    operational_engine = create_async_engine(database_url)
    lock_engine = create_async_engine(database_url, poolclass=NullPool)
    pooled_lock_engine = create_async_engine(database_url)
    try:
        operational_factory = async_sessionmaker(operational_engine)
        lock_factory = async_sessionmaker(lock_engine)

        SqlAlchemyProviderStore(
            operational_factory,
            lock_session_factory=lock_factory,
        )
        with pytest.raises(ValueError, match="NullPool"):
            SqlAlchemyProviderStore(
                operational_factory,
                lock_session_factory=async_sessionmaker(pooled_lock_engine),
            )
    finally:
        await pooled_lock_engine.dispose()
        await lock_engine.dispose()
        await operational_engine.dispose()


@pytest.mark.asyncio
async def test_sql_store_rejects_distinct_factories_bound_to_same_engine() -> None:
    """Catch two factory objects silently sharing the same operational connection source."""
    shared_engine = create_async_engine(
        "postgresql+asyncpg://unused:unused@localhost/unused", poolclass=NullPool
    )
    try:
        with pytest.raises(ValueError, match="distinct engine"):
            SqlAlchemyProviderStore(
                async_sessionmaker(shared_engine),
                lock_session_factory=async_sessionmaker(shared_engine),
            )
    finally:
        await shared_engine.dispose()


@pytest.mark.asyncio
async def test_sql_cache_operations_never_consume_lock_sessions() -> None:
    """Catch cache reads/writes occupying the dedicated singleflight connection path."""
    store, operational_factory, lock_factory = _make_store()

    assert await store.get("a" * 64) is None
    await store.put(
        "a" * 64,
        operation="generate",
        model_id="solar-pro4",
        response={"choices": []},
    )

    assert operational_factory.calls == 2
    assert lock_factory.calls == 0


@pytest.mark.asyncio
async def test_sql_singleflight_error_exits_transaction_without_manual_unlock() -> None:
    """Catch an exceptional protected section leaking a session-scoped advisory lock."""
    lock_session = _StatementSession()
    store, _, _ = _make_store(lock_session=lock_session)

    with pytest.raises(RuntimeError, match="boom"):
        async with store.singleflight("a" * 64):
            raise RuntimeError("boom")

    assert lock_session.transaction_events == [
        "transaction-enter",
        "transaction-exit:RuntimeError",
    ]
    assert lock_session.session_events == ["session-enter", "session-exit:RuntimeError"]
    assert len(lock_session.events) == 1
    assert lock_session.events[0][0] == "xact-lock"


@pytest.mark.asyncio
async def test_sql_singleflight_cancellation_exits_transaction() -> None:
    """Catch cancellation bypassing transaction cleanup while the lock is held."""
    lock_session = _StatementSession()
    store, _, _ = _make_store(lock_session=lock_session)
    entered = asyncio.Event()

    async def holder() -> None:
        async with store.singleflight("a" * 64):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(holder())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert lock_session.transaction_events == [
        "transaction-enter",
        "transaction-exit:CancelledError",
    ]
    assert lock_session.session_events == ["session-enter", "session-exit:CancelledError"]
    assert len(lock_session.events) == 1
    assert lock_session.events[0][0] == "xact-lock"
