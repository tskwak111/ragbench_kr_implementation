"""Deterministic provider request hashing contracts."""

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

    async def __aenter__(self) -> "_StatementSession":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def begin(self) -> _Transaction:
        return _Transaction()

    async def execute(self, statement: Any) -> None:
        self.statement = statement


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
