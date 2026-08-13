"""Guarded, cached, retry-bounded Upstage HTTP gateway."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from email.utils import parsedate_to_datetime
from random import uniform
from typing import Any, Protocol
from uuid import UUID, uuid4

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragbench.core.hashing import canonical_json_hash
from ragbench.core.money import BudgetGuard, Reservation, Usage
from ragbench.db.models import ApiCacheEntry
from ragbench.providers.base import (
    EmbedRequest,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    ParsedDocument,
    ParseRequest,
    ProviderGateway,
)
from ragbench.providers.upstage.pricing import PriceBook, PricingRequest

LOGGER = logging.getLogger(__name__)
CACHE_SCHEMA_VERSION = "provider-cache-v1"


class ProviderHTTPError(RuntimeError):
    """Provider rejected a request after the permitted retry policy."""

    def __init__(self, status_code: int, message: str = "provider request failed") -> None:
        super().__init__(f"{message} (status={status_code})")
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class CacheKeyParts:
    operation: str
    model_id: str
    provider_params: dict[str, Any]
    prompt_hash: str | None
    context_hash: str | None
    document_sha256: str | None
    schema_version: str = CACHE_SCHEMA_VERSION

    def digest(self) -> str:
        """Hash all response-affecting, non-secret request dimensions."""
        return canonical_json_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class CachedResponse:
    response: dict[str, Any]
    expires_at: datetime | None = None


class ProviderStore(Protocol):
    async def get(self, cache_key: str) -> CachedResponse | None: ...

    async def put(
        self,
        cache_key: str,
        *,
        operation: str,
        model_id: str,
        response: dict[str, Any],
    ) -> None: ...


class MemoryProviderStore:
    """Deterministic raw response store for offline and contract testing."""

    def __init__(self) -> None:
        self.entries: dict[str, CachedResponse] = {}

    async def get(self, cache_key: str) -> CachedResponse | None:
        cached = self.entries.get(cache_key)
        if cached and cached.expires_at and cached.expires_at <= datetime.now(UTC):
            return None
        return cached

    async def put(
        self,
        cache_key: str,
        *,
        operation: str,
        model_id: str,
        response: dict[str, Any],
    ) -> None:
        del operation, model_id
        self.entries[cache_key] = CachedResponse(dict(response))


class SqlAlchemyProviderStore:
    """PostgreSQL raw response cache shared by concurrent gateway processes."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get(self, cache_key: str) -> CachedResponse | None:
        async with self._session_factory() as session:
            entry = await session.scalar(
                select(ApiCacheEntry).where(ApiCacheEntry.cache_key == cache_key)
            )
            if entry is None:
                return None
            if entry.expires_at is not None and entry.expires_at <= datetime.now(UTC):
                return None
            return CachedResponse(dict(entry.response_snapshot), entry.expires_at)

    async def put(
        self,
        cache_key: str,
        *,
        operation: str,
        model_id: str,
        response: dict[str, Any],
    ) -> None:
        statement = (
            insert(ApiCacheEntry)
            .values(
                cache_key=cache_key,
                operation=operation,
                provider_model_id=model_id,
                response_snapshot=response,
            )
            .on_conflict_do_update(
                index_elements=[ApiCacheEntry.cache_key],
                set_={
                    "operation": operation,
                    "provider_model_id": model_id,
                    "response_snapshot": response,
                    "expires_at": None,
                },
            )
        )
        async with self._session_factory() as session, session.begin():
            await session.execute(statement)


Sleep = Callable[[float], Awaitable[None]]
Jitter = Callable[[float], float]


class UpstageGateway(ProviderGateway):
    """Only supported path for paid Upstage provider requests."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        price_book: PriceBook,
        budget_guard: BudgetGuard,
        store: ProviderStore,
        max_concurrency: int = 5,
        max_retries: int = 5,
        max_backoff_seconds: float = 8.0,
        timeout: httpx.Timeout | None = None,
        client: httpx.AsyncClient | None = None,
        sleep: Sleep | None = None,
        jitter: Jitter | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._price_book = price_book
        self._budget_guard = budget_guard
        self._store = store
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._key_locks: dict[str, asyncio.Lock] = {}
        self.max_retries = max_retries
        self.max_backoff_seconds = max_backoff_seconds
        self._sleep = sleep or asyncio.sleep
        self._jitter = jitter or (lambda upper: uniform(0, upper))
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout or httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0),
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        if request.max_output_tokens is None:
            raise ValueError("max_output_tokens is required to calculate an upper bound")
        if request.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if request.input_tokens <= 0:
            raise ValueError("input_tokens must be a positive conservative upper bound")
        _reject_reserved_params(request.provider_params, {"model", "messages", "max_tokens"})
        key = self.cache_key_for_generate(request)
        cached = await self._store.get(key)
        if cached is not None:
            return await self._cached_generation(request, cached)
        async with self._key_lock(key):
            cached = await self._store.get(key)
            if cached is not None:
                return await self._cached_generation(request, cached)
            projected = self._price_book.estimate(
                PricingRequest(
                    operation="generate",
                    model_id=request.model_id,
                    input_tokens=request.input_tokens,
                    output_tokens=request.max_output_tokens,
                )
            )
            correlation_id, reservation = await self._reserve(projected)
            payload: dict[str, Any] = {
                "model": request.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "\n\n".join((*request.context, request.prompt)),
                    }
                ],
                "max_tokens": request.max_output_tokens,
                **request.provider_params,
            }
            try:
                response = await self._request(
                    "chat/completions", correlation_id=correlation_id, json_payload=payload
                )
            except BaseException:
                await self._budget_guard.release(reservation.id)
                raise
            projected_usage = Usage(request.input_tokens, request.max_output_tokens, 0, projected)
            raw: dict[str, Any] | None = None
            try:
                raw = self._decode_json(response)
                result = self._generation_response(raw)
                usage = self._generation_usage(request, raw)
            except BaseException:
                usage = self._safe_generation_usage(request, raw, projected_usage)
                await self._budget_guard.settle(
                    reservation.id,
                    operation="generate",
                    model_id=request.model_id,
                    usage=usage,
                    cache_hit=False,
                    reconciliation_required=True,
                )
                raise
            await self._budget_guard.settle(
                reservation.id,
                operation="generate",
                model_id=request.model_id,
                usage=usage,
                cache_hit=False,
            )
            await self._store.put(
                key, operation="generate", model_id=request.model_id, response=raw
            )
            return result

    def cache_key_for_generate(self, request: GenerateRequest) -> str:
        """Return the deterministic key used by generation cache lookups."""
        return CacheKeyParts(
            operation="generate",
            model_id=request.model_id,
            provider_params={
                **request.provider_params,
                "max_output_tokens": request.max_output_tokens,
            },
            prompt_hash=canonical_json_hash(request.prompt),
            context_hash=canonical_json_hash(request.context),
            document_sha256=None,
        ).digest()

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        _reject_reserved_params(request.provider_params, {"model", "input"})
        key = CacheKeyParts(
            operation="embed",
            model_id=request.model_id,
            provider_params=request.provider_params,
            prompt_hash=canonical_json_hash(request.texts),
            context_hash=None,
            document_sha256=None,
        ).digest()
        price_request = PricingRequest(
            operation="embed", model_id=request.model_id, input_tokens=request.input_tokens
        )
        projected = self._price_book.estimate(price_request)
        cached = await self._store.get(key)
        if cached is not None:
            return await self._cached_embedding(request, cached)
        async with self._key_lock(key):
            cached = await self._store.get(key)
            if cached is not None:
                return await self._cached_embedding(request, cached)
            correlation_id, reservation = await self._reserve(projected)
            try:
                response = await self._request(
                    "embeddings",
                    json_payload={
                        "model": request.model_id,
                        "input": list(request.texts),
                        **request.provider_params,
                    },
                    correlation_id=correlation_id,
                )
            except BaseException:
                await self._budget_guard.release(reservation.id)
                raise
            try:
                raw = self._decode_json(response)
                result = self._embedding_response(raw)
            except BaseException:
                await self._budget_guard.settle(
                    reservation.id,
                    operation="embed",
                    model_id=request.model_id,
                    usage=Usage(request.input_tokens, 0, 0, projected),
                    cache_hit=False,
                    reconciliation_required=True,
                )
                raise
            await self._budget_guard.settle(
                reservation.id,
                operation="embed",
                model_id=request.model_id,
                usage=Usage(request.input_tokens, 0, 0, projected),
                cache_hit=False,
            )
            await self._store.put(key, operation="embed", model_id=request.model_id, response=raw)
            return result

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        _reject_reserved_params(
            request.provider_params,
            {"model", "document"},
        )
        params = {"mode": request.mode, **request.provider_params}
        key = CacheKeyParts(
            operation="parse",
            model_id=request.model_id,
            provider_params=params,
            prompt_hash=None,
            context_hash=None,
            document_sha256=request.document_sha256,
        ).digest()
        projected = self._price_book.estimate(
            PricingRequest(
                operation="parse",
                model_id=request.model_id,
                billable_pages=request.billable_pages,
                mode=request.mode,
            )
        )
        cached = await self._store.get(key)
        if cached is not None:
            return await self._cached_parse(request, cached)
        async with self._key_lock(key):
            cached = await self._store.get(key)
            if cached is not None:
                return await self._cached_parse(request, cached)
            correlation_id, reservation = await self._reserve(projected)
            try:
                response = await self._request(
                    "document-digitization",
                    correlation_id=correlation_id,
                    data={"model": request.model_id, **request.provider_params},
                    files={
                        "document": (
                            "document.bin",
                            request.content,
                            "application/octet-stream",
                        )
                    },
                )
            except BaseException:
                await self._budget_guard.release(reservation.id)
                raise
            try:
                raw = self._decode_json(response)
                result = ParsedDocument(raw)
            except BaseException:
                await self._budget_guard.settle(
                    reservation.id,
                    operation="parse",
                    model_id=request.model_id,
                    usage=Usage(0, 0, request.billable_pages, projected),
                    cache_hit=False,
                    reconciliation_required=True,
                )
                raise
            await self._budget_guard.settle(
                reservation.id,
                operation="parse",
                model_id=request.model_id,
                usage=Usage(0, 0, request.billable_pages, projected),
                cache_hit=False,
            )
            await self._store.put(key, operation="parse", model_id=request.model_id, response=raw)
            return result

    async def _reserve(self, projected: Decimal) -> tuple[UUID, Reservation]:
        correlation_id = uuid4()
        reservation = await self._budget_guard.reserve(
            correlation_id=correlation_id, projected_cost=projected
        )
        return correlation_id, reservation

    def _key_lock(self, cache_key: str) -> asyncio.Lock:
        return self._key_locks.setdefault(cache_key, asyncio.Lock())

    async def _record_cache_hit(self, *, operation: str, model_id: str, usage: Usage) -> None:
        await self._budget_guard.record_usage(
            correlation_id=uuid4(),
            operation=operation,
            model_id=model_id,
            usage=usage,
            cache_hit=True,
        )

    async def _cached_generation(
        self, request: GenerateRequest, cached: CachedResponse
    ) -> GenerateResponse:
        result = self._generation_response(cached.response)
        usage = self._generation_usage(request, cached.response)
        await self._record_cache_hit(
            operation="generate",
            model_id=request.model_id,
            usage=Usage(usage.input_tokens, usage.output_tokens, 0, Decimal("0")),
        )
        return result

    async def _cached_embedding(
        self, request: EmbedRequest, cached: CachedResponse
    ) -> EmbedResponse:
        result = self._embedding_response(cached.response)
        await self._record_cache_hit(
            operation="embed",
            model_id=request.model_id,
            usage=Usage(request.input_tokens, 0, 0, Decimal("0")),
        )
        return result

    async def _cached_parse(self, request: ParseRequest, cached: CachedResponse) -> ParsedDocument:
        await self._record_cache_hit(
            operation="parse",
            model_id=request.model_id,
            usage=Usage(0, 0, request.billable_pages, Decimal("0")),
        )
        return ParsedDocument(cached.response)

    async def _request(
        self,
        path: str,
        *,
        correlation_id: UUID,
        json_payload: dict[str, Any] | None = None,
        data: dict[str, str] | None = None,
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            try:
                async with self._semaphore:
                    response = await self._client.post(
                        path,
                        json=json_payload,
                        data=data,
                        files=files,
                        headers={"X-Correlation-ID": str(correlation_id)},
                    )
            except httpx.RequestError:
                if attempt >= self.max_retries:
                    raise
                await self._sleep(self._backoff(attempt, None))
                continue
            if 200 <= response.status_code < 300:
                return response
            if response.status_code != 429 and response.status_code < 500:
                raise ProviderHTTPError(response.status_code)
            if attempt >= self.max_retries:
                raise ProviderHTTPError(response.status_code)
            await self._sleep(self._backoff(attempt, response.headers.get("Retry-After")))
            LOGGER.warning(
                "retrying provider request",
                extra={
                    "correlation_id": str(correlation_id),
                    "status_code": response.status_code,
                    "attempt": attempt + 1,
                },
            )
        raise RuntimeError("unreachable retry state")

    @staticmethod
    def _decode_json(response: httpx.Response) -> dict[str, Any]:
        decoded = response.json()
        if not isinstance(decoded, dict):
            raise ValueError("provider JSON response must be an object")
        return decoded

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        if retry_after is not None:
            delay = _retry_after_seconds(retry_after)
            if delay is not None:
                return min(self.max_backoff_seconds, max(0.0, delay))
        upper_bound = min(self.max_backoff_seconds, 2.0**attempt)
        return min(self.max_backoff_seconds, max(0.0, self._jitter(upper_bound)))

    def _generation_usage(self, request: GenerateRequest, raw: dict[str, Any]) -> Usage:
        raw_usage = raw.get("usage", {})
        if not isinstance(raw_usage, dict):
            raw_usage = {}
        input_tokens = int(raw_usage.get("prompt_tokens", request.input_tokens))
        output_tokens = int(raw_usage.get("completion_tokens", request.max_output_tokens or 0))
        cost = self._price_book.estimate(
            PricingRequest(
                operation="generate",
                model_id=request.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
        return Usage(input_tokens, output_tokens, 0, cost)

    def _safe_generation_usage(
        self,
        request: GenerateRequest,
        raw: dict[str, Any] | None,
        fallback: Usage,
    ) -> Usage:
        if raw is None:
            return fallback
        try:
            return self._generation_usage(request, raw)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _generation_response(raw: dict[str, Any]) -> GenerateResponse:
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("provider response is missing choices")
        first = choices[0]
        if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
            raise ValueError("provider response is missing a message")
        content = first["message"].get("content")
        if not isinstance(content, str):
            raise ValueError("provider response message is missing content")
        return GenerateResponse(content, raw)

    @staticmethod
    def _embedding_response(raw: dict[str, Any]) -> EmbedResponse:
        data = raw.get("data")
        if not isinstance(data, list):
            raise ValueError("provider response is missing embedding data")
        vectors: list[tuple[float, ...]] = []
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise ValueError("provider response contains invalid embedding data")
            vectors.append(tuple(float(value) for value in item["embedding"]))
        return EmbedResponse(tuple(vectors), raw)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _retry_after_seconds(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0.0, (parsed - datetime.now(UTC)).total_seconds())


def _reject_reserved_params(params: dict[str, Any], reserved: set[str]) -> None:
    conflicts = sorted(params.keys() & reserved)
    if conflicts:
        raise ValueError(f"reserved provider parameter cannot be overridden: {conflicts[0]}")


__all__ = [
    "EmbedRequest",
    "EmbedResponse",
    "GenerateRequest",
    "GenerateResponse",
    "MemoryProviderStore",
    "ParseRequest",
    "ParsedDocument",
    "ProviderHTTPError",
    "SqlAlchemyProviderStore",
    "UpstageGateway",
]
