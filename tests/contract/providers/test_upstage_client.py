"""Offline HTTP contracts for the guarded Upstage gateway."""

from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import respx

from ragbench.core.money import BudgetGuard, MemoryBudgetRepository
from ragbench.providers.upstage.client import (
    EmbedRequest,
    GenerateRequest,
    MemoryProviderStore,
    ParseRequest,
    ProviderHTTPError,
    UpstageGateway,
)
from ragbench.providers.upstage.pricing import PriceBook

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_URL = "https://api.upstage.ai/v1"


def _gateway(
    *,
    repository: MemoryBudgetRepository | None = None,
    store: MemoryProviderStore | None = None,
    sleep: object | None = None,
    max_retries: int = 3,
) -> UpstageGateway:
    return UpstageGateway(
        api_key="secret-api-key",
        base_url=BASE_URL,
        price_book=PriceBook.from_yaml(PROJECT_ROOT / "configs" / "prices.yaml"),
        budget_guard=BudgetGuard(
            repository or MemoryBudgetRepository(), hard_limit=Decimal("135.00")
        ),
        store=store or MemoryProviderStore(),
        max_retries=max_retries,
        sleep=sleep if callable(sleep) else None,
        jitter=lambda upper_bound: upper_bound,
    )


def _request() -> GenerateRequest:
    return GenerateRequest(
        model_id="solar-pro4",
        prompt="한국어 질문",
        context=("근거 문장",),
        provider_params={"temperature": 0},
        input_tokens=10,
        max_output_tokens=20,
    )


@pytest.mark.asyncio
@respx.mock
async def test_identical_second_call_is_cache_hit_without_http_traffic() -> None:
    """Catch bypassing deterministic response cache on an identical paid request."""
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "답변"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )
    )
    repository = MemoryBudgetRepository()
    store = MemoryProviderStore()
    gateway = _gateway(repository=repository, store=store)

    first = await gateway.generate(_request())
    second = await gateway.generate(_request())
    await gateway.aclose()

    assert first.content == "답변"
    assert second.content == "답변"
    assert route.call_count == 1
    assert [usage.cache_hit for usage in repository.usages] == [False, True]


@pytest.mark.asyncio
@respx.mock
async def test_different_output_bounds_do_not_alias_in_cache() -> None:
    """Catch omitting the response-affecting output bound from cache key material."""
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "답변"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )
    )
    gateway = _gateway()
    request = _request()

    await gateway.generate(request)
    await gateway.generate(
        GenerateRequest(
            model_id=request.model_id,
            prompt=request.prompt,
            context=request.context,
            provider_params=request.provider_params,
            input_tokens=request.input_tokens,
            max_output_tokens=10,
        )
    )
    await gateway.aclose()

    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_429_honors_retry_after_with_bounded_backoff() -> None:
    """Catch retrying immediately or exceeding the configured retry delay ceiling."""
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0.5"}),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "답변"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                },
            ),
        ]
    )
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    gateway = _gateway(sleep=record_sleep)
    await gateway.generate(_request())
    await gateway.aclose()

    assert route.call_count == 2
    assert delays == [0.5]
    assert all(delay <= gateway.max_backoff_seconds for delay in delays)


@pytest.mark.parametrize("status_code", [400, 401])
@pytest.mark.asyncio
@respx.mock
async def test_non_retryable_client_errors_are_attempted_once(status_code: int) -> None:
    """Catch retrying malformed or unauthorized requests."""
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(status_code, json={"error": "rejected"})
    )
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    gateway = _gateway(sleep=record_sleep)
    with pytest.raises(ProviderHTTPError) as error:
        await gateway.generate(_request())
    await gateway.aclose()

    assert error.value.status_code == status_code
    assert route.call_count == 1
    assert delays == []


@pytest.mark.asyncio
async def test_generate_requires_an_output_upper_bound() -> None:
    """Catch dispatching a generation request whose worst-case cost cannot be reserved."""
    gateway = _gateway()
    request = _request()

    with pytest.raises(ValueError, match="max_output_tokens"):
        await gateway.generate(
            GenerateRequest(
                model_id=request.model_id,
                prompt=request.prompt,
                context=request.context,
                provider_params=request.provider_params,
                input_tokens=request.input_tokens,
                max_output_tokens=None,
            )
        )
    await gateway.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_provider_params_cannot_override_reserved_generation_fields() -> None:
    """Catch dispatching a model or token limit different from the reserved request."""
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "답변"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )
    )
    gateway = _gateway()
    request = _request()

    with pytest.raises(ValueError, match="reserved provider parameter"):
        await gateway.generate(
            GenerateRequest(
                model_id=request.model_id,
                prompt=request.prompt,
                context=request.context,
                provider_params={"max_tokens": 100_000},
                input_tokens=request.input_tokens,
                max_output_tokens=request.max_output_tokens,
            )
        )
    await gateway.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_embed_second_call_uses_cache_and_records_zero_cost_usage() -> None:
    """Catch an embed path that skips cache lookup, reservation, or cache-hit usage."""
    route = respx.post(f"{BASE_URL}/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
    )
    repository = MemoryBudgetRepository()
    gateway = _gateway(repository=repository)
    request = EmbedRequest(model_id="embedding-query", texts=("문장",), input_tokens=2)

    first = await gateway.embed(request)
    second = await gateway.embed(request)
    await gateway.aclose()

    assert first.embeddings == ((0.1, 0.2),)
    assert second.embeddings == ((0.1, 0.2),)
    assert route.call_count == 1
    assert [usage.cache_hit for usage in repository.usages] == [False, True]


@pytest.mark.asyncio
@respx.mock
async def test_parse_cache_key_uses_document_hash_and_records_page_usage() -> None:
    """Catch a parse path that omits document identity or page usage evidence."""
    route = respx.post(f"{BASE_URL}/document-ai/document-parse").mock(
        return_value=httpx.Response(200, json={"content": "parsed"})
    )
    repository = MemoryBudgetRepository()
    gateway = _gateway(repository=repository)
    request = ParseRequest(
        model_id="document-parse",
        document_sha256="a" * 64,
        content=b"pdf bytes",
        billable_pages=2,
    )

    await gateway.parse(request)
    await gateway.parse(request)
    await gateway.aclose()

    assert route.call_count == 1
    assert [record.usage.billable_pages for record in repository.usages] == [2, 2]
    assert [record.cache_hit for record in repository.usages] == [False, True]
