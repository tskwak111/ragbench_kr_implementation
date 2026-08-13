"""Offline HTTP contracts for the guarded Upstage gateway."""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from email.utils import format_datetime
from pathlib import Path
from typing import Any

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
    ProviderStore,
    UpstageGateway,
)
from ragbench.providers.upstage.pricing import PriceBook

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_URL = "https://api.upstage.ai/v1"


def _gateway(
    *,
    repository: MemoryBudgetRepository | None = None,
    store: ProviderStore | None = None,
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
async def test_generate_requires_a_positive_input_upper_bound() -> None:
    """Catch reserving generation with no conservative input-token upper bound."""
    gateway = _gateway()
    request = _request()

    with pytest.raises(ValueError, match="input_tokens"):
        await gateway.generate(
            GenerateRequest(
                model_id=request.model_id,
                prompt=request.prompt,
                context=request.context,
                provider_params=request.provider_params,
                input_tokens=0,
                max_output_tokens=request.max_output_tokens,
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
    route = respx.post(f"{BASE_URL}/document-digitization").mock(
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


@pytest.mark.asyncio
@respx.mock
async def test_parse_uses_official_document_digitization_multipart_contract() -> None:
    """Catch sending document bytes as invented JSON instead of multipart form data."""
    route = respx.post(f"{BASE_URL}/document-digitization").mock(
        return_value=httpx.Response(200, json={"content": "parsed"})
    )
    gateway = _gateway()

    await gateway.parse(
        ParseRequest(
            model_id="document-parse",
            document_sha256="b" * 64,
            content=b"%PDF-review-fixture",
            billable_pages=1,
            provider_params={"ocr": "force", "base64_encoding": "['table']"},
        )
    )
    await gateway.aclose()

    sent = route.calls[0].request
    body = sent.content.decode("utf-8")
    assert sent.url.path == "/v1/document-digitization"
    assert sent.headers["content-type"].startswith("multipart/form-data; boundary=")
    assert 'name="document"' in body
    assert "%PDF-review-fixture" in body
    assert 'name="model"' in body and "document-parse" in body
    assert 'name="ocr"' in body and "force" in body
    assert 'name="base64_encoding"' in body and "['table']" in body


@pytest.mark.parametrize(
    ("mode", "expected_cost"),
    [("standard", Decimal("0.010000")), ("enhanced", Decimal("0.030000"))],
)
@pytest.mark.asyncio
@respx.mock
async def test_parse_mode_is_consistent_in_form_and_pricing(
    mode: str, expected_cost: Decimal
) -> None:
    """Catch pricing one parse mode while dispatching another mode on the wire."""
    route = respx.post(f"{BASE_URL}/document-digitization").mock(
        return_value=httpx.Response(200, json={"content": "parsed"})
    )
    repository = MemoryBudgetRepository()
    gateway = _gateway(repository=repository)

    await gateway.parse(
        ParseRequest(
            model_id="document-parse",
            document_sha256=("c" if mode == "standard" else "d") * 64,
            content=b"%PDF-mode",
            billable_pages=1,
            mode=mode,
        )
    )
    await gateway.aclose()

    body = route.calls[0].request.content.decode("utf-8")
    assert 'name="mode"' in body and mode in body
    assert repository.usages[0].usage.estimated_cost_usd == expected_cost


@pytest.mark.asyncio
async def test_parse_provider_params_cannot_override_mode() -> None:
    """Catch generic provider parameters overriding the priced and cached parse mode."""
    gateway = _gateway()

    with pytest.raises(ValueError, match="reserved provider parameter.*mode"):
        await gateway.parse(
            ParseRequest(
                model_id="document-parse",
                document_sha256="e" * 64,
                content=b"%PDF-mode-override",
                billable_pages=1,
                mode="standard",
                provider_params={"mode": "enhanced"},
            )
        )
    await gateway.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_parse_modes_do_not_alias_in_cache() -> None:
    """Catch omitting the authoritative parse mode from deterministic cache material."""
    route = respx.post(f"{BASE_URL}/document-digitization").mock(
        return_value=httpx.Response(200, json={"content": "parsed"})
    )
    gateway = _gateway()
    common = {
        "model_id": "document-parse",
        "document_sha256": "f" * 64,
        "content": b"%PDF-same-document",
        "billable_pages": 1,
    }

    await gateway.parse(ParseRequest(**common, mode="standard"))
    await gateway.parse(ParseRequest(**common, mode="enhanced"))
    await gateway.aclose()

    assert route.call_count == 2


class _FailingStore(MemoryProviderStore):
    async def put(
        self,
        cache_key: str,
        *,
        operation: str,
        model_id: str,
        response: dict[str, Any],
    ) -> None:
        raise RuntimeError("cache unavailable")


@pytest.mark.asyncio
@respx.mock
async def test_cache_write_failure_does_not_release_paid_usage() -> None:
    """Catch treating cache persistence failure as if the paid request never happened."""
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "답변"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )
    )
    repository = MemoryBudgetRepository()
    gateway = _gateway(repository=repository, store=_FailingStore())

    with pytest.raises(RuntimeError, match="cache unavailable"):
        await gateway.generate(_request())
    await gateway.aclose()

    reservation = next(iter(repository.reservations.values()))
    assert reservation.status == "settled"
    assert len(repository.usages) == 1


@pytest.mark.asyncio
@respx.mock
async def test_malformed_paid_response_is_accounted_and_not_cached() -> None:
    """Catch poisoning cache and releasing budget after a malformed successful response."""
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json={"usage": {"prompt_tokens": "invalid"}})
    )
    repository = MemoryBudgetRepository()
    store = MemoryProviderStore()
    gateway = _gateway(repository=repository, store=store)

    with pytest.raises(ValueError, match="choices"):
        await gateway.generate(_request())
    await gateway.aclose()

    reservation = next(iter(repository.reservations.values()))
    assert reservation.status == "reconciliation_required"
    assert len(repository.usages) == 1
    assert store.entries == {}


@pytest.mark.asyncio
@respx.mock
async def test_actual_usage_over_reservation_is_accounted_for_reconciliation() -> None:
    """Catch releasing an already-paid response whose actual input exceeds its upper bound."""
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "답변"}}],
                "usage": {"prompt_tokens": 1_000, "completion_tokens": 4},
            },
        )
    )
    repository = MemoryBudgetRepository()
    gateway = _gateway(repository=repository)

    await gateway.generate(_request())
    await gateway.aclose()

    reservation = next(iter(repository.reservations.values()))
    assert reservation.status == "reconciliation_required"
    assert repository.usages[0].usage.input_tokens == 1_000
    assert repository.settled_cost == repository.usages[0].usage.estimated_cost_usd


@pytest.mark.asyncio
async def test_cache_hit_records_zero_cost_even_when_budget_is_near_cap() -> None:
    """Catch blocking a free cache hit by reserving its full paid upper bound first."""
    repository = MemoryBudgetRepository(settled_cost=Decimal("134.999999"))
    store = MemoryProviderStore()
    gateway = _gateway(repository=repository, store=store)
    request = _request()
    await store.put(
        gateway.cache_key_for_generate(request),
        operation="generate",
        model_id=request.model_id,
        response={
            "choices": [{"message": {"content": "cached"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        },
    )

    response = await gateway.generate(request)
    await gateway.aclose()

    assert response.content == "cached"
    assert repository.reservations == {}
    assert repository.usages[-1].cache_hit is True
    assert repository.usages[-1].usage.estimated_cost_usd == Decimal("0")


@pytest.mark.asyncio
@respx.mock
async def test_concurrent_identical_misses_coalesce_to_one_paid_call() -> None:
    """Catch a cache stampede that bills concurrent identical requests more than once."""
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
    gateway = _gateway(repository=repository)

    responses = await asyncio.gather(*(gateway.generate(_request()) for _ in range(8)))
    await gateway.aclose()

    assert [response.content for response in responses] == ["답변"] * 8
    assert route.call_count == 1
    assert len(repository.reservations) == 1
    assert [usage.cache_hit for usage in repository.usages].count(True) == 7


@pytest.mark.asyncio
@respx.mock
async def test_two_gateways_sharing_store_coalesce_identical_misses() -> None:
    """Catch gateway-instance-local locking that permits duplicate paid calls."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_response(request: httpx.Request) -> httpx.Response:
        del request
        started.set()
        await release.wait()
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "답변"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    route = respx.post(f"{BASE_URL}/chat/completions").mock(side_effect=delayed_response)
    repository = MemoryBudgetRepository()
    store = MemoryProviderStore()
    first = _gateway(repository=repository, store=store)
    second = _gateway(repository=repository, store=store)

    first_task = asyncio.create_task(first.generate(_request()))
    second_task = asyncio.create_task(second.generate(_request()))
    await started.wait()
    await asyncio.sleep(0)
    release.set()
    responses = await asyncio.gather(first_task, second_task)
    await first.aclose()
    await second.aclose()

    assert [response.content for response in responses] == ["답변", "답변"]
    assert route.call_count == 1
    assert store.singleflight_lock_count == 0


@pytest.mark.asyncio
@respx.mock
async def test_5xx_and_network_errors_retry_then_succeed() -> None:
    """Catch omitting retry for transient server and transport failures."""
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(503),
            httpx.ConnectError("connection reset"),
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

    assert route.call_count == 3
    assert delays == [1.0, 2.0]


@pytest.mark.asyncio
@respx.mock
async def test_http_date_retry_after_is_honored() -> None:
    """Catch ignoring the HTTP-date form of Retry-After."""
    retry_at = datetime.now(UTC) + timedelta(seconds=3)
    respx.post(f"{BASE_URL}/chat/completions").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": format_datetime(retry_at)}),
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

    assert len(delays) == 1
    assert 0 < delays[0] <= 3


@pytest.mark.asyncio
@respx.mock
async def test_redirect_is_terminal_and_not_a_success() -> None:
    """Catch accepting a 3xx response as successful provider JSON."""
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(302, headers={"Location": "/elsewhere"})
    )
    gateway = _gateway()

    with pytest.raises(ProviderHTTPError) as error:
        await gateway.generate(_request())
    await gateway.aclose()

    assert error.value.status_code == 302
    assert route.call_count == 1


def test_nonpositive_concurrency_is_rejected() -> None:
    """Catch constructing a permanently blocked gateway semaphore."""
    with pytest.raises(ValueError, match="max_concurrency"):
        _gateway_with_concurrency(0)


def _gateway_with_concurrency(max_concurrency: int) -> UpstageGateway:
    return UpstageGateway(
        api_key="secret-api-key",
        base_url=BASE_URL,
        price_book=PriceBook.from_yaml(PROJECT_ROOT / "configs" / "prices.yaml"),
        budget_guard=BudgetGuard(MemoryBudgetRepository(), hard_limit=Decimal("135.00")),
        store=MemoryProviderStore(),
        max_concurrency=max_concurrency,
    )
