"""Tiny opt-in Upstage smoke tests; excluded unless both live gates are set."""

from __future__ import annotations

import hashlib
import os
from decimal import Decimal
from pathlib import Path

import pytest

if os.environ.get("RUN_LIVE_UPSTAGE_TESTS") != "1":
    pytest.skip(
        "set RUN_LIVE_UPSTAGE_TESTS=1 and select pytest -m live to enable provider smoke tests",
        allow_module_level=True,
    )
if not os.environ.get("UPSTAGE_API_KEY"):
    pytest.skip(
        "UPSTAGE_API_KEY is required for live provider smoke tests", allow_module_level=True
    )

from ragbench.cli import SMOKE_PDF
from ragbench.core.config import Settings
from ragbench.core.money import BudgetGuard, MemoryBudgetRepository
from ragbench.providers.base import EmbedRequest, GenerateRequest, ParseRequest
from ragbench.providers.upstage.client import MemoryProviderStore, UpstageGateway
from ragbench.providers.upstage.pricing import PriceBook, PricingRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_SOLAR_OUTPUT_TOKENS = 16
MAX_SOLAR_COST_USD = Decimal("0.000028")
MAX_PARSE_COST_USD = Decimal("0.011000")
MAX_EMBED_COST_USD = Decimal("0.000002")

pytestmark = pytest.mark.live


@pytest.fixture
def live_gateway() -> UpstageGateway:
    """A small-budget real gateway whose cache makes a repeat input cost-free."""
    settings = Settings()
    prices = PriceBook.from_yaml(PROJECT_ROOT / "configs" / "prices.yaml")
    prices.verify_paid_batch()
    return UpstageGateway(
        api_key=settings.upstage_api_key or "",
        base_url=settings.upstage_base_url,
        price_book=prices,
        budget_guard=BudgetGuard(MemoryBudgetRepository(), hard_limit=Decimal("0.020000")),
        store=MemoryProviderStore(),
        billing_cost_multiplier=settings.billing_cost_multiplier,
        max_concurrency=1,
        max_retries=0,
    )


@pytest.mark.asyncio
async def test_live_solar_smoke_is_bounded_and_cached(live_gateway: UpstageGateway) -> None:
    """One Korean generation smoke call and its identical no-charge cache repeat."""
    request = GenerateRequest(
        model_id="solar-pro4",
        prompt="대한민국의 수도는 어디인가요? 한 단어로 답하세요.",
        input_tokens=20,
        max_output_tokens=MAX_SOLAR_OUTPUT_TOKENS,
    )
    projected = PriceBook.from_yaml(PROJECT_ROOT / "configs" / "prices.yaml").estimate(
        PricingRequest("generate", request.model_id, 20, MAX_SOLAR_OUTPUT_TOKENS)
    )
    assert projected <= MAX_SOLAR_COST_USD

    try:
        first = await live_gateway.generate(request)
        second = await live_gateway.generate(request)
    finally:
        await live_gateway.aclose()

    assert first.content
    assert second.raw_response == first.raw_response


@pytest.mark.asyncio
async def test_live_parse_smoke_is_one_page_and_cached(live_gateway: UpstageGateway) -> None:
    """One locally generated one-page PDF parse smoke call and cache repeat."""
    request = ParseRequest(
        model_id="document-parse",
        document_sha256=hashlib.sha256(SMOKE_PDF).hexdigest(),
        content=SMOKE_PDF,
        billable_pages=1,
    )
    projected = PriceBook.from_yaml(PROJECT_ROOT / "configs" / "prices.yaml").estimate(
        PricingRequest("parse", request.model_id, billable_pages=1)
    )
    assert projected <= MAX_PARSE_COST_USD

    try:
        first = await live_gateway.parse(request)
        second = await live_gateway.parse(request)
    finally:
        await live_gateway.aclose()

    assert second.raw_response == first.raw_response


@pytest.mark.asyncio
async def test_live_embed_smoke_is_short_and_cached(live_gateway: UpstageGateway) -> None:
    """One short Korean embedding smoke call and cache repeat."""
    request = EmbedRequest(
        model_id="embedding-query",
        texts=("한국어 임베딩 스모크 테스트",),
        input_tokens=50,
    )
    projected = PriceBook.from_yaml(PROJECT_ROOT / "configs" / "prices.yaml").estimate(
        PricingRequest("embed", request.model_id, input_tokens=request.input_tokens)
    )
    assert projected <= MAX_EMBED_COST_USD

    try:
        first = await live_gateway.embed(request)
        second = await live_gateway.embed(request)
    finally:
        await live_gateway.aclose()

    assert second.raw_response == first.raw_response
