"""Pricing and hard budget enforcement contracts."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from ragbench.core.money import (
    BudgetExceededError,
    BudgetGuard,
    MemoryBudgetRepository,
    SqlAlchemyBudgetRepository,
    Usage,
)
from ragbench.providers.upstage.pricing import (
    AmbiguousPromotionError,
    PriceBook,
    PricingRequest,
    StalePriceBookError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def price_book() -> PriceBook:
    return PriceBook.from_yaml(PROJECT_ROOT / "configs" / "prices.yaml")


def test_price_book_estimates_parse_chat_and_embed(price_book: PriceBook) -> None:
    """Catch applying token rates to pages or the wrong model generation rate."""
    assert price_book.estimate(
        PricingRequest(operation="parse", model_id="document-parse", billable_pages=3)
    ) == Decimal("0.030000")
    assert price_book.estimate(
        PricingRequest(
            operation="generate",
            model_id="solar-pro4",
            input_tokens=2_000_000,
            output_tokens=500_000,
        )
    ) == Decimal("1.200000")
    assert price_book.estimate(
        PricingRequest(
            operation="embed",
            model_id="embedding-query",
            input_tokens=1_000_000,
            requested_at=datetime(2026, 8, 24, tzinfo=UTC),
        )
    ) == Decimal("0.020000")


def test_embed_2_promotion_is_free_only_through_configured_deadline(
    price_book: PriceBook,
) -> None:
    """Catch extending a temporary promotion beyond its configured UTC deadline."""
    assert price_book.estimate(
        PricingRequest(
            operation="embed",
            model_id="embedding-query",
            input_tokens=5_000_000,
            requested_at=datetime(2026, 8, 23, tzinfo=UTC),
        )
    ) == Decimal("0.000000")


def test_price_preflight_rejects_stale_paid_snapshot(price_book: PriceBook) -> None:
    """Catch approving paid work from a price snapshot older than 24 hours."""
    now = price_book.verified_at + timedelta(hours=24, microseconds=1)

    with pytest.raises(StalePriceBookError):
        price_book.verify_paid_batch(now=now)


def test_price_preflight_rejects_promotion_without_a_promotional_rate() -> None:
    """Catch approving a promotion whose advertised free/paid status is ambiguous."""
    book = PriceBook(
        {
            "schema_version": "test-v1",
            "verified_at": "2026-08-13T00:00:00Z",
            "vat_excluded": True,
            "models": {
                "embedding-query": {
                    "embedding": {
                        "input_usd_per_million": "0.02",
                        "promotion": {"ends_at_exclusive": "2026-08-24T00:00:00Z"},
                    }
                }
            },
        }
    )

    with pytest.raises(AmbiguousPromotionError):
        book.verify_paid_batch(now=datetime(2026, 8, 13, 1, tzinfo=UTC))


@pytest.mark.asyncio
async def test_budget_guard_rejects_projection_that_reaches_hard_limit() -> None:
    """Catch accepting a reservation at the strict hard-cap boundary."""
    repository = MemoryBudgetRepository(settled_cost=Decimal("134.00"))
    guard = BudgetGuard(repository, hard_limit=Decimal("135.00"))

    with pytest.raises(BudgetExceededError):
        await guard.reserve(correlation_id=uuid4(), projected_cost=Decimal("1.00"))


@pytest.mark.asyncio
async def test_failed_reservation_is_released_for_later_work() -> None:
    """Catch leaked open reservations permanently consuming project budget."""
    repository = MemoryBudgetRepository(settled_cost=Decimal("100.00"))
    guard = BudgetGuard(repository, hard_limit=Decimal("135.00"))
    failed = await guard.reserve(correlation_id=uuid4(), projected_cost=Decimal("30.00"))

    await guard.release(failed.id)
    replacement = await guard.reserve(correlation_id=uuid4(), projected_cost=Decimal("30.00"))
    await guard.settle(
        replacement.id,
        operation="generate",
        model_id="solar-pro4",
        usage=Usage(
            input_tokens=1,
            output_tokens=1,
            billable_pages=0,
            estimated_cost_usd=Decimal("0.000002"),
        ),
        cache_hit=False,
    )

    assert repository.reservations[failed.id].status == "released"
    assert repository.reservations[replacement.id].status == "settled"


class _Transaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> None:
        return None


class _PersistedReservation:
    def __init__(self, reservation_id: UUID) -> None:
        self.id = reservation_id
        self.correlation_id = uuid4()
        self.reserved_cost_usd = Decimal("1")
        self.status = "open"
        self.settled_cost_usd: Decimal | None = None
        self.settled_at: datetime | None = None


class _RecordingSession:
    def __init__(self, reservation: _PersistedReservation) -> None:
        self.reservation = reservation
        self.events: list[str] = []

    async def __aenter__(self) -> "_RecordingSession":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    def begin(self) -> _Transaction:
        return _Transaction()

    async def execute(self, statement: Any, params: Any = None) -> None:
        del params
        rendered = str(statement)
        self.events.append("lock" if "pg_advisory_xact_lock" in rendered else "execute")

    async def scalar(self, statement: Any) -> _PersistedReservation:
        del statement
        self.events.append("read-reservation")
        return self.reservation

    def add(self, instance: Any) -> None:
        del instance
        self.events.append("add-usage")


class _RecordingFactory:
    def __init__(self, session: _RecordingSession) -> None:
        self.session = session

    def __call__(self) -> _RecordingSession:
        return self.session


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["settle", "release"])
async def test_sql_budget_transitions_acquire_global_lock_before_read(
    transition: str,
) -> None:
    """Catch settlement/release racing reservation accounting without the shared lock."""
    reservation = _PersistedReservation(uuid4())
    session = _RecordingSession(reservation)
    repository = SqlAlchemyBudgetRepository(_RecordingFactory(session))

    if transition == "settle":
        await repository.settle(
            reservation.id,
            operation="generate",
            model_id="solar-pro4",
            usage=Usage(1, 1, 0, Decimal("0.1")),
            cache_hit=False,
        )
    else:
        await repository.release(reservation.id)

    assert session.events[0:2] == ["lock", "read-reservation"]
