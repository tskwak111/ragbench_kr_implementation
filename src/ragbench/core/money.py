"""Atomic project-budget reservations and settled provider usage."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragbench.db.models import ApiUsage, BudgetReservation


class BudgetExceededError(RuntimeError):
    """Raised before work when its upper-bound projection reaches the hard cap."""


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    billable_pages: int
    estimated_cost_usd: Decimal


@dataclass(frozen=True, slots=True)
class Reservation:
    id: UUID
    correlation_id: UUID
    reserved_cost_usd: Decimal
    status: str = "open"


@dataclass(frozen=True, slots=True)
class UsageRecord:
    correlation_id: UUID
    operation: str
    model_id: str
    usage: Usage
    cache_hit: bool


class BudgetRepository(Protocol):
    async def reserve_atomic(
        self, *, correlation_id: UUID, projected_cost: Decimal, hard_limit: Decimal
    ) -> Reservation: ...

    async def settle(
        self,
        reservation_id: UUID,
        *,
        operation: str,
        model_id: str,
        usage: Usage,
        cache_hit: bool,
        reconciliation_required: bool = False,
    ) -> None: ...

    async def release(self, reservation_id: UUID) -> None: ...

    async def record_cache_hit(
        self,
        *,
        correlation_id: UUID,
        operation: str,
        model_id: str,
        usage: Usage,
    ) -> None: ...


@dataclass(slots=True)
class _MemoryReservation:
    id: UUID
    correlation_id: UUID
    reserved_cost_usd: Decimal
    status: str = "open"
    settled_cost_usd: Decimal | None = None


class MemoryBudgetRepository:
    """Concurrency-safe budget store for offline tests and dry-run tooling."""

    def __init__(self, *, settled_cost: Decimal = Decimal("0")) -> None:
        self.settled_cost = settled_cost
        self.reservations: dict[UUID, _MemoryReservation] = {}
        self.usages: list[UsageRecord] = []
        self._lock = asyncio.Lock()

    async def reserve_atomic(
        self, *, correlation_id: UUID, projected_cost: Decimal, hard_limit: Decimal
    ) -> Reservation:
        async with self._lock:
            open_cost = sum(
                (
                    item.reserved_cost_usd
                    for item in self.reservations.values()
                    if item.status == "open"
                ),
                Decimal("0"),
            )
            if self.settled_cost + open_cost + projected_cost >= hard_limit:
                raise BudgetExceededError("projected provider cost reaches the hard budget limit")
            reservation = _MemoryReservation(uuid4(), correlation_id, projected_cost)
            self.reservations[reservation.id] = reservation
            return Reservation(
                reservation.id, reservation.correlation_id, reservation.reserved_cost_usd
            )

    async def settle(
        self,
        reservation_id: UUID,
        *,
        operation: str,
        model_id: str,
        usage: Usage,
        cache_hit: bool,
        reconciliation_required: bool = False,
    ) -> None:
        async with self._lock:
            reservation = self.reservations[reservation_id]
            if reservation.status != "open":
                raise RuntimeError("only open reservations can be settled")
            reservation.status = (
                "reconciliation_required"
                if reconciliation_required
                or usage.estimated_cost_usd > reservation.reserved_cost_usd
                else "settled"
            )
            reservation.settled_cost_usd = usage.estimated_cost_usd
            self.settled_cost += usage.estimated_cost_usd
            self.usages.append(
                UsageRecord(reservation.correlation_id, operation, model_id, usage, cache_hit)
            )

    async def release(self, reservation_id: UUID) -> None:
        async with self._lock:
            reservation = self.reservations[reservation_id]
            if reservation.status == "open":
                reservation.status = "released"
                reservation.settled_cost_usd = Decimal("0")

    async def record_cache_hit(
        self,
        *,
        correlation_id: UUID,
        operation: str,
        model_id: str,
        usage: Usage,
    ) -> None:
        _require_zero_cost_cache_hit(usage)
        async with self._lock:
            self.usages.append(UsageRecord(correlation_id, operation, model_id, usage, True))


class SqlAlchemyBudgetRepository:
    """PostgreSQL budget store serialized by a transaction-scoped advisory lock."""

    _LOCK_ID = 7_220_243_364_734_926_986

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def _acquire_budget_lock(self, session: AsyncSession) -> None:
        await session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_id)"),
            {"lock_id": self._LOCK_ID},
        )

    async def reserve_atomic(
        self, *, correlation_id: UUID, projected_cost: Decimal, hard_limit: Decimal
    ) -> Reservation:
        async with self._session_factory() as session, session.begin():
            await self._acquire_budget_lock(session)
            settled = await session.scalar(
                select(func.coalesce(func.sum(ApiUsage.estimated_cost_usd), Decimal("0")))
            )
            open_cost = await session.scalar(
                select(
                    func.coalesce(func.sum(BudgetReservation.reserved_cost_usd), Decimal("0"))
                ).where(BudgetReservation.status == "open")
            )
            committed = Decimal(settled or 0)
            reserved = Decimal(open_cost or 0)
            if committed + reserved + projected_cost >= hard_limit:
                raise BudgetExceededError("projected provider cost reaches the hard budget limit")
            reservation_id = uuid4()
            session.add(
                BudgetReservation(
                    id=reservation_id,
                    correlation_id=correlation_id,
                    reserved_cost_usd=projected_cost,
                    status="open",
                )
            )
        return Reservation(reservation_id, correlation_id, projected_cost)

    async def settle(
        self,
        reservation_id: UUID,
        *,
        operation: str,
        model_id: str,
        usage: Usage,
        cache_hit: bool,
        reconciliation_required: bool = False,
    ) -> None:
        async with self._session_factory() as session, session.begin():
            await self._acquire_budget_lock(session)
            reservation = await session.scalar(
                select(BudgetReservation)
                .where(BudgetReservation.id == reservation_id)
                .with_for_update()
            )
            if reservation is None or reservation.status != "open":
                raise RuntimeError("only open reservations can be settled")
            reservation.status = (
                "reconciliation_required"
                if reconciliation_required
                or usage.estimated_cost_usd > reservation.reserved_cost_usd
                else "settled"
            )
            reservation.settled_cost_usd = usage.estimated_cost_usd
            reservation.settled_at = datetime.now(UTC)
            session.add(
                ApiUsage(
                    correlation_id=reservation.correlation_id,
                    operation=operation,
                    provider_model_id=model_id,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    billable_pages=usage.billable_pages,
                    estimated_cost_usd=usage.estimated_cost_usd,
                    cache_hit=cache_hit,
                )
            )

    async def release(self, reservation_id: UUID) -> None:
        async with self._session_factory() as session, session.begin():
            await self._acquire_budget_lock(session)
            reservation = await session.scalar(
                select(BudgetReservation)
                .where(BudgetReservation.id == reservation_id)
                .with_for_update()
            )
            if reservation is not None and reservation.status == "open":
                reservation.status = "released"
                reservation.settled_cost_usd = Decimal("0")
                reservation.settled_at = datetime.now(UTC)

    async def record_cache_hit(
        self,
        *,
        correlation_id: UUID,
        operation: str,
        model_id: str,
        usage: Usage,
    ) -> None:
        _require_zero_cost_cache_hit(usage)
        async with self._session_factory() as session, session.begin():
            session.add(
                ApiUsage(
                    correlation_id=correlation_id,
                    operation=operation,
                    provider_model_id=model_id,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    billable_pages=usage.billable_pages,
                    estimated_cost_usd=usage.estimated_cost_usd,
                    cache_hit=True,
                )
            )


class BudgetGuard:
    """Enforce one immutable hard cap around all provider work."""

    def __init__(self, repository: BudgetRepository, *, hard_limit: Decimal) -> None:
        self._repository = repository
        self.hard_limit = hard_limit

    async def reserve(self, *, correlation_id: UUID, projected_cost: Decimal) -> Reservation:
        if projected_cost < 0:
            raise ValueError("projected_cost cannot be negative")
        return await self._repository.reserve_atomic(
            correlation_id=correlation_id,
            projected_cost=projected_cost,
            hard_limit=self.hard_limit,
        )

    async def settle(
        self,
        reservation_id: UUID,
        *,
        operation: str,
        model_id: str,
        usage: Usage,
        cache_hit: bool,
        reconciliation_required: bool = False,
    ) -> None:
        await self._repository.settle(
            reservation_id,
            operation=operation,
            model_id=model_id,
            usage=usage,
            cache_hit=cache_hit,
            reconciliation_required=reconciliation_required,
        )

    async def release(self, reservation_id: UUID) -> None:
        await self._repository.release(reservation_id)

    async def record_cache_hit(
        self,
        *,
        correlation_id: UUID,
        operation: str,
        model_id: str,
        usage: Usage,
    ) -> None:
        _require_zero_cost_cache_hit(usage)
        await self._repository.record_cache_hit(
            correlation_id=correlation_id,
            operation=operation,
            model_id=model_id,
            usage=usage,
        )


def _require_zero_cost_cache_hit(usage: Usage) -> None:
    if usage.estimated_cost_usd != Decimal("0"):
        raise ValueError("cache-hit accounting is restricted to zero-cost usage")
