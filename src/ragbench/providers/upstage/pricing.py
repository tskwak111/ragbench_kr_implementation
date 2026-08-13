"""Versioned Upstage pricing snapshot and upper-bound estimates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any, Self

import yaml

MONEY_QUANTUM = Decimal("0.000001")
ONE_MILLION = Decimal(1_000_000)


class PriceBookError(ValueError):
    """Base exception for invalid or unusable price snapshots."""


class StalePriceBookError(PriceBookError):
    """Raised when a paid batch uses a snapshot older than 24 hours."""


class AmbiguousPromotionError(PriceBookError):
    """Raised when temporary pricing lacks a decisive UTC boundary or paid rate."""


@dataclass(frozen=True, slots=True)
class PricingRequest:
    operation: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    billable_pages: int = 0
    mode: str = "standard"
    requested_at: datetime | None = None


class PriceBook:
    """Loaded configuration snapshot; prices exclude 10% VAT."""

    def __init__(self, snapshot: dict[str, Any]) -> None:
        self._snapshot = snapshot
        self.schema_version = str(snapshot["schema_version"])
        self.verified_at = _parse_utc(str(snapshot["verified_at"]))
        self.vat_excluded = bool(snapshot["vat_excluded"])

    @classmethod
    def from_yaml(cls, path: Path | str) -> Self:
        with Path(path).open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if not isinstance(loaded, dict):
            raise PriceBookError("price snapshot must be a YAML mapping")
        return cls(loaded)

    def estimate(self, request: PricingRequest) -> Decimal:
        if min(request.input_tokens, request.output_tokens, request.billable_pages) < 0:
            raise PriceBookError("usage counts cannot be negative")
        models = self._models()
        try:
            model = models[request.model_id]
        except KeyError as error:
            raise PriceBookError(f"unknown priced model: {request.model_id}") from error
        if request.operation == "parse":
            modes = _mapping(model, "modes")
            rate = _decimal(_mapping(modes, request.mode)["usd_per_page"])
            return _money(rate * request.billable_pages)
        if request.operation == "generate":
            rates = _mapping(model, "generation")
            total = (
                _decimal(rates["input_usd_per_million"]) * request.input_tokens
                + _decimal(rates["output_usd_per_million"]) * request.output_tokens
            ) / ONE_MILLION
            return _money(total)
        if request.operation == "embed":
            rates = _mapping(model, "embedding")
            requested_at = request.requested_at or datetime.now(UTC)
            promotion = _mapping(rates, "promotion")
            if requested_at < _parse_utc(str(promotion["ends_at_exclusive"])):
                rate = _decimal(promotion["input_usd_per_million"])
            else:
                rate = _decimal(rates["input_usd_per_million"])
            return _money(rate * request.input_tokens / ONE_MILLION)
        raise PriceBookError(f"unsupported priced operation: {request.operation}")

    def verify_paid_batch(self, *, now: datetime | None = None) -> dict[str, Any]:
        checked_at = now or datetime.now(UTC)
        if checked_at - self.verified_at > timedelta(hours=24):
            raise StalePriceBookError("price snapshot is older than 24 hours")
        for model in self._models().values():
            if "embedding" not in model:
                continue
            rates = _mapping(model, "embedding")
            promotion = _mapping(rates, "promotion")
            if (
                promotion.get("ends_at_exclusive") is None
                or promotion.get("input_usd_per_million") is None
                or rates.get("input_usd_per_million") is None
            ):
                raise AmbiguousPromotionError("embedding promotion status is ambiguous")
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return dict(self._snapshot)

    def _models(self) -> dict[str, dict[str, Any]]:
        raw = _mapping(self._snapshot, "models")
        return {str(key): _mapping(raw, str(key)) for key in raw}


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise PriceBookError(f"price snapshot field {key!r} must be a mapping")
    return item


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


def _money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANTUM, rounding=ROUND_CEILING)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise PriceBookError("price timestamps must include a UTC offset")
    return parsed.astimezone(UTC)
