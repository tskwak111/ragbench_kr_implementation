"""CLI contracts for offline operator checks and guarded smoke paths."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from ragbench.cli import CheckResult, CommandServices, PreflightProbes, build_app
from ragbench.core.config import Settings
from ragbench.providers.base import (
    EmbedRequest,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    ParsedDocument,
    ParseRequest,
)
from ragbench.providers.upstage.pricing import PriceBook

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = CliRunner()


class FakeGateway:
    """Offline gateway double that records the one command operation it receives."""

    def __init__(self) -> None:
        self.requests: list[object] = []
        self.closed = False

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.requests.append(request)
        return GenerateResponse(content="확인", raw_response={"id": "fake-solar"})

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        self.requests.append(request)
        return ParsedDocument(raw_response={"id": "fake-parse"})

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        self.requests.append(request)
        return EmbedResponse(embeddings=((0.1, 0.2),), raw_response={"id": "fake-embed"})

    async def aclose(self) -> None:
        self.closed = True


def _check(name: str, ok: bool = True, detail: str = "ready") -> CheckResult:
    return CheckResult(name=name, ok=ok, detail=detail)


def _services(
    *,
    secret: str | None = "secret-value-that-must-not-appear",
    live_enabled: bool = False,
    gateway: FakeGateway | None = None,
) -> CommandServices:
    probes = PreflightProbes(
        docker=lambda: _check("docker", detail="Docker daemon available"),
        database=lambda: _check("database", detail="PostgreSQL ready"),
        migration=lambda: _check("migration", detail="at head"),
        cache=lambda: _check("cache", detail="writable"),
        budget=lambda: _check("budget", detail="134.500000 USD remaining"),
    )
    return CommandServices(
        settings=Settings(upstage_api_key=secret),
        price_book=PriceBook.from_yaml(PROJECT_ROOT / "configs" / "prices.yaml"),
        probes=probes,
        usage_status=lambda: {"settled_cost_usd": "0.500000", "remaining_usd": "134.500000"},
        gateway_factory=(lambda: gateway) if gateway is not None else None,
        live_enabled=lambda: live_enabled,
        now=lambda: datetime(2026, 8, 13, tzinfo=UTC),
    )


def test_preflight_json_reports_all_offline_checks_without_leaking_secret() -> None:
    """Catch omitting an operational gate or serializing the credential itself."""
    result = RUNNER.invoke(build_app(_services()), ["preflight", "--offline", "--json"])

    assert result.exit_code == 0
    assert "secret-value-that-must-not-appear" not in result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["offline"] is True
    assert payload["checks"] == [
        {"name": "docker", "ok": True, "detail": "Docker daemon available"},
        {"name": "database", "ok": True, "detail": "PostgreSQL ready"},
        {"name": "migration", "ok": True, "detail": "at head"},
        {"name": "cache", "ok": True, "detail": "writable"},
        {"name": "prices", "ok": True, "detail": "price snapshot is fresh"},
        {"name": "budget", "ok": True, "detail": "134.500000 USD remaining"},
        {"name": "secret", "ok": True, "detail": "configured"},
    ]


def test_preflight_reports_unavailable_docker_and_missing_secret_without_hiding_other_checks() -> (
    None
):
    """Catch treating an unavailable local prerequisite as a successful preflight."""
    services = _services(secret=None)
    services.probes.docker = lambda: _check("docker", False, "docker executable unavailable")

    result = RUNNER.invoke(build_app(services), ["preflight", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["checks"][0] == {
        "name": "docker",
        "ok": False,
        "detail": "docker executable unavailable",
    }
    assert payload["checks"][-1] == {"name": "secret", "ok": False, "detail": "missing"}


def test_smoke_dry_run_reports_bounded_projection_without_constructing_gateway() -> None:
    """Catch building a paid client or dispatching before the operator sees the cap."""
    result = RUNNER.invoke(build_app(_services()), ["smoke", "solar", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "command": "smoke solar",
        "dry_run": True,
        "model_id": "solar-pro4",
        "projected_max_usd": "0.000246",
        "requires": ["--execute", "--approve", "RUN_LIVE_UPSTAGE_TESTS=1", "UPSTAGE_API_KEY"],
    }


def test_smoke_execute_uses_fake_gateway_once_only_after_all_live_guards() -> None:
    """Catch duplicate smoke dispatch or bypassing the explicit operator/live gates."""
    gateway = FakeGateway()
    result = RUNNER.invoke(
        build_app(_services(live_enabled=True, gateway=gateway)),
        ["smoke", "embed", "--execute", "--approve", "--json"],
    )

    assert result.exit_code == 0
    assert len(gateway.requests) == 1
    assert isinstance(gateway.requests[0], EmbedRequest)
    assert gateway.closed is True
    payload = json.loads(result.output)
    assert payload["executed"] is True
    assert payload["operation"] == "embed"
    assert payload["provider_response_id"] == "fake-embed"
    assert payload["projected_max_usd"] == "0.000000"


def test_prices_verify_and_usage_status_are_machine_readable() -> None:
    """Catch leaving Task 3 price verification or budget visibility outside the CLI."""
    app = build_app(_services())

    prices = RUNNER.invoke(app, ["prices", "verify", "--json"])
    usage = RUNNER.invoke(app, ["usage", "status", "--json"])

    assert prices.exit_code == 0
    assert json.loads(prices.output)["verified_at"] == "2026-08-13T00:00:00Z"
    assert usage.exit_code == 0
    assert json.loads(usage.output) == {
        "remaining_usd": "134.500000",
        "settled_cost_usd": "0.500000",
    }
