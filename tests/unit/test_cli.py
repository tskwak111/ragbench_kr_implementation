"""CLI contracts for offline operator checks and guarded smoke paths."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import ragbench.cli as cli
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
from ragbench.rag.citations import GenerationSchemaError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = CliRunner()


class FakeGateway:
    """Offline gateway double that records the one command operation it receives."""

    def __init__(self) -> None:
        self.requests: list[object] = []
        self.closed = False

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.requests.append(request)
        return GenerateResponse(
            content="확인",
            raw_response={"id": "fake-solar", "correlation_id": "correlation-solar"},
            correlation_id="correlation-solar",
        )

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        self.requests.append(request)
        return ParsedDocument(
            raw_response={"id": "fake-parse", "correlation_id": "correlation-parse"},
            correlation_id="correlation-parse",
        )

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        self.requests.append(request)
        return EmbedResponse(
            embeddings=((0.1, 0.2),),
            raw_response={"id": "fake-embed", "correlation_id": "correlation-embed"},
            correlation_id="correlation-embed",
        )

    async def aclose(self) -> None:
        self.closed = True


def _check(name: str, ok: bool = True, detail: str = "ready") -> CheckResult:
    return CheckResult(name=name, ok=ok, detail=detail)


def _services(
    *,
    secret: str | None = "secret-value-that-must-not-appear",
    live_enabled: bool = False,
    gateway: FakeGateway | None = None,
    query_runner: Any | None = None,
) -> CommandServices:
    probes = PreflightProbes(
        docker=lambda: _check("docker", detail="Docker daemon available"),
        database=lambda: _check("database", detail="PostgreSQL ready"),
        migration=lambda: _check("migration", detail="at head"),
        cache=lambda: _check("cache", detail="writable"),
        budget=lambda: _check("budget", detail="134.500000 USD remaining"),
    )
    return CommandServices(
        settings=Settings(upstage_api_key=secret, run_live_upstage_tests=False),
        price_book=PriceBook.from_yaml(PROJECT_ROOT / "configs" / "prices.yaml"),
        probes=probes,
        usage_status=lambda: {"settled_cost_usd": "0.500000", "remaining_usd": "134.500000"},
        gateway_factory=(lambda: gateway) if gateway is not None else None,
        live_enabled=lambda: live_enabled,
        now=lambda: datetime(2026, 8, 13, tzinfo=UTC),
        query_runner=query_runner,
    )


def test_query_cli_returns_injected_grounded_service_contract_without_live_wiring() -> None:
    """Catch omitting a machine-readable query path or constructing provider HTTP in the CLI."""
    calls: list[str] = []

    async def answer(question: str) -> dict[str, object]:
        calls.append(question)
        return {
            "question": question,
            "answer": "100억 원",
            "evidence": [{"chunk_id": "chunk-1"}],
            "citations": [{"citation_id": "C1", "chunk_id": "chunk-1"}],
            "latency_ms": 3,
            "usage": {"input_tokens": 21, "output_tokens": 4},
            "experiment_id": "exp-1",
            "config_id": "cfg-1",
            "cached": True,
            "model_id": "solar-pro4",
            "correlation_id": "corr-1",
        }

    result = RUNNER.invoke(
        build_app(_services(secret=None, query_runner=answer)),
        ["query", "매출은?", "--json"],
    )

    assert result.exit_code == 0
    assert calls == ["매출은?"]
    payload = json.loads(result.output)
    assert payload["answer"] == "100억 원"
    assert payload["citations"] == [{"citation_id": "C1", "chunk_id": "chunk-1"}]
    assert payload["cached"] is True


def test_query_cli_fails_closed_when_no_grounded_service_is_configured() -> None:
    """Catch accidental direct-provider fallback when application RAG wiring is absent."""
    result = RUNNER.invoke(build_app(_services()), ["query", "질문", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.output) == {
        "ok": False,
        "error": "grounded query service unavailable",
    }


def test_query_cli_preserves_generation_schema_error_code_without_raw_model_output() -> None:
    """Catch erasing the stable failure classification or leaking malformed provider content."""

    async def broken_answer(question: str) -> dict[str, object]:
        raise GenerationSchemaError(f"malformed provider output for {question}")

    result = RUNNER.invoke(
        build_app(_services(query_runner=broken_answer)),
        ["query", "private malformed body", "--json"],
    )

    assert result.exit_code == 1
    assert json.loads(result.output) == {
        "ok": False,
        "error": "grounded query failed",
        "error_code": "GENERATION_SCHEMA_ERROR",
    }
    assert "private malformed body" not in result.output


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
        "projected_max_usd": "0.000271",
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
    assert payload["model_id"] == "embedding-query"
    assert payload["provider_response_id"] == "fake-embed"
    assert payload["correlation_id"] == "correlation-embed"
    assert payload["usage_correlation_id"] == "correlation-embed"
    assert payload["executed_at_utc"] == "2026-08-13T00:00:00Z"
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


def test_smoke_rechecks_stale_prices_before_gateway_construction() -> None:
    """Catch dispatching a paid request after the price snapshot ages since dry-run."""
    gateway_calls = 0
    snapshot = deepcopy(PriceBook.from_yaml(PROJECT_ROOT / "configs" / "prices.yaml").snapshot())
    snapshot["verified_at"] = "2026-08-10T00:00:00Z"
    services = _services(live_enabled=True)
    services.price_book = PriceBook(snapshot)

    def never_construct() -> FakeGateway:
        nonlocal gateway_calls
        gateway_calls += 1
        return FakeGateway()

    services.gateway_factory = never_construct
    result = RUNNER.invoke(
        build_app(services), ["smoke", "solar", "--execute", "--approve", "--json"]
    )

    assert result.exit_code == 1
    assert gateway_calls == 0
    payload = json.loads(result.output)
    assert payload["executed"] is False
    assert payload["blockers"] == ["price snapshot is older than 24 hours"]


def test_usage_status_operational_failure_is_valid_redacted_json() -> None:
    """Catch a database failure producing a traceback instead of an operator-safe result."""
    services = _services()

    def unavailable() -> dict[str, str]:
        raise OSError("secret-value-that-must-not-appear")

    services.usage_status = unavailable
    result = RUNNER.invoke(build_app(services), ["usage", "status", "--json"])

    assert result.exit_code == 1
    assert "secret-value-that-must-not-appear" not in result.output
    assert json.loads(result.output) == {"error": "usage status unavailable", "ok": False}


def test_main_serializes_settings_construction_failure_when_json_requested(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catch invalid live environment settings failing before the CLI can emit JSON."""
    monkeypatch.setattr(
        cli, "build_app", lambda: (_ for _ in ()).throw(ValueError("api key: secret"))
    )
    monkeypatch.setattr(sys, "argv", ["ragbench", "preflight", "--json"])

    with pytest.raises(SystemExit) as exited:
        cli.main()

    assert exited.value.code == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "configuration unavailable",
        "ok": False,
    }


def test_main_redacts_missing_key_live_settings_error_as_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Catch RUN_LIVE_UPSTAGE_TESTS validation leaking details before a JSON command starts."""
    monkeypatch.setenv("RUN_LIVE_UPSTAGE_TESTS", "1")
    monkeypatch.delenv("UPSTAGE_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["ragbench", "preflight", "--json"])

    with pytest.raises(SystemExit) as exited:
        cli.main()

    assert exited.value.code == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "configuration unavailable",
        "ok": False,
    }


@pytest.mark.asyncio
async def test_managed_gateway_disposes_both_engines_when_gateway_close_fails() -> None:
    """Catch a failed HTTP cleanup preventing normal or lock-engine disposal."""
    events: list[str] = []

    class ClosingGateway:
        async def aclose(self) -> None:
            events.append("gateway")
            raise RuntimeError("gateway close failed")

    class Engine:
        async def dispose(self) -> None:
            events.append("engine")

    class Factory:
        def __init__(self) -> None:
            self.kw: dict[str, Any] = {"bind": Engine()}

    with pytest.raises(RuntimeError, match="gateway close failed"):
        await cli._ManagedGateway(ClosingGateway(), Factory(), Factory()).aclose()  # type: ignore[arg-type]

    assert events == ["gateway", "engine", "engine"]


def test_smoke_pdf_is_a_one_page_pdf_with_cross_reference_table() -> None:
    """Catch using an invalid PDF byte fragment for a billed parse smoke operation."""
    assert cli.SMOKE_PDF.startswith(b"%PDF-1.4\n")
    assert b"xref\n" in cli.SMOKE_PDF
    assert b"/Count 1" in cli.SMOKE_PDF
    assert cli.SMOKE_PDF.rstrip().endswith(b"%%EOF")


def test_smoke_gateway_forces_one_http_attempt_for_retryable_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch smoke wiring inheriting normal retries for 429 or 5xx provider responses."""
    retries: list[int] = []
    events: list[str] = []

    class Engine:
        async def dispose(self) -> None:
            events.append("engine")

    class Factory:
        def __init__(self) -> None:
            self.kw: dict[str, Any] = {"bind": Engine()}

    class Gateway:
        def __init__(self, **kwargs: Any) -> None:
            retries.append(kwargs["max_retries"])

        async def aclose(self) -> None:
            events.append("gateway")

    monkeypatch.setattr(cli, "create_session_factory", lambda settings: Factory())
    monkeypatch.setattr(cli, "create_lock_session_factory", lambda settings: Factory())
    monkeypatch.setattr(cli, "SqlAlchemyProviderStore", lambda **kwargs: object())
    monkeypatch.setattr(cli, "UpstageGateway", Gateway)

    managed = cli._build_gateway(Settings(upstage_api_key="redacted", max_retries=5))
    assert retries == [0]
    import asyncio

    asyncio.run(managed.aclose())
    assert events == ["gateway", "engine", "engine"]


def test_gateway_construction_failure_disposes_created_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch a lock-factory construction error leaking the already-created normal engine."""
    events: list[str] = []

    class Engine:
        async def dispose(self) -> None:
            events.append("engine")

    class Factory:
        kw: dict[str, Any] = {"bind": Engine()}

    monkeypatch.setattr(cli, "create_session_factory", lambda settings: Factory())
    monkeypatch.setattr(
        cli,
        "create_lock_session_factory",
        lambda settings: (_ for _ in ()).throw(RuntimeError("lock factory failed")),
    )

    with pytest.raises(RuntimeError, match="lock factory failed"):
        cli._build_gateway(Settings(upstage_api_key="redacted"))

    assert events == ["engine"]
