"""Operator-facing preflight and explicitly guarded provider smoke commands."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Protocol

import typer
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragbench.benchmark.splits import (
    GoldAccessError,
    GoldMetadata,
    ImmutableSnapshotError,
    authorize_gold_access,
    load_sealed_gold,
    public_gold_metadata,
)
from ragbench.core.config import Settings
from ragbench.core.money import BudgetGuard, SqlAlchemyBudgetRepository
from ragbench.db.models import ApiUsage
from ragbench.db.session import create_lock_session_factory, create_session_factory
from ragbench.providers.base import EmbedRequest, GenerateRequest, ParseRequest, ProviderGateway
from ragbench.providers.upstage.client import SqlAlchemyProviderStore, UpstageGateway
from ragbench.providers.upstage.pricing import PriceBook, PriceBookError, PricingRequest
from ragbench.rag.citations import CitationValidationError, GenerationSchemaError
from ragbench.rag.service import RagAnswer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICE_PATH = PROJECT_ROOT / "configs" / "prices.yaml"
SMOKE_MAX_OUTPUT_TOKENS = 200
SMOKE_SOLAR_INPUT_TOKENS = 20
SMOKE_EMBED_INPUT_TOKENS = 50


def build_smoke_pdf() -> bytes:
    """Create a standards-valid, locally generated one-page PDF for bounded parse smoke tests."""
    stream = b"BT /F1 12 Tf 72 720 Td (RAGBench provider smoke) Tj ET\n"
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    )
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode())
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    output.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
    trailer = f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    output.extend(trailer.encode())
    return bytes(output)


SMOKE_PDF = build_smoke_pdf()


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One independently observable operational prerequisite."""

    name: str
    ok: bool
    detail: str


@dataclass(slots=True)
class PreflightProbes:
    """Injectable local probes; none contacts a provider or reads a secret value."""

    docker: Callable[[], CheckResult]
    database: Callable[[], CheckResult]
    migration: Callable[[], CheckResult]
    cache: Callable[[], CheckResult]
    budget: Callable[[], CheckResult]


class GatewayFactory(Protocol):
    def __call__(self) -> ProviderGateway: ...


class QueryRunner(Protocol):
    async def __call__(self, question: str) -> RagAnswer | dict[str, Any]: ...


@dataclass(slots=True)
class CommandServices:
    """Dependencies for a CLI instance, replaceable by deterministic offline fakes."""

    settings: Settings
    price_book: PriceBook
    probes: PreflightProbes
    usage_status: Callable[[], dict[str, str]]
    gateway_factory: GatewayFactory | None
    live_enabled: Callable[[], bool]
    now: Callable[[], datetime]
    query_runner: QueryRunner | None = None


def default_services() -> CommandServices:
    """Build local-only command services without constructing a provider client."""
    settings = Settings()
    return CommandServices(
        settings=settings,
        price_book=PriceBook.from_yaml(PRICE_PATH),
        probes=PreflightProbes(
            docker=_probe_docker,
            database=lambda: _probe_database(settings),
            migration=lambda: _probe_migration(settings),
            cache=lambda: _probe_cache(settings.cache_dir),
            budget=lambda: _probe_budget(settings),
        ),
        usage_status=lambda: _usage_status(settings),
        gateway_factory=lambda: _build_gateway(settings),
        live_enabled=lambda: os.environ.get("RUN_LIVE_UPSTAGE_TESTS") == "1",
        now=lambda: datetime.now(UTC),
    )


def build_app(services: CommandServices | None = None) -> typer.Typer:
    """Create the installable command application with injectable dependencies."""
    active = services or default_services()
    app = typer.Typer(no_args_is_help=True, add_completion=False)
    smoke = typer.Typer(no_args_is_help=True)
    prices = typer.Typer(no_args_is_help=True)
    usage = typer.Typer(no_args_is_help=True)
    gold = typer.Typer(no_args_is_help=True)

    @app.command()
    def query(
        question: str = typer.Argument(..., help="Question to answer from configured evidence."),
        as_json: bool = typer.Option(False, "--json", help="Emit one JSON object."),
    ) -> None:
        """Run the configured grounded RAG service; never construct a provider fallback."""
        if active.query_runner is None:
            _emit({"ok": False, "error": "grounded query service unavailable"}, as_json)
            raise typer.Exit(code=1)
        try:
            result = asyncio.run(active.query_runner(question))
        except (GenerationSchemaError, CitationValidationError) as error:
            _emit(
                {"ok": False, "error": "grounded query failed", "error_code": error.code},
                as_json,
            )
            raise typer.Exit(code=1) from None
        except Exception:
            _emit({"ok": False, "error": "grounded query failed"}, as_json)
            raise typer.Exit(code=1) from None
        _emit(asdict(result) if isinstance(result, RagAnswer) else result, as_json)

    @app.command()
    def preflight(
        offline: bool = typer.Option(False, help="Do not permit live-provider execution."),
        as_json: bool = typer.Option(False, "--json", help="Emit one JSON object."),
    ) -> None:
        """Report local readiness without constructing an Upstage client."""
        checks = [
            active.probes.docker(),
            active.probes.database(),
            active.probes.migration(),
            active.probes.cache(),
            _price_check(active.price_book, active.now),
            active.probes.budget(),
            CheckResult(
                "secret",
                bool(active.settings.upstage_api_key),
                "configured" if active.settings.upstage_api_key else "missing",
            ),
        ]
        _emit(
            {"ok": all(check.ok for check in checks), "offline": offline, "checks": checks}, as_json
        )
        if not all(check.ok for check in checks):
            raise typer.Exit(code=1)

    @prices.command("verify")
    def prices_verify(
        as_json: bool = typer.Option(False, "--json", help="Emit one JSON object."),
    ) -> None:
        """Verify that the local paid-rate snapshot can authorize a batch."""
        try:
            snapshot = active.price_book.verify_paid_batch(now=active.now())
        except PriceBookError as error:
            _emit({"ok": False, "error": str(error)}, as_json)
            raise typer.Exit(code=1) from error
        _emit(snapshot, as_json)

    @usage.command("status")
    def usage_status(
        as_json: bool = typer.Option(False, "--json", help="Emit one JSON object."),
    ) -> None:
        """Show settled provider cost and remaining hard-budget capacity."""
        try:
            _emit(active.usage_status(), as_json)
        except Exception:
            _emit({"ok": False, "error": "usage status unavailable"}, as_json)
            raise typer.Exit(code=1) from None

    @gold.command("verify")
    def gold_verify(
        sealed_path: Annotated[Path, typer.Argument(help="Restricted sealed gold JSONL path.")],
        metadata_path: Annotated[Path, typer.Argument(help="Public gold metadata JSON path.")],
        execute: bool = typer.Option(
            False, help="Explicitly authorize a sealed integrity check; never previews content."
        ),
        as_json: bool = typer.Option(False, "--json", help="Emit one JSON object."),
    ) -> None:
        """Verify a sealed gold snapshot while emitting public metadata only."""
        try:
            authorization = authorize_gold_access(command="evaluate-gold", explicit=execute)
            metadata = GoldMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
            load_sealed_gold(
                sealed_path,
                metadata=metadata,
                authorization=authorization,
            )
        except GoldAccessError:
            _emit({"ok": False, "error": "gold access denied"}, as_json)
            raise typer.Exit(code=1) from None
        except (ImmutableSnapshotError, OSError, ValueError):
            _emit({"ok": False, "error": "sealed gold verification failed"}, as_json)
            raise typer.Exit(code=1) from None
        _emit({"ok": True, **public_gold_metadata(metadata)}, as_json)

    def add_smoke_command(name: str, operation: str) -> None:
        @smoke.command(name)
        def run_smoke(
            execute: bool = typer.Option(False, help="Allow exactly one provider operation."),
            approve: bool = typer.Option(
                False, help="Confirm operator approval for this operation."
            ),
            as_json: bool = typer.Option(False, "--json", help="Emit one JSON object."),
        ) -> None:
            """Preview a bounded request, then require both explicit live gates."""
            request, projected = _smoke_request(operation, active)
            preview = {
                "command": f"smoke {name}",
                "dry_run": not execute,
                "model_id": request.model_id,
                "projected_max_usd": _money_text(projected),
                "requires": [
                    "--execute",
                    "--approve",
                    "RUN_LIVE_UPSTAGE_TESTS=1",
                    "UPSTAGE_API_KEY",
                ],
            }
            if not execute:
                _emit(preview, as_json)
                return
            reasons = _execution_blockers(active, approve)
            try:
                active.price_book.verify_paid_batch(now=active.now())
            except PriceBookError as error:
                reasons.append(str(error))
            if reasons:
                _emit({**preview, "executed": False, "blockers": reasons}, as_json)
                raise typer.Exit(code=1)
            assert active.gateway_factory is not None
            try:
                gateway = active.gateway_factory()
                result = asyncio.run(_execute_once(gateway, operation, request, active.now))
            except Exception:
                _emit(
                    {**preview, "executed": False, "error": "provider smoke unavailable"}, as_json
                )
                raise typer.Exit(code=1) from None
            _emit(
                {
                    **preview,
                    "dry_run": False,
                    "executed": True,
                    "operation": operation,
                    **result,
                },
                as_json,
            )

    add_smoke_command("solar", "generate")
    add_smoke_command("parse", "parse")
    add_smoke_command("embed", "embed")
    app.add_typer(prices, name="prices")
    app.add_typer(smoke, name="smoke")
    app.add_typer(usage, name="usage")
    app.add_typer(gold, name="gold")
    return app


def _price_check(price_book: PriceBook, now: Callable[[], datetime]) -> CheckResult:
    try:
        price_book.verify_paid_batch(now=now())
    except PriceBookError as error:
        return CheckResult("prices", False, str(error))
    return CheckResult("prices", True, "price snapshot is fresh")


def _smoke_request(
    operation: str, services: CommandServices
) -> tuple[GenerateRequest | ParseRequest | EmbedRequest, Decimal]:
    request: GenerateRequest | ParseRequest | EmbedRequest
    if operation == "generate":
        request = GenerateRequest(
            model_id=services.settings.upstage_solar_pro4_model_id,
            prompt="대한민국의 수도는 어디인가요? 한 단어로 답하세요.",
            input_tokens=SMOKE_SOLAR_INPUT_TOKENS,
            max_output_tokens=SMOKE_MAX_OUTPUT_TOKENS,
        )
        pricing = PricingRequest(
            operation="generate",
            model_id=request.model_id,
            input_tokens=request.input_tokens,
            output_tokens=SMOKE_MAX_OUTPUT_TOKENS,
        )
    elif operation == "parse":
        request = ParseRequest(
            model_id=services.settings.upstage_document_parse_model_id,
            document_sha256=hashlib.sha256(SMOKE_PDF).hexdigest(),
            content=SMOKE_PDF,
            billable_pages=1,
        )
        pricing = PricingRequest(operation="parse", model_id=request.model_id, billable_pages=1)
    elif operation == "embed":
        request = EmbedRequest(
            model_id=services.settings.upstage_embedding_model_id,
            texts=("한국어 임베딩 스모크 테스트",),
            input_tokens=SMOKE_EMBED_INPUT_TOKENS,
        )
        pricing = PricingRequest(
            operation="embed",
            model_id=request.model_id,
            input_tokens=request.input_tokens,
            requested_at=services.now(),
        )
    else:  # pragma: no cover - private fixed command registration
        raise ValueError(f"unknown smoke operation: {operation}")
    net = services.price_book.estimate(pricing)
    return request, (net * services.settings.billing_cost_multiplier).quantize(Decimal("0.000001"))


def _execution_blockers(services: CommandServices, approve: bool) -> list[str]:
    blockers: list[str] = []
    if not approve:
        blockers.append("--approve is required")
    if not services.live_enabled():
        blockers.append("RUN_LIVE_UPSTAGE_TESTS=1 is required")
    if not services.settings.upstage_api_key:
        blockers.append("UPSTAGE_API_KEY is required")
    if services.gateway_factory is None:
        blockers.append("provider gateway is unavailable")
    return blockers


async def _execute_once(
    gateway: ProviderGateway,
    operation: str,
    request: Any,
    now: Callable[[], datetime],
) -> dict[str, Any]:
    """Perform one gateway request and close owned HTTP resources on every outcome."""
    started = perf_counter()
    response: Any
    try:
        if operation == "generate":
            response = await gateway.generate(request)
        elif operation == "parse":
            response = await gateway.parse(request)
        else:
            response = await gateway.embed(request)
    finally:
        closer = getattr(gateway, "aclose", None)
        if closer is not None:
            await closer()
    raw = response.raw_response
    return {
        "provider_response_id": raw.get("id"),
        "correlation_id": response.correlation_id,
        "usage_correlation_id": response.correlation_id,
        "executed_at_utc": now().isoformat().replace("+00:00", "Z"),
        "latency_ms": round((perf_counter() - started) * 1000),
    }


def _emit(payload: Any, as_json: bool) -> None:
    normalized = _normalize(payload)
    if as_json:
        typer.echo(json.dumps(normalized, ensure_ascii=False, sort_keys=True))
        return
    if isinstance(normalized, dict) and "checks" in normalized:
        for check in normalized["checks"]:
            typer.echo(f"{'OK' if check['ok'] else 'FAIL'} {check['name']}: {check['detail']}")
        return
    typer.echo(json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True))


def _normalize(value: Any) -> Any:
    if isinstance(value, CheckResult):
        return asdict(value)
    if isinstance(value, Decimal):
        return _money_text(value)
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _money_text(value: Decimal) -> str:
    return f"{value:.6f}"


def _probe_docker() -> CheckResult:
    if shutil.which("docker") is None:
        return CheckResult("docker", False, "docker executable unavailable")
    return CheckResult("docker", True, "docker executable available")


def _probe_database(settings: Settings) -> CheckResult:
    try:
        asyncio.run(_database_ready(settings))
    except Exception as error:  # local diagnostics must finish even when PostgreSQL is absent
        return CheckResult("database", False, f"database unavailable: {type(error).__name__}")
    return CheckResult("database", True, "PostgreSQL ready")


async def _database_ready(settings: Settings) -> None:
    factory = create_session_factory(settings)
    engine = factory.kw["bind"]
    try:
        async with factory() as session:
            await session.execute(text("SELECT 1"))
    finally:
        await engine.dispose()


def _probe_migration(settings: Settings) -> CheckResult:
    try:
        at_head = asyncio.run(_migration_at_head(settings))
    except Exception as error:  # database availability is reported separately too
        return CheckResult(
            "migration", False, f"migration check unavailable: {type(error).__name__}"
        )
    if not at_head:
        return CheckResult("migration", False, "database revision is not at migration head")
    return CheckResult("migration", True, "at head")


async def _migration_at_head(settings: Settings) -> bool:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    script = ScriptDirectory.from_config(config)
    heads = set(script.get_heads())
    factory = create_session_factory(settings)
    engine = factory.kw["bind"]
    try:
        async with factory() as session:
            current = set(
                (await session.execute(text("SELECT version_num FROM alembic_version"))).scalars()
            )
    finally:
        await engine.dispose()
    return current == heads


def _probe_cache(cache_dir: Path) -> CheckResult:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        probe = cache_dir / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        return CheckResult(
            "cache", False, f"not writable: {error.strerror or type(error).__name__}"
        )
    return CheckResult("cache", True, "writable")


def _probe_budget(settings: Settings) -> CheckResult:
    try:
        status = _usage_status(settings)
    except Exception as error:
        return CheckResult("budget", False, f"budget check unavailable: {type(error).__name__}")
    remaining = Decimal(status["remaining_usd"])
    if remaining <= 0:
        return CheckResult("budget", False, "no budget remaining")
    return CheckResult("budget", True, f"{status['remaining_usd']} USD remaining")


def _usage_status(settings: Settings) -> dict[str, str]:
    settled = asyncio.run(_settled_usage(settings))
    remaining = settings.max_project_budget_usd - settled
    return {"settled_cost_usd": _money_text(settled), "remaining_usd": _money_text(remaining)}


async def _settled_usage(settings: Settings) -> Decimal:
    factory = create_session_factory(settings)
    engine = factory.kw["bind"]
    try:
        async with factory() as session:
            value = await session.scalar(
                select(text("coalesce(sum(estimated_cost_usd), 0)")).select_from(ApiUsage)
            )
            return Decimal(value or 0)
    finally:
        await engine.dispose()


def _build_gateway(settings: Settings) -> ProviderGateway:
    """Wire the production gateway only after all explicit live guards have passed."""
    if not settings.upstage_api_key:
        raise RuntimeError("UPSTAGE_API_KEY is required")
    session_factory: async_sessionmaker[AsyncSession] | None = None
    lock_session_factory: async_sessionmaker[AsyncSession] | None = None
    gateway: UpstageGateway | None = None
    try:
        session_factory = create_session_factory(settings)
        lock_session_factory = create_lock_session_factory(settings)
        store = SqlAlchemyProviderStore(
            session_factory=session_factory,
            lock_session_factory=lock_session_factory,
            max_lock_connections=settings.max_lock_connections,
        )
        gateway = UpstageGateway(
            api_key=settings.upstage_api_key,
            base_url=settings.upstage_base_url,
            price_book=PriceBook.from_yaml(PRICE_PATH),
            budget_guard=BudgetGuard(
                SqlAlchemyBudgetRepository(session_factory),
                hard_limit=settings.max_project_budget_usd,
            ),
            store=store,
            billing_cost_multiplier=settings.billing_cost_multiplier,
            max_concurrency=settings.max_concurrency,
            max_retries=0,
        )
        return _ManagedGateway(gateway, session_factory, lock_session_factory)
    except BaseException:
        asyncio.run(_dispose_resources(gateway, session_factory, lock_session_factory))
        raise


class _ManagedGateway:
    """Close the HTTP client and both distinct SQLAlchemy engines as one lifecycle."""

    def __init__(
        self,
        gateway: UpstageGateway,
        session_factory: async_sessionmaker[AsyncSession],
        lock_session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._gateway = gateway
        self._session_factory = session_factory
        self._lock_session_factory = lock_session_factory

    async def generate(self, request: GenerateRequest) -> Any:
        return await self._gateway.generate(request)

    async def parse(self, request: ParseRequest) -> Any:
        return await self._gateway.parse(request)

    async def embed(self, request: EmbedRequest) -> Any:
        return await self._gateway.embed(request)

    async def aclose(self) -> None:
        await _dispose_resources(self._gateway, self._session_factory, self._lock_session_factory)


async def _dispose_resources(
    gateway: Any | None,
    session_factory: Any | None,
    lock_session_factory: Any | None,
) -> None:
    """Close independently owned HTTP and SQL resources even when an earlier close fails."""
    try:
        if gateway is not None:
            await gateway.aclose()
    finally:
        try:
            if session_factory is not None:
                await session_factory.kw["bind"].dispose()
        finally:
            if lock_session_factory is not None:
                await lock_session_factory.kw["bind"].dispose()


def main() -> None:
    """Console-script entry point."""
    try:
        build_app()()
    except Exception:
        if "--json" in sys.argv:
            typer.echo(json.dumps({"ok": False, "error": "configuration unavailable"}))
        else:
            typer.echo("configuration unavailable", err=True)
        raise SystemExit(1) from None
