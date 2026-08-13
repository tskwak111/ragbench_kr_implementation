"""Resumable, evidence-preserving Standard and Enhanced parse orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from time import perf_counter
from typing import Any, Literal, Protocol

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragbench.core.hashing import canonical_json_hash
from ragbench.db.models import Document, ParseRun
from ragbench.ingestion.manifest import DocumentRecord
from ragbench.providers.base import ParseRequest, ProviderGateway
from ragbench.providers.upstage.pricing import MONEY_QUANTUM, PriceBook, PricingRequest

ParseMode = Literal["standard", "enhanced"]


class ParseIntegrityError(RuntimeError):
    """Raised when approval, cache, provider output, or mode parity is unsafe."""


@dataclass(frozen=True, slots=True)
class CorpusSnapshot:
    """Resolved immutable corpus content used by a parse batch."""

    snapshot_id: str
    documents: tuple[DocumentRecord, ...]

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise ValueError("snapshot_id cannot be empty")
        if len({item.sha256 for item in self.documents}) != len(self.documents):
            raise ValueError("snapshot document hashes must be unique")


@dataclass(frozen=True, slots=True)
class ParseCheckpoint:
    snapshot_id: str
    document_id: str
    source_sha256: str
    expected_pages: int
    provider_model_id: str
    provider_model_version: str
    mode: ParseMode
    status: Literal["succeeded", "failed"]
    raw_response: dict[str, Any] | None
    raw_response_hash: str | None
    markdown: str
    html: str
    elements: tuple[dict[str, Any], ...]
    page_mappings: tuple[dict[str, Any], ...]
    latency_ms: int
    cost_usd: Decimal
    correlation_id: str | None = None
    error: str | None = None

    @property
    def cache_integrity_ok(self) -> bool:
        if self.status != "succeeded" or self.raw_response is None or not self.raw_response_hash:
            return False
        content = self.raw_response.get("content")
        elements = self.raw_response.get("elements")
        pages = self.raw_response.get("pages")
        if (
            not isinstance(content, dict)
            or not isinstance(elements, list)
            or not isinstance(pages, list)
        ):
            return False
        return (
            canonical_json_hash(self.raw_response) == self.raw_response_hash
            and self.markdown == str(content.get("markdown", ""))
            and self.html == str(content.get("html", ""))
            and self.elements == tuple(dict(item) for item in elements if isinstance(item, dict))
            and self.page_mappings == tuple(dict(item) for item in pages if isinstance(item, dict))
        )

    @classmethod
    def success_for_test(
        cls,
        snapshot: CorpusSnapshot,
        document: DocumentRecord,
        *,
        mode: ParseMode,
        raw: dict[str, Any] | None = None,
    ) -> ParseCheckpoint:
        payload = raw or {
            "model_version": "v1",
            "content": {"markdown": "cached", "html": "<p>cached</p>"},
            "elements": [],
            "pages": [
                {"page": page, "source_page": page}
                for page in range(1, document.page_count + 1)
            ],
            "usage": {"pages": document.page_count},
        }
        content = payload.get("content", {})
        assert isinstance(content, dict)
        pages = payload.get("pages", [])
        assert isinstance(pages, list)
        elements = payload.get("elements", [])
        assert isinstance(elements, list)
        return cls(
            snapshot.snapshot_id,
            document.document_id,
            document.sha256,
            document.page_count,
            "document-parse",
            str(payload.get("model_version", "v1")),
            mode,
            "succeeded",
            payload,
            canonical_json_hash(payload),
            str(content.get("markdown", "")),
            str(content.get("html", "")),
            tuple(dict(item) for item in elements),
            tuple(dict(item) for item in pages),
            0,
            Decimal("0"),
        )


class ParseRepository(Protocol):
    async def get(
        self, snapshot_id: str, source_sha256: str, mode: ParseMode
    ) -> ParseCheckpoint | None: ...

    async def put(self, checkpoint: ParseCheckpoint) -> None: ...


class MemoryParseRepository:
    """Deterministic checkpoint repository for offline orchestration tests."""

    def __init__(self) -> None:
        self.checkpoints: dict[tuple[str, str, ParseMode], ParseCheckpoint] = {}
        self._lock = asyncio.Lock()

    async def get(
        self, snapshot_id: str, source_sha256: str, mode: ParseMode
    ) -> ParseCheckpoint | None:
        return self.checkpoints.get((snapshot_id, source_sha256, mode))

    async def put(self, checkpoint: ParseCheckpoint) -> None:
        key = (checkpoint.snapshot_id, checkpoint.source_sha256, checkpoint.mode)
        async with self._lock:
            self.checkpoints[key] = checkpoint

    async def save_success_for_test(
        self, snapshot: CorpusSnapshot, document: DocumentRecord, *, mode: ParseMode
    ) -> None:
        checkpoint = ParseCheckpoint.success_for_test(snapshot, document, mode=mode)
        await self.put(replace(checkpoint, provider_model_version="2026-08-01"))

    def corrupt_raw_response(self, snapshot_id: str, source_sha256: str, mode: ParseMode) -> None:
        key = (snapshot_id, source_sha256, mode)
        checkpoint = self.checkpoints[key]
        assert checkpoint.raw_response is not None
        corrupt = dict(checkpoint.raw_response)
        corrupt["corrupted"] = True
        self.checkpoints[key] = replace(checkpoint, raw_response=corrupt)

    def replace_page_mappings(
        self,
        snapshot_id: str,
        source_sha256: str,
        mode: ParseMode,
        pages: tuple[dict[str, Any], ...],
    ) -> None:
        key = (snapshot_id, source_sha256, mode)
        self.checkpoints[key] = replace(self.checkpoints[key], page_mappings=pages)


class SqlAlchemyParseRepository:
    """PostgreSQL implementation of atomic, idempotent parse checkpoints."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get(
        self, snapshot_id: str, source_sha256: str, mode: ParseMode
    ) -> ParseCheckpoint | None:
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(ParseRun, Document)
                    .join(Document, Document.id == ParseRun.document_id)
                    .where(
                        ParseRun.corpus_snapshot_id == snapshot_id,
                        Document.sha256 == source_sha256,
                        ParseRun.mode == mode,
                    )
                    .order_by(ParseRun.created_at.desc())
                    .limit(1)
                )
            ).one_or_none()
        if row is None:
            return None
        run, document = row
        expected_pages = int(run.config_snapshot["expected_pages"])
        return ParseCheckpoint(
            run.corpus_snapshot_id,
            str(run.config_snapshot["document_id"]),
            document.sha256,
            expected_pages,
            run.provider_model_id,
            run.provider_model_version,
            _mode(run.mode),
            "succeeded" if run.status == "succeeded" else "failed",
            dict(run.raw_response) if run.raw_response is not None else None,
            run.raw_response_hash,
            run.markdown,
            run.html,
            tuple(dict(item) for item in run.elements),
            tuple(dict(item) for item in run.page_mappings),
            run.latency_ms,
            run.cost_usd,
            run.config_snapshot.get("correlation_id"),
            run.error,
        )

    async def put(self, checkpoint: ParseCheckpoint) -> None:
        document_values = {
            "title": checkpoint.document_id,
            "sha256": checkpoint.source_sha256,
            "source_uri": f"corpus:{checkpoint.snapshot_id}/{checkpoint.document_id}",
            "metadata_snapshot": {
                "document_id": checkpoint.document_id,
                "expected_pages": checkpoint.expected_pages,
            },
        }
        async with self._session_factory() as session, session.begin():
            await session.execute(
                insert(Document)
                .values(**document_values)
                .on_conflict_do_nothing(index_elements=[Document.sha256])
            )
            document_id = await session.scalar(
                select(Document.id).where(Document.sha256 == checkpoint.source_sha256)
            )
            if document_id is None:
                raise RuntimeError("document checkpoint identity could not be persisted")
            config = {
                "document_id": checkpoint.document_id,
                "expected_pages": checkpoint.expected_pages,
                "correlation_id": checkpoint.correlation_id,
            }
            values = {
                "document_id": document_id,
                "corpus_snapshot_id": checkpoint.snapshot_id,
                "provider_model_id": checkpoint.provider_model_id,
                "provider_model_version": checkpoint.provider_model_version,
                "mode": checkpoint.mode,
                "status": checkpoint.status,
                "config_snapshot": config,
                "raw_response_hash": checkpoint.raw_response_hash,
                "raw_response": checkpoint.raw_response,
                "markdown": checkpoint.markdown,
                "html": checkpoint.html,
                "elements": list(checkpoint.elements),
                "page_mappings": list(checkpoint.page_mappings),
                "latency_ms": checkpoint.latency_ms,
                "cost_usd": checkpoint.cost_usd,
                "error": checkpoint.error,
            }
            statement = insert(ParseRun).values(**values)
            await session.execute(
                statement.on_conflict_do_update(
                    constraint="uq_parse_run_checkpoint",
                    set_={key: value for key, value in values.items() if key != "document_id"},
                )
            )


@dataclass(frozen=True, slots=True)
class PlannedDocument:
    document_id: str
    source_sha256: str
    pages: int
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class ParsePlan:
    snapshot_id: str
    mode: ParseMode
    model_id: str
    model_version: str
    price_snapshot_hash: str
    documents: tuple[PlannedDocument, ...]
    projected_new_calls: int
    projected_billable_pages: int
    base_cost_usd: Decimal
    vat_buffer: Decimal
    worst_case_cost_usd: Decimal
    settled_cost_usd: Decimal
    post_run_remaining_budget_usd: Decimal
    corrupt_checkpoints: int
    plan_hash: str


@dataclass(frozen=True, slots=True)
class ParseFailure:
    document_id: str
    error: str


@dataclass(frozen=True, slots=True)
class ParseSummary:
    snapshot_id: str
    mode: ParseMode
    plan_hash: str
    total_documents: int
    total_pages: int
    cache_hits: int
    succeeded_documents: int
    succeeded_pages: int
    failed_documents: int
    failed_pages: int
    retried_corrupt_checkpoints: int
    estimated_new_cost_usd: Decimal
    failures: tuple[ParseFailure, ...]


@dataclass(frozen=True, slots=True)
class ParseAuthorization:
    """Exact operator confirmation for one immutable paid plan."""

    confirmed_plan_hash: str


class ParserPipeline:
    """Plans and executes document-level parses exclusively through a gateway."""

    def __init__(
        self,
        *,
        snapshots: Mapping[str, CorpusSnapshot],
        gateway: ProviderGateway,
        repository: ParseRepository,
        price_book: PriceBook,
        model_id: str,
        model_version: str,
        hard_budget_usd: Decimal,
        settled_cost_usd: Callable[[], Decimal],
        vat_buffer: Decimal = Decimal("0.10"),
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        authorization: ParseAuthorization | None = None,
    ) -> None:
        if vat_buffer < 0:
            raise ValueError("vat_buffer cannot be negative")
        self._snapshots = snapshots
        self._gateway = gateway
        self._repository = repository
        self._price_book = price_book
        self._model_id = model_id
        self._model_version = model_version
        self._hard_budget_usd = hard_budget_usd
        self._settled_cost_usd = settled_cost_usd
        self._vat_buffer = vat_buffer
        self._now = now
        self._authorization = authorization

    async def plan_corpus(
        self, snapshot_id: str, mode: str, resume: bool = True
    ) -> ParsePlan:
        selected_mode = _mode(mode)
        snapshot = self._snapshot(snapshot_id)
        if selected_mode == "enhanced":
            await self._require_standard_parity(snapshot)
        planned: list[PlannedDocument] = []
        corrupt = 0
        for document in snapshot.documents:
            checkpoint = await self._repository.get(snapshot_id, document.sha256, selected_mode)
            valid = bool(
                resume
                and checkpoint
                and checkpoint.cache_integrity_ok
                and self._checkpoint_matches(checkpoint, document)
            )
            if resume and checkpoint and checkpoint.status == "succeeded" and not valid:
                corrupt += 1
            planned.append(
                PlannedDocument(document.document_id, document.sha256, document.page_count, valid)
            )
        new_documents = [item for item in planned if not item.cache_hit]
        billable_pages = sum(item.pages for item in new_documents)
        base = self._price_book.estimate(
            PricingRequest(
                operation="parse",
                model_id=self._model_id,
                billable_pages=billable_pages,
                mode=selected_mode,
            )
        )
        worst = _money(base * (Decimal("1") + self._vat_buffer))
        settled = self._settled_cost_usd()
        remaining = _money(self._hard_budget_usd - settled - worst)
        material = {
            "schema_version": "parse-plan-v1",
            "snapshot_id": snapshot_id,
            "mode": selected_mode,
            "model_id": self._model_id,
            "model_version": self._model_version,
            "price_snapshot_hash": canonical_json_hash(self._price_book.snapshot()),
            "documents": planned,
            "projected_new_calls": len(new_documents),
            "projected_billable_pages": billable_pages,
            "base_cost_usd": base,
            "vat_buffer": self._vat_buffer,
            "worst_case_cost_usd": worst,
            "settled_cost_usd": settled,
            "post_run_remaining_budget_usd": remaining,
        }
        plan_hash = canonical_json_hash(material)
        return ParsePlan(
            snapshot_id,
            selected_mode,
            self._model_id,
            self._model_version,
            str(material["price_snapshot_hash"]),
            tuple(planned),
            len(new_documents),
            billable_pages,
            base,
            self._vat_buffer,
            worst,
            settled,
            remaining,
            corrupt,
            plan_hash,
        )

    async def parse_corpus(
        self, snapshot_id: str, mode: str, resume: bool = True
    ) -> ParseSummary:
        plan = await self.plan_corpus(snapshot_id, mode, resume)
        self._price_book.verify_paid_batch(now=self._now())
        if self._authorization is None or self._authorization.confirmed_plan_hash != plan.plan_hash:
            raise ParseIntegrityError("paid execution requires the exact current plan hash")
        if plan.post_run_remaining_budget_usd <= 0:
            raise ParseIntegrityError("planned parse reaches the hard budget limit")
        snapshot = self._snapshot(snapshot_id)
        by_hash = {document.sha256: document for document in snapshot.documents}
        failures: list[ParseFailure] = []
        successes = 0
        succeeded_pages = 0
        for item in plan.documents:
            if item.cache_hit:
                continue
            document = by_hash[item.source_sha256]
            started = perf_counter()
            try:
                response = await self._gateway.parse(
                    ParseRequest(
                        model_id=self._model_id,
                        document_sha256=document.sha256,
                        content=document.local_path.read_bytes(),
                        billable_pages=document.page_count,
                        mode=plan.mode,
                    )
                )
                latency_ms = max(0, round((perf_counter() - started) * 1000))
                checkpoint = self._normalize(
                    snapshot, document, plan.mode, response.raw_response, response.correlation_id,
                    latency_ms,
                )
                await self._repository.put(checkpoint)
                successes += 1
                succeeded_pages += document.page_count
            except Exception as error:
                latency_ms = max(0, round((perf_counter() - started) * 1000))
                await self._repository.put(
                    ParseCheckpoint(
                        snapshot_id,
                        document.document_id,
                        document.sha256,
                        document.page_count,
                        self._model_id,
                        self._model_version,
                        plan.mode,
                        "failed",
                        None,
                        None,
                        "",
                        "",
                        (),
                        (),
                        latency_ms,
                        Decimal("0"),
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                failures.append(ParseFailure(document.document_id, str(error)))
        failed_ids = {failure.document_id for failure in failures}
        return ParseSummary(
            snapshot_id,
            plan.mode,
            plan.plan_hash,
            len(plan.documents),
            sum(item.pages for item in plan.documents),
            sum(item.cache_hit for item in plan.documents),
            successes,
            succeeded_pages,
            len(failures),
            sum(item.pages for item in plan.documents if item.document_id in failed_ids),
            plan.corrupt_checkpoints,
            plan.base_cost_usd,
            tuple(failures),
        )

    async def _require_standard_parity(self, snapshot: CorpusSnapshot) -> None:
        for document in snapshot.documents:
            standard = await self._repository.get(snapshot.snapshot_id, document.sha256, "standard")
            if standard is None:
                raise ParseIntegrityError(
                    f"enhanced mode requires successful Standard source {document.document_id}"
                )
            page_set = {
                int(item.get("source_page", item.get("page", 0)))
                for item in standard.page_mappings
            }
            if page_set != set(range(1, document.page_count + 1)):
                raise ParseIntegrityError(
                    f"enhanced mode page set differs for {document.document_id}"
                )
            if not standard.cache_integrity_ok:
                raise ParseIntegrityError(
                    f"enhanced mode requires intact Standard source {document.document_id}"
                )
            if not self._checkpoint_matches(standard, document):
                raise ParseIntegrityError(
                    f"enhanced mode source hash/model differs for {document.document_id}"
                )

    def _normalize(
        self,
        snapshot: CorpusSnapshot,
        document: DocumentRecord,
        mode: ParseMode,
        raw: dict[str, Any],
        correlation_id: str | None,
        latency_ms: int,
    ) -> ParseCheckpoint:
        if str(raw.get("model_version", "")) != self._model_version:
            raise ParseIntegrityError("provider model version differs from the approved plan")
        content = raw.get("content")
        pages = raw.get("pages")
        elements = raw.get("elements")
        usage = raw.get("usage")
        if not isinstance(content, dict) or not isinstance(pages, list):
            raise ParseIntegrityError("provider response lacks content or page mappings")
        if not isinstance(elements, list) or not all(isinstance(item, dict) for item in elements):
            raise ParseIntegrityError("provider response elements are malformed")
        if not isinstance(usage, dict) or usage.get("pages") != document.page_count:
            raise ParseIntegrityError("provider billable page count does not match the corpus")
        source_pages = {int(item.get("source_page", item.get("page", 0))) for item in pages}
        if source_pages != set(range(1, document.page_count + 1)):
            raise ParseIntegrityError("provider page count or page set does not match the corpus")
        cost = self._price_book.estimate(
            PricingRequest(
                operation="parse",
                model_id=self._model_id,
                billable_pages=document.page_count,
                mode=mode,
            )
        )
        return ParseCheckpoint(
            snapshot.snapshot_id,
            document.document_id,
            document.sha256,
            document.page_count,
            self._model_id,
            self._model_version,
            mode,
            "succeeded",
            dict(raw),
            canonical_json_hash(raw),
            str(content.get("markdown", "")),
            str(content.get("html", "")),
            tuple(dict(item) for item in elements),
            tuple(dict(item) for item in pages if isinstance(item, dict)),
            latency_ms,
            cost,
            correlation_id,
        )

    def _checkpoint_matches(
        self, checkpoint: ParseCheckpoint, document: DocumentRecord
    ) -> bool:
        return (
            checkpoint.source_sha256 == document.sha256
            and checkpoint.expected_pages == document.page_count
            and checkpoint.provider_model_id == self._model_id
            and checkpoint.provider_model_version == self._model_version
        )

    def _snapshot(self, snapshot_id: str) -> CorpusSnapshot:
        try:
            return self._snapshots[snapshot_id]
        except KeyError as error:
            raise ParseIntegrityError(f"unknown corpus snapshot: {snapshot_id}") from error


def _mode(value: str) -> ParseMode:
    if value not in {"standard", "enhanced"}:
        raise ValueError("mode must be 'standard' or 'enhanced'")
    return value  # type: ignore[return-value]


def _money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANTUM, rounding=ROUND_CEILING)


def execution_blockers(
    *, execute: bool, confirm_plan: str | None, live_enabled: bool, api_key_present: bool
) -> tuple[str, ...]:
    """Return independent fail-closed gates for the paid CLI path."""
    if not execute:
        return ()
    blockers: list[str] = []
    if not confirm_plan:
        blockers.append("--confirm-plan is required")
    if not live_enabled:
        blockers.append("RUN_LIVE_UPSTAGE_TESTS=1 is required")
    if not api_key_present:
        blockers.append("UPSTAGE_API_KEY is required")
    return tuple(blockers)


async def parse_corpus(
    snapshot_id: str,
    mode: str,
    resume: bool = True,
    *,
    pipeline: ParserPipeline,
) -> ParseSummary:
    """Stable functional entry point backed by an explicitly wired pipeline."""
    return await pipeline.parse_corpus(snapshot_id, mode, resume)
