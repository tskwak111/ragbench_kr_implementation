"""Resumable, evidence-preserving Standard and Enhanced parse orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal
from time import perf_counter
from typing import Any, Literal, Protocol

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragbench.core.hashing import canonical_json_hash
from ragbench.db.models import Document, ParseRun
from ragbench.ingestion.manifest import DocumentRecord
from ragbench.providers.base import ParseRequest, ProviderGateway
from ragbench.providers.upstage.pricing import MONEY_QUANTUM, PriceBook, PricingRequest

ParseMode = Literal["standard", "enhanced"]
_SQL_LOCAL_LOCKS: dict[str, asyncio.Lock] = {}


@dataclass(slots=True)
class _Flight:
    lock: asyncio.Lock
    users: int = 0


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
    status: Literal["succeeded", "failed", "reconciliation_required"]
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
        content = _content_fields(self.raw_response)
        elements = self.raw_response.get("elements", [])
        if not isinstance(elements, list):
            return False
        try:
            mappings = _page_mappings(self.raw_response, self.expected_pages)
        except ParseIntegrityError:
            return False
        return (
            canonical_json_hash(self.raw_response) == self.raw_response_hash
            and self.markdown == content[0]
            and self.html == content[1]
            and self.elements == tuple(dict(item) for item in elements if isinstance(item, dict))
            and self.page_mappings == mappings
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
                {"page": page, "source_page": page} for page in range(1, document.page_count + 1)
            ],
            "usage": {"pages": document.page_count},
        }
        markdown, html = _content_fields(payload)
        elements = payload.get("elements", [])
        assert isinstance(elements, list)
        return cls(
            snapshot.snapshot_id,
            document.document_id,
            document.sha256,
            document.page_count,
            "document-parse",
            _provider_version(payload) or "v1",
            mode,
            "succeeded",
            payload,
            canonical_json_hash(payload),
            markdown,
            html,
            tuple(dict(item) for item in elements),
            _page_mappings(payload, document.page_count),
            0,
            Decimal("0"),
        )


class ParseRepository(Protocol):
    async def get(
        self,
        snapshot_id: str,
        source_sha256: str,
        mode: ParseMode,
        provider_model_id: str,
        provider_model_version: str,
    ) -> ParseCheckpoint | None: ...

    async def put(self, checkpoint: ParseCheckpoint) -> None: ...

    def singleflight(self, identity: str) -> AbstractAsyncContextManager[None]: ...


class MemoryParseRepository:
    """Deterministic checkpoint repository for offline orchestration tests."""

    def __init__(self) -> None:
        self.checkpoints: dict[tuple[str, str, ParseMode, str, str], ParseCheckpoint] = {}
        self._lock = asyncio.Lock()
        self._flight_locks: dict[str, _Flight] = {}

    async def get(
        self,
        snapshot_id: str,
        source_sha256: str,
        mode: ParseMode,
        provider_model_id: str,
        provider_model_version: str,
    ) -> ParseCheckpoint | None:
        matches = [
            value
            for key, value in self.checkpoints.items()
            if key[:3] == (snapshot_id, source_sha256, mode)
        ]
        matches = [value for value in matches if value.provider_model_id == provider_model_id]
        matches = [
            value for value in matches if value.provider_model_version == provider_model_version
        ]
        return matches[-1] if matches else None

    async def put(self, checkpoint: ParseCheckpoint) -> None:
        key = (
            checkpoint.snapshot_id,
            checkpoint.source_sha256,
            checkpoint.mode,
            checkpoint.provider_model_id,
            checkpoint.provider_model_version,
        )
        async with self._lock:
            self.checkpoints[key] = checkpoint

    async def save_success_for_test(
        self, snapshot: CorpusSnapshot, document: DocumentRecord, *, mode: ParseMode
    ) -> None:
        checkpoint = ParseCheckpoint.success_for_test(snapshot, document, mode=mode)
        await self.put(replace(checkpoint, provider_model_version="2026-08-01"))

    def corrupt_raw_response(self, snapshot_id: str, source_sha256: str, mode: ParseMode) -> None:
        checkpoint = self._find(snapshot_id, source_sha256, mode)
        assert checkpoint.raw_response is not None
        corrupt = dict(checkpoint.raw_response)
        corrupt["corrupted"] = True
        key = (
            snapshot_id,
            source_sha256,
            mode,
            checkpoint.provider_model_id,
            checkpoint.provider_model_version,
        )
        self.checkpoints[key] = replace(checkpoint, raw_response=corrupt)

    def replace_page_mappings(
        self,
        snapshot_id: str,
        source_sha256: str,
        mode: ParseMode,
        pages: tuple[dict[str, Any], ...],
    ) -> None:
        checkpoint = self._find(snapshot_id, source_sha256, mode)
        key = (
            snapshot_id,
            source_sha256,
            mode,
            checkpoint.provider_model_id,
            checkpoint.provider_model_version,
        )
        self.checkpoints[key] = replace(checkpoint, page_mappings=pages)

    def _find(self, snapshot_id: str, source_sha256: str, mode: ParseMode) -> ParseCheckpoint:
        return next(
            value
            for key, value in self.checkpoints.items()
            if key[:3] == (snapshot_id, source_sha256, mode)
        )

    @asynccontextmanager
    async def singleflight(self, identity: str) -> AsyncIterator[None]:
        flight = self._flight_locks.setdefault(identity, _Flight(asyncio.Lock()))
        flight.users += 1
        acquired = False
        try:
            await flight.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                flight.lock.release()
            flight.users -= 1
            if flight.users == 0 and self._flight_locks.get(identity) is flight:
                del self._flight_locks[identity]

    @property
    def singleflight_lock_count(self) -> int:
        return len(self._flight_locks)


class SqlAlchemyParseRepository:
    """PostgreSQL implementation of atomic, idempotent parse checkpoints."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        lock_session_factory: async_sessionmaker[AsyncSession] | None = None,
        max_lock_connections: int = 2,
    ) -> None:
        if max_lock_connections <= 0:
            raise ValueError("max_lock_connections must be positive")
        self._session_factory = session_factory
        self._lock_session_factory = lock_session_factory
        self._lock_permits = asyncio.Semaphore(max_lock_connections)

    async def get(
        self,
        snapshot_id: str,
        source_sha256: str,
        mode: ParseMode,
        provider_model_id: str,
        provider_model_version: str,
    ) -> ParseCheckpoint | None:
        async with self._session_factory() as session:
            predicates = [
                ParseRun.corpus_snapshot_id == snapshot_id,
                Document.sha256 == source_sha256,
                ParseRun.mode == mode,
            ]
            predicates.append(ParseRun.provider_model_id == provider_model_id)
            predicates.append(ParseRun.provider_model_version == provider_model_version)
            row = (
                await session.execute(
                    select(ParseRun, Document)
                    .join(Document, Document.id == ParseRun.document_id)
                    .where(
                        *predicates,
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
            run.status,
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

    @asynccontextmanager
    async def singleflight(self, identity: str) -> AsyncIterator[None]:
        lock = _SQL_LOCAL_LOCKS.setdefault(identity, asyncio.Lock())
        async with lock:
            if self._lock_session_factory is None:
                yield
                return
            lock_id = int.from_bytes(
                hashlib.sha256(identity.encode()).digest()[:8], "big", signed=True
            )
            async with (
                self._lock_permits,
                self._lock_session_factory() as session,
                session.begin(),
            ):
                await session.execute(
                    text("SELECT pg_advisory_xact_lock(:lock_id)"), {"lock_id": lock_id}
                )
                yield


@dataclass(frozen=True, slots=True)
class PlannedDocument:
    document_id: str
    source_sha256: str
    pages: int
    cache_hit: bool
    source_size: int
    source_mtime_ns: int


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
    cached_successes: int
    cached_success_pages: int
    new_successes: int
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

    async def plan_corpus(self, snapshot_id: str, mode: str, resume: bool = True) -> ParsePlan:
        selected_mode = _mode(mode)
        snapshot = self._snapshot(snapshot_id)
        if selected_mode == "enhanced":
            await self._require_standard_parity(snapshot)
        planned: list[PlannedDocument] = []
        corrupt = 0
        for document in snapshot.documents:
            checkpoint = await self._repository.get(
                snapshot_id, document.sha256, selected_mode, self._model_id, self._model_version
            )
            valid = bool(
                resume
                and checkpoint
                and checkpoint.cache_integrity_ok
                and self._checkpoint_matches(checkpoint, document)
            )
            if resume and checkpoint and checkpoint.status == "succeeded" and not valid:
                corrupt += 1
            source_size, source_mtime_ns = _source_metadata(document)
            planned.append(
                PlannedDocument(
                    document.document_id,
                    document.sha256,
                    document.page_count,
                    valid,
                    source_size,
                    source_mtime_ns,
                )
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

    async def parse_corpus(self, snapshot_id: str, mode: str, resume: bool = True) -> ParseSummary:
        plan = await self.plan_corpus(snapshot_id, mode, resume)
        self._price_book.verify_paid_batch(now=self._now())
        if self._authorization is None or self._authorization.confirmed_plan_hash != plan.plan_hash:
            raise ParseIntegrityError("paid execution requires the exact current plan hash")
        if plan.post_run_remaining_budget_usd <= 0:
            raise ParseIntegrityError("planned parse reaches the hard budget limit")
        snapshot = self._snapshot(snapshot_id)
        by_hash = {document.sha256: document for document in snapshot.documents}
        failures: list[ParseFailure] = []
        new_successes = 0
        new_success_pages = 0
        dynamic_cache_hits = sum(item.cache_hit for item in plan.documents)
        dynamic_cached_pages = sum(item.pages for item in plan.documents if item.cache_hit)
        for item in plan.documents:
            if item.cache_hit:
                continue
            document = by_hash[item.source_sha256]
            identity = canonical_json_hash(
                [snapshot_id, document.sha256, plan.mode, self._model_id, self._model_version]
            )
            async with self._repository.singleflight(identity):
                existing = await self._repository.get(
                    snapshot_id, document.sha256, plan.mode, self._model_id, self._model_version
                )
                if resume and existing and existing.cache_integrity_ok:
                    dynamic_cache_hits += 1
                    dynamic_cached_pages += document.page_count
                    continue
                started = perf_counter()
                response = None
                try:
                    source_bytes = _read_verified_source(document)
                    response = await self._gateway.parse(
                        ParseRequest(
                            model_id=self._model_id,
                            document_sha256=document.sha256,
                            content=source_bytes,
                            billable_pages=document.page_count,
                            mode=plan.mode,
                        )
                    )
                    latency_ms = max(0, round((perf_counter() - started) * 1000))
                    checkpoint = self._normalize(
                        snapshot,
                        document,
                        plan.mode,
                        response.raw_response,
                        response.correlation_id,
                        latency_ms,
                    )
                    await self._repository.put(checkpoint)
                    new_successes += 1
                    new_success_pages += document.page_count
                except Exception as error:
                    latency_ms = max(0, round((perf_counter() - started) * 1000))
                    raw = response.raw_response if response is not None else None
                    paid = raw is not None
                    version = _provider_version(raw) if raw is not None else None
                    await self._repository.put(
                        ParseCheckpoint(
                            snapshot_id,
                            document.document_id,
                            document.sha256,
                            document.page_count,
                            self._model_id,
                            version or self._model_version,
                            plan.mode,
                            "reconciliation_required" if paid else "failed",
                            dict(raw) if raw is not None else None,
                            canonical_json_hash(raw) if raw is not None else None,
                            *_content_fields(raw or {}),
                            _evidence_elements(raw or {}),
                            _safe_page_mappings(raw or {}, document.page_count) if paid else (),
                            latency_ms,
                            _money(
                                self._price_book.estimate(
                                    PricingRequest(
                                        operation="parse",
                                        model_id=self._model_id,
                                        billable_pages=document.page_count,
                                        mode=plan.mode,
                                    )
                                )
                                * (Decimal("1") + self._vat_buffer)
                            )
                            if paid
                            else Decimal("0"),
                            response.correlation_id if response is not None else None,
                            f"{type(error).__name__}: {error}",
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
            dynamic_cache_hits,
            dynamic_cache_hits,
            dynamic_cached_pages,
            new_successes,
            dynamic_cache_hits + new_successes,
            dynamic_cached_pages + new_success_pages,
            len(failures),
            sum(item.pages for item in plan.documents if item.document_id in failed_ids),
            plan.corrupt_checkpoints,
            plan.base_cost_usd,
            tuple(failures),
        )

    async def _require_standard_parity(self, snapshot: CorpusSnapshot) -> None:
        for document in snapshot.documents:
            standard = await self._repository.get(
                snapshot.snapshot_id,
                document.sha256,
                "standard",
                self._model_id,
                self._model_version,
            )
            if standard is None:
                raise ParseIntegrityError(
                    f"enhanced mode requires successful Standard source {document.document_id}"
                )
            page_set = {
                int(item.get("source_page", item.get("page", 0))) for item in standard.page_mappings
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
        resolved_version = _provider_version(raw)
        if resolved_version != self._model_version:
            raise ParseIntegrityError("provider model version differs from the approved plan")
        elements = raw.get("elements", [])
        if not isinstance(elements, list) or not all(isinstance(item, dict) for item in elements):
            raise ParseIntegrityError("provider response elements are malformed")
        has_provider_page_count = _validate_provider_page_count(raw, document.page_count)
        markdown, html = _content_fields(raw)
        if "elements" not in raw and not has_provider_page_count and not markdown and not html:
            raise ParseIntegrityError("provider response contains no recognized parse evidence")
        mappings = _page_mappings(raw, document.page_count)
        cost = _money(
            self._price_book.estimate(
                PricingRequest(
                    operation="parse",
                    model_id=self._model_id,
                    billable_pages=document.page_count,
                    mode=mode,
                )
            )
            * (Decimal("1") + self._vat_buffer)
        )
        return ParseCheckpoint(
            snapshot.snapshot_id,
            document.document_id,
            document.sha256,
            document.page_count,
            self._model_id,
            resolved_version,
            mode,
            "succeeded",
            dict(raw),
            canonical_json_hash(raw),
            markdown,
            html,
            tuple(dict(item) for item in elements),
            mappings,
            latency_ms,
            cost,
            correlation_id,
        )

    def _checkpoint_matches(self, checkpoint: ParseCheckpoint, document: DocumentRecord) -> bool:
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


def _provider_version(raw: dict[str, Any] | None) -> str | None:
    if raw is None:
        return None
    value = raw.get("model", raw.get("model_version"))
    return str(value) if isinstance(value, str) and value else None


def _evidence_elements(raw: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    elements = raw.get("elements")
    if not isinstance(elements, list):
        return ()
    return tuple(dict(value) for value in elements if isinstance(value, dict))


def _content_fields(raw: dict[str, Any]) -> tuple[str, str]:
    content = raw.get("content")
    markdown = raw.get("markdown", "")
    html = raw.get("html", "")
    if isinstance(content, dict):
        markdown = content.get("markdown", markdown)
        html = content.get("html", html)
    elif isinstance(content, str) and not markdown:
        markdown = content
    element_markdown: list[str] = []
    element_html: list[str] = []
    elements = raw.get("elements")
    if isinstance(elements, list):
        for element in elements:
            if not isinstance(element, dict):
                continue
            element_content = element.get("content")
            if isinstance(element_content, dict):
                candidate_markdown = element_content.get("markdown")
                candidate_html = element_content.get("html")
            else:
                candidate_markdown = element_content
                candidate_html = None
            if isinstance(candidate_markdown, str) and candidate_markdown:
                element_markdown.append(candidate_markdown)
            if isinstance(candidate_html, str) and candidate_html:
                element_html.append(candidate_html)
    if not markdown:
        markdown = "\n\n".join(element_markdown)
    if not html:
        html = "\n".join(element_html)
    return (
        markdown if isinstance(markdown, str) else "",
        html if isinstance(html, str) else "",
    )


def _page_mappings(raw: dict[str, Any], expected_pages: int) -> tuple[dict[str, Any], ...]:
    metadata_by_page: dict[int, dict[str, Any]] = {}
    candidates = raw.get("pages")
    if candidates is not None and not isinstance(candidates, list):
        raise ParseIntegrityError("provider page metadata is malformed")
    if isinstance(candidates, list):
        for item in candidates:
            if not isinstance(item, dict):
                raise ParseIntegrityError("provider page metadata is malformed")
            page = item.get("source_page", item.get("page_number", item.get("page")))
            if not isinstance(page, int):
                raise ParseIntegrityError("provider page metadata lacks a page number")
            if page not in range(1, expected_pages + 1):
                raise ParseIntegrityError("provider page metadata is outside the manifest page set")
            metadata_by_page[page] = dict(item)
    elements = raw.get("elements")
    element_indexes: dict[int, list[int]] = {page: [] for page in range(1, expected_pages + 1)}
    if isinstance(elements, list):
        for index, item in enumerate(elements):
            if not isinstance(item, dict):
                raise ParseIntegrityError("provider response elements are malformed")
            page = item.get("page", item.get("page_number"))
            if not isinstance(page, int):
                raise ParseIntegrityError("provider element lacks a page number")
            if page not in element_indexes:
                raise ParseIntegrityError("provider element page is outside the manifest page set")
            element_indexes[page].append(index)
    mappings: list[dict[str, Any]] = []
    for page in range(1, expected_pages + 1):
        mapping: dict[str, Any] = {
            "page": page,
            "source_page": page,
            "element_count": len(element_indexes[page]),
            "element_indexes": element_indexes[page],
        }
        if page in metadata_by_page:
            mapping["provider_page_metadata"] = metadata_by_page[page]
        mappings.append(mapping)
    return tuple(mappings)


def _safe_page_mappings(raw: dict[str, Any], expected_pages: int) -> tuple[dict[str, Any], ...]:
    try:
        return _page_mappings(raw, expected_pages)
    except ParseIntegrityError:
        return tuple(
            {
                "page": page,
                "source_page": page,
                "element_count": 0,
                "element_indexes": [],
            }
            for page in range(1, expected_pages + 1)
        )


def _validate_provider_page_count(raw: dict[str, Any], expected_pages: int) -> bool:
    observed: list[Any] = []
    usage = raw.get("usage")
    if usage is not None:
        if not isinstance(usage, dict):
            raise ParseIntegrityError("provider usage is malformed")
        for key in ("billable_pages", "pages", "page_count"):
            if key in usage:
                observed.append(usage[key])
    for key in ("billable_pages", "page_count"):
        if key in raw:
            observed.append(raw[key])
    if any(not isinstance(value, int) for value in observed):
        raise ParseIntegrityError("provider billable page count is malformed")
    if any(value != expected_pages for value in observed):
        raise ParseIntegrityError("provider billable page count does not match the corpus")
    return bool(observed)


def _read_verified_source(document: DocumentRecord) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(document.local_path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ParseIntegrityError("source is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != document.sha256:
        raise ParseIntegrityError("source bytes no longer match the approved manifest")
    return payload


def _source_metadata(document: DocumentRecord) -> tuple[int, int]:
    metadata = document.local_path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ParseIntegrityError("source is not a regular file")
    return metadata.st_size, metadata.st_mtime_ns


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
