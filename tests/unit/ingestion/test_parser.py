"""Offline tests for dual-mode corpus parse planning and normalization."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from ragbench.ingestion.manifest import DocumentRecord
from ragbench.ingestion.parser import (
    CorpusSnapshot,
    MemoryParseRepository,
    ParseAuthorization,
    ParseIntegrityError,
    ParserPipeline,
    execution_blockers,
)
from ragbench.providers.base import ParsedDocument, ParseRequest
from ragbench.providers.upstage.pricing import PriceBook


def _document(tmp_path: Path, name: str, pages: int) -> DocumentRecord:
    path = tmp_path / f"{name}.pdf"
    payload = f"pdf-{name}".encode()
    path.write_bytes(payload)
    import hashlib

    return DocumentRecord(
        document_id=name,
        title=name,
        organization=f"org-{name}",
        year=2025,
        document_type="report",
        language="ko",
        sector="public",
        content_stratum="mixed",
        template_family=f"family-{name}",
        license="reviewed",
        redistribution_status="nonredistributable",
        local_path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        page_count=pages,
        inclusion_rationale="test fixture",
    )


def _prices() -> PriceBook:
    return PriceBook(
        {
            "schema_version": "test-prices-v1",
            "verified_at": "2026-08-14T00:00:00Z",
            "vat_excluded": True,
            "models": {
                "document-parse": {
                    "modes": {
                        "standard": {"usd_per_page": "0.01"},
                        "enhanced": {"usd_per_page": "0.03"},
                    }
                }
            },
        }
    )


class RecordingGateway:
    def __init__(self, responses: list[dict[str, object] | Exception]) -> None:
        self.responses = responses
        self.requests: list[ParseRequest] = []

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ParsedDocument(response, "correlation-test")


def _response(pages: int, *, markdown: str = "ok") -> dict[str, object]:
    return {
        "model_version": "2026-08-01",
        "content": {"markdown": markdown, "html": f"<p>{markdown}</p>"},
        "elements": [{"category": "paragraph", "content": markdown, "page": 1}],
        "pages": [{"page": page, "source_page": page} for page in range(1, pages + 1)],
        "usage": {"pages": pages},
    }


def _provider_response(pages: int, *, model: str = "2026-08-01") -> dict[str, object]:
    return {
        "model": model,
        "content": {"markdown": "whole markdown", "html": "<p>whole html</p>"},
        "elements": [
            {
                "id": page,
                "page": page,
                "category": "paragraph",
                "content": {"markdown": f"page {page}", "html": f"<p>{page}</p>"},
            }
            for page in range(1, pages + 1)
        ],
    }


def _pipeline(
    snapshot: CorpusSnapshot,
    gateway: RecordingGateway,
    repository: MemoryParseRepository,
    *,
    authorization: ParseAuthorization | None = None,
    settled: Decimal = Decimal("5"),
    model_version: str = "2026-08-01",
) -> ParserPipeline:
    return ParserPipeline(
        snapshots={snapshot.snapshot_id: snapshot},
        gateway=gateway,
        repository=repository,
        price_book=_prices(),
        model_id="document-parse",
        model_version=model_version,
        hard_budget_usd=Decimal("100"),
        settled_cost_usd=lambda: settled,
        vat_buffer=Decimal("0.10"),
        now=lambda: datetime(2026, 8, 14, 1, tzinfo=UTC),
        authorization=authorization,
    )


@pytest.mark.asyncio
async def test_dry_run_plan_lists_documents_cache_calls_cost_and_stable_hash(
    tmp_path: Path,
) -> None:
    """Catch a plan that hides billable scope or hashes mutable display fields."""
    documents = (_document(tmp_path, "a", 2), _document(tmp_path, "b", 3))
    snapshot = CorpusSnapshot("snapshot-1", documents)
    repository = MemoryParseRepository()
    await repository.save_success_for_test(snapshot, documents[0], mode="standard")
    pipeline = _pipeline(snapshot, RecordingGateway([]), repository)

    first = await pipeline.plan_corpus("snapshot-1", "standard", resume=True)
    second = await pipeline.plan_corpus("snapshot-1", "standard", resume=True)

    assert [(item.document_id, item.pages, item.cache_hit) for item in first.documents] == [
        ("a", 2, True),
        ("b", 3, False),
    ]
    assert first.projected_new_calls == 1
    assert first.projected_billable_pages == 3
    assert first.base_cost_usd == Decimal("0.030000")
    assert first.worst_case_cost_usd == Decimal("0.033000")
    assert first.post_run_remaining_budget_usd == Decimal("94.967000")
    assert first.plan_hash == second.plan_hash


@pytest.mark.asyncio
async def test_success_persists_normalized_evidence_and_reconciles_pages(tmp_path: Path) -> None:
    """Catch dropping response evidence or treating a document request as page requests."""
    document = _document(tmp_path, "a", 1)
    snapshot = CorpusSnapshot("snapshot-1", (document,))
    repository = MemoryParseRepository()
    gateway = RecordingGateway([_response(1)])
    planner = _pipeline(snapshot, gateway, repository)
    plan = await planner.plan_corpus("snapshot-1", "standard")
    pipeline = _pipeline(
        snapshot,
        gateway,
        repository,
        authorization=ParseAuthorization(plan.plan_hash),
    )

    summary = await pipeline.parse_corpus("snapshot-1", "standard")
    checkpoint = await repository.get(
        "snapshot-1", document.sha256, "standard", "document-parse", "2026-08-01"
    )

    assert summary.succeeded_documents == 1
    assert summary.succeeded_pages == 1
    assert len(gateway.requests) == 1
    assert gateway.requests[0].billable_pages == 1
    assert checkpoint is not None
    assert checkpoint.provider_model_id == "document-parse"
    assert checkpoint.provider_model_version == "2026-08-01"
    assert checkpoint.markdown == "ok"
    assert checkpoint.html == "<p>ok</p>"
    assert checkpoint.elements[0]["category"] == "paragraph"
    assert checkpoint.page_mappings == (
        {
            "page": 1,
            "source_page": 1,
            "element_count": 1,
            "element_indexes": [0],
            "provider_page_metadata": {"page": 1, "source_page": 1},
        },
    )
    assert checkpoint.raw_response_hash is not None
    assert checkpoint.latency_ms >= 0
    assert checkpoint.cost_usd == Decimal("0.011000")


@pytest.mark.asyncio
async def test_real_provider_schema_derives_pages_from_elements(tmp_path: Path) -> None:
    """Catch requiring invented top-level pages/usage fields from provider output."""
    document = _document(tmp_path, "a", 2)
    snapshot = CorpusSnapshot("snapshot-1", (document,))
    repository = MemoryParseRepository()
    gateway = RecordingGateway([_provider_response(2)])
    plan = await _pipeline(snapshot, gateway, repository).plan_corpus("snapshot-1", "standard")
    pipeline = _pipeline(
        snapshot, gateway, repository, authorization=ParseAuthorization(plan.plan_hash)
    )

    summary = await pipeline.parse_corpus("snapshot-1", "standard")
    checkpoint = await repository.get(
        "snapshot-1", document.sha256, "standard", "document-parse", "2026-08-01"
    )

    assert summary.failed_documents == 0
    assert checkpoint is not None
    assert checkpoint.provider_model_version == "2026-08-01"
    assert checkpoint.page_mappings == (
        {"page": 1, "source_page": 1, "element_count": 1, "element_indexes": [0]},
        {"page": 2, "source_page": 2, "element_count": 1, "element_indexes": [1]},
    )
    assert checkpoint.elements[1]["category"] == "paragraph"


@pytest.mark.asyncio
async def test_provider_schema_derives_content_from_elements(tmp_path: Path) -> None:
    """Catch discarding element content when aggregate fields are absent."""
    document = _document(tmp_path, "a", 1)
    snapshot = CorpusSnapshot("snapshot-1", (document,))
    repository = MemoryParseRepository()
    raw = {
        "model": "2026-08-01",
        "elements": [
            {
                "page_number": 1,
                "category": "paragraph",
                "content": {"markdown": "element md", "html": "<p>element</p>"},
            }
        ],
    }
    gateway = RecordingGateway([raw])
    plan = await _pipeline(snapshot, gateway, repository).plan_corpus("snapshot-1", "standard")
    summary = await _pipeline(
        snapshot, gateway, repository, authorization=ParseAuthorization(plan.plan_hash)
    ).parse_corpus("snapshot-1", "standard")
    checkpoint = await repository.get(
        "snapshot-1", document.sha256, "standard", "document-parse", "2026-08-01"
    )
    assert summary.failed_documents == 0
    assert checkpoint is not None
    assert checkpoint.markdown == "element md"
    assert checkpoint.html == "<p>element</p>"


@pytest.mark.asyncio
async def test_sparse_elements_map_every_manifest_page_without_fabricating_content(
    tmp_path: Path,
) -> None:
    """Catch treating an element-free middle page as a provider page-set failure."""
    document = _document(tmp_path, "a", 3)
    snapshot = CorpusSnapshot("snapshot-1", (document,))
    repository = MemoryParseRepository()
    raw = {
        "model": "2026-08-01",
        "elements": [
            {"page": 1, "category": "paragraph", "content": "first"},
            {"page": 3, "category": "paragraph", "content": "third"},
        ],
        "pages": [
            {"page_number": 1, "width": 100},
            {"page_number": 3, "width": 120},
        ],
        "usage": {"billable_pages": 3},
    }
    gateway = RecordingGateway([raw])
    plan = await _pipeline(snapshot, gateway, repository).plan_corpus("snapshot-1", "standard")
    summary = await _pipeline(
        snapshot, gateway, repository, authorization=ParseAuthorization(plan.plan_hash)
    ).parse_corpus("snapshot-1", "standard")
    checkpoint = await repository.get(
        "snapshot-1", document.sha256, "standard", "document-parse", "2026-08-01"
    )

    assert summary.failed_documents == 0
    assert checkpoint is not None
    assert checkpoint.page_mappings == (
        {
            "page": 1,
            "source_page": 1,
            "element_count": 1,
            "element_indexes": [0],
            "provider_page_metadata": {"page_number": 1, "width": 100},
        },
        {
            "page": 2,
            "source_page": 2,
            "element_count": 0,
            "element_indexes": [],
        },
        {
            "page": 3,
            "source_page": 3,
            "element_count": 1,
            "element_indexes": [1],
            "provider_page_metadata": {"page_number": 3, "width": 120},
        },
    )


@pytest.mark.asyncio
async def test_entirely_blank_document_maps_all_manifest_pages_empty(tmp_path: Path) -> None:
    """Catch rejecting a valid all-blank response with an authoritative billable count."""
    document = _document(tmp_path, "a", 2)
    snapshot = CorpusSnapshot("snapshot-1", (document,))
    repository = MemoryParseRepository()
    raw = {
        "model": "2026-08-01",
        "usage": {"billable_pages": 2},
    }
    gateway = RecordingGateway([raw])
    plan = await _pipeline(snapshot, gateway, repository).plan_corpus("snapshot-1", "standard")
    summary = await _pipeline(
        snapshot, gateway, repository, authorization=ParseAuthorization(plan.plan_hash)
    ).parse_corpus("snapshot-1", "standard")
    checkpoint = await repository.get(
        "snapshot-1", document.sha256, "standard", "document-parse", "2026-08-01"
    )

    assert summary.failed_documents == 0
    assert checkpoint is not None
    assert checkpoint.markdown == ""
    assert checkpoint.html == ""
    assert checkpoint.page_mappings == (
        {"page": 1, "source_page": 1, "element_count": 0, "element_indexes": []},
        {"page": 2, "source_page": 2, "element_count": 0, "element_indexes": []},
    )


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (
            {
                "model": "2026-08-01",
                "elements": [{"page": 4, "category": "paragraph", "content": "outside"}],
                "usage": {"billable_pages": 3},
            },
            "outside",
        ),
        (
            {
                "model": "2026-08-01",
                "elements": [{"page": 1, "category": "paragraph", "content": "first"}],
                "usage": {"billable_pages": 2},
            },
            "billable page count",
        ),
    ],
)
@pytest.mark.asyncio
async def test_provider_page_drift_is_reconciliation_required(
    tmp_path: Path, raw: dict[str, object], error: str
) -> None:
    """Catch accepting out-of-range elements or a provider billable-count mismatch."""
    document = _document(tmp_path, "a", 3)
    snapshot = CorpusSnapshot("snapshot-1", (document,))
    repository = MemoryParseRepository()
    gateway = RecordingGateway([raw])
    plan = await _pipeline(snapshot, gateway, repository).plan_corpus("snapshot-1", "standard")
    summary = await _pipeline(
        snapshot, gateway, repository, authorization=ParseAuthorization(plan.plan_hash)
    ).parse_corpus("snapshot-1", "standard")
    checkpoint = await repository.get(
        "snapshot-1", document.sha256, "standard", "document-parse", "2026-08-01"
    )

    assert summary.failed_documents == 1
    assert error in summary.failures[0].error
    assert checkpoint is not None
    assert checkpoint.status == "reconciliation_required"
    assert checkpoint.raw_response == raw


@pytest.mark.asyncio
async def test_paid_schema_failure_preserves_raw_response_and_cost(tmp_path: Path) -> None:
    """Catch replacing paid malformed-response evidence with an empty zero-cost failure."""
    document = _document(tmp_path, "a", 1)
    snapshot = CorpusSnapshot("snapshot-1", (document,))
    repository = MemoryParseRepository()
    raw = {"model": "unexpected", "elements": [{"page": 1, "content": "text"}]}
    gateway = RecordingGateway([raw])
    plan = await _pipeline(snapshot, gateway, repository).plan_corpus("snapshot-1", "standard")
    pipeline = _pipeline(
        snapshot, gateway, repository, authorization=ParseAuthorization(plan.plan_hash)
    )

    summary = await pipeline.parse_corpus("snapshot-1", "standard")
    checkpoint = await repository.get(
        "snapshot-1", document.sha256, "standard", "document-parse", "unexpected"
    )

    assert summary.failed_documents == 1
    assert checkpoint is not None
    assert checkpoint.status == "reconciliation_required"
    assert checkpoint.raw_response == raw
    assert checkpoint.raw_response_hash is not None
    assert checkpoint.correlation_id == "correlation-test"
    assert checkpoint.provider_model_version == "unexpected"
    assert checkpoint.cost_usd == Decimal("0.011000")


@pytest.mark.asyncio
async def test_paid_model_alias_can_be_reconciled_without_another_provider_call(
    tmp_path: Path,
) -> None:
    """Catch re-billing valid raw evidence after an alias resolves to a concrete model."""
    document = _document(tmp_path, "a", 1)
    snapshot = CorpusSnapshot("snapshot-1", (document,))
    repository = MemoryParseRepository()
    gateway = RecordingGateway([_provider_response(1, model="document-parse-260630")])
    plan = await _pipeline(snapshot, gateway, repository).plan_corpus("snapshot-1", "standard")
    failed = await _pipeline(
        snapshot,
        gateway,
        repository,
        authorization=ParseAuthorization(plan.plan_hash),
    ).parse_corpus("snapshot-1", "standard")
    assert failed.failed_documents == 1

    reconciled = await _pipeline(
        snapshot,
        gateway,
        repository,
        model_version="document-parse-260630",
    ).reconcile_paid_checkpoint("snapshot-1", document.document_id, "standard")

    assert len(gateway.requests) == 1
    assert reconciled.status == "succeeded"
    assert reconciled.provider_model_version == "document-parse-260630"
    assert reconciled.cache_integrity_ok


@pytest.mark.asyncio
async def test_execute_rejects_wrong_plan_hash_and_page_count(tmp_path: Path) -> None:
    """Catch stale approval and provider/local page-set drift."""
    document = _document(tmp_path, "a", 2)
    snapshot = CorpusSnapshot("snapshot-1", (document,))
    repository = MemoryParseRepository()
    gateway = RecordingGateway([_response(1)])
    pipeline = _pipeline(
        snapshot,
        gateway,
        repository,
        authorization=ParseAuthorization("wrong"),
    )
    with pytest.raises(ParseIntegrityError, match="plan hash"):
        await pipeline.parse_corpus("snapshot-1", "standard")
    assert gateway.requests == []

    plan = await _pipeline(snapshot, gateway, repository).plan_corpus("snapshot-1", "standard")
    pipeline = _pipeline(
        snapshot,
        gateway,
        repository,
        authorization=ParseAuthorization(plan.plan_hash),
    )
    summary = await pipeline.parse_corpus("snapshot-1", "standard")
    assert summary.failed_documents == 1
    assert "page count" in summary.failures[0].error


def test_cli_paid_execution_requires_all_independent_live_gates() -> None:
    """Catch a plan confirmation alone enabling paid execution."""
    assert execution_blockers(
        execute=True,
        confirm_plan="abc",
        live_enabled=False,
        api_key_present=True,
    ) == ("RUN_LIVE_UPSTAGE_TESTS=1 is required",)
    assert execution_blockers(
        execute=True,
        confirm_plan=None,
        live_enabled=True,
        api_key_present=False,
    ) == ("--confirm-plan is required", "UPSTAGE_API_KEY is required")
