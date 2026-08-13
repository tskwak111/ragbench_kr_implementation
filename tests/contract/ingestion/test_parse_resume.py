"""Offline contract tests for parser checkpoint recovery and dual-mode parity."""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
import respx

from ragbench.core.money import BudgetGuard, MemoryBudgetRepository
from ragbench.ingestion.manifest import DocumentRecord
from ragbench.ingestion.parser import (
    CorpusSnapshot,
    MemoryParseRepository,
    ParseAuthorization,
    ParseCheckpoint,
    ParseIntegrityError,
    ParserPipeline,
)
from ragbench.providers.base import ParsedDocument, ParseRequest
from ragbench.providers.upstage.client import MemoryProviderStore, UpstageGateway
from ragbench.providers.upstage.pricing import PriceBook

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _doc(tmp_path: Path, name: str) -> DocumentRecord:
    path = tmp_path / f"{name}.pdf"
    content = name.encode()
    path.write_bytes(content)
    return DocumentRecord(
        document_id=name,
        title=name,
        organization=name,
        year=2025,
        document_type="report",
        language="ko",
        sector="corporate",
        content_stratum="text_heavy",
        template_family=name,
        license="reviewed",
        redistribution_status="nonredistributable",
        local_path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        page_count=1,
        inclusion_rationale="fixture",
    )


def _raw(label: str) -> dict[str, object]:
    return {
        "model_version": "v1",
        "content": {"markdown": label, "html": f"<p>{label}</p>"},
        "elements": [],
        "pages": [{"page": 1, "source_page": 1}],
        "usage": {"pages": 1},
    }


class Gateway:
    def __init__(self, outcomes: list[dict[str, object] | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    async def parse(self, request: ParseRequest) -> ParsedDocument:
        self.calls.append(request.document_sha256)
        await asyncio.sleep(0)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return ParsedDocument(outcome)


def _book() -> PriceBook:
    return PriceBook(
        {
            "schema_version": "v1",
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


def _pipeline(
    snapshot: CorpusSnapshot,
    repository: MemoryParseRepository,
    gateway: Gateway,
    confirmation: str | None = None,
) -> ParserPipeline:
    return ParserPipeline(
        snapshots={snapshot.snapshot_id: snapshot},
        gateway=gateway,
        repository=repository,
        price_book=_book(),
        model_id="document-parse",
        model_version="v1",
        hard_budget_usd=Decimal("10"),
        settled_cost_usd=lambda: Decimal("0"),
        vat_buffer=Decimal("0.10"),
        now=lambda: datetime(2026, 8, 14, 1, tzinfo=UTC),
        authorization=ParseAuthorization(confirmation) if confirmation else None,
    )


async def _authorized(
    snapshot: CorpusSnapshot,
    repository: MemoryParseRepository,
    gateway: Gateway,
    mode: str = "standard",
) -> ParserPipeline:
    plan = await _pipeline(snapshot, repository, gateway).plan_corpus(snapshot.snapshot_id, mode)
    return _pipeline(snapshot, repository, gateway, plan.plan_hash)


@pytest.mark.asyncio
async def test_partial_failure_is_retained_and_resume_only_retries_failure(tmp_path: Path) -> None:
    """Catch all-or-nothing batches or resume that re-bills completed documents."""
    documents = (_doc(tmp_path, "a"), _doc(tmp_path, "b"))
    snapshot = CorpusSnapshot("snapshot", documents)
    repository = MemoryParseRepository()
    gateway = Gateway([_raw("a"), RuntimeError("temporary")])
    first = await _authorized(snapshot, repository, gateway)
    summary = await first.parse_corpus("snapshot", "standard")
    assert (summary.succeeded_documents, summary.failed_documents) == (1, 1)

    gateway.outcomes.append(_raw("b"))
    resumed = await _authorized(snapshot, repository, gateway)
    second = await resumed.parse_corpus("snapshot", "standard", resume=True)
    assert (second.cache_hits, second.succeeded_documents, second.failed_documents) == (1, 2, 0)
    assert gateway.calls == [documents[0].sha256, documents[1].sha256, documents[1].sha256]


@pytest.mark.asyncio
async def test_duplicate_invocation_is_idempotent(tmp_path: Path) -> None:
    """Catch duplicate successful checkpoints or repeat provider calls."""
    document = _doc(tmp_path, "a")
    snapshot = CorpusSnapshot("snapshot", (document,))
    repository = MemoryParseRepository()
    gateway = Gateway([_raw("a")])
    first = await _authorized(snapshot, repository, gateway)
    await first.parse_corpus("snapshot", "standard")
    duplicate = await _authorized(snapshot, repository, gateway)
    summary = await duplicate.parse_corpus("snapshot", "standard")
    assert summary.cache_hits == 1
    assert len(repository.checkpoints) == 1
    assert len(gateway.calls) == 1


@pytest.mark.asyncio
async def test_concurrent_pipelines_share_checkpoint_singleflight(tmp_path: Path) -> None:
    """Catch concurrent orchestration billing the same checkpoint twice."""
    document = _doc(tmp_path, "a")
    snapshot = CorpusSnapshot("snapshot", (document,))
    repository = MemoryParseRepository()
    gateway = Gateway([_raw("a")])
    first, second = await asyncio.gather(
        _authorized(snapshot, repository, gateway),
        _authorized(snapshot, repository, gateway),
    )
    summaries = await asyncio.gather(
        first.parse_corpus("snapshot", "standard"),
        second.parse_corpus("snapshot", "standard"),
    )

    assert gateway.calls == [document.sha256]
    assert sum(summary.cache_hits for summary in summaries) == 1
    assert sum(summary.new_successes for summary in summaries) == 1
    assert repository.singleflight_lock_count == 0


@pytest.mark.asyncio
async def test_corrupted_success_checkpoint_is_rejected_and_retried(tmp_path: Path) -> None:
    """Catch trusting cached raw bytes whose hash no longer matches their checkpoint."""
    document = _doc(tmp_path, "a")
    snapshot = CorpusSnapshot("snapshot", (document,))
    repository = MemoryParseRepository()
    await repository.put(
        ParseCheckpoint.success_for_test(snapshot, document, mode="standard", raw=_raw("old"))
    )
    repository.corrupt_raw_response(snapshot.snapshot_id, document.sha256, "standard")
    gateway = Gateway([_raw("new")])
    pipeline = await _authorized(snapshot, repository, gateway)

    summary = await pipeline.parse_corpus("snapshot", "standard", resume=True)

    assert summary.retried_corrupt_checkpoints == 1
    assert gateway.calls == [document.sha256]
    saved = await repository.get("snapshot", document.sha256, "standard", "document-parse", "v1")
    assert saved is not None and saved.markdown == "new"


@pytest.mark.asyncio
async def test_resume_summary_includes_cached_success_pages(tmp_path: Path) -> None:
    """Catch summaries that make whole-corpus success rate impossible to compute."""
    documents = (_doc(tmp_path, "a"), _doc(tmp_path, "b"))
    snapshot = CorpusSnapshot("snapshot", documents)
    repository = MemoryParseRepository()
    await repository.put(
        ParseCheckpoint.success_for_test(snapshot, documents[0], mode="standard", raw=_raw("a"))
    )
    gateway = Gateway([_raw("b")])
    summary = await (await _authorized(snapshot, repository, gateway)).parse_corpus(
        "snapshot", "standard"
    )
    assert summary.cached_successes == 1
    assert summary.cached_success_pages == 1
    assert summary.new_successes == 1
    assert summary.succeeded_documents == 2
    assert summary.succeeded_pages == 2


@pytest.mark.asyncio
async def test_source_replacement_after_plan_never_reaches_gateway(tmp_path: Path) -> None:
    """Catch dispatching bytes that no longer match the approved manifest and plan."""
    document = _doc(tmp_path, "a")
    snapshot = CorpusSnapshot("snapshot", (document,))
    repository = MemoryParseRepository()

    class ExplodingGateway(Gateway):
        async def parse(self, request: ParseRequest) -> ParsedDocument:
            raise AssertionError("gateway/cache must not be consulted")

    gateway = ExplodingGateway([])
    plan = await _pipeline(snapshot, repository, gateway).plan_corpus("snapshot", "standard")
    document.local_path.write_bytes(b"replacement")
    pipeline = _pipeline(snapshot, repository, gateway, plan.plan_hash)

    replacement_plan = await pipeline.plan_corpus("snapshot", "standard")
    assert replacement_plan.plan_hash != plan.plan_hash
    with pytest.raises(ParseIntegrityError, match="plan hash"):
        await pipeline.parse_corpus("snapshot", "standard")

    assert gateway.calls == []
    checkpoint = await repository.get(
        "snapshot", document.sha256, "standard", "document-parse", "v1"
    )
    assert checkpoint is None


@pytest.mark.asyncio
async def test_enhanced_requires_identical_standard_source_and_page_set(tmp_path: Path) -> None:
    """Catch comparing modes over different document hashes or page mappings."""
    document = _doc(tmp_path, "a")
    snapshot = CorpusSnapshot("snapshot", (document,))
    repository = MemoryParseRepository()
    await repository.put(
        ParseCheckpoint.success_for_test(snapshot, document, mode="standard", raw=_raw("std"))
    )
    repository.replace_page_mappings(
        "snapshot", document.sha256, "standard", ({"page": 2, "source_page": 2},)
    )

    with pytest.raises(ParseIntegrityError, match="page set"):
        await _pipeline(snapshot, repository, Gateway([])).plan_corpus("snapshot", "enhanced")


def test_offline_migration_adds_complete_idempotent_parse_evidence_schema() -> None:
    """Catch deploying checkpoints without the evidence or uniqueness needed for resume."""
    completed = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    sql = completed.stdout.lower()
    for column in (
        "corpus_snapshot_id",
        "provider_model_version",
        "raw_response",
        "markdown",
        "html",
        "elements",
        "page_mappings",
        "latency_ms",
        "cost_usd",
        "error",
    ):
        assert column in sql
    assert "uq_parse_run_checkpoint" in sql


@pytest.mark.asyncio
@respx.mock
async def test_real_gateway_response_flows_through_standard_and_enhanced_pipeline(
    tmp_path: Path,
) -> None:
    """Catch an adapter contract that works only with a pipeline-specific fake schema."""
    document = _doc(tmp_path, "a")
    snapshot = CorpusSnapshot("snapshot", (document,))
    repository = MemoryParseRepository()
    route = respx.post("https://api.upstage.ai/v1/document-digitization").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "v1",
                "content": {"markdown": "provider", "html": "<p>provider</p>"},
                "elements": [{"page": 1, "category": "paragraph", "content": "provider"}],
            },
        )
    )
    gateway = UpstageGateway(
        api_key="offline",
        base_url="https://api.upstage.ai/v1",
        price_book=_book(),
        budget_guard=BudgetGuard(MemoryBudgetRepository(), hard_limit=Decimal("10")),
        store=MemoryProviderStore(),
    )
    try:
        for mode in ("standard", "enhanced"):
            plan = await ParserPipeline(
                snapshots={"snapshot": snapshot},
                gateway=gateway,
                repository=repository,
                price_book=_book(),
                model_id="document-parse",
                model_version="v1",
                hard_budget_usd=Decimal("10"),
                settled_cost_usd=lambda: Decimal("0"),
            ).plan_corpus("snapshot", mode)
            pipeline = ParserPipeline(
                snapshots={"snapshot": snapshot},
                gateway=gateway,
                repository=repository,
                price_book=_book(),
                model_id="document-parse",
                model_version="v1",
                hard_budget_usd=Decimal("10"),
                settled_cost_usd=lambda: Decimal("0"),
                authorization=ParseAuthorization(plan.plan_hash),
                now=lambda: datetime(2026, 8, 14, 1, tzinfo=UTC),
            )
            summary = await pipeline.parse_corpus("snapshot", mode)
            assert summary.failed_documents == 0
    finally:
        await gateway.aclose()
    assert route.call_count == 2
