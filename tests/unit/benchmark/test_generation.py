"""Contracts for deterministic, gated synthetic benchmark generation."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from ragbench.benchmark.generation import (
    DEFAULT_QUOTAS,
    BenchmarkGenerator,
    Difficulty,
    EvidenceSpan,
    FileBatchRepository,
    GenerationAuthorization,
    GenerationConfig,
    GenerationPlanner,
    GeneratorMetadata,
    MemoryBatchRepository,
    QuestionCandidate,
    QuestionType,
    SourceUnit,
    SourceWindow,
    ValidationDecision,
    ValidationStatus,
    controlled_unanswerable,
    generation_campaign_hash,
    generation_execution_blockers,
    parse_candidate_json,
)
from ragbench.core.hashing import canonical_json_hash
from ragbench.providers.base import GenerateRequest, GenerateResponse
from ragbench.providers.upstage.pricing import PriceBook


def _windows() -> tuple[SourceWindow, ...]:
    return tuple(
        SourceWindow(
            window_id=f"doc-{doc}-p{page}",
            document_id=f"doc-{doc}",
            document_title=f"문서 {doc}",
            page_start=page,
            page_end=page,
            chunk_ids=(f"chunk-{doc}-{page}",),
            content=(
                f"문서 {doc}의 {page}페이지 근거입니다. "
                f"매출은 {doc * 100 + page}원입니다."
            ),
            source_units=(
                SourceUnit(
                    page=page,
                    chunk_id=f"chunk-{doc}-{page}",
                    content=(
                        f"문서 {doc}의 {page}페이지 근거입니다. "
                        f"매출은 {doc * 100 + page}원입니다."
                    ),
                ),
            ),
        )
        for doc in range(1, 7)
        for page in range(1, 11)
    )


def _candidate(*, plan_hash: str, batch_id: str = "batch-0000") -> QuestionCandidate:
    return QuestionCandidate(
        candidate_id="candidate-1",
        question="첫 문서의 매출은 얼마인가?",
        gold_answer="  101원  ",
        evidence_spans=(
            EvidenceSpan(
                text="매출은 101원입니다.",
                document_id="doc-1",
                page=1,
                chunk_id="chunk-1-1",
            ),
        ),
        question_type=QuestionType.NUMERIC_TABLE,
        difficulty=Difficulty.EASY,
        answerable=True,
        generator=GeneratorMetadata(
            model_id="solar-pro3",
            prompt_version="benchmark-v1",
            plan_hash=plan_hash,
            batch_id=batch_id,
            source_window_hash="a" * 64,
            reasoning_kind="single-source lookup",
        ),
        validation=ValidationStatus(decision=ValidationDecision.UNVALIDATED),
    )


def test_question_candidate_is_strict_frozen_and_normalizes_gold_answer() -> None:
    """Catch mutable records, ignored provider fields, or unstable answer whitespace."""
    candidate = _candidate(plan_hash="b" * 64)

    assert candidate.gold_answer == "101원"
    with pytest.raises(ValidationError):
        candidate.question = "변경"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Extra inputs"):
        QuestionCandidate.model_validate({**candidate.model_dump(), "unexpected": True})


def test_parse_candidate_json_rejects_invalid_json_and_missing_evidence() -> None:
    """Catch malformed provider JSON and answerable rows with no traceable evidence."""
    with pytest.raises(ValueError, match="valid candidate JSON"):
        parse_candidate_json("not-json")

    payload = _candidate(plan_hash="b" * 64).model_dump(mode="json")
    payload["evidence_spans"] = []
    with pytest.raises(ValueError, match="valid candidate JSON"):
        parse_candidate_json(json.dumps(payload, ensure_ascii=False))


def test_malformed_unanswerable_candidate_is_rejected() -> None:
    """Catch unanswerables that retain a gold answer or omit absence assertions."""
    payload = _candidate(plan_hash="b" * 64).model_dump()
    payload.update(
        {
            "question_type": QuestionType.UNANSWERABLE,
            "answerable": False,
            "evidence_spans": (),
            "asserted_absent_facts": (),
        }
    )

    with pytest.raises(ValidationError, match="unanswerable"):
        QuestionCandidate.model_validate(payload)


def test_planner_hits_exact_1500_quotas_and_balances_documents_and_pages() -> None:
    """Catch a planner that overproduces easy rows from the first document/page."""
    planner = GenerationPlanner(
        GenerationConfig(per_document_cap=260, per_page_cap=30, batch_size=12)
    )

    plan = planner.plan(_windows(), corpus_snapshot_id="corpus-v1", model_id="solar-pro3")

    counts = {kind: 0 for kind in DEFAULT_QUOTAS}
    doc_counts: dict[str, int] = {}
    page_counts: dict[tuple[str, int], int] = {}
    for job in plan.jobs:
        counts[job.question_type] += 1
        doc_counts[job.window.document_id] = doc_counts.get(job.window.document_id, 0) + 1
        page_key = (job.window.document_id, job.window.page_start)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1

    assert len(plan.jobs) == 1500
    assert counts == DEFAULT_QUOTAS
    assert max(doc_counts.values()) - min(doc_counts.values()) <= 1
    assert max(page_counts.values()) <= 30
    assert {job.window.page_start for job in plan.jobs} == set(range(1, 11))
    assert plan.plan_hash == planner.plan(
        _windows(), corpus_snapshot_id="corpus-v1", model_id="solar-pro3"
    ).plan_hash


def test_planner_rejects_unbounded_or_insufficient_sources() -> None:
    """Catch paid planning that silently truncates oversized windows or violates caps."""
    oversized = _windows()[0].model_copy(update={"content": "가" * 20_001})
    planner = GenerationPlanner(GenerationConfig(source_window_max_chars=20_000))

    with pytest.raises(ValueError, match="bounded"):
        planner.plan((oversized,), corpus_snapshot_id="corpus-v1", model_id="solar-pro3")
    with pytest.raises(ValueError, match="capacity"):
        planner.plan(_windows()[:2], corpus_snapshot_id="corpus-v1", model_id="solar-pro3")


def test_execution_requires_live_paid_plan_price_and_budget_gates() -> None:
    """Catch one-switch provider execution, stale pricing, plan drift, or overspend."""
    plan = GenerationPlanner(
        GenerationConfig(per_document_cap=260, per_page_cap=30)
    ).plan(_windows(), corpus_snapshot_id="corpus-v1", model_id="solar-pro3")
    fresh = PriceBook(
        {
            "schema_version": "prices-v1",
            "verified_at": "2026-08-14T00:00:00Z",
            "vat_excluded": True,
            "models": {
                "solar-pro3": {
                    "generation": {
                        "input_usd_per_million": "0.15",
                        "output_usd_per_million": "0.60",
                    }
                }
            },
        }
    )

    blockers = generation_execution_blockers(
        plan,
        authorization=GenerationAuthorization(
            execute=True,
            confirm_paid=False,
            live_enabled=True,
            confirmed_plan_hash=plan.plan_hash,
        ),
        price_book=fresh,
        projected_cost_usd=Decimal("2"),
        remaining_budget_usd=Decimal("10"),
        now=datetime(2026, 8, 14, 1, tzinfo=UTC),
    )
    assert blockers == ("paid generation requires explicit price confirmation",)

    authorized = GenerationAuthorization(True, True, True, plan.plan_hash)
    assert generation_execution_blockers(
        plan,
        authorization=authorized,
        price_book=fresh,
        projected_cost_usd=Decimal("2"),
        remaining_budget_usd=Decimal("10"),
        now=datetime(2026, 8, 14, 1, tzinfo=UTC),
    ) == ()
    assert "remaining budget" in generation_execution_blockers(
        plan,
        authorization=authorized,
        price_book=fresh,
        projected_cost_usd=Decimal("10"),
        remaining_budget_usd=Decimal("10"),
        now=datetime(2026, 8, 14, 1, tzinfo=UTC),
    )[0]


class _FakeGateway:
    def __init__(self, response: QuestionCandidate) -> None:
        self.response = response
        self.requests: list[GenerateRequest] = []

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.requests.append(request)
        return GenerateResponse(
            content=json.dumps(
                {"candidates": [self.response.model_dump(mode="json")]}, ensure_ascii=False
            ),
            raw_response={"usage": {"prompt_tokens": 20, "completion_tokens": 20}},
            correlation_id="corr-1",
            cache_hit=False,
        )


@pytest.mark.asyncio
async def test_generation_uses_gateway_and_resumes_completed_batches() -> None:
    """Catch direct provider bypasses and repeat billing for a completed immutable batch."""
    config = GenerationConfig(
        quotas={QuestionType.FACT: 1},
        per_document_cap=1,
        per_page_cap=1,
        batch_size=1,
    )
    plan = GenerationPlanner(config).plan(
        (_windows()[0],), corpus_snapshot_id="corpus-v1", model_id="solar-pro3"
    )
    authoritative_window_hash = canonical_json_hash(
        plan.batches[0].jobs[0].window.model_dump(mode="json")
    )
    candidate = _candidate(plan_hash=plan.plan_hash).model_copy(
        update={
            "question_type": QuestionType.FACT,
            "generator": _candidate(plan_hash=plan.plan_hash).generator.model_copy(
                update={"source_window_hash": authoritative_window_hash}
            ),
        }
    )
    gateway = _FakeGateway(candidate)
    repository = MemoryBatchRepository()
    generator = BenchmarkGenerator(gateway, repository, config=config)

    first = await generator.generate_batch(plan, plan.batches[0])
    second = await generator.generate_batch(plan, plan.batches[0])

    assert first == second
    assert first[0].question == candidate.question
    assert len(gateway.requests) == 1
    assert gateway.requests[0].context == ()
    assert "문서 1의 1페이지 근거" in gateway.requests[0].prompt
    assert gateway.requests[0].max_output_tokens == config.max_output_tokens
    assert first[0].generator.correlation_id == "corr-1"
    assert first[0].generator.cache_hit is False


@pytest.mark.asyncio
async def test_generation_rejects_evidence_outside_the_assigned_source_window() -> None:
    """Catch a model citing corpus evidence that was not present in its bounded prompt job."""
    config = GenerationConfig(
        quotas={QuestionType.FACT: 1}, per_document_cap=1, per_page_cap=1, batch_size=1
    )
    plan = GenerationPlanner(config).plan(
        (_windows()[0],), corpus_snapshot_id="corpus-v1", model_id="solar-pro3"
    )
    candidate = _candidate(plan_hash=plan.plan_hash).model_copy(
        update={
            "question_type": QuestionType.FACT,
            "evidence_spans": (
                EvidenceSpan(
                    text="다른 문서 근거",
                    document_id="doc-2",
                    page=1,
                    chunk_id="chunk-2-1",
                ),
            ),
        }
    )

    generator = BenchmarkGenerator(_FakeGateway(candidate), MemoryBatchRepository(), config=config)
    with pytest.raises(ValueError, match="assigned source window"):
        await generator.generate_batch(plan, plan.batches[0])


@pytest.mark.asyncio
async def test_generation_revalidates_server_identity_when_resuming_checkpoint() -> None:
    """Catch a hash-valid legacy checkpoint bypassing current server-owned provenance rules."""
    config = GenerationConfig(
        quotas={QuestionType.FACT: 1}, per_document_cap=1, per_page_cap=1, batch_size=1
    )
    plan = GenerationPlanner(config).plan(
        (_windows()[0],), corpus_snapshot_id="corpus-v1", model_id="solar-pro3"
    )
    repository = MemoryBatchRepository()
    await repository.save_candidates(
        plan.plan_hash,
        plan.batches[0].batch_id,
        (
            _candidate(plan_hash=plan.plan_hash).model_copy(
                update={"question_type": QuestionType.FACT}
            ),
        ),
    )
    generator = BenchmarkGenerator(
        _FakeGateway(_candidate(plan_hash=plan.plan_hash)), repository, config=config
    )

    with pytest.raises(RuntimeError, match="current plan provenance"):
        await generator.generate_batch(plan, plan.batches[0])


def test_controlled_unanswerable_requires_transformed_fact_to_be_absent() -> None:
    """Catch negative examples whose supposedly absent fact actually exists in the snapshot."""
    window = _windows()[0]
    candidate = controlled_unanswerable(
        question="첫 문서의 매출은 999원인가?",
        original_fact="매출은 101원",
        asserted_absent_fact="매출은 999원",
        document_windows=(window,),
        metadata=GeneratorMetadata(
            model_id="offline-transform",
            prompt_version="controlled-v1",
            plan_hash="b" * 64,
            batch_id="batch-1",
            source_window_hash="a" * 64,
            reasoning_kind="controlled fact substitution",
        ),
    )
    assert candidate.answerable is False
    assert candidate.evidence_spans == ()
    assert candidate.unanswerable_transform is not None
    assert candidate.unanswerable_transform.target_document_id == "doc-1"

    with pytest.raises(ValueError, match="present"):
        controlled_unanswerable(
            question="매출은 101원인가?",
            original_fact="매출은 101원",
            asserted_absent_fact="매출은 101원",
            document_windows=(window,),
            metadata=candidate.generator,
        )

    with pytest.raises(ValueError, match="anchor"):
        controlled_unanswerable(
            question="CEO는 홍길동인가?",
            original_fact="매출은 101원",
            asserted_absent_fact="CEO는 홍길동",
            document_windows=(window,),
            metadata=candidate.generator,
        )


def test_replacement_plan_refills_only_quota_deficits_with_new_global_ids() -> None:
    """Catch stopping after one rejected batch or reusing model-controlled IDs across attempts."""
    config = GenerationConfig(
        quotas={QuestionType.FACT: 2, QuestionType.UNANSWERABLE: 1},
        per_document_cap=10,
        per_page_cap=10,
        batch_size=2,
    )
    planner = GenerationPlanner(config)
    initial = planner.plan(
        _windows()[:3], corpus_snapshot_id="corpus-v1", model_id="solar-pro3"
    )
    replacement = planner.plan_replacements(
        initial,
        accepted_counts={QuestionType.FACT: 1, QuestionType.UNANSWERABLE: 1},
        attempt=1,
    )

    assert tuple(job.question_type for job in replacement.jobs) == (QuestionType.FACT,)
    assert replacement.plan_hash != initial.plan_hash
    assert replacement.batches[0].batch_id.startswith("replacement-0001-")


def test_replacement_plans_preserve_campaign_caps_and_campaign_confirmation_identity() -> None:
    """Catch replacement calls escaping approved rounds or resetting page/document caps."""
    config = GenerationConfig(
        quotas={QuestionType.FACT: 1},
        per_document_cap=2,
        per_page_cap=2,
        batch_size=1,
    )
    planner = GenerationPlanner(config)
    initial = planner.plan(
        (_windows()[0],), corpus_snapshot_id="corpus-v1", model_id="solar-pro3"
    )
    replacement = planner.plan_replacements(
        initial,
        accepted_counts={},
        attempt=1,
        prior_plans=(initial,),
    )
    with pytest.raises(ValueError, match="capacity"):
        planner.plan_replacements(
            initial,
            accepted_counts={},
            attempt=2,
            prior_plans=(initial, replacement),
        )

    first = generation_campaign_hash(
        initial, max_replacement_rounds=1, allow_reduced_scope=False
    )
    second = generation_campaign_hash(
        initial, max_replacement_rounds=2, allow_reduced_scope=False
    )
    assert first != second


@pytest.mark.asyncio
async def test_file_batch_repository_survives_restart_and_detects_corruption(
    tmp_path: Path,
) -> None:
    """Catch non-resumable paid batches or reuse of a corrupted checkpoint."""
    repository = FileBatchRepository(tmp_path / "batches")
    plan_hash = "b" * 64
    candidate = _candidate(plan_hash=plan_hash)
    await repository.save_candidates(plan_hash, "batch-0000", (candidate,))

    restarted = FileBatchRepository(tmp_path / "batches")
    stored = await restarted.get(plan_hash, "batch-0000")
    assert stored is not None
    assert stored.candidates == (candidate,)

    checkpoint = next((tmp_path / "batches").glob("*.json"))
    checkpoint.write_text(checkpoint.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity"):
        await restarted.get(plan_hash, "batch-0000")


def test_generate_benchmark_cli_is_dry_run_by_default(tmp_path: Path) -> None:
    """Catch the ordinary CLI path constructing a paid gateway or executing generation."""
    windows_path = tmp_path / "windows.jsonl"
    windows_path.write_text(
        "\n".join(window.model_dump_json() for window in _windows()) + "\n",
        encoding="utf-8",
    )
    script = Path(__file__).parents[3] / "scripts" / "generate_benchmark.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(windows_path),
            "--corpus-snapshot-id",
            "corpus-v1",
            "--model-id",
            "solar-pro3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {
        "batch_count": 125,
        "candidate_target": 1500,
        "campaign_hash": payload["campaign_hash"],
        "live_executed": False,
        "mode": "dry-run",
        "plan_hash": payload["plan_hash"],
    }
    assert len(payload["plan_hash"]) == 64
    assert len(payload["campaign_hash"]) == 64
