"""Contracts for conservative, deterministic benchmark validation."""

from __future__ import annotations

from collections import Counter

import pytest

from ragbench.benchmark.generation import (
    DEFAULT_QUOTAS,
    Difficulty,
    EvidenceSpan,
    GeneratorMetadata,
    QuestionCandidate,
    QuestionType,
    SourceUnit,
    SourceWindow,
    ValidationDecision,
    ValidationStatus,
)
from ragbench.benchmark.validation import (
    CompletionLevel,
    ValidationConfig,
    completion_level,
    quota_deficits,
    report_payload,
    validate_candidates,
)


def _metadata(candidate_id: str) -> GeneratorMetadata:
    return GeneratorMetadata(
        model_id="solar-pro3",
        prompt_version="benchmark-v1",
        plan_hash="b" * 64,
        batch_id="batch-1",
        source_window_hash="a" * 64,
        reasoning_kind=f"lookup:{candidate_id}",
    )


def _candidate(
    candidate_id: str,
    *,
    question: str = "2025년 매출은 얼마인가?",
    answer: str = "123원",
    evidence: str = "2025년 매출은 123원이다.",
    page: int = 1,
    kind: QuestionType = QuestionType.NUMERIC_TABLE,
) -> QuestionCandidate:
    return QuestionCandidate(
        candidate_id=candidate_id,
        question=question,
        gold_answer=answer,
        evidence_spans=(
            EvidenceSpan(text=evidence, document_id="doc-1", page=page, chunk_id="c1"),
        ),
        question_type=kind,
        difficulty=Difficulty.MEDIUM,
        answerable=True,
        generator=_metadata(candidate_id),
        validation=ValidationStatus(decision=ValidationDecision.UNVALIDATED),
    )


def _corpus() -> tuple[SourceWindow, ...]:
    return (
        SourceWindow(
            window_id="w1",
            document_id="doc-1",
            document_title="보고서",
            page_start=1,
            page_end=1,
            chunk_ids=("c1",),
            content="2025년 매출은 123원이다. 영업이익은 45원이다.",
            source_units=(
                SourceUnit(
                    page=1,
                    chunk_id="c1",
                    content="2025년 매출은 123원이다. 영업이익은 45원이다.",
                ),
            ),
        ),
        SourceWindow(
            window_id="w2",
            document_id="doc-1",
            document_title="보고서",
            page_start=2,
            page_end=2,
            chunk_ids=("c2",),
            content="직원 수는 30명이다.",
            source_units=(SourceUnit(page=2, chunk_id="c2", content="직원 수는 30명이다."),),
        ),
    )


@pytest.mark.parametrize(
    ("candidate", "rule"),
    [
        (_candidate("missing", evidence="문서에 없는 근거"), "evidence_not_found"),
        (_candidate("number", answer="999원"), "numeric_mismatch"),
        (_candidate("page", page=99), "impossible_page"),
        (
            _candidate("leak", question="정답 123원은 무엇을 뜻하는가?"),
            "answer_leaked_in_question",
        ),
    ],
)
def test_validation_rejects_untraceable_or_contaminated_items(
    candidate: QuestionCandidate, rule: str
) -> None:
    """Catch acceptance when evidence, numbers, pages, or question leakage are invalid."""
    report = validate_candidates((candidate,), _corpus())

    assert report.accepted_count == 0
    assert report.rejected_count == 1
    assert rule in report.items[0].validation.rule_codes


def test_validation_accepts_whitespace_fuzzy_verbatim_evidence() -> None:
    """Catch false rejection caused only by normalized whitespace around a source span."""
    candidate = _candidate("fuzzy", evidence="2025년   매출은 123원이다")

    report = validate_candidates((candidate,), _corpus())

    assert report.accepted_count == 1
    assert report.items[0].validation.decision is ValidationDecision.ACCEPTED


def test_validation_groups_exact_and_offline_semantic_like_duplicates() -> None:
    """Catch paraphrase families surviving because only byte equality is checked."""
    candidates = (
        _candidate("first"),
        _candidate("exact", question=" 2025년 매출은 얼마인가? "),
        _candidate("near", question="2025년 매출은 얼마인가요?"),
    )

    report = validate_candidates(candidates, _corpus())

    assert report.accepted_count == 1
    assert Counter(
        rule for item in report.items for rule in item.validation.rule_codes
    )["duplicate_question"] == 2
    assert report.duplicate_groups == (("first", "exact", "near"),)


def test_unanswerable_absence_and_contamination_are_checked_across_document() -> None:
    """Catch an unanswerable whose transformed fact appears on another source page."""
    candidate = QuestionCandidate(
        candidate_id="negative",
        question="직원 수는 30명인가?",
        gold_answer=None,
        evidence_spans=(),
        question_type=QuestionType.UNANSWERABLE,
        difficulty=Difficulty.HARD,
        answerable=False,
        asserted_absent_facts=("직원 수는 30명",),
        unanswerable_transform={
            "target_document_id": "doc-1",
            "original_fact": "매출은 123원",
            "transformed_fact": "직원 수는 30명",
        },
        generator=_metadata("negative"),
        validation=ValidationStatus(decision=ValidationDecision.UNVALIDATED),
    )

    report = validate_candidates((candidate,), _corpus(), contamination_terms=("비밀셋",))

    assert "asserted_absent_fact_present" in report.items[0].validation.rule_codes


def test_quota_document_caps_and_report_distributions_are_deterministic() -> None:
    """Catch accepted output drifting beyond configured sampling limits or opaque reports."""
    candidates = tuple(
        _candidate(
            f"c{index}",
            question=f"2025년 매출 항목 {index}은 얼마인가?",
            kind=QuestionType.FACT,
        )
        for index in range(3)
    )
    config = ValidationConfig(
        quotas={QuestionType.FACT: 2},
        per_document_cap=1,
        duplicate_similarity_threshold=1.0,
    )

    report = validate_candidates(candidates, _corpus(), config=config)

    assert report.accepted_count == 1
    assert report.rejected_count == 2
    assert report.rejection_counts == {"per_document_cap_exceeded": 2}
    assert report.type_distribution == {"fact": 1}
    assert report.document_distribution == {"doc-1": 1}
    assert report.rejection_samples == {
        "per_document_cap_exceeded": ("c1", "c2")
    }


def test_completion_floor_distinguishes_normal_dod_from_emergency_only() -> None:
    """Catch the emergency 800-item escape hatch being reported as normal completion."""
    exact = {kind.value: count for kind, count in DEFAULT_QUOTAS.items()}
    normal = {
        "fact": 200,
        "numeric_table": 200,
        "comparison": 167,
        "multihop": 167,
        "unanswerable": 133,
        "complex_summary": 133,
    }
    assert completion_level(exact) is CompletionLevel.TARGET
    assert completion_level(normal) is CompletionLevel.NORMAL_FLOOR
    skewed = dict(normal)
    skewed["fact"] += skewed.pop("unanswerable")
    assert completion_level(skewed) is CompletionLevel.EMERGENCY_ONLY
    assert completion_level({"fact": 799}) is CompletionLevel.INSUFFICIENT
    assert quota_deficits(normal, normal_scope=True) == {}


def test_report_payload_is_stable_and_contains_rejection_samples() -> None:
    """Catch a validation artifact that omits why representative rows were rejected."""
    report = validate_candidates(
        (_candidate("bad", answer="999원"),),
        _corpus(),
    )

    assert report_payload(report) == {
        "accepted_count": 0,
        "completion_level": "insufficient",
        "difficulty_distribution": {},
        "document_distribution": {},
        "duplicate_groups": [],
        "quota_deficits": {
            "comparison": 250,
            "complex_summary": 200,
            "fact": 300,
            "multihop": 250,
            "numeric_table": 300,
            "unanswerable": 200,
        },
        "rejected_count": 1,
        "rejection_counts": {"numeric_mismatch": 1},
        "rejection_samples": {"numeric_mismatch": ["bad"]},
        "type_distribution": {},
        "validation_run_hash": report.validation_run_hash,
        "validation_config": report.validation_config,
    }


def test_exact_page_chunk_unit_and_complex_support_fail_closed() -> None:
    """Catch wrong-unit provenance and hallucinated summaries with one shared token."""
    multi = SourceWindow(
        window_id="multi",
        document_id="doc-1",
        document_title="보고서",
        page_start=1,
        page_end=2,
        chunk_ids=("c1", "c2"),
        content="1쪽 실제 근거\n2쪽 다른 회사 자료",
        source_units=(
            SourceUnit(page=1, chunk_id="c1", content="1쪽 실제 근거"),
            SourceUnit(page=2, chunk_id="c2", content="2쪽 다른 회사 자료"),
        ),
    )
    wrong_unit = _candidate("wrong", evidence="1쪽 실제 근거", page=2).model_copy(
        update={
            "evidence_spans": (
                EvidenceSpan(text="1쪽 실제 근거", document_id="doc-1", page=2, chunk_id="c2"),
            )
        }
    )
    hallucinated = _candidate(
        "hallucinated",
        question="회사의 장기 전망을 요약하라",
        answer="회사는 화성 기지를 설립한다",
        evidence="회사는 매출을 공개했다",
        kind=QuestionType.COMPLEX_SUMMARY,
    )

    wrong_report = validate_candidates((wrong_unit,), (multi,))
    hallucinated_report = validate_candidates((hallucinated,), _corpus())
    assert "evidence_not_found" in wrong_report.items[0].validation.rule_codes
    assert "answer_not_supported" in hallucinated_report.items[0].validation.rule_codes


def test_validation_identity_changes_with_rules_and_global_candidate_ids_are_unique() -> None:
    """Catch artifact overwrites after rule changes and ambiguous duplicate candidate IDs."""
    first = _candidate("same", question="매출 항목 A는 얼마인가?")
    second = _candidate("same", question="매출 항목 B는 얼마인가?")
    report = validate_candidates((first, second), _corpus())
    changed = validate_candidates(
        (first,), _corpus(), config=ValidationConfig(evidence_similarity_threshold=0.90)
    )

    assert "duplicate_candidate_id" in report.items[1].validation.rule_codes
    assert report.validation_run_hash != changed.validation_run_hash
