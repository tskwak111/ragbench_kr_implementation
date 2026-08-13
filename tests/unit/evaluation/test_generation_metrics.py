from __future__ import annotations

import pytest

from ragbench.evaluation.generation import (
    GenerationCase,
    MaterialClaim,
    aggregate_generation,
    evaluate_generation,
    match_answer,
    normalize_answer,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  대한민국은\u3000서울이다. ", "대한민국은서울이다"),
        ("ＡＢＣ， １２３！", "abc123"),
        ("서울—부산", "서울부산"),
    ],
)
def test_normalize_answer_removes_unicode_whitespace_and_punctuation(
    raw: str, expected: str
) -> None:
    assert normalize_answer(raw) == expected


@pytest.mark.parametrize(
    ("prediction", "gold"),
    [
        ("12.5%", "0.125"),
        ("₩1,500", "1,500원"),
        ("1.5만원", "15,000원"),
        ("2.0 kg", "2000 g"),
    ],
)
def test_numeric_match_handles_percent_commas_currency_and_units(
    prediction: str, gold: str
) -> None:
    result = match_answer(prediction, gold)
    assert result.numeric
    assert result.correct


def test_alias_match_is_explicit_and_does_not_become_exact() -> None:
    result = match_answer("서울특별시", "서울", aliases=("서울특별시",))
    assert result.alias
    assert result.correct
    assert not result.exact


def test_malformed_comma_grouping_is_not_a_numeric_match() -> None:
    assert not match_answer("1,2", "12").numeric


def test_claim_citation_metrics_use_claim_support_mapping() -> None:
    metric = evaluate_generation(
        GenerationCase(
            question_id="q1",
            question_type="fact",
            expected_answer="정답",
            aliases=(),
            answerable=True,
            predicted_answer="정답",
            abstained=False,
            claims=(
                MaterialClaim("c1", True, frozenset({"e1"})),
                MaterialClaim("c2", False, frozenset()),
            ),
            claim_citations={"c1": ("e1", "e9"), "c2": ()},
        )
    )
    assert metric.faithfulness == pytest.approx(0.5)
    assert metric.citation_precision == pytest.approx(0.5)
    assert metric.citation_recall == pytest.approx(0.5)


def test_citation_denominators_are_explicit_when_empty() -> None:
    metric = evaluate_generation(
        GenerationCase(
            question_id="q-empty",
            question_type="open",
            expected_answer=None,
            aliases=(),
            answerable=True,
            predicted_answer="설명",
            abstained=False,
            claims=(),
            claim_citations={},
        )
    )
    assert metric.deterministic_correctness is None
    assert metric.faithfulness is None
    assert metric.citation_precision is None
    assert metric.citation_recall is None


@pytest.mark.parametrize(
    ("answerable", "abstained", "correct", "false_answer", "false_abstention"),
    [
        (False, True, 1.0, 0.0, 0.0),
        (False, False, 0.0, 1.0, 0.0),
        (True, True, 0.0, 0.0, 1.0),
        (True, False, 1.0, 0.0, 0.0),
    ],
)
def test_abstention_outcomes_are_reported_separately(
    answerable: bool,
    abstained: bool,
    correct: float,
    false_answer: float,
    false_abstention: float,
) -> None:
    metric = evaluate_generation(
        GenerationCase(
            question_id=f"q-{answerable}-{abstained}",
            question_type="fact",
            expected_answer="답" if answerable else None,
            aliases=(),
            answerable=answerable,
            predicted_answer="답" if not abstained else "근거 부족",
            abstained=abstained,
            claims=(),
            claim_citations={},
        )
    )
    assert metric.abstention_accuracy == correct
    assert metric.false_answer == false_answer
    assert metric.false_abstention == false_abstention


def test_generation_aggregate_reports_rates_without_hiding_classes() -> None:
    cases = tuple(
        GenerationCase(
            question_id=f"q{index}",
            question_type="fact",
            expected_answer="답" if answerable else None,
            aliases=(),
            answerable=answerable,
            predicted_answer="답",
            abstained=abstained,
            claims=(),
            claim_citations={},
        )
        for index, (answerable, abstained) in enumerate(
            ((True, False), (True, True), (False, False), (False, True))
        )
    )
    aggregate = aggregate_generation(cases)
    assert aggregate.question_count == 4
    assert aggregate.abstention_accuracy == 0.5
    assert aggregate.false_answer_rate == 0.5
    assert aggregate.false_abstention_rate == 0.5
    with pytest.raises(TypeError):
        aggregate.by_question_type["fact"] = aggregate  # type: ignore[index]
    with pytest.raises(TypeError):
        aggregate.by_question_type["fact"].by_question_type["injected"] = aggregate  # type: ignore[index]


def test_generation_case_freezes_citation_mapping_for_reproducibility() -> None:
    mutable = {"c1": ("e1",)}
    case = GenerationCase(
        question_id="q-frozen",
        question_type="fact",
        expected_answer="답",
        aliases=(),
        answerable=True,
        predicted_answer="답",
        abstained=False,
        claims=(MaterialClaim("c1", True, frozenset({"e1"})),),
        claim_citations=mutable,
    )
    mutable["c1"] = ("e9",)
    assert case.claim_citations == {"c1": ("e1",)}
    with pytest.raises(TypeError):
        case.claim_citations["c1"] = ()  # type: ignore[index]


def test_generation_case_defensively_freezes_all_caller_owned_collections() -> None:
    evidence = {"e1"}
    aliases = ["별칭"]
    claims = [MaterialClaim("c1", True, evidence)]  # type: ignore[arg-type]
    case = GenerationCase(
        question_id="q-all-frozen",
        question_type="fact",
        expected_answer="답",
        aliases=aliases,  # type: ignore[arg-type]
        answerable=True,
        predicted_answer="별칭",
        abstained=False,
        claims=claims,  # type: ignore[arg-type]
        claim_citations={"c1": ("e1",)},
    )
    evidence.clear()
    aliases.clear()
    claims.clear()
    assert case.aliases == ("별칭",)
    assert len(case.claims) == 1
    assert case.claims[0].supporting_evidence_ids == frozenset({"e1"})
    assert evaluate_generation(case).deterministic_correctness == 1.0


def test_supported_claim_requires_at_least_one_supporting_evidence_unit() -> None:
    with pytest.raises(ValueError, match="supported claim"):
        MaterialClaim("c1", True, frozenset())
