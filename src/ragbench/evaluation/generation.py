"""Deterministic, provider-free generation and citation metrics."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import MappingProxyType

_NUMBER = re.compile(
    r"\A(?P<prefix>[₩￦$])?(?P<number>[+-]?(?:\d+(?:\.\d+)?|\.\d+))"
    r"(?P<unit>%|퍼센트|만원|천원|원|kg|킬로그램|g|그램)?\Z",
    re.IGNORECASE,
)
_GROUPED_NUMBER = re.compile(
    r"\A[₩￦$]?[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"(?:%|퍼센트|만원|천원|원|kg|킬로그램|g|그램)?\Z",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class AnswerMatch:
    exact: bool
    numeric: bool
    alias: bool

    @property
    def correct(self) -> bool:
        return self.exact or self.numeric or self.alias


@dataclass(frozen=True, slots=True)
class MaterialClaim:
    claim_id: str
    supported: bool
    supporting_evidence_ids: frozenset[str]

    def __post_init__(self) -> None:
        if not self.claim_id.strip():
            raise ValueError("claim ID cannot be blank")
        if any(not item.strip() for item in self.supporting_evidence_ids):
            raise ValueError("supporting evidence IDs cannot be blank")
        if self.supported != bool(self.supporting_evidence_ids):
            raise ValueError("supported claim must name supporting evidence units")


@dataclass(frozen=True, slots=True)
class GenerationCase:
    question_id: str
    question_type: str
    expected_answer: str | None
    aliases: tuple[str, ...]
    answerable: bool
    predicted_answer: str
    abstained: bool
    claims: tuple[MaterialClaim, ...]
    claim_citations: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not self.question_id.strip() or not self.question_type.strip():
            raise ValueError("generation case identity cannot be blank")
        if self.expected_answer is not None and not self.expected_answer.strip():
            raise ValueError("expected answer cannot be blank")
        if not self.answerable and self.expected_answer is not None:
            raise ValueError("unanswerable case cannot have an expected answer")
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim IDs must be unique")
        if not set(self.claim_citations).issubset(claim_ids):
            raise ValueError("citations may only map declared claims")
        for citations in self.claim_citations.values():
            if any(not item.strip() for item in citations) or len(citations) != len(set(citations)):
                raise ValueError("citation IDs must be nonblank and unique per claim")
        object.__setattr__(
            self,
            "claim_citations",
            MappingProxyType(
                {claim_id: tuple(citations) for claim_id, citations in self.claim_citations.items()}
            ),
        )


@dataclass(frozen=True, slots=True)
class GenerationMetric:
    question_id: str
    question_type: str
    deterministic_correctness: float | None
    exact_match: float | None
    numeric_match: float | None
    alias_match: float | None
    faithfulness: float | None
    citation_precision: float | None
    citation_recall: float | None
    abstention_accuracy: float
    false_answer: float
    false_abstention: float


@dataclass(frozen=True, slots=True)
class GenerationAggregate:
    question_count: int
    deterministic_scorable_count: int
    deterministic_correctness: float | None
    faithfulness: float | None
    citation_precision: float | None
    citation_recall: float | None
    abstention_accuracy: float
    false_answer_rate: float | None
    false_abstention_rate: float | None
    by_question_type: Mapping[str, GenerationAggregate]


def normalize_answer(value: str) -> str:
    """Normalize Unicode width/case and remove all punctuation and whitespace."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith(("P", "Z"))
        and not character.isspace()
    )


def _numeric_value(value: str) -> tuple[Decimal, str] | None:
    compact = unicodedata.normalize("NFKC", value).casefold()
    compact = "".join(character for character in compact if not character.isspace())
    if "," in compact and _GROUPED_NUMBER.fullmatch(compact) is None:
        return None
    compact = compact.replace(",", "")
    match = _NUMBER.fullmatch(compact)
    if match is None:
        return None
    try:
        number = Decimal(match.group("number"))
    except InvalidOperation:
        return None
    prefix = match.group("prefix")
    unit = (match.group("unit") or "").casefold()
    if prefix == "$":
        dimension, multiplier = "usd", Decimal(1)
    elif prefix in {"₩", "￦"}:
        dimension, multiplier = "krw", Decimal(1)
    elif unit in {"원", "만원", "천원"}:
        dimension = "krw"
        multiplier = {"원": Decimal(1), "천원": Decimal(1000), "만원": Decimal(10000)}[unit]
    elif unit in {"kg", "킬로그램", "g", "그램"}:
        dimension = "mass-g"
        multiplier = Decimal(1000) if unit in {"kg", "킬로그램"} else Decimal(1)
    elif unit in {"%", "퍼센트"}:
        dimension, multiplier = "scalar", Decimal("0.01")
    else:
        dimension, multiplier = "scalar", Decimal(1)
    return number * multiplier, dimension


def match_answer(
    prediction: str, gold: str, *, aliases: Sequence[str] = ()
) -> AnswerMatch:
    """Return independent exact, numeric, and allowlisted-alias decisions."""
    prediction_normalized = normalize_answer(prediction)
    gold_normalized = normalize_answer(gold)
    exact = bool(prediction_normalized) and prediction_normalized == gold_normalized
    left, right = _numeric_value(prediction), _numeric_value(gold)
    numeric = left is not None and right is not None and left == right
    alias = any(
        prediction_normalized == normalize_answer(candidate)
        and prediction_normalized != gold_normalized
        for candidate in aliases
    )
    return AnswerMatch(exact, numeric, alias)


def evaluate_generation(case: GenerationCase) -> GenerationMetric:
    """Score one response while preserving undefined denominator states as ``None``."""
    match: AnswerMatch | None = None
    if case.answerable and not case.abstained and case.expected_answer is not None:
        match = match_answer(case.predicted_answer, case.expected_answer, aliases=case.aliases)

    faithfulness = (
        sum(claim.supported for claim in case.claims) / len(case.claims)
        if case.claims
        else None
    )
    claims_by_id = {claim.claim_id: claim for claim in case.claims}
    citations = [
        (claim_id, citation_id)
        for claim_id, citation_ids in case.claim_citations.items()
        for citation_id in citation_ids
    ]
    supported_citations = sum(
        citation_id in claims_by_id[claim_id].supporting_evidence_ids
        for claim_id, citation_id in citations
    )
    citation_precision = supported_citations / len(citations) if citations else None
    citation_recall = (
        sum(
            any(
                citation in claim.supporting_evidence_ids
                for citation in case.claim_citations.get(claim.claim_id, ())
            )
            for claim in case.claims
        )
        / len(case.claims)
        if case.claims
        else None
    )
    correct_abstention = case.abstained == (not case.answerable)
    return GenerationMetric(
        case.question_id,
        case.question_type,
        float(match.correct) if match is not None else None,
        float(match.exact) if match is not None else None,
        float(match.numeric) if match is not None else None,
        float(match.alias) if match is not None else None,
        faithfulness,
        citation_precision,
        citation_recall,
        float(correct_abstention),
        float(not case.answerable and not case.abstained),
        float(case.answerable and case.abstained),
    )


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def _aggregate(cases: Sequence[GenerationCase], *, include_groups: bool) -> GenerationAggregate:
    if not cases:
        raise ValueError("generation aggregate requires at least one case")
    metrics = tuple(evaluate_generation(case) for case in cases)
    answerable = [metric for metric, case in zip(metrics, cases, strict=True) if case.answerable]
    unanswerable = [
        metric for metric, case in zip(metrics, cases, strict=True) if not case.answerable
    ]
    grouped: dict[str, list[GenerationCase]] = defaultdict(list)
    if include_groups:
        for case in cases:
            grouped[case.question_type].append(case)
    return GenerationAggregate(
        len(cases),
        sum(metric.deterministic_correctness is not None for metric in metrics),
        _mean_defined([metric.deterministic_correctness for metric in metrics]),
        _mean_defined([metric.faithfulness for metric in metrics]),
        _mean_defined([metric.citation_precision for metric in metrics]),
        _mean_defined([metric.citation_recall for metric in metrics]),
        sum(metric.abstention_accuracy for metric in metrics) / len(metrics),
        sum(metric.false_answer for metric in unanswerable) / len(unanswerable)
        if unanswerable
        else None,
        sum(metric.false_abstention for metric in answerable) / len(answerable)
        if answerable
        else None,
        {name: _aggregate(grouped[name], include_groups=False) for name in sorted(grouped)},
    )


def aggregate_generation(cases: Sequence[GenerationCase]) -> GenerationAggregate:
    """Aggregate overall and type-specific metrics with class-conditional error rates."""
    identifiers = [case.question_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("generation question IDs must be unique")
    return _aggregate(cases, include_groups=True)
