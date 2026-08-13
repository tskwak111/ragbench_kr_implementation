"""Blind, provenance-constrained judge parsing and human calibration support."""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ragbench.chunking.tokenizer import encoding
from ragbench.core.hashing import canonical_json_hash
from ragbench.providers.base import GenerateRequest, ProviderGateway

JUDGE_RUBRIC_VERSION = "judge-v1"
JUDGE_RUBRIC = {
    "version": JUDGE_RUBRIC_VERSION,
    "correctness": "Score 0..1 against the supplied gold answer and gold evidence only.",
    "faithfulness": "Return one support judgment for every supplied material answer claim.",
    "citations": "Return one support judgment for every model citation.",
    "benchmark_defect": "Flag only defects demonstrated by supplied evidence.",
    "provenance": "Every rationale must cite only supplied evidence IDs; never use outside facts.",
}
JUDGE_RUBRIC_HASH = canonical_json_hash(JUDGE_RUBRIC)
_TOKEN_MARGIN = 16


class JudgeParseError(ValueError):
    """Judge output violated strict schema or supplied-provenance boundaries."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class EvidenceUnit(_StrictModel):
    evidence_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("evidence_id", "text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence fields cannot be blank")
        return value.strip()


class JudgeInput(_StrictModel):
    question: str = Field(min_length=1)
    gold_answer: str | None
    gold_evidence: tuple[EvidenceUnit, ...]
    model_answer: str = Field(min_length=1)
    answer_claims: tuple[str, ...]
    model_citation_ids: tuple[str, ...]
    retrieved_context: tuple[EvidenceUnit, ...]

    @field_validator("question", "model_answer")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("judge input text cannot be blank")
        return value.strip()

    @model_validator(mode="after")
    def _identities_are_unambiguous(self) -> JudgeInput:
        evidence_ids = [
            item.evidence_id for item in (*self.gold_evidence, *self.retrieved_context)
        ]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique across judge input")
        claim_ids = _claim_ids(self.answer_claims)
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("answer claim IDs must be unique")
        if any(not item.strip() for item in self.model_citation_ids) or len(
            self.model_citation_ids
        ) != len(set(self.model_citation_ids)):
            raise ValueError("model citation IDs must be nonblank and unique")
        return self


class ClaimJudgment(_StrictModel):
    claim_id: str = Field(min_length=1)
    supported: bool
    evidence_ids: tuple[str, ...]
    rationale: str = Field(min_length=1)


class CitationJudgment(_StrictModel):
    citation_id: str = Field(min_length=1)
    claim_ids: tuple[str, ...]
    supported: bool
    evidence_ids: tuple[str, ...]
    rationale: str = Field(min_length=1)


class JudgeRecord(_StrictModel):
    correctness: float = Field(ge=0, le=1)
    correctness_evidence_ids: tuple[str, ...]
    claims: tuple[ClaimJudgment, ...]
    citations: tuple[CitationJudgment, ...]
    benchmark_defect: bool
    benchmark_defect_evidence_ids: tuple[str, ...]
    rationale: str = Field(min_length=1)


class JudgeEvaluation(JudgeRecord):
    model_id: str
    rubric_version: str
    rubric_hash: str
    temperature: float | None
    cached: bool | None
    correlation_id: str | None


class JudgeConfig(_StrictModel):
    model_id: str = Field(min_length=1)
    generator_model_id: str = Field(min_length=1)
    rubric_version: str
    temperature: float | None
    max_output_tokens: int = Field(gt=0)
    same_model_unavailability_reason: str | None = None
    temperature_unsupported_reason: str | None = None

    @field_validator("model_id", "generator_model_id", "rubric_version")
    @classmethod
    def _identity_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("judge configuration identities cannot be blank")
        return value.strip()

    @model_validator(mode="after")
    def _judge_policy(self) -> JudgeConfig:
        if self.rubric_version != JUDGE_RUBRIC_VERSION:
            raise ValueError("unknown judge rubric version")
        if self.model_id == self.generator_model_id and not (
            self.same_model_unavailability_reason
            and self.same_model_unavailability_reason.strip()
        ):
            raise ValueError("judge model must be distinct from generator when available")
        if self.temperature is not None and self.temperature != 0:
            raise ValueError("judge temperature must be zero where supported")
        if self.temperature is None and not (
            self.temperature_unsupported_reason and self.temperature_unsupported_reason.strip()
        ):
            raise ValueError("unsupported temperature requires an explicit reason")
        return self


def _claim_ids(claims: Sequence[str]) -> tuple[str, ...]:
    identifiers: list[str] = []
    for claim in claims:
        identifier, separator, text = claim.partition(":")
        if not separator or not identifier.strip() or not text.strip():
            raise ValueError("answer claims must use '<claim_id>: <claim>' format")
        identifiers.append(identifier.strip())
    return tuple(identifiers)


def render_judge_prompt(value: JudgeInput) -> str:
    """Render a blind rubric payload; generator configuration cannot enter this API."""
    payload = {
        "rubric": JUDGE_RUBRIC,
        "instructions": {
            "output": "Return one strict JSON object matching the rubric fields.",
            "outside_knowledge": "Forbidden. Cite only supplied evidence IDs in rationales.",
            "coverage": (
                "Judge every supplied claim and every supplied model citation exactly once."
            ),
        },
        "case": value.model_dump(mode="json"),
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)


def parse_judge_response(raw: str, value: JudgeInput) -> JudgeRecord:
    """Parse once, without JSON repair, then enforce exact claim/citation provenance."""
    try:
        record = JudgeRecord.model_validate_json(raw)
    except (ValidationError, ValueError):
        raise JudgeParseError("judge response is not strict rubric JSON") from None
    expected_claim_ids = set(_claim_ids(value.answer_claims))
    returned_claim_ids = [item.claim_id for item in record.claims]
    if len(returned_claim_ids) != len(set(returned_claim_ids)) or set(
        returned_claim_ids
    ) != expected_claim_ids:
        raise JudgeParseError("judge must evaluate every supplied claim exactly once")
    expected_citations = set(value.model_citation_ids)
    returned_citations = [item.citation_id for item in record.citations]
    if len(returned_citations) != len(set(returned_citations)) or set(
        returned_citations
    ) != expected_citations:
        raise JudgeParseError("judge must evaluate every supplied citation exactly once")
    allowed_evidence = {
        item.evidence_id for item in (*value.gold_evidence, *value.retrieved_context)
    }
    cited_evidence = [
        *record.correctness_evidence_ids,
        *record.benchmark_defect_evidence_ids,
        *(evidence for claim in record.claims for evidence in claim.evidence_ids),
        *(evidence for citation in record.citations for evidence in citation.evidence_ids),
    ]
    cited_claims = [claim_id for citation in record.citations for claim_id in citation.claim_ids]
    if any(item not in allowed_evidence for item in cited_evidence):
        raise JudgeParseError("judge invented an evidence ID")
    if any(item not in expected_claim_ids for item in cited_claims):
        raise JudgeParseError("judge invented a claim ID")
    if any(claim.supported and not claim.evidence_ids for claim in record.claims) or any(
        citation.supported and not citation.evidence_ids for citation in record.citations
    ):
        raise JudgeParseError("supported judge decisions require cited supplied evidence")
    if record.correctness > 0 and not record.correctness_evidence_ids:
        raise JudgeParseError("positive correctness requires cited supplied evidence")
    if record.benchmark_defect and not record.benchmark_defect_evidence_ids:
        raise JudgeParseError("benchmark defect requires cited supplied evidence")
    rationale_references = (
        (record.correctness_evidence_ids, record.rationale),
        (record.benchmark_defect_evidence_ids, record.rationale),
        *((claim.evidence_ids, claim.rationale) for claim in record.claims),
        *((citation.evidence_ids, citation.rationale) for citation in record.citations),
    )
    if any(
        not all(_rationale_mentions(evidence_id, rationale) for evidence_id in evidence_ids)
        for evidence_ids, rationale in rationale_references
    ):
        raise JudgeParseError("each cited evidence ID must appear in its rationale")
    if any(not item.strip() for item in cited_evidence) or any(
        len(items) != len(set(items))
        for items in (
            record.correctness_evidence_ids,
            record.benchmark_defect_evidence_ids,
            *(claim.evidence_ids for claim in record.claims),
            *(citation.evidence_ids for citation in record.citations),
        )
    ):
        raise JudgeParseError("judge evidence references must be nonblank and unique")
    return record


def _rationale_mentions(evidence_id: str, rationale: str) -> bool:
    # Korean postpositions are commonly attached directly to Latin evidence IDs (for example
    # ``e1이``), so boundaries apply only to the ASCII ID alphabet.
    return re.search(
        rf"(?<![A-Za-z0-9_-]){re.escape(evidence_id)}(?![A-Za-z0-9_-])",
        rationale,
    ) is not None


class JudgeRunner:
    """Execute judge requests only through the cached, metered provider gateway."""

    def __init__(self, gateway: ProviderGateway) -> None:
        self._gateway = gateway

    async def evaluate(self, value: JudgeInput, config: JudgeConfig) -> JudgeEvaluation:
        prompt = render_judge_prompt(value)
        provider_params = {} if config.temperature is None else {"temperature": config.temperature}
        response = await self._gateway.generate(
            GenerateRequest(
                model_id=config.model_id,
                prompt=prompt,
                context=(),
                provider_params=provider_params,
                input_tokens=len(encoding().encode(prompt)) + _TOKEN_MARGIN,
                max_output_tokens=config.max_output_tokens,
            )
        )
        record = parse_judge_response(response.content, value)
        return JudgeEvaluation.model_validate(
            {
                **record.model_dump(mode="python"),
                "model_id": config.model_id,
                "rubric_version": config.rubric_version,
                "rubric_hash": JUDGE_RUBRIC_HASH,
                "temperature": config.temperature,
                "cached": response.cache_hit,
                "correlation_id": response.correlation_id,
            }
        )


@dataclass(frozen=True, slots=True)
class CalibrationCandidate:
    response_id: str
    system_id: str
    question_type: str
    judge_human_disagreement: bool
    known_failure: bool

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.response_id, self.system_id, self.question_type)
        ):
            raise ValueError("calibration candidate identity cannot be blank")


@dataclass(frozen=True, slots=True)
class HumanCalibrationPlan:
    responses: tuple[CalibrationCandidate, ...]
    seed: int
    stratum_counts: Mapping[str, int]
    requires_real_human_labels: bool = True


def plan_human_calibration(
    candidates: Sequence[CalibrationCandidate], *, sample_size: int, seed: int
) -> HumanCalibrationPlan:
    """Select 100–300 responses balanced by system/type and retaining hard cases."""
    if not 100 <= sample_size <= 300:
        raise ValueError("human calibration sample must contain 100–300 responses")
    if len(candidates) < sample_size:
        raise ValueError("calibration pool is smaller than requested sample")
    identifiers = [item.response_id for item in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("calibration response IDs must be unique")
    if not any(item.judge_human_disagreement for item in candidates) or not any(
        item.known_failure for item in candidates
    ):
        raise ValueError("calibration pool must include disagreements and known failures")
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[CalibrationCandidate]] = defaultdict(list)
    for item in candidates:
        strata[(item.system_id, item.question_type)].append(item)
    systems = {item.system_id for item in candidates}
    question_types = {item.question_type for item in candidates}
    expected_strata = {(system, kind) for system in systems for kind in question_types}
    if strata.keys() != expected_strata:
        raise ValueError("calibration pool must cover every system/type stratum")
    keys = sorted(strata)
    for key in keys:
        rows = strata[key]
        rng.shuffle(rows)
        rows.sort(key=lambda row: not (row.judge_human_disagreement or row.known_failure))
    selected: list[CalibrationCandidate] = []
    while len(selected) < sample_size:
        progressed = False
        for key in keys:
            if strata[key]:
                selected.append(strata[key].pop(0))
                progressed = True
                if len(selected) == sample_size:
                    break
        if not progressed:
            raise ValueError("balanced calibration sample cannot be satisfied")
    if not any(item.judge_human_disagreement for item in selected) or not any(
        item.known_failure for item in selected
    ):
        raise ValueError("selected sample must retain disagreements and known failures")
    rng.shuffle(selected)
    counts: dict[str, int] = defaultdict(int)
    for item in selected:
        counts[f"{item.system_id}::{item.question_type}"] += 1
    if max(counts.values()) - min(counts.values()) > 1:
        raise ValueError("available strata cannot satisfy a balanced calibration sample")
    return HumanCalibrationPlan(tuple(selected), seed, dict(sorted(counts.items())))


@dataclass(frozen=True, slots=True)
class CalibrationPair:
    response_id: str
    question_type: str
    judge_score: float
    human_score: float
    reviewer_id: str
    human_attested: bool

    def __post_init__(self) -> None:
        if any(
            not item.strip()
            for item in (self.response_id, self.question_type, self.reviewer_id)
        ):
            raise ValueError("calibration label identity cannot be blank")
        if any(
            not math.isfinite(value) or not 0 <= value <= 1
            for value in (self.judge_score, self.human_score)
        ):
            raise ValueError("calibration scores must be finite values from zero to one")
        if not isinstance(self.human_attested, bool):
            raise ValueError("human attestation must be a boolean")


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    sample_size: int
    spearman_correlation: float
    threshold_agreement: float
    threshold_f1: float
    mean_bias_by_question_type: Mapping[str, float]
    binary_threshold: float
    status: str
    is_final_authority: bool


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[ordered[position]] = average_rank
        start = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    if left_ss == 0 or right_ss == 0:
        raise ValueError("calibration correlation is undefined for constant scores")
    return numerator / math.sqrt(left_ss * right_ss)


def calibrate_judge(
    pairs: Sequence[CalibrationPair], *, binary_threshold: float
) -> CalibrationReport:
    """Compare judge scores with real human labels; never promote judge to authority."""
    if not 0 < binary_threshold < 1:
        raise ValueError("binary threshold must be between zero and one")
    if not 100 <= len(pairs) <= 300:
        raise ValueError("judge calibration requires 100–300 real human labels")
    identifiers = [pair.response_id for pair in pairs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("calibration response IDs must be unique")
    nonhuman = {"synthetic", "fixture", "automated", "model", "judge"}
    if any(
        not pair.human_attested
        or any(marker in pair.reviewer_id.casefold() for marker in nonhuman)
        for pair in pairs
    ):
        raise ValueError("judge calibration requires real human reviewer labels")
    judge = [pair.judge_score for pair in pairs]
    human = [pair.human_score for pair in pairs]
    spearman = _correlation(_ranks(judge), _ranks(human))
    predicted = [value >= binary_threshold for value in judge]
    expected = [value >= binary_threshold for value in human]
    agreement = sum(
        left == right for left, right in zip(predicted, expected, strict=True)
    ) / len(pairs)
    true_positive = sum(left and right for left, right in zip(predicted, expected, strict=True))
    false_positive = sum(
        left and not right for left, right in zip(predicted, expected, strict=True)
    )
    false_negative = sum(
        not left and right for left, right in zip(predicted, expected, strict=True)
    )
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator if denominator else 0.0
    grouped: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.question_type].append(pair.judge_score - pair.human_score)
    bias = {name: sum(values) / len(values) for name, values in sorted(grouped.items())}
    return CalibrationReport(
        len(pairs),
        spearman,
        agreement,
        f1,
        bias,
        binary_threshold,
        "calibrated-assistant-only",
        False,
    )
