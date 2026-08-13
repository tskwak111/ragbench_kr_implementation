"""Conservative, deterministic validation for synthetic benchmark candidates."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum

from ragbench.benchmark.generation import (
    DEFAULT_QUOTAS,
    QuestionCandidate,
    QuestionType,
    SourceWindow,
    ValidationDecision,
    ValidationStatus,
)


class CompletionLevel(StrEnum):
    TARGET = "target"
    NORMAL_FLOOR = "normal_floor"
    EMERGENCY_ONLY = "emergency_only"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    quotas: Mapping[QuestionType, int] | None = None
    per_document_cap: int = 300
    duplicate_similarity_threshold: float = 0.92
    evidence_similarity_threshold: float = 0.96
    rejection_sample_limit: int = 5
    contamination_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        quotas = dict(self.quotas or DEFAULT_QUOTAS)
        if any(count <= 0 for count in quotas.values()):
            raise ValueError("validation quotas must be positive")
        if self.per_document_cap <= 0:
            raise ValueError("per_document_cap must be positive")
        if not 0 <= self.duplicate_similarity_threshold <= 1:
            raise ValueError("duplicate threshold must be between zero and one")
        if not 0 <= self.evidence_similarity_threshold <= 1:
            raise ValueError("evidence threshold must be between zero and one")
        if self.rejection_sample_limit <= 0:
            raise ValueError("rejection sample limit must be positive")
        if any(not term.strip() for term in self.contamination_terms):
            raise ValueError("contamination terms cannot be blank")
        object.__setattr__(self, "quotas", quotas)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    items: tuple[QuestionCandidate, ...]
    accepted_count: int
    rejected_count: int
    rejection_counts: dict[str, int]
    rejection_samples: dict[str, tuple[str, ...]]
    type_distribution: dict[str, int]
    difficulty_distribution: dict[str, int]
    document_distribution: dict[str, int]
    duplicate_groups: tuple[tuple[str, ...], ...]


def report_payload(report: ValidationReport) -> dict[str, object]:
    """Return a stable public validation summary without benchmark content."""
    return {
        "accepted_count": report.accepted_count,
        "completion_level": completion_level(report.accepted_count).value,
        "difficulty_distribution": report.difficulty_distribution,
        "document_distribution": report.document_distribution,
        "duplicate_groups": [list(group) for group in report.duplicate_groups],
        "rejected_count": report.rejected_count,
        "rejection_counts": report.rejection_counts,
        "rejection_samples": {
            key: list(value) for key, value in report.rejection_samples.items()
        },
        "type_distribution": report.type_distribution,
    }


def completion_level(valid_count: int) -> CompletionLevel:
    if valid_count < 0:
        raise ValueError("valid_count cannot be negative")
    if valid_count >= 1_500:
        return CompletionLevel.TARGET
    if valid_count >= 1_000:
        return CompletionLevel.NORMAL_FLOOR
    if valid_count >= 800:
        return CompletionLevel.EMERGENCY_ONLY
    return CompletionLevel.INSUFFICIENT


def validate_candidates(
    candidates: Sequence[QuestionCandidate],
    corpus: Sequence[SourceWindow],
    *,
    config: ValidationConfig | None = None,
    contamination_terms: Sequence[str] = (),
) -> ValidationReport:
    """Validate in stable input order and reject conservatively on every failed rule."""
    active_config = config or ValidationConfig()
    windows = tuple(corpus)
    pages = _index_pages(windows)
    document_text = _document_text(windows)
    duplicate_groups, duplicate_ids = _duplicate_groups(
        candidates, active_config.duplicate_similarity_threshold
    )
    accepted_type_counts: Counter[QuestionType] = Counter()
    accepted_document_counts: Counter[str] = Counter()
    validated: list[QuestionCandidate] = []

    for candidate in candidates:
        rules: list[str] = []
        evidence_documents = tuple(
            dict.fromkeys(span.document_id for span in candidate.evidence_spans)
        )
        if candidate.candidate_id in duplicate_ids:
            rules.append("duplicate_question")
        rules.extend(_evidence_rules(candidate, pages, active_config))
        rules.extend(_answer_rules(candidate))
        rules.extend(_unanswerable_rules(candidate, document_text))
        searchable = " ".join(
            filter(None, (candidate.question, candidate.gold_answer or ""))
        )
        active_contamination_terms = (
            *active_config.contamination_terms,
            *contamination_terms,
        )
        if any(
            _search_text(term) in _search_text(searchable)
            for term in active_contamination_terms
        ):
            rules.append("contamination_detected")

        if not rules:
            quota = active_config.quotas.get(candidate.question_type, 0)  # type: ignore[union-attr]
            if accepted_type_counts[candidate.question_type] >= quota:
                rules.append("type_quota_exceeded")
            elif any(
                accepted_document_counts[document_id] >= active_config.per_document_cap
                for document_id in evidence_documents
            ):
                rules.append("per_document_cap_exceeded")

        unique_rules = tuple(dict.fromkeys(rules))
        decision = (
            ValidationDecision.REJECTED if unique_rules else ValidationDecision.ACCEPTED
        )
        item = candidate.model_copy(
            update={"validation": ValidationStatus(decision=decision, rule_codes=unique_rules)}
        )
        validated.append(item)
        if decision is ValidationDecision.ACCEPTED:
            accepted_type_counts[candidate.question_type] += 1
            for document_id in evidence_documents:
                accepted_document_counts[document_id] += 1

    accepted = tuple(
        item for item in validated if item.validation.decision is ValidationDecision.ACCEPTED
    )
    rejections = Counter(
        rule for item in validated for rule in item.validation.rule_codes
    )
    samples: dict[str, list[str]] = defaultdict(list)
    for item in validated:
        for rule in item.validation.rule_codes:
            if len(samples[rule]) < active_config.rejection_sample_limit:
                samples[rule].append(item.candidate_id)
    return ValidationReport(
        items=tuple(validated),
        accepted_count=len(accepted),
        rejected_count=len(validated) - len(accepted),
        rejection_counts=dict(sorted(rejections.items())),
        rejection_samples={key: tuple(value) for key, value in sorted(samples.items())},
        type_distribution=dict(
            sorted(Counter(item.question_type.value for item in accepted).items())
        ),
        difficulty_distribution=dict(
            sorted(Counter(item.difficulty.value for item in accepted).items())
        ),
        document_distribution=dict(sorted(accepted_document_counts.items())),
        duplicate_groups=duplicate_groups,
    )


def _index_pages(
    windows: Sequence[SourceWindow],
) -> dict[tuple[str, int], tuple[SourceWindow, ...]]:
    indexed: dict[tuple[str, int], list[SourceWindow]] = defaultdict(list)
    for window in windows:
        for page in range(window.page_start, window.page_end + 1):
            indexed[(window.document_id, page)].append(window)
    return {key: tuple(value) for key, value in indexed.items()}


def _document_text(windows: Sequence[SourceWindow]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for window in windows:
        grouped[window.document_id].append(window.content)
    return {key: "\n".join(value) for key, value in grouped.items()}


def _evidence_rules(
    candidate: QuestionCandidate,
    pages: Mapping[tuple[str, int], tuple[SourceWindow, ...]],
    config: ValidationConfig,
) -> list[str]:
    rules: list[str] = []
    for span in candidate.evidence_spans:
        page_windows = pages.get((span.document_id, span.page))
        if page_windows is None:
            rules.append("impossible_page")
            continue
        chunk_windows = tuple(
            window for window in page_windows if span.chunk_id in window.chunk_ids
        )
        if not chunk_windows:
            rules.append("impossible_chunk")
            continue
        if not any(
            _evidence_matches(span.text, window.content, config.evidence_similarity_threshold)
            for window in chunk_windows
        ):
            rules.append("evidence_not_found")
    return rules


def _evidence_matches(span: str, source: str, threshold: float) -> bool:
    normalized_span = _search_text(span)
    normalized_source = _search_text(source)
    if not normalized_span:
        return False
    if normalized_span in normalized_source:
        return True
    if len(normalized_span) > len(normalized_source):
        return False
    window_size = len(normalized_span)
    step = max(1, window_size // 10)
    return any(
        SequenceMatcher(
            None, normalized_span, normalized_source[start : start + window_size]
        ).ratio()
        >= threshold
        for start in range(0, len(normalized_source) - window_size + 1, step)
    )


def _answer_rules(candidate: QuestionCandidate) -> list[str]:
    if not candidate.answerable or candidate.gold_answer is None:
        return []
    rules: list[str] = []
    evidence = " ".join(span.text for span in candidate.evidence_spans)
    if _search_text(candidate.gold_answer) in _search_text(candidate.question):
        rules.append("answer_leaked_in_question")
    answer_numbers = _numbers(candidate.gold_answer)
    evidence_numbers = _numbers(evidence)
    if answer_numbers and not answer_numbers.issubset(evidence_numbers):
        rules.append("numeric_mismatch")
    elif (
        candidate.question_type in {QuestionType.FACT, QuestionType.NUMERIC_TABLE}
        and _search_text(candidate.gold_answer) not in _search_text(evidence)
    ):
        rules.append("answer_not_supported")
    elif candidate.question_type not in {QuestionType.FACT, QuestionType.NUMERIC_TABLE}:
        answer_tokens = _content_tokens(candidate.gold_answer)
        evidence_tokens = _content_tokens(evidence)
        if answer_tokens and not answer_tokens.intersection(evidence_tokens):
            rules.append("answer_not_supported")
    return rules


def _unanswerable_rules(
    candidate: QuestionCandidate, document_text: Mapping[str, str]
) -> list[str]:
    if candidate.answerable:
        return []
    documents = tuple(document_text.values())
    return [
        "asserted_absent_fact_present"
        for fact in candidate.asserted_absent_facts
        if any(_search_text(fact) in _search_text(content) for content in documents)
    ]


def _duplicate_groups(
    candidates: Sequence[QuestionCandidate], threshold: float
) -> tuple[tuple[tuple[str, ...], ...], frozenset[str]]:
    fingerprints = [_question_fingerprint(item.question) for item in candidates]
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if SequenceMatcher(None, fingerprints[left], fingerprints[right]).ratio() >= threshold:
                union(left, right)
    grouped: dict[int, list[str]] = defaultdict(list)
    grouped_indexes: dict[int, list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        root = find(index)
        grouped[root].append(candidate.candidate_id)
        grouped_indexes[root].append(index)
    groups = tuple(tuple(ids) for _, ids in sorted(grouped.items()) if len(ids) > 1)
    rejected = frozenset(
        candidates[index].candidate_id
        for root, indexes in grouped_indexes.items()
        if len(indexes) > 1
        for index in indexes[1:]
    )
    return groups, rejected


def _question_fingerprint(value: str) -> str:
    normalized = _search_text(value)
    for suffix in ("인가요", "인가", "입니까", "인가요"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _search_text(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", unicodedata.normalize("NFKC", value).lower())


def _numbers(value: str) -> set[str]:
    return {match.replace(",", "") for match in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", value)}


def _content_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[0-9a-z가-힣]+", unicodedata.normalize("NFKC", value).lower())
        if len(token) >= 2
    }
