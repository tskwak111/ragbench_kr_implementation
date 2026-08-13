"""Strict generation schema and server-side citation tests."""

import pytest

from ragbench.rag.citations import (
    CitationValidationError,
    GenerationSchemaError,
    parse_generated_answer,
    resolve_citations,
    validate_answer_policy,
)
from ragbench.rag.context import ContextBuilder, RetrievedChunk
from ragbench.retrieval.base import SearchHit


def _bundle():  # type: ignore[no-untyped-def]
    return ContextBuilder().build(
        (
            RetrievedChunk(
                SearchHit("chunk-7", 0.9, 1, "dense"),
                "doc-1",
                "감사보고서",
                7,
                8,
                ("재무", "매출"),
                "매출은 100억 원이다.",
            ),
        ),
        token_budget=2_000,
    )


def test_resolves_only_included_citation_ids_to_server_side_provenance() -> None:
    """Catch trusting model-supplied source metadata or non-retrieved chunk IDs."""
    payload = parse_generated_answer(
        '{"answer":"100억 원","citations":["C1"],"abstained":false}'
    )

    citations = resolve_citations(payload.citations, _bundle())

    assert len(citations) == 1
    assert citations[0].citation_id == "C1"
    assert citations[0].chunk_id == "chunk-7"
    assert citations[0].document_id == "doc-1"
    assert citations[0].document_title == "감사보고서"
    assert citations[0].page_start == 7
    assert citations[0].page_end == 8
    assert citations[0].section_path == ("재무", "매출")


@pytest.mark.parametrize("citation_id", ["C2", "chunk-7", "C999"])
def test_unknown_or_unsupported_citation_fails(citation_id: str) -> None:
    """Catch citations that bypass the included-context allowlist."""
    with pytest.raises(CitationValidationError, match="not in included context"):
        resolve_citations((citation_id,), _bundle())


def test_duplicate_citation_is_rejected_instead_of_semantically_repaired() -> None:
    """Catch silently rewriting malformed model evidence claims."""
    with pytest.raises(CitationValidationError, match="duplicate"):
        resolve_citations(("C1", "C1"), _bundle())


def test_parser_performs_at_most_one_syntactic_fence_repair() -> None:
    """Catch rejecting a common purely syntactic JSON fence without changing semantics."""
    payload = parse_generated_answer(
        '```json\n{"answer":"근거 부족","citations":[],"abstained":true}\n```'
    )

    assert payload.answer == "근거 부족"
    assert payload.citations == ()
    assert payload.abstained is True


@pytest.mark.parametrize(
    "raw",
    [
        '{"answer":"x","citations":["C1"]}',
        '{"answer":"x","citations":[],"abstained":"false"}',
        '{"answer":"x","citations":[],"abstained":false,"source":"C1"}',
        "not json at all",
        '```json\n{"answer":\n```',
    ],
)
def test_persistent_syntax_or_schema_failure_has_stable_error_code(raw: str) -> None:
    """Catch permissive parsing, semantic repair, or unstable failure classification."""
    with pytest.raises(GenerationSchemaError) as caught:
        parse_generated_answer(raw)

    assert caught.value.code == "GENERATION_SCHEMA_ERROR"


def test_context_only_prompt_policy_requires_citations_or_clean_abstention() -> None:
    """Catch unsupported non-abstained answers and citations attached to abstentions."""
    no_citation = parse_generated_answer(
        '{"answer":"답","citations":[],"abstained":false}'
    )
    bad_abstention = parse_generated_answer(
        '{"answer":"모름","citations":["C1"],"abstained":true}'
    )

    with pytest.raises(CitationValidationError, match="requires at least one"):
        validate_answer_policy(no_citation, prompt_version="v2")
    with pytest.raises(CitationValidationError, match="cannot include citations"):
        validate_answer_policy(bad_abstention, prompt_version="v3")
