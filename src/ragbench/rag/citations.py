"""Strict generated-answer parsing and server-owned citation provenance."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ragbench.rag.context import ContextBundle

GENERATION_SCHEMA_ERROR = "GENERATION_SCHEMA_ERROR"
CITATION_VALIDATION_ERROR = "CITATION_VALIDATION_ERROR"
_JSON_FENCE = re.compile(r"\A```(?:json)?\s*\n(?P<body>.*)\n```\s*\Z", re.DOTALL)


class GenerationSchemaError(RuntimeError):
    """The provider output remained invalid after one purely syntactic repair attempt."""

    code = GENERATION_SCHEMA_ERROR


class CitationValidationError(RuntimeError):
    """The model made a citation claim outside the included evidence allowlist."""

    code = CITATION_VALIDATION_ERROR


class GeneratedAnswer(BaseModel):
    """Only model-authored fields accepted from a generation response."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    answer: str = Field(min_length=1)
    citations: tuple[str, ...]
    abstained: bool

    @field_validator("answer")
    @classmethod
    def answer_cannot_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer cannot be blank")
        return value

    @field_validator("citations")
    @classmethod
    def citation_ids_cannot_be_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not citation_id.strip() for citation_id in value):
            raise ValueError("citation IDs cannot be blank")
        return value


@dataclass(frozen=True, slots=True)
class ResolvedCitation:
    """Citation enriched exclusively from the server-side context bundle."""

    citation_id: str
    chunk_id: str
    document_id: str
    document_title: str
    page_start: int
    page_end: int
    section_path: tuple[str, ...]


def parse_generated_answer(raw: str) -> GeneratedAnswer:
    """Parse strict JSON, allowing one repair that only removes a full Markdown JSON fence."""
    try:
        return GeneratedAnswer.model_validate_json(raw)
    except (ValidationError, ValueError):
        repaired = _remove_single_json_fence(raw)
        if repaired is None:
            raise GenerationSchemaError(GENERATION_SCHEMA_ERROR) from None
        try:
            return GeneratedAnswer.model_validate_json(repaired)
        except (ValidationError, ValueError):
            raise GenerationSchemaError(GENERATION_SCHEMA_ERROR) from None


def validate_answer_policy(answer: GeneratedAnswer, *, prompt_version: str) -> None:
    """Enforce citation/abstention semantics without repairing the model's claims."""
    if prompt_version not in {"v1", "v2", "v3"}:
        raise ValueError("unknown prompt version")
    if answer.abstained and answer.citations:
        raise CitationValidationError("an abstained answer cannot include citations")
    if prompt_version in {"v2", "v3"} and not answer.abstained and not answer.citations:
        raise CitationValidationError(
            "a non-abstained context-only answer requires at least one citation"
        )


def resolve_citations(
    citation_ids: Sequence[str], context: ContextBundle
) -> tuple[ResolvedCitation, ...]:
    """Resolve only citation IDs assigned to complete records included in this exact bundle."""
    if len(set(citation_ids)) != len(citation_ids):
        raise CitationValidationError("duplicate citation IDs are not allowed")
    allowed = {item.citation_id: item for item in context.items}
    output: list[ResolvedCitation] = []
    for citation_id in citation_ids:
        item = allowed.get(citation_id)
        if item is None:
            raise CitationValidationError(f"citation {citation_id!r} is not in included context")
        output.append(
            ResolvedCitation(
                citation_id=item.citation_id,
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                document_title=item.document_title,
                page_start=item.page_start,
                page_end=item.page_end,
                section_path=item.section_path,
            )
        )
    return tuple(output)


def _remove_single_json_fence(raw: str) -> str | None:
    match = _JSON_FENCE.fullmatch(raw.strip())
    if match is None:
        return None
    return match.group("body")
