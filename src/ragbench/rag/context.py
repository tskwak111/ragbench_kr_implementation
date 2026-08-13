"""Deterministic, token-bounded serialization of untrusted retrieved evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from ragbench.chunking.tokenizer import encoding
from ragbench.retrieval.base import SearchHit

_DOCUMENT_OPEN = "<UNTRUSTED_DOCUMENT>"
_DOCUMENT_CLOSE = "</UNTRUSTED_DOCUMENT>"
_DOCUMENT_SEPARATOR = "\n"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One retrieved chunk plus immutable human-auditable source provenance."""

    hit: SearchHit
    document_id: str
    document_title: str
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    content: str

    def __post_init__(self) -> None:
        identity = (self.hit.chunk_id, self.document_id, self.document_title)
        if any(not value.strip() for value in identity):
            raise ValueError("retrieved chunk provenance identity cannot be blank")
        if self.hit.rank <= 0 or not math.isfinite(self.hit.score):
            raise ValueError("retrieved chunk provenance rank and score must be valid")
        if self.page_start <= 0 or self.page_end < self.page_start:
            raise ValueError("retrieved chunk provenance page range is invalid")
        if not self.content.strip():
            raise ValueError("retrieved chunk provenance content cannot be blank")
        if any(not section.strip() for section in self.section_path):
            raise ValueError("retrieved chunk provenance section cannot be blank")
        object.__setattr__(self, "section_path", tuple(self.section_path))


@dataclass(frozen=True, slots=True)
class ContextItem:
    """One context item after stable citation assignment and safe serialization."""

    citation_id: str
    chunk_id: str
    document_id: str
    document_title: str
    page_start: int
    page_end: int
    section_path: tuple[str, ...]
    content: str
    retrieval_rank: int
    retrieval_score: float
    retriever: str
    serialized: str
    token_count: int


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """A stable top-ranked prefix that fits the exact serialized token budget."""

    items: tuple[ContextItem, ...]
    text: str
    token_count: int
    token_budget: int
    duplicate_chunk_ids: tuple[str, ...]
    truncated_chunk_ids: tuple[str, ...]


class ContextBuilder:
    """Build a deterministic, unambiguous context without splitting provenance records.

    Rows are ordered by retrieval rank, descending score, then stable source identity. Duplicate
    chunk IDs keep the first row under that ordering. Budgeting retains the largest top-ranked
    prefix whose complete serialized records fit; a record is never split and lower-ranked rows
    never replace a higher-ranked record that does not fit.
    """

    def build(
        self, hits: tuple[RetrievedChunk, ...] | list[RetrievedChunk], token_budget: int
    ) -> ContextBundle:
        if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < 0:
            raise ValueError("token_budget must be a nonnegative integer")
        ordered = sorted(
            hits,
            key=lambda row: (
                row.hit.rank,
                -row.hit.score,
                row.hit.chunk_id,
                row.document_id,
                row.page_start,
                row.page_end,
                row.section_path,
                row.content,
            ),
        )
        unique: list[RetrievedChunk] = []
        seen: dict[str, RetrievedChunk] = {}
        duplicates: set[str] = set()
        for row in ordered:
            prior = seen.get(row.hit.chunk_id)
            if prior is not None:
                if _source_identity(prior) != _source_identity(row):
                    raise ValueError("conflicting duplicate chunk provenance")
                duplicates.add(row.hit.chunk_id)
                continue
            seen[row.hit.chunk_id] = row
            unique.append(row)

        included: list[ContextItem] = []
        serialized: list[str] = []
        truncated: list[str] = []
        tokenizer = encoding()
        for index, row in enumerate(unique, start=1):
            citation_id = f"C{index}"
            item_text = _serialize(row, citation_id)
            candidate_text = _DOCUMENT_SEPARATOR.join((*serialized, item_text))
            candidate_tokens = len(tokenizer.encode(candidate_text))
            if candidate_tokens > token_budget:
                truncated.extend(item.hit.chunk_id for item in unique[index - 1 :])
                break
            serialized.append(item_text)
            included.append(
                ContextItem(
                    citation_id=citation_id,
                    chunk_id=row.hit.chunk_id,
                    document_id=row.document_id,
                    document_title=row.document_title,
                    page_start=row.page_start,
                    page_end=row.page_end,
                    section_path=row.section_path,
                    content=row.content,
                    retrieval_rank=row.hit.rank,
                    retrieval_score=row.hit.score,
                    retriever=row.hit.retriever,
                    serialized=item_text,
                    token_count=len(tokenizer.encode(item_text)),
                )
            )
        text = _DOCUMENT_SEPARATOR.join(serialized)
        return ContextBundle(
            items=tuple(included),
            text=text,
            token_count=len(tokenizer.encode(text)),
            token_budget=token_budget,
            duplicate_chunk_ids=tuple(sorted(duplicates)),
            truncated_chunk_ids=tuple(truncated),
        )


def _serialize(row: RetrievedChunk, citation_id: str) -> str:
    payload = {
        "citation_id": citation_id,
        "chunk_id": row.hit.chunk_id,
        "document_id": row.document_id,
        "document_title": row.document_title,
        "page_range": {"start": row.page_start, "end": row.page_end},
        "section": list(row.section_path),
        "content_is_untrusted_data": True,
        "content": row.content,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    # JSON quoting alone does not prevent content from visually forging the surrounding sentinel.
    # Escaping markup characters leaves the payload reversible while making sentinel boundaries
    # unambiguous to both parsers and model prompts.
    encoded = encoded.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return f"{_DOCUMENT_OPEN}\n{encoded}\n{_DOCUMENT_CLOSE}"


def _source_identity(row: RetrievedChunk) -> tuple[object, ...]:
    return (
        row.document_id,
        row.document_title,
        row.page_start,
        row.page_end,
        row.section_path,
        row.content,
    )
