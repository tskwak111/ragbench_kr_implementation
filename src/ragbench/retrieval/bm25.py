"""Dependency-free Korean BM25 baseline over one immutable chunk snapshot.

The tokenizer applies Unicode NFKC normalization and case folding, retains comma-grouped and
decimal numeric strings, and otherwise splits conservatively at whitespace/punctuation. Korean
particles are intentionally not segmented: this is a reproducible lexical baseline, not a
morphological analyzer. Repeated query tokens are treated once so duplicated wording does not
multiply identical evidence.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from ragbench.retrieval.base import SearchFilter, SearchHit

_TOKEN_PATTERN = re.compile(
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|[^\W\d_]+",
    flags=re.UNICODE,
)


def baseline_tokenize(text: str) -> tuple[str, ...]:
    """Return normalized conservative lexical tokens without morphology dependencies."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(match.group(0) for match in _TOKEN_PATTERN.finditer(normalized))


@dataclass(frozen=True, slots=True)
class BM25Document:
    chunk_id: str
    document_id: str
    content: str

    def __post_init__(self) -> None:
        if not self.chunk_id or not self.document_id or not self.content:
            raise ValueError("BM25 document identity and content cannot be empty")


@dataclass(frozen=True, slots=True)
class BM25IndexSnapshot:
    """Exact identity and immutable documents used to build one sparse index."""

    search_filter: SearchFilter
    documents: tuple[BM25Document, ...]

    def __post_init__(self) -> None:
        documents = tuple(self.documents)
        if len({document.chunk_id for document in documents}) != len(documents):
            raise ValueError("duplicate chunk IDs are not allowed in a BM25 snapshot")
        object.__setattr__(self, "documents", documents)


class BM25Retriever:
    """Standard Okapi BM25 with explicit, validated ``k1`` and ``b`` parameters."""

    def __init__(self, snapshot: BM25IndexSnapshot, *, k1: float = 1.2, b: float = 0.75) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be finite and positive")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be finite and between 0 and 1")
        self._snapshot = snapshot
        self._k1 = float(k1)
        self._b = float(b)
        self._tokens = tuple(baseline_tokenize(document.content) for document in snapshot.documents)
        self._term_frequencies = tuple(Counter(tokens) for tokens in self._tokens)
        self._document_frequencies = Counter(
            term for tokens in self._tokens for term in set(tokens)
        )
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens) if self._tokens else 0.0
        )

    async def search(self, query: str, *, top_k: int, filter: SearchFilter) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if filter != self._snapshot.search_filter and (
            filter.corpus_snapshot_id != self._snapshot.search_filter.corpus_snapshot_id
            or filter.parse_snapshot_id != self._snapshot.search_filter.parse_snapshot_id
            or filter.chunk_strategy != self._snapshot.search_filter.chunk_strategy
            or filter.embedding_snapshot_id != self._snapshot.search_filter.embedding_snapshot_id
        ):
            raise ValueError("search filter snapshot identity does not match BM25 index snapshot")
        query_terms = tuple(dict.fromkeys(baseline_tokenize(query)))
        if not query_terms or not self._snapshot.documents:
            return []

        allowed_documents = set(filter.document_ids)
        row_count = len(self._snapshot.documents)
        scored: list[tuple[str, float]] = []
        for document, frequencies, tokens in zip(
            self._snapshot.documents, self._term_frequencies, self._tokens, strict=True
        ):
            if allowed_documents and document.document_id not in allowed_documents:
                continue
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                document_frequency = self._document_frequencies[term]
                inverse_document_frequency = math.log(
                    1 + (row_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                length_ratio = len(tokens) / self._average_length
                denominator = term_frequency + self._k1 * (1 - self._b + self._b * length_ratio)
                score += inverse_document_frequency * (
                    term_frequency * (self._k1 + 1) / denominator
                )
            if score > 0:
                scored.append((document.chunk_id, score))

        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            SearchHit(chunk_id, score, rank, "bm25")
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        ]
