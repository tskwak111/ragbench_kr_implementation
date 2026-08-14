"""Shared retrieval contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SearchFilter:
    corpus_snapshot_id: str
    parse_snapshot_id: str
    chunk_strategy: str
    embedding_snapshot_id: str
    document_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        identity = (
            self.corpus_snapshot_id,
            self.parse_snapshot_id,
            self.chunk_strategy,
            self.embedding_snapshot_id,
        )
        if any(not value.strip() for value in identity):
            raise ValueError("search filter snapshot identity cannot be blank")
        if any(not document_id.strip() for document_id in self.document_ids):
            raise ValueError("search filter document identity cannot be blank")
        object.__setattr__(self, "document_ids", tuple(sorted(set(self.document_ids))))


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    score: float
    rank: int
    retriever: str
    evidence: RetrievalEvidence | None = None


@dataclass(frozen=True, slots=True)
class RetrievalEvidence:
    """Auditable component evidence attached to a fused retrieval hit."""

    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float | None
    sparse_score: float | None
    fused_score: float


class Retriever(Protocol):
    async def search(self, query: str, *, top_k: int, filter: SearchFilter) -> list[SearchHit]: ...
