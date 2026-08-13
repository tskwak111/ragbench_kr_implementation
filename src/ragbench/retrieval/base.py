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
        object.__setattr__(self, "document_ids", tuple(sorted(set(self.document_ids))))


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    score: float
    rank: int
    retriever: str


class Retriever(Protocol):
    async def search(
        self, query: str, *, top_k: int, filter: SearchFilter
    ) -> list[SearchHit]: ...
