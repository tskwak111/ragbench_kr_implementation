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
