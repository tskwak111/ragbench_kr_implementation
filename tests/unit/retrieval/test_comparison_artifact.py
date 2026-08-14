"""Executable check for the public-safe fixed Korean retriever comparison artifact."""

import json
from pathlib import Path

import numpy as np
import pytest

from ragbench.retrieval.base import SearchFilter
from ragbench.retrieval.bm25 import BM25Document, BM25IndexSnapshot, BM25Retriever
from ragbench.retrieval.dense import cosine_top_k
from ragbench.retrieval.service import HybridRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class FixedDenseRetriever:
    def __init__(
        self,
        search_filter: SearchFilter,
        chunk_ids: tuple[str, ...],
        vectors: np.ndarray,
        query_vectors: dict[str, np.ndarray],
    ) -> None:
        self._filter = search_filter
        self._chunk_ids = chunk_ids
        self._vectors = vectors
        self._query_vectors = query_vectors

    async def search(self, query: str, *, top_k: int, filter: SearchFilter):  # type: ignore[no-untyped-def]
        if filter != self._filter:
            raise ValueError("snapshot mismatch")
        return cosine_top_k(self._query_vectors[query], self._vectors, top_k, self._chunk_ids)


@pytest.mark.asyncio
async def test_fixed_korean_fact_numeric_paraphrase_comparison_is_reproducible() -> None:
    """Catch undocumented comparison drift without relying on provider, DB, or private corpus."""
    payload = json.loads(
        (PROJECT_ROOT / "artifacts/retrieval/korean-fixed-comparison.json").read_text()
    )
    assert payload["fixture_scope"] == "synthetic-public-safe-offline"
    search_filter = SearchFilter(**payload["search_filter"])
    documents = tuple(
        BM25Document(row["chunk_id"], row["document_id"], row["content"])
        for row in payload["documents"]
    )
    sparse = BM25Retriever(BM25IndexSnapshot(search_filter, documents))
    chunk_ids = tuple(row["chunk_id"] for row in payload["documents"])
    dense = FixedDenseRetriever(
        search_filter,
        chunk_ids,
        np.asarray([row["vector"] for row in payload["documents"]], dtype=np.float64),
        {row["query"]: np.asarray(row["vector"], dtype=np.float64) for row in payload["queries"]},
    )
    hybrid = HybridRetriever(dense, sparse)

    actual: list[dict[str, object]] = []
    for row in payload["queries"]:
        query = row["query"]
        dense_hits = await dense.search(query, top_k=3, filter=search_filter)
        sparse_hits = await sparse.search(query, top_k=3, filter=search_filter)
        hybrid_hits = await hybrid.search(query, top_k=3, filter=search_filter)
        actual.append(
            {
                "type": row["type"],
                "query": query,
                "dense": [hit.chunk_id for hit in dense_hits],
                "bm25": [hit.chunk_id for hit in sparse_hits],
                "hybrid": [hit.chunk_id for hit in hybrid_hits],
            }
        )

    assert actual == payload["expected"]
