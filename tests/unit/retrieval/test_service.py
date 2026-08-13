"""Hybrid retrieval orchestration and evidence tests."""

from collections.abc import Sequence

import pytest

from ragbench.retrieval.base import SearchFilter, SearchHit
from ragbench.retrieval.service import HybridRetriever


class RecordingRetriever:
    def __init__(self, name: str, rows: Sequence[tuple[str, float]]) -> None:
        self.name = name
        self.rows = tuple(rows)
        self.calls: list[tuple[str, int, SearchFilter]] = []
        self.filter_objects: list[SearchFilter] = []

    async def search(
        self, query: str, *, top_k: int, filter: SearchFilter
    ) -> list[SearchHit]:
        self.calls.append((query, top_k, filter))
        self.filter_objects.append(filter)
        return [
            SearchHit(chunk_id, score, rank, self.name)
            for rank, (chunk_id, score) in enumerate(self.rows[:top_k], start=1)
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("top_k", "expected_fetch"), [(1, 20), (5, 20), (6, 24)])
async def test_hybrid_overfetches_each_branch_by_exact_policy(
    top_k: int, expected_fetch: int
) -> None:
    """Catch retrieval-depth drift from max(20, 4*top_k)."""
    dense = RecordingRetriever("dense", [("a", 0.9)])
    sparse = RecordingRetriever("bm25", [("a", 2.0)])
    hybrid = HybridRetriever(dense, sparse)
    search_filter = SearchFilter("corpus", "parse", "heading", "embed", ("doc-a",))

    await hybrid.search("질문", top_k=top_k, filter=search_filter)

    assert dense.calls == [("질문", expected_fetch, search_filter)]
    assert sparse.calls == [("질문", expected_fetch, search_filter)]
    assert dense.filter_objects[0] is search_filter
    assert sparse.filter_objects[0] is search_filter


@pytest.mark.asyncio
async def test_hybrid_records_component_ranks_scores_and_fused_score() -> None:
    """Catch fused rankings that discard auditable dense/sparse evidence."""
    dense = RecordingRetriever("dense", [("a", 0.9), ("b", 0.8), ("c", 0.7)])
    sparse = RecordingRetriever("bm25", [("b", 4.0), ("d", 3.0), ("a", 2.0)])
    hybrid = HybridRetriever(dense, sparse, rrf_k=60, weights=(1.0, 2.0))

    hits = await hybrid.search(
        "질문", top_k=3, filter=SearchFilter("corpus", "parse", "heading", "embed")
    )

    assert [hit.chunk_id for hit in hits] == ["b", "a", "d"]
    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert all(hit.retriever == "hybrid-rrf" for hit in hits)
    evidence = {hit.chunk_id: hit.evidence for hit in hits}
    assert evidence["b"] is not None
    assert evidence["b"].dense_rank == 2
    assert evidence["b"].sparse_rank == 1
    assert evidence["b"].dense_score == 0.8
    assert evidence["b"].sparse_score == 4.0
    assert evidence["b"].fused_score == pytest.approx(1 / 62 + 2 / 61)
    assert evidence["d"] is not None
    assert evidence["d"].dense_rank is None
    assert evidence["d"].dense_score is None


@pytest.mark.asyncio
async def test_hybrid_rejects_invalid_top_k_before_branch_calls() -> None:
    """Catch pointless branch work for invalid top-k requests."""
    dense = RecordingRetriever("dense", [])
    sparse = RecordingRetriever("bm25", [])

    with pytest.raises(ValueError, match="positive"):
        await HybridRetriever(dense, sparse).search(
            "질문", top_k=0, filter=SearchFilter("c", "p", "s", "e")
        )

    assert dense.calls == []
    assert sparse.calls == []
