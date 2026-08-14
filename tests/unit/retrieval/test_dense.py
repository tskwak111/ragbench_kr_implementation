"""Dense retriever contracts independent of PostgreSQL availability."""

import pytest

from ragbench.retrieval.base import SearchFilter
from ragbench.retrieval.dense import DenseRetriever


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    async def embed_query(
        self, query: str, *, snapshot_id: str, input_tokens: int
    ) -> tuple[float, ...]:
        self.calls.append((query, snapshot_id, input_tokens))
        return (1.0, 0.0)


class FakeSearchRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[float, ...], int, SearchFilter]] = []

    async def search(
        self, query_vector: tuple[float, ...], *, top_k: int, filter: SearchFilter
    ) -> list[tuple[str, float]]:
        self.calls.append((query_vector, top_k, filter))
        return [("chunk-a", 1.0), ("chunk-b", 0.25)]


@pytest.mark.asyncio
async def test_dense_retriever_embeds_as_query_and_passes_all_exact_filters() -> None:
    """Catch query/document mode drift or omission of snapshot provenance filters."""
    embeddings = FakeEmbeddingService()
    repository = FakeSearchRepository()
    retriever = DenseRetriever(embeddings, repository, token_counter=lambda _: 7)
    search_filter = SearchFilter(
        "corpus-a", "parse-a", "fixed-300", "snapshot-a", ("doc-b", "doc-a")
    )

    hits = await retriever.search("질문", top_k=2, filter=search_filter)

    assert embeddings.calls == [("질문", "snapshot-a", 7)]
    assert repository.calls == [((1.0, 0.0), 2, search_filter)]
    assert [(hit.chunk_id, hit.score, hit.rank, hit.retriever) for hit in hits] == [
        ("chunk-a", 1.0, 1, "dense"),
        ("chunk-b", 0.25, 2, "dense"),
    ]


@pytest.mark.asyncio
async def test_dense_retriever_rejects_invalid_top_k_before_embedding() -> None:
    """Catch provider calls for a meaningless retrieval request."""
    embeddings = FakeEmbeddingService()
    retriever = DenseRetriever(embeddings, FakeSearchRepository(), token_counter=lambda _: 1)

    with pytest.raises(ValueError, match="positive"):
        await retriever.search("질문", top_k=0, filter=SearchFilter("c", "p", "s", "e"))

    assert embeddings.calls == []


def test_search_filter_normalizes_document_ids_and_defines_empty_as_no_restriction() -> None:
    """Catch nondeterministic document predicates or treating empty as match-nothing."""
    filtered = SearchFilter("c", "p", "s", "e", ("doc-b", "doc-a", "doc-b"))
    unrestricted = SearchFilter("c", "p", "s", "e", ())

    assert filtered.document_ids == ("doc-a", "doc-b")
    assert unrestricted.document_ids == ()
