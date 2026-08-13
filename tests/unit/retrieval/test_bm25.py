"""Behavior tests for the dependency-free Korean BM25 baseline."""

import pytest

from ragbench.retrieval.base import SearchFilter
from ragbench.retrieval.bm25 import (
    BM25Document,
    BM25IndexSnapshot,
    BM25Retriever,
    baseline_tokenize,
)

FILTER = SearchFilter("corpus-a", "parse-a", "fixed-300", "embed-a")


def _retriever(*documents: BM25Document) -> BM25Retriever:
    return BM25Retriever(BM25IndexSnapshot(FILTER, documents))


def test_baseline_tokenizer_normalizes_unicode_case_and_conservative_boundaries() -> None:
    """Catch compatibility characters, case, punctuation, or numeric units joining tokens."""
    assert baseline_tokenize("ＡI, 매출은 1,234.50원 / 성장률 12.5%") == (
        "ai",
        "매출은",
        "1,234.50",
        "원",
        "성장률",
        "12.5",
    )


@pytest.mark.asyncio
async def test_bm25_preserves_exact_numeric_terms_and_document_filter() -> None:
    """Catch comma/decimal destruction or omission of the shared document predicate."""
    retriever = _retriever(
        BM25Document("chunk-b", "doc-b", "2024년 매출은 1,234.50 원이다"),
        BM25Document("chunk-a", "doc-a", "2024년 매출은 1234.50 원이다"),
        BM25Document("chunk-c", "doc-a", "2023년 매출은 1,234.50 원이다"),
    )
    restricted = SearchFilter("corpus-a", "parse-a", "fixed-300", "embed-a", ("doc-a",))

    hits = await retriever.search("1,234.50", top_k=5, filter=restricted)

    assert [hit.chunk_id for hit in hits] == ["chunk-c"]
    assert all(hit.retriever == "bm25" for hit in hits)
    assert [hit.rank for hit in hits] == [1]


@pytest.mark.asyncio
async def test_bm25_empty_query_returns_no_hits_without_dividing_by_zero() -> None:
    """Catch empty-token queries entering scoring and producing NaN or arbitrary rows."""
    retriever = _retriever(BM25Document("chunk-a", "doc-a", "내용"))

    assert await retriever.search(" --- ", top_k=3, filter=FILTER) == []


@pytest.mark.asyncio
async def test_bm25_score_matches_hand_calculated_okapi_formula() -> None:
    """Catch changes to standard k1/b length normalization or IDF math."""
    retriever = _retriever(
        BM25Document("chunk-a", "doc-a", "사과 사과"),
        BM25Document("chunk-b", "doc-b", "사과 배 배 배"),
    )

    hits = await retriever.search("사과", top_k=2, filter=FILTER)

    assert [hit.chunk_id for hit in hits] == ["chunk-a", "chunk-b"]
    assert [hit.score for hit in hits] == pytest.approx(
        [0.27662581030806904, 0.16044296997868007], abs=1e-14
    )


@pytest.mark.asyncio
async def test_bm25_repeated_query_terms_do_not_multiply_the_same_evidence() -> None:
    """Catch accidental query-term frequency inflation in the documented binary-query baseline."""
    retriever = _retriever(
        BM25Document("chunk-a", "doc-a", "기준 기준 보고서"),
        BM25Document("chunk-b", "doc-b", "기준 보고서"),
    )

    single = await retriever.search("기준", top_k=2, filter=FILTER)
    repeated = await retriever.search("기준 기준 기준", top_k=2, filter=FILTER)

    assert [(hit.chunk_id, hit.score) for hit in repeated] == [
        (hit.chunk_id, hit.score) for hit in single
    ]


@pytest.mark.asyncio
async def test_bm25_breaks_equal_score_ties_by_stable_chunk_id() -> None:
    """Catch rankings that depend on source document insertion order."""
    retriever = _retriever(
        BM25Document("chunk-z", "doc-z", "동일 길이"),
        BM25Document("chunk-a", "doc-a", "동일 길이"),
    )

    hits = await retriever.search("동일", top_k=2, filter=FILTER)

    assert [hit.chunk_id for hit in hits] == ["chunk-a", "chunk-z"]


@pytest.mark.asyncio
async def test_bm25_rejects_snapshot_identity_drift_and_invalid_parameters() -> None:
    """Catch sparse retrieval silently searching a different immutable chunk universe."""
    retriever = _retriever(BM25Document("chunk-a", "doc-a", "내용"))

    with pytest.raises(ValueError, match="snapshot"):
        await retriever.search(
            "내용", top_k=1, filter=SearchFilter("other", "parse-a", "fixed-300", "embed-a")
        )
    with pytest.raises(ValueError, match="positive"):
        await retriever.search("내용", top_k=0, filter=FILTER)
    with pytest.raises(ValueError, match="k1"):
        BM25Retriever(BM25IndexSnapshot(FILTER, ()), k1=0)
    with pytest.raises(ValueError, match="b"):
        BM25Retriever(BM25IndexSnapshot(FILTER, ()), b=1.1)


def test_bm25_snapshot_rejects_duplicate_ids_and_is_immutable() -> None:
    """Catch ambiguous evidence IDs or a mutable index identity/data boundary."""
    document = BM25Document("chunk-a", "doc-a", "내용")
    with pytest.raises(ValueError, match="duplicate"):
        BM25IndexSnapshot(FILTER, (document, document))
    with pytest.raises(AttributeError):
        document.content = "변경"  # type: ignore[misc]


def test_search_filter_rejects_blank_snapshot_or_document_identity() -> None:
    """Catch an index/search request whose immutable evidence identity is ambiguous."""
    with pytest.raises(ValueError, match="identity"):
        SearchFilter("", "parse", "fixed", "embed")
    with pytest.raises(ValueError, match="document"):
        SearchFilter("corpus", "parse", "fixed", "embed", ("",))
