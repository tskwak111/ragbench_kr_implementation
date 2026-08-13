"""Configured-PostgreSQL parity between NumPy and pgvector cosine ranking."""

import os

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragbench.embeddings.repository import SqlAlchemyEmbeddingRepository
from ragbench.retrieval.base import SearchFilter
from ragbench.retrieval.dense import cosine_top_k


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pgvector_matches_numpy_for_multiple_ties_within_tolerance() -> None:
    """Catch distance conversion or tie policy divergence on a configured database fixture."""
    database_url = os.getenv("RAGBENCH_TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("RAGBENCH_TEST_DATABASE_URL is not configured")
    fixture_snapshot = os.getenv("RAGBENCH_PARITY_SNAPSHOT_ID")
    if fixture_snapshot is None:
        pytest.skip("RAGBENCH_PARITY_SNAPSHOT_ID is not configured")

    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    chunk_ids = ("00000000-0000-0000-0000-000000000001",
                 "00000000-0000-0000-0000-000000000002",
                 "00000000-0000-0000-0000-000000000003",
                 "00000000-0000-0000-0000-000000000004")
    search_filter = SearchFilter(
        os.environ["RAGBENCH_PARITY_CORPUS_SNAPSHOT_ID"],
        os.environ["RAGBENCH_PARITY_PARSE_SNAPSHOT_ID"],
        os.environ["RAGBENCH_PARITY_CHUNK_STRATEGY"],
        fixture_snapshot,
    )
    engine = create_async_engine(database_url)
    repository = SqlAlchemyEmbeddingRepository(async_sessionmaker(engine, expire_on_commit=False))
    try:
        for query in (np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([0.0, 1.0])):
            expected = cosine_top_k(query, matrix, 4, chunk_ids)
            actual = await repository.search(tuple(query), top_k=4, filter=search_filter)
            assert [chunk_id for chunk_id, _ in actual] == [hit.chunk_id for hit in expected]
            assert [score for _, score in actual] == pytest.approx(
                [hit.score for hit in expected], abs=1e-5
            )
    finally:
        await engine.dispose()
