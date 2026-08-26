"""Self-contained NumPy/pgvector parity on configured PostgreSQL."""

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ragbench.core.config import Settings
from ragbench.db.session import require_distinct_database
from ragbench.embeddings.repository import (
    ChunkEmbeddingInput,
    EmbeddingSnapshot,
    SqlAlchemyEmbeddingRepository,
    chunk_manifest_hash,
    dense_search_spec,
    embedding_index_plan,
)
from ragbench.retrieval.base import SearchFilter
from ragbench.retrieval.dense import cosine_top_k

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DIMENSION = 2001
SNAPSHOT_ID = UUID("00000000-0000-0000-0000-000000000801")


def _alembic_config(database_url: str) -> Config:
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


async def _reset_schema(database_url: str) -> None:
    require_distinct_database(database_url, Settings().database_url)
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            await connection.execute(text("CREATE SCHEMA public"))
    finally:
        await engine.dispose()


def _fixture() -> tuple[tuple[ChunkEmbeddingInput, ...], np.ndarray]:
    chunks: list[ChunkEmbeddingInput] = []
    vectors: list[np.ndarray] = []
    for index in range(100):
        group = index % 4
        within_group = index // 4
        chunks.append(
            ChunkEmbeddingInput(
                f"chunk-{index:02d}",
                "doc-a" if index < 50 else "doc-b",
                f"문서 조각 {index}",
                3,
            )
        )
        vector = np.zeros(DIMENSION)
        tie_group = within_group // 2
        angle = (tie_group - 6) * 0.05
        vector[group * 2] = np.cos(angle)
        vector[group * 2 + 1] = np.sin(angle)
        # Each pair is an exact full-vector tie; distinct groups have a tail large enough
        # to remain ordered after pgvector's float32 storage and 1e-5 score tolerance.
        vector[-1] = ((tie_group * 7) % 9 - 4) * 0.02
        vectors.append(vector / np.linalg.norm(vector))
    return tuple(chunks), np.vstack(vectors)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pgvector_subvector_rerank_matches_numpy_for_50_queries_and_uses_index() -> None:
    """Catch non-self-contained parity, insufficient reranking, tie drift, or unused HNSW DDL."""
    database_url = os.getenv("RAGBENCH_TEST_DATABASE_URL")
    if database_url is None:
        pytest.skip("RAGBENCH_TEST_DATABASE_URL is not configured")
    await _reset_schema(database_url)
    await asyncio.to_thread(command.upgrade, _alembic_config(database_url), "head")
    engine = create_async_engine(database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    repository = SqlAlchemyEmbeddingRepository(factory)
    chunks, matrix = _fixture()
    plan = embedding_index_plan(DIMENSION)
    snapshot = EmbeddingSnapshot(
        str(SNAPSHOT_ID),
        "corpus-parity",
        "parse-parity",
        "fixed-parity",
        "embedding-passage",
        "embedding-query",
        DIMENSION,
        "l2",
        len(chunks),
        chunk_manifest_hash(chunks),
        plan.strategy,
        plan.candidate_factor,
        datetime(2026, 8, 14, tzinfo=UTC),
    )
    try:
        # Concurrent identical registration/persistence exercises snapshot advisory serialization.
        await asyncio.gather(
            repository.create_snapshot(snapshot, chunks),
            repository.create_snapshot(snapshot, chunks),
        )
        vector_rows = tuple(
            (chunk.chunk_id, tuple(float(value) for value in vector))
            for chunk, vector in zip(chunks, matrix, strict=True)
        )
        await asyncio.gather(
            repository.persist_batch(snapshot.snapshot_id, vector_rows[:50]),
            repository.persist_batch(snapshot.snapshot_id, vector_rows[50:]),
        )
        await repository.persist_batch(snapshot.snapshot_id, vector_rows[:1])
        completed, duplicate = await asyncio.gather(
            repository.finalize_snapshot(snapshot.snapshot_id),
            repository.finalize_snapshot(snapshot.snapshot_id),
        )
        assert completed.complete and duplicate.complete

        search_filter = SearchFilter(
            "corpus-parity", "parse-parity", "fixed-parity", str(SNAPSHOT_ID)
        )
        chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
        queries: list[np.ndarray] = []
        for index in range(50):
            group = index % 4
            query = np.zeros(DIMENSION)
            target = (index % 7) - 3
            query[group * 2] = np.cos(target * 0.05)
            query[group * 2 + 1] = np.sin(target * 0.05)
            query[-1] = ((index % 9) - 4) * 0.02
            queries.append(query / np.linalg.norm(query))
        for query in queries:
            expected = cosine_top_k(query, matrix, 5, chunk_ids)
            actual = await repository.search(tuple(query), top_k=5, filter=search_filter)
            assert [chunk_id for chunk_id, _ in actual] == [hit.chunk_id for hit in expected]
            assert [score for _, score in actual] == pytest.approx(
                [hit.score for hit in expected], abs=1e-5
            )
        document_filter = SearchFilter(
            "corpus-parity", "parse-parity", "fixed-parity", str(SNAPSHOT_ID), ("doc-a",)
        )
        filtered = await repository.search(tuple(queries[0]), top_k=20, filter=document_filter)
        assert filtered
        assert all(int(chunk_id.removeprefix("chunk-")) < 50 for chunk_id, _ in filtered)

        spec = dense_search_spec(SNAPSHOT_ID, dimension=DIMENSION, top_k=5)
        assert spec.params["candidate_k"] == 20
        assert spec.params["candidate_k"] < len(chunks)
        query_literal = "[" + ",".join(format(value, ".17g") for value in queries[0]) + "]"
        async with engine.begin() as connection:
            await connection.execute(text("ANALYZE chunk_embedding"))
            await connection.execute(text("SET LOCAL enable_seqscan = off"))
            # A 100-row fixture makes a btree scan plus sort cheaper than HNSW even
            # with sequential scans disabled. Disable explicit sorts to prove the
            # expression/predicate pair can use the ordered HNSW access path.
            await connection.execute(text("SET LOCAL enable_sort = off"))
            explain = await connection.execute(
                text("EXPLAIN " + spec.sql), {**spec.params, "query": query_literal}
            )
            plan_text = "\n".join(str(row[0]) for row in explain)
        assert completed.index_name is not None
        assert completed.index_name in plan_text
    finally:
        await engine.dispose()
