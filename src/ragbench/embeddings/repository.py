"""Persistence contracts for immutable embedding snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragbench.db.models import ChunkEmbedding
from ragbench.db.models import EmbeddingSnapshot as SnapshotRow
from ragbench.retrieval.base import SearchFilter


@dataclass(frozen=True, slots=True)
class ChunkEmbeddingInput:
    chunk_id: str
    content: str
    token_count: int

    def __post_init__(self) -> None:
        if not self.chunk_id or not self.content:
            raise ValueError("chunk identity and content cannot be empty")
        if self.token_count <= 0:
            raise ValueError("chunk token_count must be positive")


@dataclass(frozen=True, slots=True)
class EmbeddingSnapshot:
    snapshot_id: str
    corpus_snapshot_id: str
    parse_snapshot_id: str
    chunk_strategy: str
    model_id: str
    query_model_id: str
    dimension: int
    normalization: str
    expected_chunk_count: int
    created_at: datetime
    complete: bool = False
    index_name: str | None = None
    index_state: str = "pending"

    def __post_init__(self) -> None:
        required = (
            self.snapshot_id,
            self.corpus_snapshot_id,
            self.parse_snapshot_id,
            self.chunk_strategy,
            self.model_id,
            self.query_model_id,
        )
        if any(not value for value in required):
            raise ValueError("embedding snapshot identity fields cannot be empty")
        if self.dimension <= 0 or self.expected_chunk_count < 0:
            raise ValueError("embedding dimension must be positive and count nonnegative")
        if self.normalization != "l2":
            raise ValueError("only l2 normalization is supported")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must include a timezone")

    def immutable_identity(self) -> tuple[object, ...]:
        return (
            self.snapshot_id,
            self.corpus_snapshot_id,
            self.parse_snapshot_id,
            self.chunk_strategy,
            self.model_id,
            self.query_model_id,
            self.dimension,
            self.normalization,
            self.expected_chunk_count,
            self.created_at,
        )


class EmbeddingRepository(Protocol):
    async def create_snapshot(self, snapshot: EmbeddingSnapshot) -> EmbeddingSnapshot: ...

    async def get_snapshot(self, snapshot_id: str) -> EmbeddingSnapshot | None: ...

    async def completed_chunk_ids(self, snapshot_id: str) -> set[str]: ...

    async def persist_batch(
        self, snapshot_id: str, vectors: Sequence[tuple[str, tuple[float, ...]]]
    ) -> None: ...

    async def finalize_snapshot(self, snapshot_id: str) -> EmbeddingSnapshot: ...


@dataclass(frozen=True, slots=True)
class HnswIndexSpec:
    name: str
    sql: str


def hnsw_index_spec(snapshot_id: UUID, dimension: int) -> HnswIndexSpec:
    """Return injection-safe DDL for one dimension-specific partial HNSW index."""
    if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= 4000:
        raise ValueError("HNSW dimension must be an integer from 1 through 4000")
    cast_type = "vector" if dimension <= 2000 else "halfvec"
    ops = "vector_cosine_ops" if cast_type == "vector" else "halfvec_cosine_ops"
    name = f"ix_chunk_embedding_hnsw_{snapshot_id.hex}_{dimension}"
    sql = (
        f"CREATE INDEX IF NOT EXISTS {name} ON chunk_embedding USING hnsw "
        f"((embedding::{cast_type}({dimension})) {ops}) "
        f"WHERE embedding_snapshot_id = '{snapshot_id}'::uuid"
    )
    return HnswIndexSpec(name, sql)


class MemoryEmbeddingRepository:
    """Deterministic repository used by offline tests and notebook examples."""

    def __init__(self) -> None:
        self.snapshots: dict[str, EmbeddingSnapshot] = {}
        self.vectors: dict[str, dict[str, tuple[float, ...]]] = {}
        self.finalize_attempts = 0

    async def create_snapshot(self, snapshot: EmbeddingSnapshot) -> EmbeddingSnapshot:
        existing = self.snapshots.get(snapshot.snapshot_id)
        if existing is not None:
            if existing.immutable_identity() != snapshot.immutable_identity():
                raise ValueError("embedding snapshot metadata is immutable")
            return existing
        self.snapshots[snapshot.snapshot_id] = snapshot
        self.vectors[snapshot.snapshot_id] = {}
        return snapshot

    async def get_snapshot(self, snapshot_id: str) -> EmbeddingSnapshot | None:
        return self.snapshots.get(snapshot_id)

    async def completed_chunk_ids(self, snapshot_id: str) -> set[str]:
        self._require_snapshot(snapshot_id)
        return set(self.vectors[snapshot_id])

    async def persist_batch(
        self, snapshot_id: str, vectors: Sequence[tuple[str, tuple[float, ...]]]
    ) -> None:
        snapshot = self._require_snapshot(snapshot_id)
        if snapshot.complete:
            raise RuntimeError("complete embedding snapshots are immutable")
        target = self.vectors[snapshot_id]
        for chunk_id, vector in vectors:
            if len(vector) != snapshot.dimension:
                raise ValueError("embedding dimension does not match snapshot")
            existing = target.get(chunk_id)
            if existing is not None and existing != vector:
                raise ValueError("chunk embedding is immutable")
            target[chunk_id] = vector

    async def finalize_snapshot(self, snapshot_id: str) -> EmbeddingSnapshot:
        self.finalize_attempts += 1
        snapshot = self._require_snapshot(snapshot_id)
        if len(self.vectors[snapshot_id]) != snapshot.expected_chunk_count:
            raise RuntimeError("cannot complete a partial embedding snapshot")
        completed = replace(snapshot, complete=True, index_state="ready")
        self.snapshots[snapshot_id] = completed
        return completed

    def _require_snapshot(self, snapshot_id: str) -> EmbeddingSnapshot:
        try:
            return self.snapshots[snapshot_id]
        except KeyError as error:
            raise KeyError(f"unknown embedding snapshot: {snapshot_id}") from error


class SqlAlchemyEmbeddingRepository:
    """PostgreSQL storage with transactionally finalized, typed HNSW indexes."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create_snapshot(self, snapshot: EmbeddingSnapshot) -> EmbeddingSnapshot:
        snapshot_uuid = UUID(snapshot.snapshot_id)
        async with self._session_factory() as session, session.begin():
            existing = await session.get(SnapshotRow, snapshot_uuid)
            if existing is not None:
                restored = self._from_row(existing)
                if restored.immutable_identity() != snapshot.immutable_identity():
                    raise ValueError("embedding snapshot metadata is immutable")
                return restored
            session.add(
                SnapshotRow(
                    id=snapshot_uuid,
                    parse_run_id=None,
                    corpus_snapshot_id=snapshot.corpus_snapshot_id,
                    parse_snapshot_id=snapshot.parse_snapshot_id,
                    chunk_strategy=snapshot.chunk_strategy,
                    model_id=snapshot.model_id,
                    query_model_id=snapshot.query_model_id,
                    dimension=snapshot.dimension,
                    normalization=snapshot.normalization,
                    expected_chunk_count=snapshot.expected_chunk_count,
                    complete=False,
                    index_name=None,
                    index_state="pending",
                    config_snapshot={},
                    created_at=snapshot.created_at,
                )
            )
        return snapshot

    async def get_snapshot(self, snapshot_id: str) -> EmbeddingSnapshot | None:
        async with self._session_factory() as session:
            row = await session.get(SnapshotRow, UUID(snapshot_id))
            return None if row is None else self._from_row(row)

    async def completed_chunk_ids(self, snapshot_id: str) -> set[str]:
        async with self._session_factory() as session:
            values = await session.scalars(
                select(ChunkEmbedding.chunk_id).where(
                    ChunkEmbedding.embedding_snapshot_id == UUID(snapshot_id)
                )
            )
            return {str(value) for value in values}

    async def persist_batch(
        self, snapshot_id: str, vectors: Sequence[tuple[str, tuple[float, ...]]]
    ) -> None:
        snapshot_uuid = UUID(snapshot_id)
        async with self._session_factory() as session, session.begin():
            snapshot = await session.get(SnapshotRow, snapshot_uuid, with_for_update=True)
            if snapshot is None:
                raise KeyError(f"unknown embedding snapshot: {snapshot_id}")
            if snapshot.complete:
                raise RuntimeError("complete embedding snapshots are immutable")
            for chunk_id, vector in vectors:
                if len(vector) != snapshot.dimension:
                    raise ValueError("embedding dimension does not match snapshot")
                statement = insert(ChunkEmbedding).values(
                    embedding_snapshot_id=snapshot_uuid,
                    chunk_id=chunk_id,
                    dimension=snapshot.dimension,
                    embedding=vector,
                ).on_conflict_do_nothing(
                    index_elements=[
                        ChunkEmbedding.embedding_snapshot_id,
                        ChunkEmbedding.chunk_id,
                    ]
                )
                await session.execute(statement)

    async def finalize_snapshot(self, snapshot_id: str) -> EmbeddingSnapshot:
        snapshot_uuid = UUID(snapshot_id)
        async with self._session_factory() as session, session.begin():
            snapshot = await session.get(SnapshotRow, snapshot_uuid, with_for_update=True)
            if snapshot is None:
                raise KeyError(f"unknown embedding snapshot: {snapshot_id}")
            count = await session.scalar(
                select(func.count(ChunkEmbedding.id)).where(
                    ChunkEmbedding.embedding_snapshot_id == snapshot_uuid
                )
            )
            if count != snapshot.expected_chunk_count:
                raise RuntimeError("cannot complete a partial embedding snapshot")
            spec = hnsw_index_spec(snapshot_uuid, snapshot.dimension)
            snapshot.index_state = "building"
            await session.flush()
            await session.execute(text(spec.sql))
            snapshot.index_name = spec.name
            snapshot.index_state = "ready"
            snapshot.complete = True
            await session.flush()
            return self._from_row(snapshot)

    async def search(
        self, query_vector: tuple[float, ...], *, top_k: int, filter: SearchFilter
    ) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        snapshot_uuid = UUID(filter.embedding_snapshot_id)
        async with self._session_factory() as session:
            snapshot = await session.get(SnapshotRow, snapshot_uuid)
            if snapshot is None:
                raise KeyError(f"unknown embedding snapshot: {filter.embedding_snapshot_id}")
            if not snapshot.complete or snapshot.index_state != "ready":
                raise RuntimeError("embedding snapshot is incomplete")
            if (
                snapshot.corpus_snapshot_id != filter.corpus_snapshot_id
                or snapshot.parse_snapshot_id != filter.parse_snapshot_id
                or snapshot.chunk_strategy != filter.chunk_strategy
            ):
                raise ValueError("search filter does not match embedding snapshot")
            if len(query_vector) != snapshot.dimension:
                raise ValueError("query embedding dimension does not match snapshot")
            cast_type = "vector" if snapshot.dimension <= 2000 else "halfvec"
            vector_literal = "[" + ",".join(format(value, ".17g") for value in query_vector) + "]"
            distance = (
                f"ce.embedding::{cast_type}({snapshot.dimension}) <=> "
                f"CAST(:query AS {cast_type}({snapshot.dimension}))"
            )
            query = text(
                "SELECT ce.chunk_id, 1.0 - (" + distance + ") AS score "
                "FROM chunk_embedding ce "
                "WHERE ce.embedding_snapshot_id = :snapshot_id "
                "ORDER BY (" + distance + ") ASC, ce.chunk_id ASC LIMIT :top_k"
            )
            rows = await session.execute(
                query,
                {
                    "query": vector_literal,
                    "snapshot_id": snapshot_uuid,
                    "top_k": top_k,
                },
            )
            return [(str(chunk_id), float(score)) for chunk_id, score in rows]

    @staticmethod
    def _from_row(row: SnapshotRow) -> EmbeddingSnapshot:
        return EmbeddingSnapshot(
            snapshot_id=str(row.id),
            corpus_snapshot_id=row.corpus_snapshot_id,
            parse_snapshot_id=row.parse_snapshot_id,
            chunk_strategy=row.chunk_strategy,
            model_id=row.model_id,
            query_model_id=row.query_model_id,
            dimension=row.dimension,
            normalization=row.normalization,
            expected_chunk_count=row.expected_chunk_count,
            created_at=row.created_at,
            complete=row.complete,
            index_name=row.index_name,
            index_state=row.index_state,
        )
