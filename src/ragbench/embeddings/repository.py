"""Persistence contracts for immutable embedding snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ragbench.core.hashing import canonical_json_hash
from ragbench.db.models import ChunkArtifact, ChunkEmbedding
from ragbench.db.models import EmbeddingSnapshot as SnapshotRow
from ragbench.retrieval.base import SearchFilter


@dataclass(frozen=True, slots=True)
class ChunkEmbeddingInput:
    chunk_id: str
    document_id: str
    content: str
    token_count: int
    content_sha256: str = ""
    source_metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.chunk_id or not self.document_id or not self.content:
            raise ValueError("chunk identity and content cannot be empty")
        if self.token_count <= 0:
            raise ValueError("chunk token_count must be positive")
        derived = hashlib.sha256(self.content.encode()).hexdigest()
        if self.content_sha256 and self.content_sha256 != derived:
            raise ValueError("content hash does not match chunk content")
        object.__setattr__(self, "content_sha256", derived)


def frozen_source_metadata(value: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Encode JSON metadata into an immutable, deterministically ordered value."""
    return tuple(
        (str(key), json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        for key, item in sorted(value.items())
    )


def thawed_source_metadata(value: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    return {key: json.loads(item) for key, item in value}


def chunk_manifest_hash(chunks: Sequence[ChunkEmbeddingInput]) -> str:
    return canonical_json_hash(
        [asdict(chunk) for chunk in sorted(chunks, key=lambda item: item.chunk_id)]
    )


@dataclass(frozen=True, slots=True)
class EmbeddingIndexPlan:
    strategy: str
    candidate_factor: int


def embedding_index_plan(dimension: int) -> EmbeddingIndexPlan:
    if isinstance(dimension, bool) or not isinstance(dimension, int) or not 1 <= dimension <= 16000:
        raise ValueError("embedding dimension must be an integer from 1 through 16000")
    if dimension <= 2000:
        return EmbeddingIndexPlan("full-vector-hnsw", 1)
    return EmbeddingIndexPlan("subvector-2000-rerank", 4)


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
    artifact_manifest_hash: str
    index_strategy: str
    candidate_factor: int
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
        plan = embedding_index_plan(self.dimension)
        if self.expected_chunk_count < 0:
            raise ValueError("embedding dimension must be positive and count nonnegative")
        if self.index_strategy != plan.strategy or self.candidate_factor != plan.candidate_factor:
            raise ValueError("embedding index strategy does not match dimension")
        if len(self.artifact_manifest_hash) != 64:
            raise ValueError("artifact manifest hash must be a SHA-256 digest")
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
            self.artifact_manifest_hash,
            self.index_strategy,
            self.candidate_factor,
            self.created_at,
        )


class EmbeddingRepository(Protocol):
    async def create_snapshot(
        self, snapshot: EmbeddingSnapshot, chunks: Sequence[ChunkEmbeddingInput]
    ) -> EmbeddingSnapshot: ...

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
    strategy: str


def hnsw_index_spec(snapshot_id: UUID, dimension: int) -> HnswIndexSpec:
    """Return injection-safe DDL for one dimension-specific partial HNSW index."""
    plan = embedding_index_plan(dimension)
    expression = (
        f"embedding::vector({dimension})"
        if dimension <= 2000
        else "subvector(embedding, 1, 2000)::vector(2000)"
    )
    name = f"ix_chunk_embedding_hnsw_{snapshot_id.hex}_{dimension}"
    sql = (
        f"CREATE INDEX IF NOT EXISTS {name} ON chunk_embedding USING hnsw "
        f"(({expression}) vector_cosine_ops) "
        f"WHERE embedding_snapshot_id = '{snapshot_id}'::uuid"
    )
    return HnswIndexSpec(name, sql, plan.strategy)


@dataclass(frozen=True, slots=True)
class DenseSearchSpec:
    sql: str
    params: dict[str, object]


def dense_search_spec(
    snapshot_id: UUID,
    *,
    dimension: int,
    top_k: int,
    document_ids: tuple[str, ...] = (),
    candidate_factor: int | None = None,
) -> DenseSearchSpec:
    plan = embedding_index_plan(dimension)
    factor = plan.candidate_factor if candidate_factor is None else candidate_factor
    if factor < plan.candidate_factor:
        raise ValueError("candidate factor is below the index strategy minimum")
    predicate = f"ce.embedding_snapshot_id = '{snapshot_id}'::uuid"
    params: dict[str, object] = {"top_k": top_k}
    document_clause = ""
    if document_ids:
        document_clause = " AND ca.document_id = ANY(:document_ids)"
        params["document_ids"] = list(sorted(set(document_ids)))
    if plan.strategy == "full-vector-hnsw":
        full = f"ce.embedding::vector({dimension}) <=> CAST(:query AS vector({dimension}))"
        sql = (
            f"SELECT ce.chunk_id, 1.0 - ({full}) AS score FROM chunk_embedding ce "
            "JOIN chunk_artifact ca ON ca.embedding_snapshot_id = ce.embedding_snapshot_id "
            f"AND ca.chunk_id = ce.chunk_id WHERE {predicate}{document_clause} "
            f"ORDER BY {full} ASC, ce.chunk_id ASC LIMIT :top_k"
        )
    else:
        full = f"ce.embedding <=> CAST(:query AS vector({dimension}))"
        indexed = (
            "subvector(ce.embedding, 1, 2000)::vector(2000) <=> "
            f"subvector(CAST(:query AS vector({dimension})), 1, 2000)::vector(2000)"
        )
        params["candidate_k"] = max(20, factor * top_k)
        sql = (
            "WITH candidates AS (SELECT ce.chunk_id, ce.embedding FROM chunk_embedding ce "
            "JOIN chunk_artifact ca ON ca.embedding_snapshot_id = ce.embedding_snapshot_id "
            f"AND ca.chunk_id = ce.chunk_id WHERE {predicate}{document_clause} "
            f"ORDER BY {indexed} ASC LIMIT :candidate_k) "
            f"SELECT ce.chunk_id, 1.0 - ({full}) AS score FROM candidates ce "
            f"ORDER BY {full} ASC, ce.chunk_id ASC LIMIT :top_k"
        )
    return DenseSearchSpec(sql, params)


class MemoryEmbeddingRepository:
    """Deterministic repository used by offline tests and notebook examples."""

    def __init__(self) -> None:
        self.snapshots: dict[str, EmbeddingSnapshot] = {}
        self.vectors: dict[str, dict[str, tuple[float, ...]]] = {}
        self.artifacts: dict[str, dict[str, ChunkEmbeddingInput]] = {}
        self.finalize_attempts = 0

    async def create_snapshot(
        self, snapshot: EmbeddingSnapshot, chunks: Sequence[ChunkEmbeddingInput]
    ) -> EmbeddingSnapshot:
        if chunk_manifest_hash(chunks) != snapshot.artifact_manifest_hash:
            raise ValueError("chunk manifest does not match snapshot")
        existing = self.snapshots.get(snapshot.snapshot_id)
        if existing is not None:
            if existing.immutable_identity() != snapshot.immutable_identity():
                raise ValueError("embedding snapshot metadata is immutable")
            if self.artifacts[snapshot.snapshot_id] != {chunk.chunk_id: chunk for chunk in chunks}:
                raise ValueError("chunk manifest is immutable")
            return existing
        self.snapshots[snapshot.snapshot_id] = snapshot
        self.vectors[snapshot.snapshot_id] = {}
        self.artifacts[snapshot.snapshot_id] = {chunk.chunk_id: chunk for chunk in chunks}
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
                raise ValueError("chunk already has a different vector")
            if chunk_id not in self.artifacts[snapshot_id]:
                raise ValueError("vector has no registered chunk artifact")
            if any(not math.isfinite(value) for value in vector) or not any(
                value != 0 for value in vector
            ):
                raise ValueError("stored embedding must be finite and nonzero")
            target[chunk_id] = vector

    async def finalize_snapshot(self, snapshot_id: str) -> EmbeddingSnapshot:
        self.finalize_attempts += 1
        snapshot = self._require_snapshot(snapshot_id)
        if set(self.vectors[snapshot_id]) != set(self.artifacts[snapshot_id]):
            raise RuntimeError("cannot finalize without exact artifact/vector set")
        try:
            index_name = hnsw_index_spec(UUID(snapshot.snapshot_id), snapshot.dimension).name
        except ValueError:
            index_name = "memory-index"
        completed = replace(snapshot, complete=True, index_name=index_name, index_state="ready")
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

    async def create_snapshot(
        self, snapshot: EmbeddingSnapshot, chunks: Sequence[ChunkEmbeddingInput]
    ) -> EmbeddingSnapshot:
        if chunk_manifest_hash(chunks) != snapshot.artifact_manifest_hash:
            raise ValueError("chunk manifest does not match snapshot")
        snapshot_uuid = UUID(snapshot.snapshot_id)
        async with self._session_factory() as session, session.begin():
            await self._lock_snapshot(session, snapshot_uuid)
            existing = await session.get(SnapshotRow, snapshot_uuid)
            if existing is not None:
                restored = self._from_row(existing)
                if restored.immutable_identity() != snapshot.immutable_identity():
                    raise ValueError("embedding snapshot metadata is immutable")
                stored = await session.scalars(
                    select(ChunkArtifact).where(
                        ChunkArtifact.embedding_snapshot_id == snapshot_uuid
                    )
                )
                evidence = {
                    row.chunk_id: (row.document_id, row.content_sha256, row.token_count)
                    for row in stored
                }
                expected = {
                    chunk.chunk_id: (chunk.document_id, chunk.content_sha256, chunk.token_count)
                    for chunk in chunks
                }
                if evidence != expected:
                    raise ValueError("chunk manifest is immutable")
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
                    artifact_manifest_hash=snapshot.artifact_manifest_hash,
                    index_strategy=snapshot.index_strategy,
                    candidate_factor=snapshot.candidate_factor,
                    complete=False,
                    index_name=None,
                    index_state="pending",
                    config_snapshot={},
                    created_at=snapshot.created_at,
                )
            )
            session.add_all(
                [
                    ChunkArtifact(
                        embedding_snapshot_id=snapshot_uuid,
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        content_sha256=chunk.content_sha256,
                        token_count=chunk.token_count,
                        metadata_snapshot=thawed_source_metadata(chunk.source_metadata),
                    )
                    for chunk in chunks
                ]
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
            await self._lock_snapshot(session, snapshot_uuid)
            snapshot = await session.get(SnapshotRow, snapshot_uuid, with_for_update=True)
            if snapshot is None:
                raise KeyError(f"unknown embedding snapshot: {snapshot_id}")
            if snapshot.complete:
                raise RuntimeError("complete embedding snapshots are immutable")
            for chunk_id, vector in vectors:
                if len(vector) != snapshot.dimension:
                    raise ValueError("embedding dimension does not match snapshot")
                existing = await session.scalar(
                    select(ChunkEmbedding).where(
                        ChunkEmbedding.embedding_snapshot_id == snapshot_uuid,
                        ChunkEmbedding.chunk_id == chunk_id,
                    )
                )
                if existing is not None:
                    stored = np.asarray(existing.embedding, dtype=np.float32)
                    requested = np.asarray(vector, dtype=np.float32)
                    if not np.array_equal(stored, requested):
                        raise ValueError("chunk already has a different vector")
                    continue
                statement = insert(ChunkEmbedding).values(
                    embedding_snapshot_id=snapshot_uuid,
                    chunk_id=chunk_id,
                    dimension=snapshot.dimension,
                    embedding=vector,
                )
                await session.execute(statement)

    async def finalize_snapshot(self, snapshot_id: str) -> EmbeddingSnapshot:
        snapshot_uuid = UUID(snapshot_id)
        async with self._session_factory() as session, session.begin():
            await self._lock_snapshot(session, snapshot_uuid)
            snapshot = await session.get(SnapshotRow, snapshot_uuid, with_for_update=True)
            if snapshot is None:
                raise KeyError(f"unknown embedding snapshot: {snapshot_id}")
            artifacts = set(
                await session.scalars(
                    select(ChunkArtifact.chunk_id).where(
                        ChunkArtifact.embedding_snapshot_id == snapshot_uuid
                    )
                )
            )
            vectors = set(
                await session.scalars(
                    select(ChunkEmbedding.chunk_id).where(
                        ChunkEmbedding.embedding_snapshot_id == snapshot_uuid
                    )
                )
            )
            if artifacts != vectors or len(artifacts) != snapshot.expected_chunk_count:
                raise RuntimeError("cannot finalize without exact artifact/vector set")
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
            vector_literal = "[" + ",".join(format(value, ".17g") for value in query_vector) + "]"
            spec = dense_search_spec(
                snapshot_uuid,
                dimension=snapshot.dimension,
                top_k=top_k,
                document_ids=filter.document_ids,
                candidate_factor=snapshot.candidate_factor,
            )
            rows = await session.execute(text(spec.sql), {**spec.params, "query": vector_literal})
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
            artifact_manifest_hash=row.artifact_manifest_hash,
            index_strategy=row.index_strategy,
            candidate_factor=row.candidate_factor,
            created_at=row.created_at,
            complete=row.complete,
            index_name=row.index_name,
            index_state=row.index_state,
        )

    @staticmethod
    async def _lock_snapshot(session: AsyncSession, snapshot_id: UUID) -> None:
        lock_id = int.from_bytes(hashlib.sha256(snapshot_id.bytes).digest()[:8], "big", signed=True)
        await session.execute(text("SELECT pg_advisory_xact_lock(:lock_id)"), {"lock_id": lock_id})
