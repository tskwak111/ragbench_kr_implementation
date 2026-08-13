"""Immutable experiment-evidence records persisted in PostgreSQL."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    false,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from ragbench.db.base import Base, CreatedAt, UuidPk


class Document(Base):
    """Source document identified by its immutable content hash."""

    __tablename__ = "document"

    id: Mapped[UuidPk]
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[CreatedAt]


class ParseRun(Base):
    """A versioned provider parse of one document."""

    __tablename__ = "parse_run"

    id: Mapped[UuidPk]
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("document.id", ondelete="RESTRICT"), nullable=False
    )
    corpus_snapshot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_model_version: Mapped[str] = mapped_column(String(255), nullable=False)
    mode: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    raw_response_hash: Mapped[str | None] = mapped_column(String(64))
    raw_response: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    markdown: Mapped[str] = mapped_column(Text, nullable=False, default="")
    html: Mapped[str] = mapped_column(Text, nullable=False, default="")
    elements: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    page_mappings: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[CreatedAt]
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "corpus_snapshot_id",
            "provider_model_id",
            "provider_model_version",
            "mode",
            name="uq_parse_run_checkpoint",
        ),
        Index(
            "ix_parse_run_snapshot_mode_status",
            "corpus_snapshot_id",
            "mode",
            "status",
        ),
        CheckConstraint("latency_ms >= 0", name="parse_run_latency_nonnegative"),
        CheckConstraint("cost_usd >= 0", name="parse_run_cost_nonnegative"),
    )


class EmbeddingSnapshot(Base):
    """Records the model and dimensionality required to interpret chunk vectors."""

    __tablename__ = "embedding_snapshot"

    id: Mapped[UuidPk]
    parse_run_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("parse_run.id", ondelete="RESTRICT"), nullable=True
    )
    corpus_snapshot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    parse_snapshot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    query_model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    normalization: Mapped[str] = mapped_column(String(32), nullable=False)
    expected_chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    artifact_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    index_strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_factor: Mapped[int] = mapped_column(Integer, nullable=False)
    complete: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=false())
    index_name: Mapped[str | None] = mapped_column(String(128))
    index_state: Mapped[str] = mapped_column(String(32), nullable=False, server_default="pending")
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[CreatedAt]
    __table_args__ = (
        CheckConstraint("dimension > 0", name="embedding_dimension_positive"),
        CheckConstraint("expected_chunk_count >= 0", name="embedding_expected_count_nonnegative"),
        CheckConstraint("candidate_factor > 0", name="embedding_candidate_factor_positive"),
        UniqueConstraint("id", "dimension", name="uq_embedding_snapshot_id_dimension"),
    )


class Chunk(Base):
    """Deterministic content slice and its deferred-dimension vector representation."""

    __tablename__ = "chunk"

    id: Mapped[UuidPk]
    parse_run_id: Mapped[UUID] = mapped_column(
        ForeignKey("parse_run.id", ondelete="RESTRICT"), nullable=False
    )
    embedding_snapshot_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("embedding_snapshot.id", ondelete="RESTRICT")
    )
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    section_path: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[Any | None] = mapped_column(Vector(), nullable=True)
    created_at: Mapped[CreatedAt]
    # Task 8 creates an HNSW index only after populated vectors establish the selected dimension.
    __table_args__ = (
        CheckConstraint("ordinal >= 0", name="chunk_ordinal_nonnegative"),
        CheckConstraint("page_start > 0", name="chunk_page_start_positive"),
        CheckConstraint("page_end >= page_start", name="chunk_page_order"),
        CheckConstraint("token_count >= 0", name="chunk_token_count_nonnegative"),
        Index("ix_chunk_parse_run_strategy_ordinal", "parse_run_id", "strategy", "ordinal"),
    )


class ChunkArtifact(Base):
    """Immutable chunk identity and source evidence scoped to an embedding snapshot."""

    __tablename__ = "chunk_artifact"

    embedding_snapshot_id: Mapped[UUID] = mapped_column(
        ForeignKey("embedding_snapshot.id", ondelete="RESTRICT"), primary_key=True
    )
    chunk_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(512), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[CreatedAt]
    __table_args__ = (
        CheckConstraint("token_count > 0", name="chunk_artifact_token_count_positive"),
    )


class ChunkEmbedding(Base):
    """One immutable vector for a chunk within a versioned embedding snapshot."""

    __tablename__ = "chunk_embedding"

    id: Mapped[UuidPk]
    embedding_snapshot_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(512), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(), nullable=False)
    created_at: Mapped[CreatedAt]
    __table_args__ = (
        ForeignKeyConstraint(
            ["embedding_snapshot_id", "dimension"],
            ["embedding_snapshot.id", "embedding_snapshot.dimension"],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["embedding_snapshot_id", "chunk_id"],
            ["chunk_artifact.embedding_snapshot_id", "chunk_artifact.chunk_id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint("vector_dims(embedding) = dimension", name="chunk_embedding_dimension"),
        CheckConstraint(
            "vector_norm(embedding) > 0 AND "
            "vector_norm(embedding) < 'Infinity'::float8",
            name="chunk_embedding_finite_nonzero",
        ),
        UniqueConstraint(
            "embedding_snapshot_id", "chunk_id", name="uq_chunk_embedding_snapshot_chunk"
        ),
    )


class Question(Base):
    """Benchmark question and its fixed split/type labels."""

    __tablename__ = "question"

    id: Mapped[UuidPk]
    document_id: Mapped[UUID | None] = mapped_column(ForeignKey("document.id", ondelete="RESTRICT"))
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(Text)
    split: Mapped[str] = mapped_column(String(32), nullable=False)
    question_type: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[CreatedAt]
    __table_args__ = (Index("ix_question_split_question_type", "split", "question_type"),)


class Experiment(Base):
    """An immutable benchmark configuration and lifecycle status."""

    __tablename__ = "experiment"

    id: Mapped[UuidPk]
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[CreatedAt]


class ExperimentResponse(Base):
    """Generated answer evidence for an experiment-question pair."""

    __tablename__ = "experiment_response"

    id: Mapped[UuidPk]
    experiment_id: Mapped[UUID] = mapped_column(
        ForeignKey("experiment.id", ondelete="RESTRICT"), nullable=False
    )
    question_id: Mapped[UUID] = mapped_column(
        ForeignKey("question.id", ondelete="RESTRICT"), nullable=False
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    citations_snapshot: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[CreatedAt]
    __table_args__ = (
        Index(
            "ix_experiment_response_experiment_question",
            "experiment_id",
            "question_id",
            unique=True,
        ),
    )


class RetrievalResult(Base):
    """Ranked retrieved chunk evidence attached to a concrete response."""

    __tablename__ = "retrieval_result"

    id: Mapped[UuidPk]
    experiment_response_id: Mapped[UUID] = mapped_column(
        ForeignKey("experiment_response.id", ondelete="RESTRICT"), nullable=False
    )
    embedding_snapshot_id: Mapped[UUID | None] = mapped_column(nullable=True)
    legacy_chunk_id: Mapped[UUID | None] = mapped_column(nullable=True)
    chunk_id: Mapped[str] = mapped_column(String(512), nullable=False)
    retriever: Mapped[str] = mapped_column(String(64), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    created_at: Mapped[CreatedAt]
    __table_args__ = (
        CheckConstraint("rank > 0", name="retrieval_rank_positive"),
        CheckConstraint(
            "((legacy_chunk_id IS NOT NULL AND embedding_snapshot_id IS NULL "
            "AND chunk_id = legacy_chunk_id::text) OR "
            "(legacy_chunk_id IS NULL AND embedding_snapshot_id IS NOT NULL))",
            name="retrieval_result_exactly_one_evidence_mode",
        ),
        ForeignKeyConstraint(
            ["legacy_chunk_id"], ["chunk.id"], ondelete="RESTRICT"
        ),
        ForeignKeyConstraint(
            ["embedding_snapshot_id", "chunk_id"],
            ["chunk_artifact.embedding_snapshot_id", "chunk_artifact.chunk_id"],
            ondelete="RESTRICT",
        ),
    )


class Metric(Base):
    """Measured quality metric retained with its calculation context."""

    __tablename__ = "metric"

    id: Mapped[UuidPk]
    experiment_id: Mapped[UUID] = mapped_column(
        ForeignKey("experiment.id", ondelete="RESTRICT"), nullable=False
    )
    experiment_response_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("experiment_response.id", ondelete="RESTRICT")
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    metadata_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[CreatedAt]


class ApiUsage(Base):
    """Settled or cache-hit provider usage with a request correlation identifier."""

    __tablename__ = "api_usage"

    id: Mapped[UuidPk]
    correlation_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False, index=True)
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    billable_pages: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    estimated_cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    cache_hit: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    created_at: Mapped[CreatedAt]


class ApiCacheEntry(Base):
    """Private provider response cache keyed by deterministic request content."""

    __tablename__ = "api_cache_entry"

    id: Mapped[UuidPk]
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_model_id: Mapped[str] = mapped_column(String(255), nullable=False)
    response_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[CreatedAt]


class BudgetReservation(Base):
    """Open or settled worst-case cost reservation for budget enforcement."""

    __tablename__ = "budget_reservation"

    id: Mapped[UuidPk]
    correlation_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False, unique=True)
    reserved_cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    settled_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[CreatedAt]
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
