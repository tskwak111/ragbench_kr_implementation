"""Add versioned embedding vectors and dimension-specific index state.

Revision ID: 20260814_0003
Revises: 20260814_0002
Create Date: 2026-08-14 00:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "20260814_0003"
down_revision: str | None = "20260814_0002"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

UUID = postgresql.UUID(as_uuid=True)
UUID_PK = sa.text("gen_random_uuid()")


def upgrade() -> None:
    """Separate immutable chunk vectors from chunks and track HNSW readiness."""
    op.alter_column("embedding_snapshot", "parse_run_id", nullable=True)
    additions = (
        sa.Column("corpus_snapshot_id", sa.String(64)),
        sa.Column("parse_snapshot_id", sa.String(64)),
        sa.Column("chunk_strategy", sa.String(64)),
        sa.Column("query_model_id", sa.String(255)),
        sa.Column("normalization", sa.String(32), server_default="l2"),
        sa.Column("expected_chunk_count", sa.Integer(), server_default="0"),
        sa.Column("complete", sa.Boolean(), server_default=sa.false()),
        sa.Column("index_name", sa.String(128)),
        sa.Column("index_state", sa.String(32), server_default="pending"),
    )
    for column in additions:
        op.add_column("embedding_snapshot", column)
    op.execute(
        "UPDATE embedding_snapshot SET corpus_snapshot_id = repeat('0', 64), "
        "parse_snapshot_id = repeat('0', 64), chunk_strategy = 'legacy-unknown', "
        "query_model_id = model_id"
    )
    for column in (
        "corpus_snapshot_id",
        "parse_snapshot_id",
        "chunk_strategy",
        "query_model_id",
        "normalization",
        "expected_chunk_count",
        "complete",
        "index_state",
    ):
        op.alter_column("embedding_snapshot", column, nullable=False)
    op.create_check_constraint(
        "embedding_expected_count_nonnegative",
        "embedding_snapshot",
        "expected_chunk_count >= 0",
    )
    op.create_unique_constraint(
        "uq_embedding_snapshot_id_dimension", "embedding_snapshot", ["id", "dimension"]
    )
    op.create_table(
        "chunk_embedding",
        sa.Column("id", UUID, primary_key=True, nullable=False, server_default=UUID_PK),
        sa.Column("embedding_snapshot_id", UUID, nullable=False),
        sa.Column("chunk_id", sa.String(512), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "vector_dims(embedding) = dimension",
            name="ck_chunk_embedding_chunk_embedding_dimension",
        ),
        sa.ForeignKeyConstraint(
            ["embedding_snapshot_id", "dimension"],
            ["embedding_snapshot.id", "embedding_snapshot.dimension"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "embedding_snapshot_id", "chunk_id", name="uq_chunk_embedding_snapshot_chunk"
        ),
    )


def downgrade() -> None:
    """Remove Task 8 vectors without dropping pgvector itself."""
    op.drop_table("chunk_embedding")
    op.drop_constraint("uq_embedding_snapshot_id_dimension", "embedding_snapshot", type_="unique")
    op.drop_constraint("embedding_expected_count_nonnegative", "embedding_snapshot", type_="check")
    for column in (
        "index_state",
        "index_name",
        "complete",
        "expected_chunk_count",
        "normalization",
        "query_model_id",
        "chunk_strategy",
        "parse_snapshot_id",
        "corpus_snapshot_id",
    ):
        op.drop_column("embedding_snapshot", column)
    op.alter_column("embedding_snapshot", "parse_run_id", nullable=False)
