"""Harden dense index manifests, filtering, and retrieval evidence.

Revision ID: 20260814_0004
Revises: 20260814_0003
Create Date: 2026-08-14 00:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260814_0004"
down_revision: str | None = "20260814_0003"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())


def upgrade() -> None:
    """Add immutable chunk evidence and make retrieval IDs artifact-compatible."""
    op.add_column("embedding_snapshot", sa.Column("artifact_manifest_hash", sa.String(64)))
    op.add_column("embedding_snapshot", sa.Column("index_strategy", sa.String(64)))
    op.add_column(
        "embedding_snapshot", sa.Column("candidate_factor", sa.Integer(), server_default="1")
    )
    op.execute(
        "UPDATE embedding_snapshot SET artifact_manifest_hash = repeat('0', 64), "
        "index_strategy = CASE WHEN dimension <= 2000 THEN 'full-vector-hnsw' "
        "ELSE 'subvector-2000-rerank' END, "
        "candidate_factor = CASE WHEN dimension <= 2000 THEN 1 ELSE 4 END"
    )
    for column in ("artifact_manifest_hash", "index_strategy", "candidate_factor"):
        op.alter_column("embedding_snapshot", column, nullable=False)
    op.create_check_constraint(
        "embedding_candidate_factor_positive", "embedding_snapshot", "candidate_factor > 0"
    )
    op.create_table(
        "chunk_artifact",
        sa.Column("embedding_snapshot_id", UUID, primary_key=True, nullable=False),
        sa.Column("chunk_id", sa.String(512), primary_key=True, nullable=False),
        sa.Column("document_id", sa.String(512), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("metadata_snapshot", JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "token_count > 0", name="ck_chunk_artifact_chunk_artifact_token_count_positive"
        ),
        sa.ForeignKeyConstraint(
            ["embedding_snapshot_id"], ["embedding_snapshot.id"], ondelete="RESTRICT"
        ),
    )
    op.execute(
        "INSERT INTO chunk_artifact "
        "(embedding_snapshot_id, chunk_id, document_id, content_sha256, token_count, "
        "metadata_snapshot) SELECT embedding_snapshot_id, chunk_id, 'legacy-unknown', "
        "repeat('0', 64), 1, '{\"legacy_backfill\": true}'::jsonb FROM chunk_embedding"
    )
    op.create_foreign_key(
        "fk_chunk_embedding_artifact",
        "chunk_embedding",
        "chunk_artifact",
        ["embedding_snapshot_id", "chunk_id"],
        ["embedding_snapshot_id", "chunk_id"],
        ondelete="RESTRICT",
    )
    op.create_check_constraint(
        "chunk_embedding_finite_nonzero",
        "chunk_embedding",
        "vector_norm(embedding) > 0 AND vector_norm(embedding) < 'Infinity'::float8",
    )
    op.drop_constraint("fk_retrieval_result_chunk_id_chunk", "retrieval_result", type_="foreignkey")
    op.alter_column(
        "retrieval_result",
        "chunk_id",
        type_=sa.String(512),
        postgresql_using="chunk_id::text",
    )
    op.add_column("retrieval_result", sa.Column("embedding_snapshot_id", UUID, nullable=True))
    op.create_foreign_key(
        "fk_retrieval_result_artifact",
        "retrieval_result",
        "chunk_artifact",
        ["embedding_snapshot_id", "chunk_id"],
        ["embedding_snapshot_id", "chunk_id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    """Restore UUID retrieval IDs only when all persisted IDs remain valid UUID text."""
    op.drop_constraint("fk_retrieval_result_artifact", "retrieval_result", type_="foreignkey")
    op.drop_column("retrieval_result", "embedding_snapshot_id")
    op.execute(
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM retrieval_result WHERE chunk_id !~* "
        "'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$') "
        "THEN RAISE EXCEPTION 'cannot downgrade non-UUID retrieval chunk IDs'; END IF; END $$"
    )
    op.alter_column(
        "retrieval_result", "chunk_id", type_=UUID, postgresql_using="chunk_id::uuid"
    )
    op.create_foreign_key(
        "fk_retrieval_result_chunk_id_chunk",
        "retrieval_result",
        "chunk",
        ["chunk_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.drop_constraint("chunk_embedding_finite_nonzero", "chunk_embedding", type_="check")
    op.drop_constraint("fk_chunk_embedding_artifact", "chunk_embedding", type_="foreignkey")
    op.drop_table("chunk_artifact")
    op.drop_constraint(
        "embedding_candidate_factor_positive", "embedding_snapshot", type_="check"
    )
    for column in ("candidate_factor", "index_strategy", "artifact_manifest_hash"):
        op.drop_column("embedding_snapshot", column)
