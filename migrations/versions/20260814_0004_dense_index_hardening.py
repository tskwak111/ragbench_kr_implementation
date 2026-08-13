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
        "candidate_factor = CASE WHEN dimension <= 2000 THEN 1 ELSE 4 END, "
        "index_state = 'pending' WHERE complete = false OR index_name IS NULL"
    )
    op.execute(
        r"""
DO $migration$
DECLARE
    snapshot record;
    expected_name text;
BEGIN
    FOR snapshot IN
        SELECT id, dimension, index_name
        FROM embedding_snapshot
        WHERE complete = true
        ORDER BY id
        FOR UPDATE
    LOOP
        IF snapshot.index_name IS NULL THEN
            RAISE EXCEPTION 'completed embedding snapshot % has no index', snapshot.id;
        END IF;
        IF snapshot.dimension > 4000 THEN
            RAISE EXCEPTION 'v0003 could not complete dimension > 4000 snapshot %', snapshot.id;
        END IF;
        expected_name := 'ix_chunk_embedding_hnsw_' ||
            replace(snapshot.id::text, '-', '') || '_' || snapshot.dimension::text;
        IF snapshot.index_name <> expected_name THEN
            RAISE EXCEPTION 'unexpected stored embedding index name %', snapshot.index_name;
        END IF;
        IF snapshot.dimension > 2000 THEN
            EXECUTE 'DROP INDEX IF EXISTS ' || quote_ident(snapshot.index_name);
            EXECUTE 'CREATE INDEX ' || quote_ident(expected_name) ||
                ' ON chunk_embedding USING hnsw ' ||
                '((subvector(embedding, 1, 2000)::vector(2000)) vector_cosine_ops) ' ||
                'WHERE embedding_snapshot_id = ' || quote_literal(snapshot.id) || '::uuid';
            UPDATE embedding_snapshot
            SET index_strategy = 'subvector-2000-rerank',
                candidate_factor = 4,
                index_state = 'ready'
            WHERE id = snapshot.id;
        ELSE
            UPDATE embedding_snapshot
            SET index_strategy = 'full-vector-hnsw',
                candidate_factor = 1,
                index_state = 'ready'
            WHERE id = snapshot.id;
        END IF;
    END LOOP;
END
$migration$;
"""
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
    op.add_column("retrieval_result", sa.Column("legacy_chunk_id", UUID, nullable=True))
    op.execute("UPDATE retrieval_result SET legacy_chunk_id = chunk_id")
    op.drop_constraint("fk_retrieval_result_chunk_id_chunk", "retrieval_result", type_="foreignkey")
    op.alter_column(
        "retrieval_result",
        "chunk_id",
        type_=sa.String(512),
        postgresql_using="chunk_id::text",
    )
    op.add_column("retrieval_result", sa.Column("embedding_snapshot_id", UUID, nullable=True))
    op.create_foreign_key(
        "fk_retrieval_result_legacy_chunk",
        "retrieval_result",
        "chunk",
        ["legacy_chunk_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_retrieval_result_artifact",
        "retrieval_result",
        "chunk_artifact",
        ["embedding_snapshot_id", "chunk_id"],
        ["embedding_snapshot_id", "chunk_id"],
        ondelete="RESTRICT",
    )
    op.create_check_constraint(
        "retrieval_result_exactly_one_evidence_mode",
        "retrieval_result",
        "((legacy_chunk_id IS NOT NULL AND embedding_snapshot_id IS NULL) OR "
        "(legacy_chunk_id IS NULL AND embedding_snapshot_id IS NOT NULL))",
    )


def downgrade() -> None:
    """Restore UUID retrieval IDs only when all persisted IDs remain valid UUID text."""
    op.execute(
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM retrieval_result "
        "WHERE embedding_snapshot_id IS NOT NULL OR legacy_chunk_id IS NULL) "
        "THEN RAISE EXCEPTION 'cannot downgrade artifact-mode retrieval results'; END IF; END $$"
    )
    op.execute(
        r"""
DO $migration$
DECLARE
    snapshot record;
    expected_name text;
BEGIN
    IF EXISTS (
        SELECT 1 FROM embedding_snapshot
        WHERE complete = true AND dimension > 4000
    ) THEN
        RAISE EXCEPTION 'cannot downgrade completed snapshots above dimension 4000';
    END IF;
    FOR snapshot IN
        SELECT id, dimension, index_name
        FROM embedding_snapshot
        WHERE complete = true AND dimension > 2000
        ORDER BY id
        FOR UPDATE
    LOOP
        expected_name := 'ix_chunk_embedding_hnsw_' ||
            replace(snapshot.id::text, '-', '') || '_' || snapshot.dimension::text;
        IF snapshot.index_name IS NULL OR snapshot.index_name <> expected_name THEN
            RAISE EXCEPTION 'incompatible completed embedding index state for %', snapshot.id;
        END IF;
        EXECUTE 'DROP INDEX IF EXISTS ' || quote_ident(snapshot.index_name);
        EXECUTE 'CREATE INDEX ' || quote_ident(expected_name) ||
            ' ON chunk_embedding USING hnsw ' ||
            '((embedding::halfvec(' || snapshot.dimension::text || ')) halfvec_cosine_ops) ' ||
            'WHERE embedding_snapshot_id = ' || quote_literal(snapshot.id) || '::uuid';
    END LOOP;
END
$migration$;
"""
    )
    op.drop_constraint(
        "retrieval_result_exactly_one_evidence_mode", "retrieval_result", type_="check"
    )
    op.drop_constraint("fk_retrieval_result_artifact", "retrieval_result", type_="foreignkey")
    op.drop_constraint("fk_retrieval_result_legacy_chunk", "retrieval_result", type_="foreignkey")
    op.drop_column("retrieval_result", "embedding_snapshot_id")
    op.execute("UPDATE retrieval_result SET chunk_id = legacy_chunk_id::text")
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
    op.drop_column("retrieval_result", "legacy_chunk_id")
    op.drop_constraint("chunk_embedding_finite_nonzero", "chunk_embedding", type_="check")
    op.drop_constraint("fk_chunk_embedding_artifact", "chunk_embedding", type_="foreignkey")
    op.drop_table("chunk_artifact")
    op.drop_constraint(
        "embedding_candidate_factor_positive", "embedding_snapshot", type_="check"
    )
    for column in ("candidate_factor", "index_strategy", "artifact_manifest_hash"):
        op.drop_column("embedding_snapshot", column)
