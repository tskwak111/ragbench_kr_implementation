"""Create PostgreSQL-backed experiment evidence persistence.

Revision ID: 20260813_0001
Revises:
Create Date: 2026-08-13 00:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "20260813_0001"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())
MONEY = sa.Numeric(12, 6)
UUID_PK = sa.text("gen_random_uuid()")


def upgrade() -> None:
    """Install pgvector and all tables needed to preserve experiment evidence."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "document",
        sa.Column("id", UUID, primary_key=True, nullable=False, server_default=UUID_PK),
        sa.Column("title", sa.String(512), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("metadata_snapshot", JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("sha256", name="uq_document_sha256"),
    )
    op.create_index("ix_document_sha256", "document", ["sha256"])

    op.create_table(
        "parse_run",
        sa.Column("id", UUID, primary_key=True, nullable=False, server_default=UUID_PK),
        sa.Column("document_id", UUID, nullable=False),
        sa.Column("provider_model_id", sa.String(255), nullable=False),
        sa.Column("mode", sa.String(64), nullable=False),
        sa.Column("status", sa.String(64), nullable=False),
        sa.Column("config_snapshot", JSONB, nullable=False),
        sa.Column("raw_response_hash", sa.String(64)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["document.id"], ondelete="RESTRICT"),
    )
    op.create_table(
        "embedding_snapshot",
        sa.Column("id", UUID, primary_key=True, nullable=False, server_default=UUID_PK),
        sa.Column("parse_run_id", UUID, nullable=False),
        sa.Column("model_id", sa.String(255), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("config_snapshot", JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "dimension > 0", name="ck_embedding_snapshot_embedding_dimension_positive"
        ),
        sa.ForeignKeyConstraint(["parse_run_id"], ["parse_run.id"], ondelete="RESTRICT"),
    )
    op.create_table(
        "chunk",
        sa.Column("id", UUID, primary_key=True, nullable=False, server_default=UUID_PK),
        sa.Column("parse_run_id", UUID, nullable=False),
        sa.Column("embedding_snapshot_id", UUID),
        sa.Column("strategy", sa.String(64), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=False),
        sa.Column("page_end", sa.Integer(), nullable=False),
        sa.Column("section_path", JSONB, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("ordinal >= 0", name="ck_chunk_chunk_ordinal_nonnegative"),
        sa.CheckConstraint("page_start > 0", name="ck_chunk_chunk_page_start_positive"),
        sa.CheckConstraint("page_end >= page_start", name="ck_chunk_chunk_page_order"),
        sa.CheckConstraint("token_count >= 0", name="ck_chunk_chunk_token_count_nonnegative"),
        sa.ForeignKeyConstraint(["parse_run_id"], ["parse_run.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["embedding_snapshot_id"], ["embedding_snapshot.id"], ondelete="RESTRICT"
        ),
    )
    op.create_index(
        "ix_chunk_parse_run_strategy_ordinal", "chunk", ["parse_run_id", "strategy", "ordinal"]
    )
    # Task 8 owns the selected embedding dimension and adds the populated-data HNSW index.

    op.create_table(
        "question",
        sa.Column("id", UUID, primary_key=True, nullable=False, server_default=UUID_PK),
        sa.Column("document_id", UUID),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("expected_answer", sa.Text()),
        sa.Column("split", sa.String(32), nullable=False),
        sa.Column("question_type", sa.String(64), nullable=False),
        sa.Column("metadata_snapshot", JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["document_id"], ["document.id"], ondelete="RESTRICT"),
    )
    op.create_index("ix_question_split_question_type", "question", ["split", "question_type"])
    op.create_table(
        "experiment",
        sa.Column("id", UUID, primary_key=True, nullable=False, server_default=UUID_PK),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(64), nullable=False),
        sa.Column("config_snapshot", JSONB, nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("name", name="uq_experiment_name"),
    )
    op.create_index("ix_experiment_status", "experiment", ["status"])
    op.create_table(
        "experiment_response",
        sa.Column("id", UUID, primary_key=True, nullable=False, server_default=UUID_PK),
        sa.Column("experiment_id", UUID, nullable=False),
        sa.Column("question_id", UUID, nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("citations_snapshot", JSONB, nullable=False),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiment.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["question_id"], ["question.id"], ondelete="RESTRICT"),
    )
    op.create_index(
        "ix_experiment_response_experiment_question",
        "experiment_response",
        ["experiment_id", "question_id"],
        unique=True,
    )
    op.create_table(
        "retrieval_result",
        sa.Column("id", UUID, primary_key=True, nullable=False, server_default=UUID_PK),
        sa.Column("experiment_response_id", UUID, nullable=False),
        sa.Column("chunk_id", UUID, nullable=False),
        sa.Column("retriever", sa.String(64), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", MONEY, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("rank > 0", name="ck_retrieval_result_retrieval_rank_positive"),
        sa.ForeignKeyConstraint(
            ["experiment_response_id"], ["experiment_response.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunk.id"], ondelete="RESTRICT"),
    )
    op.create_table(
        "metric",
        sa.Column("id", UUID, primary_key=True, nullable=False, server_default=UUID_PK),
        sa.Column("experiment_id", UUID, nullable=False),
        sa.Column("experiment_response_id", UUID),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("value", MONEY, nullable=False),
        sa.Column("metadata_snapshot", JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["experiment_id"], ["experiment.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["experiment_response_id"], ["experiment_response.id"], ondelete="RESTRICT"
        ),
    )

    op.create_table(
        "api_usage",
        sa.Column("id", UUID, primary_key=True, nullable=False, server_default=UUID_PK),
        sa.Column("correlation_id", UUID, nullable=False),
        sa.Column("operation", sa.String(64), nullable=False),
        sa.Column("provider_model_id", sa.String(255), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("billable_pages", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("estimated_cost_usd", MONEY, nullable=False),
        sa.Column("cache_hit", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_api_usage_correlation_id", "api_usage", ["correlation_id"])
    op.create_table(
        "api_cache_entry",
        sa.Column("id", UUID, primary_key=True, nullable=False, server_default=UUID_PK),
        sa.Column("cache_key", sa.String(64), nullable=False),
        sa.Column("operation", sa.String(64), nullable=False),
        sa.Column("provider_model_id", sa.String(255), nullable=False),
        sa.Column("response_snapshot", JSONB, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("cache_key", name="uq_api_cache_entry_cache_key"),
    )
    op.create_table(
        "budget_reservation",
        sa.Column("id", UUID, primary_key=True, nullable=False, server_default=UUID_PK),
        sa.Column("correlation_id", UUID, nullable=False),
        sa.Column("reserved_cost_usd", MONEY, nullable=False),
        sa.Column("settled_cost_usd", MONEY),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("settled_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("correlation_id", name="uq_budget_reservation_correlation_id"),
    )


def downgrade() -> None:
    """Remove exactly the tables and extension installed by this revision."""
    op.drop_table("budget_reservation")
    op.drop_table("api_cache_entry")
    op.drop_index("ix_api_usage_correlation_id", table_name="api_usage")
    op.drop_table("api_usage")
    op.drop_table("metric")
    op.drop_table("retrieval_result")
    op.drop_index("ix_experiment_response_experiment_question", table_name="experiment_response")
    op.drop_table("experiment_response")
    op.drop_index("ix_experiment_status", table_name="experiment")
    op.drop_table("experiment")
    op.drop_index("ix_question_split_question_type", table_name="question")
    op.drop_table("question")
    op.drop_index("ix_chunk_parse_run_strategy_ordinal", table_name="chunk")
    op.drop_table("chunk")
    op.drop_table("embedding_snapshot")
    op.drop_table("parse_run")
    op.drop_index("ix_document_sha256", table_name="document")
    op.drop_table("document")
