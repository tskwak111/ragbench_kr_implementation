"""Add complete resumable parse checkpoint evidence.

Revision ID: 20260814_0002
Revises: 20260813_0001
Create Date: 2026-08-14 00:00:00+00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260814_0002"
down_revision: str | None = "20260813_0001"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

JSONB = postgresql.JSONB(astext_type=sa.Text())
MONEY = sa.Numeric(12, 6)


def upgrade() -> None:
    """Make parse_run a durable, idempotent per-document checkpoint."""
    op.add_column("parse_run", sa.Column("corpus_snapshot_id", sa.String(64)))
    op.add_column("parse_run", sa.Column("provider_model_version", sa.String(255)))
    op.add_column("parse_run", sa.Column("raw_response", JSONB))
    op.add_column("parse_run", sa.Column("markdown", sa.Text(), server_default=""))
    op.add_column("parse_run", sa.Column("html", sa.Text(), server_default=""))
    op.add_column("parse_run", sa.Column("elements", JSONB, server_default="[]"))
    op.add_column("parse_run", sa.Column("page_mappings", JSONB, server_default="[]"))
    op.add_column("parse_run", sa.Column("latency_ms", sa.Integer(), server_default="0"))
    op.add_column("parse_run", sa.Column("cost_usd", MONEY, server_default="0"))
    op.add_column("parse_run", sa.Column("error", sa.Text()))
    op.execute("UPDATE parse_run SET corpus_snapshot_id = repeat('0', 64)")
    op.execute("UPDATE parse_run SET provider_model_version = 'legacy-unknown'")
    for column in (
        "corpus_snapshot_id",
        "provider_model_version",
        "markdown",
        "html",
        "elements",
        "page_mappings",
        "latency_ms",
        "cost_usd",
    ):
        op.alter_column("parse_run", column, nullable=False)
    op.create_check_constraint("parse_run_latency_nonnegative", "parse_run", "latency_ms >= 0")
    op.create_check_constraint("parse_run_cost_nonnegative", "parse_run", "cost_usd >= 0")
    op.create_unique_constraint(
        "uq_parse_run_checkpoint",
        "parse_run",
        [
            "document_id",
            "corpus_snapshot_id",
            "provider_model_id",
            "provider_model_version",
            "mode",
        ],
    )
    op.create_index(
        "ix_parse_run_snapshot_mode_status",
        "parse_run",
        ["corpus_snapshot_id", "mode", "status"],
    )


def downgrade() -> None:
    """Remove only Task 6 parse checkpoint additions."""
    op.drop_index("ix_parse_run_snapshot_mode_status", table_name="parse_run")
    op.drop_constraint("uq_parse_run_checkpoint", "parse_run", type_="unique")
    op.drop_constraint("parse_run_cost_nonnegative", "parse_run", type_="check")
    op.drop_constraint("parse_run_latency_nonnegative", "parse_run", type_="check")
    for column in (
        "error",
        "cost_usd",
        "latency_ms",
        "page_mappings",
        "elements",
        "html",
        "markdown",
        "raw_response",
        "provider_model_version",
        "corpus_snapshot_id",
    ):
        op.drop_column("parse_run", column)
