"""Offline SQL contracts for dense-index compatibility migrations."""

import subprocess
import sys
from pathlib import Path

from sqlalchemy import CheckConstraint

from ragbench.db.models import RetrievalResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _migration_sql(direction: str) -> str:
    command = [sys.executable, "-m", "alembic", direction]
    command += (
        ["20260814_0003:20260814_0004", "--sql"]
        if direction == "upgrade"
        else [
            "20260814_0004:20260814_0003",
            "--sql",
        ]
    )
    return subprocess.run(
        command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    ).stdout


def test_upgrade_rebuilds_completed_wide_indexes_before_relabeling() -> None:
    """Catch falsely labeling old halfvec indexes as subvector-ready."""
    sql = _migration_sql("upgrade")

    assert "quote_ident(snapshot.index_name)" in sql
    assert "DROP INDEX IF EXISTS" in sql
    assert "subvector(embedding, 1, 2000)::vector(2000)" in sql
    assert "dimension > 4000" in sql
    assert "index_state = 'pending'" in sql
    assert sql.index("CREATE INDEX") < sql.rindex("index_strategy = 'subvector-2000-rerank'")


def test_downgrade_refuses_incompatible_rows_and_restores_halfvec_indexes() -> None:
    """Catch dropping plan metadata while leaving v0004 physical indexes behind."""
    sql = _migration_sql("downgrade")

    assert "completed snapshots above dimension 4000" in sql
    assert "DROP INDEX IF EXISTS" in sql
    assert "embedding::halfvec(" in sql
    assert "halfvec_cosine_ops" in sql
    assert "artifact-mode retrieval results" in sql


def test_retrieval_result_model_enforces_exactly_one_evidence_mode() -> None:
    """Catch ambiguous or evidence-free retrieval result rows."""
    columns = RetrievalResult.__table__.columns
    checks = {
        constraint.sqltext.text
        for constraint in RetrievalResult.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert columns.legacy_chunk_id.nullable is True
    assert columns.embedding_snapshot_id.nullable is True
    assert any(
        "legacy_chunk_id IS NOT NULL AND embedding_snapshot_id IS NULL" in check
        and "chunk_id = legacy_chunk_id::text" in check
        and "legacy_chunk_id IS NULL AND embedding_snapshot_id IS NOT NULL" in check
        for check in checks
    )


def test_upgrade_binds_legacy_display_id_to_preserved_chunk_uuid() -> None:
    """Catch valid legacy foreign-key evidence paired with an unrelated display chunk ID."""
    sql = _migration_sql("upgrade")

    check = "chunk_id = legacy_chunk_id::text"
    assert check in sql
    assert sql.index("UPDATE retrieval_result SET legacy_chunk_id = chunk_id") < sql.index(check)
    assert "UPDATE retrieval_result SET chunk_id = legacy_chunk_id::text" in _migration_sql(
        "downgrade"
    )
