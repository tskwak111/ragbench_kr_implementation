"""SQL safety and lifecycle contracts for vector snapshots."""

from uuid import UUID

import pytest

from ragbench.embeddings.repository import hnsw_index_spec


@pytest.mark.parametrize(
    ("dimension", "cast_type", "ops"),
    [(1, "vector", "vector_cosine_ops"), (2000, "vector", "vector_cosine_ops"),
     (2001, "halfvec", "halfvec_cosine_ops"), (4000, "halfvec", "halfvec_cosine_ops")],
)
def test_hnsw_index_spec_selects_safe_dimension_specific_typmod(
    dimension: int, cast_type: str, ops: str
) -> None:
    """Catch an unbounded vector HNSW index or an invalid operator class."""
    snapshot_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

    spec = hnsw_index_spec(snapshot_id, dimension)

    assert spec.name == f"ix_chunk_embedding_hnsw_{snapshot_id.hex}_{dimension}"
    assert f"embedding::{cast_type}({dimension})" in spec.sql
    assert ops in spec.sql
    assert str(snapshot_id) in spec.sql


@pytest.mark.parametrize("dimension", [0, -1, 4001, True])
def test_hnsw_index_spec_rejects_unsafe_or_unsupported_dimensions(dimension: int) -> None:
    """Catch SQL interpolation with unvalidated dimensions or unsupported HNSW widths."""
    with pytest.raises(ValueError, match="dimension"):
        hnsw_index_spec(UUID(int=1), dimension)
