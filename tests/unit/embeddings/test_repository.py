"""SQL safety and lifecycle contracts for vector snapshots."""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from ragbench.embeddings.repository import (
    ChunkEmbeddingInput,
    EmbeddingSnapshot,
    MemoryEmbeddingRepository,
    chunk_manifest_hash,
    dense_search_spec,
    embedding_index_plan,
    frozen_source_metadata,
    hnsw_index_spec,
)


@pytest.mark.parametrize(
    ("dimension", "strategy", "expression"),
    [
        (1, "full-vector-hnsw", "embedding::vector(1)"),
        (2000, "full-vector-hnsw", "embedding::vector(2000)"),
        (2001, "subvector-2000-rerank", "subvector(embedding, 1, 2000)::vector(2000)"),
        (4096, "subvector-2000-rerank", "subvector(embedding, 1, 2000)::vector(2000)"),
        (16000, "subvector-2000-rerank", "subvector(embedding, 1, 2000)::vector(2000)"),
    ],
)
def test_hnsw_index_spec_selects_supported_full_or_subvector_strategy(
    dimension: int, strategy: str, expression: str
) -> None:
    """Catch rejecting 4096D after embedding or indexing it with unsupported typmods."""
    snapshot_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

    spec = hnsw_index_spec(snapshot_id, dimension)

    assert spec.name == f"ix_chunk_embedding_hnsw_{snapshot_id.hex}_{dimension}"
    assert spec.strategy == strategy
    assert expression in spec.sql
    assert "vector_cosine_ops" in spec.sql
    assert str(snapshot_id) in spec.sql


@pytest.mark.parametrize("dimension", [0, -1, 16001, True])
def test_hnsw_index_spec_rejects_unsafe_or_unsupported_dimensions(dimension: int) -> None:
    """Catch SQL interpolation with unvalidated dimensions or unsupported HNSW widths."""
    with pytest.raises(ValueError, match="dimension"):
        hnsw_index_spec(UUID(int=1), dimension)


def test_subvector_search_sql_matches_partial_index_and_reranks_full_vectors() -> None:
    """Catch SQL whose predicate or ordering cannot use the partial expression index."""
    snapshot_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    spec = dense_search_spec(
        snapshot_id,
        dimension=4096,
        top_k=3,
        document_ids=("doc-b", "doc-a"),
        candidate_factor=6,
    )

    predicate = f"ce.embedding_snapshot_id = '{snapshot_id}'::uuid"
    indexed_order = (
        "subvector(ce.embedding, 1, 2000)::vector(2000) <=> "
        "subvector(CAST(:query AS vector(4096)), 1, 2000)::vector(2000)"
    )
    assert predicate in spec.sql
    assert indexed_order in spec.sql
    assert f"ORDER BY {indexed_order} ASC LIMIT :candidate_k" in spec.sql
    assert "ce.embedding <=> CAST(:query AS vector(4096))" in spec.sql
    assert spec.params["candidate_k"] == 20
    assert spec.params["document_ids"] == ["doc-a", "doc-b"]
    assert ":snapshot_id" not in spec.sql


def test_full_vector_search_order_matches_full_vector_expression_index() -> None:
    """Catch querying an unbounded vector expression that cannot use the typmod HNSW index."""
    spec = dense_search_spec(UUID(int=8), dimension=1024, top_k=5)

    expression = "ce.embedding::vector(1024) <=> CAST(:query AS vector(1024))"
    assert f"ORDER BY {expression} ASC, ce.chunk_id ASC" in spec.sql


def _chunks() -> tuple[ChunkEmbeddingInput, ...]:
    return (
        ChunkEmbeddingInput(
            "chunk-a",
            "doc-a",
            "첫째",
            2,
            source_metadata=frozen_source_metadata({"page_start": 1}),
        ),
        ChunkEmbeddingInput(
            "chunk-b",
            "doc-b",
            "둘째",
            3,
            source_metadata=frozen_source_metadata({"page_start": 2}),
        ),
    )


def _snapshot(chunks: tuple[ChunkEmbeddingInput, ...]) -> EmbeddingSnapshot:
    plan = embedding_index_plan(4096)
    return EmbeddingSnapshot(
        "00000000-0000-0000-0000-000000000001",
        "corpus-a",
        "parse-a",
        "fixed-300-0",
        "embedding-passage",
        "embedding-query",
        4096,
        "l2",
        len(chunks),
        chunk_manifest_hash(chunks),
        plan.strategy,
        plan.candidate_factor,
        datetime(2026, 8, 14, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_manifest_registration_is_immutable_and_exact_finalization_rejects_missing_vectors(
) -> None:
    """Catch count-only completion or changing source evidence after snapshot registration."""
    chunks = _chunks()
    snapshot = _snapshot(chunks)
    repository = MemoryEmbeddingRepository()

    await repository.create_snapshot(snapshot, chunks)
    with pytest.raises(ValueError, match="manifest"):
        changed = (ChunkEmbeddingInput("chunk-a", "doc-a", "changed", 2), chunks[1])
        await repository.create_snapshot(snapshot, changed)
    await repository.persist_batch(snapshot.snapshot_id, (("chunk-a", (1.0,) * 4096),))
    with pytest.raises(RuntimeError, match="exact artifact/vector set"):
        await repository.finalize_snapshot(snapshot.snapshot_id)


@pytest.mark.asyncio
async def test_duplicate_vector_is_idempotent_only_when_values_match() -> None:
    """Catch ON CONFLICT silently accepting stale or corrupt vector values."""
    chunks = _chunks()
    snapshot = _snapshot(chunks)
    repository = MemoryEmbeddingRepository()
    await repository.create_snapshot(snapshot, chunks)
    vector = (1.0,) + (0.0,) * 4095

    await repository.persist_batch(snapshot.snapshot_id, (("chunk-a", vector),))
    await repository.persist_batch(snapshot.snapshot_id, (("chunk-a", vector),))
    with pytest.raises(ValueError, match="different vector"):
        await repository.persist_batch(
            snapshot.snapshot_id, (("chunk-a", (0.0, 1.0) + (0.0,) * 4094),)
        )


def test_chunk_source_metadata_is_copied_and_canonically_hashed() -> None:
    """Catch caller mutation or metadata order changing the immutable manifest identity."""
    metadata = {"source_block_ids": ["b", "a"], "page_start": 1}
    first = ChunkEmbeddingInput(
        "chunk-a", "doc-a", "내용", 1, source_metadata=frozen_source_metadata(metadata)
    )
    second = ChunkEmbeddingInput(
        "chunk-a",
        "doc-a",
        "내용",
        1,
        source_metadata=frozen_source_metadata(
            {"page_start": 1, "source_block_ids": ["b", "a"]}
        ),
    )
    metadata["page_start"] = 99

    assert dict(first.source_metadata)["page_start"] == "1"
    assert chunk_manifest_hash((first,)) == chunk_manifest_hash((second,))
