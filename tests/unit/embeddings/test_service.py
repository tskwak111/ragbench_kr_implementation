"""Offline contracts for resumable, guarded embedding snapshots."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from ragbench.embeddings.repository import (
    ChunkEmbeddingInput,
    EmbeddingSnapshot,
    MemoryEmbeddingRepository,
    chunk_manifest_hash,
    embedding_index_plan,
)
from ragbench.embeddings.service import EmbeddingService
from ragbench.providers.base import EmbedRequest, EmbedResponse


class RecordingGateway:
    def __init__(self, responses: list[EmbedResponse]) -> None:
        self.responses = responses
        self.requests: list[EmbedRequest] = []

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def _chunk(
    chunk_id: str, content: str, tokens: int, document_id: str = "doc-a"
) -> ChunkEmbeddingInput:
    return ChunkEmbeddingInput(chunk_id, document_id, content, tokens)


def _snapshot(
    *, chunks: tuple[ChunkEmbeddingInput, ...] | None = None, complete: bool = False
) -> EmbeddingSnapshot:
    chunks = chunks or (
        _chunk("chunk-b", "둘", 2),
        _chunk("chunk-a", "하나", 3),
        _chunk("chunk-c", "셋", 2),
    )
    plan = embedding_index_plan(2)
    return EmbeddingSnapshot(
        snapshot_id="snapshot-a",
        corpus_snapshot_id="corpus-a",
        parse_snapshot_id="parse-a",
        chunk_strategy="fixed-300-overlap-0",
        model_id="embedding-passage",
        query_model_id="embedding-query",
        dimension=2,
        normalization="l2",
        expected_chunk_count=len(chunks),
        artifact_manifest_hash=chunk_manifest_hash(chunks),
        index_strategy=plan.strategy,
        candidate_factor=plan.candidate_factor,
        created_at=datetime(2026, 8, 14, tzinfo=UTC),
        complete=complete,
        index_name="memory-index" if complete else None,
        index_state="ready" if complete else "pending",
    )


@pytest.mark.asyncio
async def test_embed_chunks_obeys_item_and_token_limits_and_preserves_order() -> None:
    """Catch oversized batches or response vectors being attached to the wrong chunk."""
    gateway = RecordingGateway(
        [
            EmbedResponse(((3.0, 4.0), (0.0, 2.0)), {}, "one", "embedding-passage"),
            EmbedResponse(((-2.0, 0.0),), {}, "two", "embedding-passage"),
        ]
    )
    repository = MemoryEmbeddingRepository()
    service = EmbeddingService(
        gateway,
        repository,
        max_batch_items=2,
        max_batch_tokens=5,
        supports_input_type=True,
    )
    chunks = (
        _chunk("chunk-b", "둘", 2),
        _chunk("chunk-a", "하나", 3),
        _chunk("chunk-c", "셋", 2),
    )

    completed = await service.embed_chunks(_snapshot(chunks=chunks), chunks)

    assert [request.texts for request in gateway.requests] == [("둘", "하나"), ("셋",)]
    assert [request.input_tokens for request in gateway.requests] == [5, 2]
    assert [request.provider_params for request in gateway.requests] == [
        {"input_type": "document"},
        {"input_type": "document"},
    ]
    assert list(repository.vectors["snapshot-a"]) == ["chunk-b", "chunk-a", "chunk-c"]
    assert repository.vectors["snapshot-a"]["chunk-b"] == pytest.approx((0.6, 0.8))
    assert completed.complete is True


@pytest.mark.asyncio
async def test_embed_chunks_resumes_only_missing_chunks_and_finalizes_last() -> None:
    """Catch resume re-embedding paid chunks or marking a partial snapshot complete."""
    repository = MemoryEmbeddingRepository()
    chunks = (_chunk("chunk-a", "a", 1), _chunk("chunk-b", "b", 1), _chunk("chunk-c", "c", 1))
    snapshot = _snapshot(chunks=chunks)
    await repository.create_snapshot(snapshot, chunks)
    await repository.persist_batch(snapshot.snapshot_id, (("chunk-b", (1.0, 0.0)),))
    gateway = RecordingGateway(
        [EmbedResponse(((0.0, 1.0), (1.0, 1.0)), {}, "resume", "embedding-passage")]
    )
    service = EmbeddingService(gateway, repository, max_batch_items=10, max_batch_tokens=10)
    completed = await service.embed_chunks(snapshot, chunks)

    assert gateway.requests[0].texts == ("a", "c")
    assert completed.complete is True
    assert repository.finalize_attempts == 1


@pytest.mark.asyncio
async def test_complete_snapshot_rejects_stale_index_plan_before_short_circuit() -> None:
    """Catch returning a v0003 halfvec index after metadata expects subvector reranking."""
    chunks = (_chunk("a", "a", 1),)
    snapshot = _snapshot(chunks=chunks, complete=True)
    stale = replace(
        snapshot,
        index_name="old-halfvec-index",
        index_state="ready",
    )
    object.__setattr__(stale, "index_strategy", "subvector-2000-rerank")
    object.__setattr__(stale, "candidate_factor", 4)
    repository = MemoryEmbeddingRepository()
    repository.snapshots[snapshot.snapshot_id] = stale
    repository.vectors[snapshot.snapshot_id] = {"a": (1.0, 0.0)}
    repository.artifacts[snapshot.snapshot_id] = {"a": chunks[0]}
    service = EmbeddingService(
        RecordingGateway([]), repository, max_batch_items=2, max_batch_tokens=10
    )

    with pytest.raises(RuntimeError, match="migration or index rebuild"):
        await service.embed_chunks(snapshot, chunks)


@pytest.mark.asyncio
async def test_complete_snapshot_requires_expected_generated_index_name() -> None:
    """Catch trusting a complete flag whose physical index identity is stale."""
    chunks = (_chunk("a", "a", 1),)
    snapshot = replace(
        _snapshot(chunks=chunks, complete=True),
        snapshot_id="00000000-0000-0000-0000-000000000888",
        index_name="wrong-index",
        index_state="ready",
    )
    repository = MemoryEmbeddingRepository()
    repository.snapshots[snapshot.snapshot_id] = snapshot
    repository.vectors[snapshot.snapshot_id] = {"a": (1.0, 0.0)}
    repository.artifacts[snapshot.snapshot_id] = {"a": chunks[0]}
    service = EmbeddingService(
        RecordingGateway([]), repository, max_batch_items=2, max_batch_tokens=10
    )

    with pytest.raises(RuntimeError, match="migration or index rebuild"):
        await service.embed_chunks(snapshot, chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (EmbedResponse(((1.0, 0.0),), {}, None, "embedding-passage"), "count"),
        (
            EmbedResponse(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), {}, None, "embedding-passage"),
            "dimension",
        ),
        (
            EmbedResponse(((float("nan"), 0.0), (1.0, 0.0)), {}, None, "embedding-passage"),
            "finite",
        ),
        (
            EmbedResponse(((0.0, 0.0), (1.0, 0.0)), {}, None, "embedding-passage"),
            "zero",
        ),
        (
            EmbedResponse(((1.0, 0.0), (1.0, 0.0)), {}, None, "different-model"),
            "model",
        ),
    ],
)
async def test_embed_chunks_rejects_malformed_provider_responses_before_persistence(
    response: EmbedResponse, message: str
) -> None:
    """Catch corrupt vectors or model drift entering a supposedly immutable snapshot."""
    repository = MemoryEmbeddingRepository()
    gateway = RecordingGateway([response])
    service = EmbeddingService(gateway, repository, max_batch_items=2, max_batch_tokens=10)
    chunks = (_chunk("a", "a", 1), _chunk("b", "b", 1))

    with pytest.raises(ValueError, match=message):
        await service.embed_chunks(_snapshot(chunks=chunks), chunks)

    assert repository.vectors["snapshot-a"] == {}
    assert repository.snapshots["snapshot-a"].complete is False


@pytest.mark.asyncio
async def test_embed_query_uses_query_mode_and_validates_snapshot_completion() -> None:
    """Catch document-mode query vectors or retrieval against an incomplete index."""
    repository = MemoryEmbeddingRepository()
    chunks = (_chunk("chunk-b", "둘", 2), _chunk("chunk-a", "하나", 3), _chunk("chunk-c", "셋", 2))
    completed_snapshot = _snapshot(chunks=chunks, complete=True)
    await repository.create_snapshot(completed_snapshot, chunks)
    gateway = RecordingGateway([EmbedResponse(((0.0, 2.0),), {}, "query", "embedding-query")])
    service = EmbeddingService(
        gateway, repository, max_batch_items=2, max_batch_tokens=10, supports_input_type=True
    )

    vector = await service.embed_query("질문", snapshot_id="snapshot-a", input_tokens=2)

    assert vector == pytest.approx((0.0, 1.0))
    assert gateway.requests[0].model_id == "embedding-query"
    assert gateway.requests[0].provider_params == {"input_type": "query"}

    await repository.create_snapshot(
        replace(_snapshot(chunks=chunks), snapshot_id="snapshot-incomplete"), chunks
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        await service.embed_query("질문", snapshot_id="snapshot-incomplete", input_tokens=2)


@pytest.mark.asyncio
async def test_embed_chunks_rejects_single_chunk_over_exact_request_limit() -> None:
    """Catch silently issuing a provider request beyond the configured token cap."""
    service = EmbeddingService(
        RecordingGateway([]),
        MemoryEmbeddingRepository(),
        max_batch_items=2,
        max_batch_tokens=3,
    )

    with pytest.raises(ValueError, match="token limit"):
        await service.embed_chunks(
            _snapshot(chunks=(_chunk("a", "oversized", 4),)),
            (_chunk("a", "oversized", 4),),
        )


def test_chunk_input_derives_and_verifies_content_hash() -> None:
    """Catch content evidence whose declared digest no longer matches the text."""
    chunk = _chunk("a", "내용", 1, "doc-a")

    assert len(chunk.content_sha256) == 64
    with pytest.raises(ValueError, match="content hash"):
        ChunkEmbeddingInput("a", "doc-a", "내용", 1, content_sha256="0" * 64)
