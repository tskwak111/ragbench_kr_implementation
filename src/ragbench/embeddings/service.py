"""Guarded batching, validation, and resume for embedding snapshots."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ragbench.embeddings.repository import (
    ChunkEmbeddingInput,
    EmbeddingRepository,
    EmbeddingSnapshot,
)
from ragbench.providers.base import EmbedRequest, EmbedResponse, ProviderGateway


class EmbeddingService:
    def __init__(
        self,
        gateway: ProviderGateway,
        repository: EmbeddingRepository,
        *,
        max_batch_items: int,
        max_batch_tokens: int,
        supports_input_type: bool = False,
    ) -> None:
        if max_batch_items <= 0 or max_batch_tokens <= 0:
            raise ValueError("embedding batch limits must be positive")
        self._gateway = gateway
        self._repository = repository
        self._max_batch_items = max_batch_items
        self._max_batch_tokens = max_batch_tokens
        self._supports_input_type = supports_input_type

    async def embed_chunks(
        self, snapshot: EmbeddingSnapshot, chunks: Sequence[ChunkEmbeddingInput]
    ) -> EmbeddingSnapshot:
        if len(chunks) != snapshot.expected_chunk_count:
            raise ValueError("chunk count does not match embedding snapshot plan")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("embedding snapshot chunks must have unique IDs")
        stored = await self._repository.create_snapshot(snapshot, chunks)
        if stored.complete:
            return stored
        completed = await self._repository.completed_chunk_ids(snapshot.snapshot_id)
        missing = [chunk for chunk in chunks if chunk.chunk_id not in completed]
        for batch in self._batches(missing):
            response = await self._gateway.embed(
                EmbedRequest(
                    model_id=snapshot.model_id,
                    texts=tuple(chunk.content for chunk in batch),
                    input_tokens=sum(chunk.token_count for chunk in batch),
                    provider_params=self._mode_params("document"),
                )
            )
            vectors = self._validate_response(
                response,
                expected_count=len(batch),
                expected_dimension=snapshot.dimension,
                expected_model_id=snapshot.model_id,
            )
            await self._repository.persist_batch(
                snapshot.snapshot_id,
                tuple(
                    (chunk.chunk_id, vector)
                    for chunk, vector in zip(batch, vectors, strict=True)
                ),
            )
        return await self._repository.finalize_snapshot(snapshot.snapshot_id)

    async def embed_query(
        self, query: str, *, snapshot_id: str, input_tokens: int
    ) -> tuple[float, ...]:
        if not query or input_tokens <= 0:
            raise ValueError("query and positive input_tokens are required")
        snapshot = await self._repository.get_snapshot(snapshot_id)
        if snapshot is None:
            raise KeyError(f"unknown embedding snapshot: {snapshot_id}")
        if not snapshot.complete:
            raise RuntimeError("embedding snapshot is incomplete")
        response = await self._gateway.embed(
            EmbedRequest(
                model_id=snapshot.query_model_id,
                texts=(query,),
                input_tokens=input_tokens,
                provider_params=self._mode_params("query"),
            )
        )
        return self._validate_response(
            response,
            expected_count=1,
            expected_dimension=snapshot.dimension,
            expected_model_id=snapshot.query_model_id,
        )[0]

    def _batches(
        self, chunks: Sequence[ChunkEmbeddingInput]
    ) -> list[tuple[ChunkEmbeddingInput, ...]]:
        batches: list[tuple[ChunkEmbeddingInput, ...]] = []
        current: list[ChunkEmbeddingInput] = []
        current_tokens = 0
        for chunk in chunks:
            if chunk.token_count > self._max_batch_tokens:
                raise ValueError("single chunk exceeds embedding request token limit")
            if current and (
                len(current) >= self._max_batch_items
                or current_tokens + chunk.token_count > self._max_batch_tokens
            ):
                batches.append(tuple(current))
                current = []
                current_tokens = 0
            current.append(chunk)
            current_tokens += chunk.token_count
        if current:
            batches.append(tuple(current))
        return batches

    def _mode_params(self, mode: str) -> dict[str, str]:
        return {"input_type": mode} if self._supports_input_type else {}

    @staticmethod
    def _validate_response(
        response: EmbedResponse,
        *,
        expected_count: int,
        expected_dimension: int,
        expected_model_id: str,
    ) -> tuple[tuple[float, ...], ...]:
        if response.model_id != expected_model_id:
            raise ValueError("embedding response model does not match snapshot model")
        if len(response.embeddings) != expected_count:
            raise ValueError("embedding response count does not match request")
        validated: list[tuple[float, ...]] = []
        for embedding in response.embeddings:
            vector = np.asarray(embedding, dtype=np.float64)
            if vector.ndim != 1 or vector.shape[0] != expected_dimension:
                raise ValueError("embedding response dimension does not match snapshot")
            if not np.all(np.isfinite(vector)):
                raise ValueError("embedding response must contain finite values")
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                raise ValueError("embedding response contains a zero vector")
            validated.append(tuple(float(value) for value in vector / norm))
        return tuple(validated)
