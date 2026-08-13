"""Dense cosine retrieval and a deterministic NumPy reference."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ragbench.retrieval.base import SearchFilter, SearchHit


class QueryEmbedder(Protocol):
    async def embed_query(
        self, query: str, *, snapshot_id: str, input_tokens: int
    ) -> tuple[float, ...]: ...


class DenseSearchRepository(Protocol):
    async def search(
        self, query_vector: tuple[float, ...], *, top_k: int, filter: SearchFilter
    ) -> list[tuple[str, float]]: ...


def cosine_top_k(
    query: npt.ArrayLike,
    matrix: npt.ArrayLike,
    k: int,
    chunk_ids: Sequence[str],
) -> list[SearchHit]:
    """Return at most ``k`` cosine hits, tie-breaking by chunk ID.

    ``k`` must be positive. Values larger than the number of rows are clamped.
    Inputs must be finite, non-zero vectors with matching dimensions.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    query_array = np.asarray(query, dtype=np.float64)
    matrix_array = np.asarray(matrix, dtype=np.float64)
    if query_array.ndim != 1:
        raise ValueError("query must be one-dimensional")
    if matrix_array.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if matrix_array.shape[1] != query_array.shape[0]:
        raise ValueError("query and matrix dimensions must match")
    if matrix_array.shape[0] != len(chunk_ids):
        raise ValueError("chunk_ids length must match matrix rows")
    if not np.all(np.isfinite(query_array)) or not np.all(np.isfinite(matrix_array)):
        raise ValueError("vectors must contain only finite values")

    query_norm = float(np.linalg.norm(query_array))
    row_norms = np.linalg.norm(matrix_array, axis=1)
    if query_norm == 0.0 or np.any(row_norms == 0.0):
        raise ValueError("zero vectors do not have cosine similarity")

    scores = (matrix_array @ query_array) / (row_norms * query_norm)
    ranked = sorted(
        ((str(chunk_id), float(score)) for chunk_id, score in zip(chunk_ids, scores, strict=True)),
        key=lambda item: (-item[1], item[0]),
    )
    return [
        SearchHit(chunk_id, score, rank, "dense")
        for rank, (chunk_id, score) in enumerate(ranked[: min(k, len(ranked))], start=1)
    ]


class DenseRetriever:
    """Retriever that embeds queries in query mode and delegates exact filtering."""

    def __init__(
        self,
        embeddings: QueryEmbedder,
        repository: DenseSearchRepository,
        *,
        token_counter: Callable[[str], int],
    ) -> None:
        self._embeddings = embeddings
        self._repository = repository
        self._token_counter = token_counter

    async def search(
        self, query: str, *, top_k: int, filter: SearchFilter
    ) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = await self._embeddings.embed_query(
            query,
            snapshot_id=filter.embedding_snapshot_id,
            input_tokens=self._token_counter(query),
        )
        rows = await self._repository.search(query_vector, top_k=top_k, filter=filter)
        return [
            SearchHit(chunk_id, float(score), rank, "dense")
            for rank, (chunk_id, score) in enumerate(rows, start=1)
        ]
