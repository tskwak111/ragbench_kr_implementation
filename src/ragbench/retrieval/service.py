"""Dense/sparse hybrid retrieval orchestration."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence

from ragbench.retrieval.base import RetrievalEvidence, Retriever, SearchFilter, SearchHit
from ragbench.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    """Over-fetch dense and sparse branches over one filter, then fuse with RRF."""

    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        *,
        rrf_k: int = 60,
        weights: Sequence[float] = (1.0, 1.0),
    ) -> None:
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 0:
            raise ValueError("rrf_k must be a nonnegative integer")
        resolved_weights = tuple(float(weight) for weight in weights)
        if len(resolved_weights) != 2:
            raise ValueError("hybrid weights must contain dense and sparse values")
        if any(not math.isfinite(weight) or weight <= 0 for weight in resolved_weights):
            raise ValueError("hybrid weights must be finite and positive")
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k
        self._weights = resolved_weights

    async def search(
        self, query: str, *, top_k: int, filter: SearchFilter
    ) -> list[SearchHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        branch_k = max(20, 4 * top_k)
        dense_hits, sparse_hits = await asyncio.gather(
            self._dense.search(query, top_k=branch_k, filter=filter),
            self._sparse.search(query, top_k=branch_k, filter=filter),
        )
        fused = reciprocal_rank_fusion(
            (dense_hits, sparse_hits), k=self._rrf_k, weights=self._weights
        )
        dense_by_id = {hit.chunk_id: hit for hit in dense_hits}
        sparse_by_id = {hit.chunk_id: hit for hit in sparse_hits}
        output: list[SearchHit] = []
        for rank, hit in enumerate(fused[:top_k], start=1):
            dense = dense_by_id.get(hit.chunk_id)
            sparse = sparse_by_id.get(hit.chunk_id)
            output.append(
                SearchHit(
                    hit.chunk_id,
                    hit.score,
                    rank,
                    "hybrid-rrf",
                    RetrievalEvidence(
                        None if dense is None else dense.rank,
                        None if sparse is None else sparse.rank,
                        None if dense is None else dense.score,
                        None if sparse is None else sparse.score,
                        hit.score,
                    ),
                )
            )
        return output
