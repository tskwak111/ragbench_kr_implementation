"""Predeclared retrieval-screen configuration grid."""

from __future__ import annotations

from collections.abc import Sequence

from ragbench.experiments.config import (
    ParseMode,
    RetrievalExperimentConfig,
    RetrieverName,
    RRFConfig,
    TopK,
)

PARSE_MODES: tuple[ParseMode, ...] = ("standard", "enhanced")
CHUNK_STRATEGIES = (
    "fixed-300-0",
    "fixed-300-100",
    "fixed-600-0",
    "fixed-600-100",
    "fixed-1000-0",
    "fixed-1000-100",
    "heading-600-100",
)
RETRIEVERS: tuple[RetrieverName, ...] = ("dense", "bm25", "hybrid")
TOP_K_VALUES: tuple[TopK, ...] = (3, 5, 10)


def require_unique_configs(
    configs: Sequence[RetrievalExperimentConfig],
) -> tuple[RetrievalExperimentConfig, ...]:
    ordered = tuple(configs)
    seen: set[str] = set()
    for config in ordered:
        if config.semantic_hash in seen:
            raise ValueError(f"duplicate semantic experiment configuration: {config.semantic_hash}")
        seen.add(config.semantic_hash)
    return ordered


def generate_core_retrieval_configs(
    *,
    corpus_snapshot_id: str,
    question_snapshot_id: str,
    code_commit: str,
    random_seed: int,
) -> tuple[RetrievalExperimentConfig, ...]:
    """Build the fixed 2 x 7 x 3 x 3 screening grid in deterministic order."""
    configs: list[RetrievalExperimentConfig] = []
    for parse_mode in PARSE_MODES:
        parse_snapshot_id = f"parse-{parse_mode}-{corpus_snapshot_id}"
        for chunk_strategy in CHUNK_STRATEGIES:
            chunk_snapshot_id = f"chunk-{parse_snapshot_id}-{chunk_strategy}"
            embedding_snapshot_id = f"embedding-{chunk_snapshot_id}"
            for retriever in RETRIEVERS:
                for top_k in TOP_K_VALUES:
                    configs.append(
                        RetrievalExperimentConfig(
                            schema_version="retrieval-screen-v1",
                            corpus_snapshot_id=corpus_snapshot_id,
                            parse_snapshot_id=parse_snapshot_id,
                            parse_mode=parse_mode,
                            chunk_snapshot_id=chunk_snapshot_id,
                            chunk_strategy=chunk_strategy,
                            embedding_snapshot_id=embedding_snapshot_id,
                            retriever=retriever,
                            rrf=RRFConfig() if retriever == "hybrid" else None,
                            top_k=top_k,
                            question_snapshot_id=question_snapshot_id,
                            question_split="dev_auto",
                            random_seed=random_seed,
                            code_commit=code_commit,
                            metric_version="retrieval-v1",
                        )
                    )
    return require_unique_configs(configs)
