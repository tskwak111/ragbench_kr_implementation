"""Predeclared retrieval-screen configuration grid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

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


@dataclass(frozen=True, slots=True)
class CoreSnapshotBinding:
    parse_mode: ParseMode
    parse_snapshot_id: str
    chunk_strategy: str
    chunk_snapshot_id: str
    embedding_snapshot_id: str

    def __post_init__(self) -> None:
        UUID(self.embedding_snapshot_id)
        if any(
            not value.strip()
            for value in (self.parse_snapshot_id, self.chunk_strategy, self.chunk_snapshot_id)
        ):
            raise ValueError("core snapshot binding identities cannot be blank")


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
    snapshot_bindings: Sequence[CoreSnapshotBinding],
) -> tuple[RetrievalExperimentConfig, ...]:
    """Build the fixed 2 x 7 x 3 x 3 screening grid in deterministic order."""
    bindings = {(row.parse_mode, row.chunk_strategy): row for row in snapshot_bindings}
    expected = {(mode, strategy) for mode in PARSE_MODES for strategy in CHUNK_STRATEGIES}
    if bindings.keys() != expected or len(bindings) != len(snapshot_bindings):
        raise ValueError("snapshot bindings must map the exact 14 parse/chunk variants")
    configs: list[RetrievalExperimentConfig] = []
    for parse_mode in PARSE_MODES:
        for chunk_strategy in CHUNK_STRATEGIES:
            binding = bindings[(parse_mode, chunk_strategy)]
            for retriever in RETRIEVERS:
                for top_k in TOP_K_VALUES:
                    configs.append(
                        RetrievalExperimentConfig(
                            schema_version="retrieval-screen-v1",
                            corpus_snapshot_id=corpus_snapshot_id,
                            parse_snapshot_id=binding.parse_snapshot_id,
                            parse_mode=parse_mode,
                            chunk_snapshot_id=binding.chunk_snapshot_id,
                            chunk_strategy=chunk_strategy,
                            embedding_snapshot_id=binding.embedding_snapshot_id,
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
