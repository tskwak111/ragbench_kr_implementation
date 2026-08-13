"""Strict immutable retrieval-screen configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragbench.core.hashing import canonical_json_hash

ParseMode = Literal["standard", "enhanced"]
RetrieverName = Literal["dense", "bm25", "hybrid"]
TopK = Literal[3, 5, 10]


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    output: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in output:
            raise ValueError(f"duplicate YAML key: {key}")
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


class RRFConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    rank_constant: int = Field(default=60, ge=0)
    dense_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    sparse_weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)


class RetrievalExperimentConfig(BaseModel):
    """All identities required to reproduce one retrieval-only screen."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["retrieval-screen-v1"]
    corpus_snapshot_id: str = Field(min_length=1)
    parse_snapshot_id: str = Field(min_length=1)
    parse_mode: ParseMode
    chunk_snapshot_id: str = Field(min_length=1)
    chunk_strategy: str = Field(min_length=1)
    embedding_snapshot_id: str = Field(min_length=1)
    retriever: RetrieverName
    rrf: RRFConfig | None
    top_k: TopK
    question_snapshot_id: str = Field(min_length=1)
    question_split: Literal["dev_auto"]
    random_seed: int = Field(ge=0)
    code_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    metric_version: Literal["retrieval-v1"]

    @field_validator(
        "corpus_snapshot_id",
        "parse_snapshot_id",
        "chunk_snapshot_id",
        "chunk_strategy",
        "embedding_snapshot_id",
        "question_snapshot_id",
        "code_commit",
    )
    @classmethod
    def _identity_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("experiment identity cannot be blank")
        return value.strip()

    @model_validator(mode="after")
    def _rrf_matches_retriever(self) -> Self:
        if (self.retriever == "hybrid") != (self.rrf is not None):
            raise ValueError("RRF parameters are required only for the hybrid retriever")
        return self

    @property
    def semantic_hash(self) -> str:
        return canonical_json_hash(self.model_dump(mode="json"))

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        if not isinstance(raw, dict):
            raise ValueError("experiment YAML root must be a mapping")
        return cls.model_validate(raw)

    def to_yaml(self, path: Path) -> None:
        with path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(
                self.model_dump(mode="json"),
                stream,
                allow_unicode=True,
                sort_keys=True,
            )
