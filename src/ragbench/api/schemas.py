"""Strict public API schemas that exclude provider and restricted benchmark payloads."""

from __future__ import annotations

import math
import re
from typing import Any, Literal, Self

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator, model_validator

_SENSITIVE_NAME = re.compile(r"(?:api[_-]?key|secret|password|credential|raw[_-]?response)", re.I)
_RESTRICTED_NAME = re.compile(r"(?:gold|sealed|restricted|expected[_-]?answer)", re.I)
_PATH_NAME = re.compile(r"(?:local[_-]?path|file[_-]?path|filesystem|directory)", re.I)
_LOCAL_PATH_VALUE = re.compile(r"^(?:/|~(?:/|$)|file:|[A-Za-z]:[\\/])", re.I)
_PUBLIC_METADATA_KEYS = {
    "content_stratum",
    "document_type",
    "downloaded_at",
    "inclusion_rationale",
    "language",
    "license",
    "license_url",
    "organization",
    "page_count",
    "redistribution_status",
    "sector",
    "sha256",
    "template_family",
    "year",
}


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ComponentHealth(PublicModel):
    ready: bool
    detail: str


class HealthResponse(PublicModel):
    status: Literal["ready", "degraded"]
    ready: bool
    components: dict[str, ComponentHealth]
    version: str


class DocumentCreate(PublicModel):
    title: str = Field(min_length=1, max_length=512)
    source_uri: AnyHttpUrl
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("title")
    @classmethod
    def _title_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title cannot be blank")
        return value.strip()

    @field_validator("metadata")
    @classmethod
    def _metadata_is_public(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_public_metadata(value)

    @model_validator(mode="after")
    def _source_has_no_embedded_credentials(self) -> Self:
        if self.source_uri.username is not None or self.source_uri.password is not None:
            raise ValueError("source URI credentials are not accepted")
        return self


class DocumentResponse(PublicModel):
    id: str
    title: str
    source_uri: AnyHttpUrl
    metadata: dict[str, Any]
    status: Literal["registered", "queued", "processing", "complete", "failed"]

    @field_validator("metadata")
    @classmethod
    def _metadata_is_public(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_public_metadata(value)

    @model_validator(mode="after")
    def _source_has_no_embedded_credentials(self) -> Self:
        if self.source_uri.username is not None or self.source_uri.password is not None:
            raise ValueError("source URI credentials are not accepted")
        return self


class SearchFilterRequest(PublicModel):
    corpus_snapshot_id: str = Field(min_length=1, max_length=512)
    parse_snapshot_id: str = Field(min_length=1, max_length=512)
    chunk_strategy: str = Field(min_length=1, max_length=128)
    embedding_snapshot_id: str = Field(min_length=1, max_length=512)
    document_ids: tuple[str, ...] = Field(default=(), max_length=1000)

    @field_validator("document_ids", mode="before")
    @classmethod
    def _document_sequence(cls, value: object) -> tuple[object, ...]:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        raise ValueError("document IDs must be a sequence")

    @field_validator(
        "corpus_snapshot_id", "parse_snapshot_id", "chunk_strategy", "embedding_snapshot_id"
    )
    @classmethod
    def _identity_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("snapshot identity cannot be blank")
        return value.strip()

    @field_validator("document_ids")
    @classmethod
    def _documents_are_stable(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("document identity cannot be blank")
        return tuple(sorted(set(value)))


class SearchRequest(PublicModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=100)
    filter: SearchFilterRequest

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value.strip()


class SearchHitResponse(PublicModel):
    chunk_id: str
    score: float
    rank: int = Field(gt=0)
    retriever: str

    @field_validator("score")
    @classmethod
    def _score_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("score must be finite")
        return value


class SearchResponse(PublicModel):
    hits: tuple[SearchHitResponse, ...]


class QueryRequest(PublicModel):
    question: str = Field(min_length=1, max_length=10_000)
    config_id: str = Field(min_length=1, max_length=512)

    @field_validator("question", "config_id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query fields cannot be blank")
        return value.strip()


class CitationResponse(PublicModel):
    citation_id: str
    chunk_id: str
    document_id: str
    document_title: str
    page_start: int = Field(gt=0)
    page_end: int = Field(gt=0)
    section_path: tuple[str, ...]

    @model_validator(mode="after")
    def _page_order(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("citation page range is invalid")
        return self


class QueryResponse(PublicModel):
    question: str
    answer: str
    abstained: bool
    citations: tuple[CitationResponse, ...]
    experiment_id: str
    config_id: str
    cached: bool


class ExperimentCreate(PublicModel):
    config: dict[str, Any]
    question_ids: tuple[str, ...] = Field(min_length=1, max_length=10_000)

    @field_validator("question_ids", mode="before")
    @classmethod
    def _question_sequence(cls, value: object) -> tuple[object, ...]:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        raise ValueError("question IDs must be a sequence")

    @field_validator("config")
    @classmethod
    def _config_is_public(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_sensitive_keys(value, restricted=True, paths=False)
        return value

    @field_validator("question_ids")
    @classmethod
    def _question_ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("question IDs cannot be blank")
        if any(_RESTRICTED_NAME.search(item) for item in normalized):
            raise ValueError("restricted question identities are not accepted")
        if len(normalized) != len(set(normalized)):
            raise ValueError("question IDs must be unique")
        return normalized


class ExperimentResponse(PublicModel):
    id: str
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["planned", "queued", "running", "completed", "failed", "cancelled"]
    question_count: int = Field(ge=0)


class ExperimentListResponse(PublicModel):
    experiments: tuple[ExperimentResponse, ...]


class ExperimentMetricsResponse(PublicModel):
    experiment_id: str
    metrics: dict[str, float]

    @field_validator("metrics")
    @classmethod
    def _metrics_are_finite(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(metric) for metric in value.values()):
            raise ValueError("metrics must be finite")
        return value


def _reject_sensitive_keys(value: Any, *, restricted: bool, paths: bool) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                _SENSITIVE_NAME.search(str(key))
                or (restricted and _RESTRICTED_NAME.search(str(key)))
                or (paths and _PATH_NAME.search(str(key)))
            ):
                raise ValueError("sensitive or restricted fields are not accepted")
            _reject_sensitive_keys(child, restricted=restricted, paths=paths)
    elif isinstance(value, list | tuple):
        for child in value:
            _reject_sensitive_keys(child, restricted=restricted, paths=paths)
    elif restricted and isinstance(value, str) and _RESTRICTED_NAME.search(value):
        raise ValueError("restricted values are not accepted")


def _validate_public_metadata(value: dict[str, Any]) -> dict[str, Any]:
    if not set(value) <= _PUBLIC_METADATA_KEYS:
        raise ValueError("document metadata contains a non-public field")
    _reject_sensitive_keys(value, restricted=True, paths=True)
    for item in value.values():
        if not isinstance(item, (str, int, bool, type(None))) or isinstance(item, float):
            raise ValueError("document metadata values must be public scalars")
        if isinstance(item, str) and (_LOCAL_PATH_VALUE.search(item) or "\x00" in item):
            raise ValueError("document metadata cannot contain local paths")
    return value
