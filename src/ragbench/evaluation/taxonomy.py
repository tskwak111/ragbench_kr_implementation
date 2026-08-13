"""Frozen failure taxonomy used by sealed analysis and public exports."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator


class FailureCategory(StrEnum):
    PARSER_ERROR = "PARSER_ERROR"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    CHUNK_BOUNDARY = "CHUNK_BOUNDARY"
    TABLE_ERROR = "TABLE_ERROR"
    RETRIEVAL_NOISE = "RETRIEVAL_NOISE"
    GENERATION_ERROR = "GENERATION_ERROR"
    HALLUCINATION = "HALLUCINATION"
    BAD_CITATION = "BAD_CITATION"
    FALSE_ABSTENTION = "FALSE_ABSTENTION"
    BENCHMARK_DEFECT = "BENCHMARK_DEFECT"


class FailureLabels(BaseModel):
    """Exactly one primary failure and an optional distinct secondary failure."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    primary: FailureCategory
    secondary: FailureCategory | None = None

    @model_validator(mode="after")
    def _labels_are_distinct(self) -> Self:
        if self.secondary is self.primary:
            raise ValueError("primary and secondary failure labels must be distinct")
        return self
