from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragbench.evaluation.taxonomy import FailureCategory, FailureLabels


def test_taxonomy_is_frozen_to_the_declared_failure_categories() -> None:
    assert tuple(category.value for category in FailureCategory) == (
        "PARSER_ERROR",
        "RETRIEVAL_MISS",
        "CHUNK_BOUNDARY",
        "TABLE_ERROR",
        "RETRIEVAL_NOISE",
        "GENERATION_ERROR",
        "HALLUCINATION",
        "BAD_CITATION",
        "FALSE_ABSTENTION",
        "BENCHMARK_DEFECT",
    )


def test_failure_labels_require_one_primary_and_at_most_one_distinct_secondary() -> None:
    labels = FailureLabels(
        primary=FailureCategory.RETRIEVAL_MISS,
        secondary=FailureCategory.CHUNK_BOUNDARY,
    )
    assert labels.primary is FailureCategory.RETRIEVAL_MISS
    assert labels.secondary is FailureCategory.CHUNK_BOUNDARY

    with pytest.raises(ValidationError, match="distinct"):
        FailureLabels(
            primary=FailureCategory.RETRIEVAL_MISS,
            secondary=FailureCategory.RETRIEVAL_MISS,
        )
    with pytest.raises(ValidationError):
        FailureLabels.model_validate({"secondary": "TABLE_ERROR"})
