"""Hand-calculated contracts for the NumPy cosine reference."""

import numpy as np
import pytest

from ragbench.retrieval.dense import cosine_top_k


def test_cosine_top_k_returns_hand_calculated_scores_in_rank_order() -> None:
    """Catch using raw dot product or ascending similarity order."""
    hits = cosine_top_k(
        np.array([1.0, 1.0]),
        np.array([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]]),
        3,
        ("axis", "same", "opposite"),
    )

    assert [hit.chunk_id for hit in hits] == ["same", "axis", "opposite"]
    assert [hit.score for hit in hits] == pytest.approx([1.0, 2**-0.5, -(2**-0.5)])
    assert all(isinstance(hit.score, float) for hit in hits)


def test_cosine_top_k_breaks_ties_by_chunk_id_independent_of_input_order() -> None:
    """Catch an unstable sort whose equal-score ranking follows storage order."""
    query = np.array([1.0, 0.0])
    matrix = np.array([[1.0, 1.0], [1.0, -1.0], [0.0, 1.0]])

    first = cosine_top_k(query, matrix, 3, ("chunk-b", "chunk-a", "chunk-c"))
    second = cosine_top_k(query, matrix[[1, 0, 2]], 3, ("chunk-a", "chunk-b", "chunk-c"))

    assert [hit.chunk_id for hit in first] == ["chunk-a", "chunk-b", "chunk-c"]
    assert first == second


@pytest.mark.parametrize(
    ("query", "matrix", "chunk_ids", "message"),
    [
        (np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]), ("a",), "one-dimensional"),
        (np.array([1.0, 0.0]), np.array([1.0, 0.0]), ("a",), "two-dimensional"),
        (np.array([1.0]), np.array([[1.0, 0.0]]), ("a",), "dimensions"),
        (np.array([np.nan, 0.0]), np.array([[1.0, 0.0]]), ("a",), "finite"),
        (np.array([1.0, 0.0]), np.array([[np.inf, 0.0]]), ("a",), "finite"),
        (np.array([0.0, 0.0]), np.array([[1.0, 0.0]]), ("a",), "zero"),
        (np.array([1.0, 0.0]), np.array([[0.0, 0.0]]), ("a",), "zero"),
        (np.array([1.0, 0.0]), np.array([[1.0, 0.0]]), (), "chunk_ids"),
    ],
)
def test_cosine_top_k_rejects_invalid_vectors(
    query: np.ndarray,
    matrix: np.ndarray,
    chunk_ids: tuple[str, ...],
    message: str,
) -> None:
    """Catch silent broadcasting, NaNs, zero-vector division, or identity drift."""
    with pytest.raises(ValueError, match=message):
        cosine_top_k(query, matrix, 1, chunk_ids)


def test_cosine_top_k_clamps_k_to_available_rows_and_rejects_nonpositive_k() -> None:
    """Catch out-of-range slicing surprises and meaningless Top-K requests."""
    hits = cosine_top_k(
        np.array([1.0, 0.0]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        99,
        ("a", "b"),
    )

    assert [hit.chunk_id for hit in hits] == ["a", "b"]
    with pytest.raises(ValueError, match="positive"):
        cosine_top_k(np.array([1.0]), np.array([[1.0]]), 0, ("a",))
