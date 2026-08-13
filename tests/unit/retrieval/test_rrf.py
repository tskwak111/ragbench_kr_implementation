"""Hand-derived reciprocal-rank-fusion behavior tests."""

import pytest

from ragbench.retrieval.base import SearchHit
from ragbench.retrieval.rrf import reciprocal_rank_fusion


def _hit(chunk_id: str, rank: int, *, score: float = 1.0, retriever: str = "test") -> SearchHit:
    return SearchHit(chunk_id, score, rank, retriever)


def test_rrf_uses_one_based_rank_and_includes_missing_candidates() -> None:
    """Catch zero-based denominators or intersection-only candidate handling."""
    hits = reciprocal_rank_fusion(
        [
            [_hit("a", 1), _hit("b", 2)],
            [_hit("b", 1), _hit("c", 2)],
        ],
        k=60,
    )

    assert [hit.chunk_id for hit in hits] == ["b", "a", "c"]
    assert [hit.score for hit in hits] == pytest.approx(
        [1 / 62 + 1 / 61, 1 / 61, 1 / 62]
    )
    assert [hit.rank for hit in hits] == [1, 2, 3]


def test_rrf_applies_weights_and_stable_id_ties() -> None:
    """Catch ignored branch weights or iteration-order tie breaking."""
    hits = reciprocal_rank_fusion(
        [[_hit("z", 1)], [_hit("a", 1)]], k=10, weights=(2.0, 2.0)
    )

    assert [(hit.chunk_id, hit.score) for hit in hits] == [
        ("a", pytest.approx(2 / 11)),
        ("z", pytest.approx(2 / 11)),
    ]


def test_rrf_rejects_duplicates_nonsequential_ranks_and_invalid_weights() -> None:
    """Catch ambiguous within-branch ranks and malformed fusion configurations."""
    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank_fusion([[_hit("a", 1), _hit("a", 2)]])
    with pytest.raises(ValueError, match="sequential"):
        reciprocal_rank_fusion([[_hit("a", 2)]])
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[_hit("a", 1)]], weights=(1.0, 2.0))
    with pytest.raises(ValueError, match="finite and positive"):
        reciprocal_rank_fusion([[_hit("a", 1)]], weights=(0.0,))
    with pytest.raises(ValueError, match="nonnegative"):
        reciprocal_rank_fusion([[_hit("a", 1)]], k=-1)


def test_rrf_empty_rankings_are_deterministically_empty() -> None:
    """Catch empty input special cases producing synthetic candidates."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []], weights=(1.0, 1.0)) == []
