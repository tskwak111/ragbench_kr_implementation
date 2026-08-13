import pytest

from ragbench.evaluation.retrieval import (
    RetrievalCase,
    aggregate_retrieval,
    evaluate_retrieval,
    paired_bootstrap_inputs,
)


def test_metrics_match_hand_calculated_multi_evidence_ranking() -> None:
    case = RetrievalCase(
        question_id="q1",
        question_type="multi_hop",
        ranked_chunk_ids=("noise", "e2", "e1", "tail"),
        evidence_chunk_ids=("e1", "e2", "e3"),
        latency_ms=12.5,
    )

    result = evaluate_retrieval(case, k=3)

    assert result.hit_at_k == 1.0
    assert result.evidence_recall_at_k == pytest.approx(2 / 3)
    assert result.mrr == 0.5


def test_no_evidence_questions_are_reported_but_not_scored_as_successes() -> None:
    result = evaluate_retrieval(RetrievalCase("q0", "unanswerable", ("noise",), (), 4.0), k=1)

    assert result.is_scorable is False
    assert result.hit_at_k is None
    assert result.evidence_recall_at_k is None
    assert result.mrr is None


def test_macro_micro_per_type_and_bootstrap_inputs_are_hand_calculated() -> None:
    cases = (
        RetrievalCase("q2", "fact", ("x", "a"), ("a",), 30.0),
        RetrievalCase("q1", "fact", ("a",), ("a", "b"), 10.0),
        RetrievalCase("q3", "unanswerable", (), (), 20.0),
    )

    aggregate = aggregate_retrieval(cases, k=1)

    assert aggregate.overall.question_count == 3
    assert aggregate.overall.scorable_count == 2
    assert aggregate.overall.no_evidence_count == 1
    assert aggregate.overall.macro_hit_at_k == 0.5
    assert aggregate.overall.macro_evidence_recall_at_k == 0.25
    assert aggregate.overall.micro_evidence_recall_at_k == pytest.approx(1 / 3)
    assert aggregate.overall.macro_mrr == 0.75
    assert aggregate.overall.mean_latency_ms == 20.0
    assert aggregate.by_question_type["fact"].scorable_count == 2
    assert aggregate.by_question_type["unanswerable"].scorable_count == 0
    assert [row.question_id for row in aggregate.bootstrap_inputs] == ["q1", "q2"]
    assert aggregate.bootstrap_inputs[0].evidence_recall_at_k == 0.5
    assert aggregate.bootstrap_inputs_hash == (
        "f134705ca2fec5c6f2e2b536d3b2042d152e7946b86db65e9659767715c63af6"
    )


@pytest.mark.parametrize("k", [0, -1])
def test_metrics_reject_nonpositive_k(k: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        evaluate_retrieval(RetrievalCase("q", "fact", (), ("a",), 0.0), k=k)


def test_case_rejects_nonfinite_latency() -> None:
    with pytest.raises(ValueError, match="finite"):
        RetrievalCase("q", "fact", (), ("a",), float("nan"))


def test_paired_bootstrap_inputs_align_by_question_identity_not_input_order() -> None:
    left = aggregate_retrieval(
        (
            RetrievalCase("q2", "fact", ("a",), ("a",), 1),
            RetrievalCase("q1", "fact", (), ("a",), 1),
        ),
        k=1,
    )
    right = aggregate_retrieval(
        (
            RetrievalCase("q1", "fact", ("a",), ("a",), 1),
            RetrievalCase("q2", "fact", (), ("a",), 1),
        ),
        k=1,
    )

    paired = paired_bootstrap_inputs(left, right)

    assert [(row.question_id, row.left_hit_at_k, row.right_hit_at_k) for row in paired] == [
        ("q1", 0.0, 1.0),
        ("q2", 1.0, 0.0),
    ]


def test_paired_bootstrap_inputs_reject_unpaired_questions() -> None:
    left = aggregate_retrieval((RetrievalCase("q1", "fact", (), ("a",), 1),), k=1)
    right = aggregate_retrieval((RetrievalCase("q2", "fact", (), ("a",), 1),), k=1)

    with pytest.raises(ValueError, match="same scorable questions"):
        paired_bootstrap_inputs(left, right)


def test_aggregation_rejects_duplicate_question_ids() -> None:
    case = RetrievalCase("q1", "fact", (), ("a",), 1)
    with pytest.raises(ValueError, match="question IDs"):
        aggregate_retrieval((case, case), k=1)
