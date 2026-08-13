import asyncio
from dataclasses import replace

import pytest

from ragbench.core.hashing import canonical_json_hash
from ragbench.experiments.config import RetrievalExperimentConfig
from ragbench.experiments.screening import (
    MemoryScreeningStore,
    RetrievalScreenRunner,
    ScreeningQuestion,
    ScreeningQuestionSnapshot,
    SnapshotBinding,
    SnapshotInventory,
)
from ragbench.retrieval.base import RetrievalEvidence, SearchFilter, SearchHit


def _config() -> RetrievalExperimentConfig:
    return RetrievalExperimentConfig.model_validate(
        {
            "schema_version": "retrieval-screen-v1",
            "corpus_snapshot_id": "corpus",
            "parse_snapshot_id": "parse",
            "parse_mode": "standard",
            "chunk_snapshot_id": "chunks",
            "chunk_strategy": "fixed-300-0",
            "embedding_snapshot_id": "embed",
            "retriever": "hybrid",
            "rrf": {"rank_constant": 60, "dense_weight": 1.0, "sparse_weight": 1.0},
            "top_k": 3,
            "question_snapshot_id": "questions",
            "question_split": "dev_auto",
            "random_seed": 7,
            "code_commit": "0bce46e",
            "metric_version": "retrieval-v1",
        }
    )


def _snapshot(*, split: str = "dev_auto") -> ScreeningQuestionSnapshot:
    questions = (
        ScreeningQuestion("q1", "첫 질문", "fact", ("c1",)),
        ScreeningQuestion("q2", "둘째 질문", "numeric", ("c2", "c3")),
    )
    return ScreeningQuestionSnapshot(
        snapshot_id="questions",
        split=split,
        content_hash=canonical_json_hash(questions),
        complete=True,
        questions=questions,
    )


def _inventory() -> SnapshotInventory:
    return SnapshotInventory(
        corpus=SnapshotBinding("corpus", None, None, True),
        parse=SnapshotBinding("parse", "corpus", "standard", True),
        chunk=SnapshotBinding("chunks", "parse", "fixed-300-0", True),
        embedding=SnapshotBinding("embed", "chunks", "fixed-300-0", True),
    )


class FakeRetriever:
    def __init__(self, *, fail_once_on: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_once_on = fail_once_on

    async def search(self, query: str, *, top_k: int, filter: SearchFilter) -> list[SearchHit]:
        self.calls.append(query)
        if query == self.fail_once_on:
            self.fail_once_on = None
            raise RuntimeError("interrupted")
        chunk_id = "c1" if query == "첫 질문" else "c2"
        return [
            SearchHit(
                chunk_id,
                0.75,
                1,
                "hybrid-rrf",
                RetrievalEvidence(2, 1, 0.5, 3.0, 0.75),
            ),
            SearchHit("noise", 0.25, 2, "hybrid-rrf"),
        ][:top_k]


@pytest.mark.asyncio
async def test_screen_persists_every_hit_and_component_score_then_aggregates() -> None:
    store = MemoryScreeningStore()
    retriever = FakeRetriever()
    ticks = iter((1.0, 1.010, 2.0, 2.025))
    runner = RetrievalScreenRunner(store=store, inventory=_inventory(), clock=lambda: next(ticks))

    result = await runner.run(_config(), _snapshot(), retriever)

    run = store.get(result.run_id)
    assert run.status == "complete"
    assert len(run.hits) == 4
    first = run.hits[0]
    assert (first.question_id, first.chunk_id, first.rank, first.score) == ("q1", "c1", 1, 0.75)
    assert (first.dense_rank, first.sparse_rank, first.fused_score) == (2, 1, 0.75)
    assert result.evaluation.overall.macro_hit_at_k == 1.0
    assert result.evaluation.overall.micro_evidence_recall_at_k == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_screen_resumes_without_retrieving_or_overwriting_completed_questions() -> None:
    store = MemoryScreeningStore()
    retriever = FakeRetriever(fail_once_on="둘째 질문")
    runner = RetrievalScreenRunner(store=store, inventory=_inventory())

    with pytest.raises(RuntimeError, match="interrupted"):
        await runner.run(_config(), _snapshot(), retriever)
    assert retriever.calls == ["첫 질문", "둘째 질문"]

    result = await runner.run(_config(), _snapshot(), retriever)

    assert retriever.calls == ["첫 질문", "둘째 질문", "둘째 질문"]
    assert store.get(result.run_id).completed_question_ids == {"q1", "q2"}


@pytest.mark.asyncio
async def test_screen_refuses_incomplete_or_mismatched_snapshot_before_retrieval() -> None:
    retriever = FakeRetriever()
    runner = RetrievalScreenRunner(store=MemoryScreeningStore(), inventory=_inventory())

    with pytest.raises(ValueError, match="incomplete"):
        await runner.run(_config(), replace(_snapshot(), complete=False), retriever)
    bad_inventory = replace(
        _inventory(), chunk=SnapshotBinding("chunks", "wrong-parse", "fixed-300-0", True)
    )
    with pytest.raises(ValueError, match="lineage"):
        await RetrievalScreenRunner(
            store=MemoryScreeningStore(), inventory=bad_inventory
        ).run(_config(), _snapshot(), retriever)
    assert retriever.calls == []


@pytest.mark.asyncio
async def test_normal_screen_refuses_gold_without_reading_or_logging_question_content() -> None:
    retriever = FakeRetriever()
    secret_prompt = "DO-NOT-LEAK-GOLD-PROMPT"
    questions = (ScreeningQuestion("gold-id", secret_prompt, "fact", ("gold-evidence",)),)
    snapshot = replace(
        _snapshot(split="test_gold"),
        content_hash=canonical_json_hash(questions),
        questions=questions,
    )

    with pytest.raises(PermissionError) as raised:
        await RetrievalScreenRunner(
            store=MemoryScreeningStore(), inventory=_inventory()
        ).run(_config(), snapshot, retriever)

    assert secret_prompt not in str(raised.value)
    assert retriever.calls == []


def test_question_snapshot_refuses_content_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="content hash"):
        replace(_snapshot(), content_hash="0" * 64)


@pytest.mark.asyncio
async def test_screen_rejects_noncontiguous_or_duplicate_rankings_atomically() -> None:
    class InvalidRetriever:
        async def search(
            self, query: str, *, top_k: int, filter: SearchFilter
        ) -> list[SearchHit]:
            return [SearchHit("same", 1.0, 2, "dense"), SearchHit("same", 0.5, 3, "dense")]

    store = MemoryScreeningStore()
    runner = RetrievalScreenRunner(store=store, inventory=_inventory())
    with pytest.raises(ValueError, match="ranking"):
        await runner.run(_config(), _snapshot(), InvalidRetriever())
    run = store.get(runner.run_identity(_config(), _snapshot()))
    assert run.hits == []
    assert run.completed_question_ids == set()


def test_runner_does_not_require_an_event_loop_for_identity() -> None:
    runner = RetrievalScreenRunner(store=MemoryScreeningStore(), inventory=_inventory())
    assert asyncio.run(asyncio.sleep(0, result=runner.run_identity(_config(), _snapshot())))
