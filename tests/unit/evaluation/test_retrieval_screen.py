import asyncio
from dataclasses import replace

import pytest

from ragbench.benchmark.splits import SnapshotName, SplitSnapshot
from ragbench.core.hashing import canonical_json_hash
from ragbench.experiments.config import RetrievalExperimentConfig
from ragbench.experiments.screening import (
    BoundRetriever,
    FileScreeningStore,
    MemoryScreeningStore,
    RetrievalScreenRunner,
    ScreeningQuestion,
    ScreeningQuestionSnapshot,
    SnapshotBinding,
    SnapshotInventory,
    authorize_development_snapshot,
)
from ragbench.retrieval.base import RetrievalEvidence, SearchFilter, SearchHit

QUESTION_SNAPSHOT_ID = "1" * 64


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
            "question_snapshot_id": QUESTION_SNAPSHOT_ID,
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
        snapshot_id=QUESTION_SNAPSHOT_ID,
        split=split,
        content_hash=canonical_json_hash({"split": split, "questions": questions}),
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


def _authorization():
    question_ids = tuple(sorted(question.question_id for question in _snapshot().questions))
    return authorize_development_snapshot(
        SplitSnapshot(
            name=SnapshotName.DEV_AUTO,
            version="v1",
            snapshot_id=QUESTION_SNAPSHOT_ID,
            seed=7,
            item_count=len(question_ids),
            membership_hash=canonical_json_hash(question_ids),
            item_ids=question_ids,
        )
    )


class FakeRetriever:
    def __init__(self, *, fail_once_on: str | None = None) -> None:
        self.calls: list[str] = []
        self.top_ks: list[int] = []
        self.fail_once_on = fail_once_on

    async def search(self, query: str, *, top_k: int, filter: SearchFilter) -> list[SearchHit]:
        self.calls.append(query)
        self.top_ks.append(top_k)
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
            SearchHit(
                "noise",
                0.25,
                2,
                "hybrid-rrf",
                RetrievalEvidence(None, 2, None, 2.0, 0.25),
            ),
        ][:top_k]


def _bound(retriever: FakeRetriever | object) -> BoundRetriever:
    return BoundRetriever(
        name="hybrid",
        retriever=retriever,  # type: ignore[arg-type]
        rrf=_config().rrf,
    )


@pytest.mark.asyncio
async def test_screen_persists_every_hit_and_component_score_then_aggregates() -> None:
    store = MemoryScreeningStore()
    retriever = FakeRetriever()
    ticks = iter((1.0, 1.010, 2.0, 2.025))
    runner = RetrievalScreenRunner(
        store=store,
        inventory=_inventory(),
        development_authorization=_authorization(),
        clock=lambda: next(ticks),
    )

    result = await runner.run(_config(), _snapshot(), _bound(retriever))

    run = store.get(result.run_id)
    assert run.status == "complete"
    assert len(run.hits) == 4
    first = run.hits[0]
    assert (first.question_id, first.chunk_id, first.rank, first.score) == ("q1", "c1", 1, 0.75)
    assert (first.dense_rank, first.sparse_rank, first.fused_score) == (2, 1, 0.75)
    assert result.evaluation.overall.macro_hit_at_k == 1.0
    assert result.evaluation.overall.micro_evidence_recall_at_k == pytest.approx(2 / 3)
    assert result.selection_evaluation.k == 5
    assert retriever.top_ks == [5, 5]


@pytest.mark.asyncio
async def test_screen_resumes_without_retrieving_or_overwriting_completed_questions() -> None:
    store = MemoryScreeningStore()
    retriever = FakeRetriever(fail_once_on="둘째 질문")
    runner = RetrievalScreenRunner(
        store=store, inventory=_inventory(), development_authorization=_authorization()
    )

    with pytest.raises(RuntimeError, match="interrupted"):
        await runner.run(_config(), _snapshot(), _bound(retriever))
    assert retriever.calls == ["첫 질문", "둘째 질문"]

    result = await runner.run(_config(), _snapshot(), _bound(retriever))

    assert retriever.calls == ["첫 질문", "둘째 질문", "둘째 질문"]
    assert store.get(result.run_id).completed_question_ids == {"q1", "q2"}


@pytest.mark.asyncio
async def test_screen_refuses_incomplete_or_mismatched_snapshot_before_retrieval() -> None:
    retriever = FakeRetriever()
    runner = RetrievalScreenRunner(
        store=MemoryScreeningStore(),
        inventory=_inventory(),
        development_authorization=_authorization(),
    )

    with pytest.raises(ValueError, match="incomplete"):
        await runner.run(_config(), replace(_snapshot(), complete=False), _bound(retriever))
    bad_inventory = replace(
        _inventory(), chunk=SnapshotBinding("chunks", "wrong-parse", "fixed-300-0", True)
    )
    with pytest.raises(ValueError, match="lineage"):
        await RetrievalScreenRunner(
            store=MemoryScreeningStore(),
            inventory=bad_inventory,
            development_authorization=_authorization(),
        ).run(_config(), _snapshot(), _bound(retriever))
    assert retriever.calls == []


@pytest.mark.asyncio
async def test_normal_screen_refuses_gold_without_reading_or_logging_question_content() -> None:
    retriever = FakeRetriever()
    secret_prompt = "DO-NOT-LEAK-GOLD-PROMPT"
    questions = (ScreeningQuestion("gold-id", secret_prompt, "fact", ("gold-evidence",)),)
    snapshot = replace(
        _snapshot(split="test_gold"),
        content_hash=canonical_json_hash({"split": "test_gold", "questions": questions}),
        questions=questions,
    )

    with pytest.raises(PermissionError) as raised:
        await RetrievalScreenRunner(
            store=MemoryScreeningStore(),
            inventory=_inventory(),
            development_authorization=_authorization(),
        ).run(_config(), snapshot, _bound(retriever))

    assert secret_prompt not in str(raised.value)
    assert retriever.calls == []


def test_question_snapshot_refuses_content_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="content hash"):
        replace(_snapshot(), content_hash="0" * 64)


@pytest.mark.asyncio
async def test_screen_rejects_noncontiguous_or_duplicate_rankings_atomically() -> None:
    class InvalidRetriever:
        async def search(self, query: str, *, top_k: int, filter: SearchFilter) -> list[SearchHit]:
            return [SearchHit("same", 1.0, 2, "dense"), SearchHit("same", 0.5, 3, "dense")]

    store = MemoryScreeningStore()
    runner = RetrievalScreenRunner(
        store=store, inventory=_inventory(), development_authorization=_authorization()
    )
    with pytest.raises(ValueError, match="ranking"):
        await runner.run(_config(), _snapshot(), _bound(InvalidRetriever()))
    run = store.get(runner.run_identity(_config(), _snapshot()))
    assert run.hits == []
    assert run.completed_question_ids == set()


def test_runner_does_not_require_an_event_loop_for_identity() -> None:
    runner = RetrievalScreenRunner(
        store=MemoryScreeningStore(),
        inventory=_inventory(),
        development_authorization=_authorization(),
    )
    assert asyncio.run(asyncio.sleep(0, result=runner.run_identity(_config(), _snapshot())))


@pytest.mark.asyncio
async def test_file_store_resumes_across_runner_process_lifetimes(tmp_path) -> None:
    retriever = FakeRetriever(fail_once_on="둘째 질문")
    first = RetrievalScreenRunner(
        store=FileScreeningStore(tmp_path / "runs"),
        inventory=_inventory(),
        development_authorization=_authorization(),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        await first.run(_config(), _snapshot(), _bound(retriever))

    second_store = FileScreeningStore(tmp_path / "runs")
    result = await RetrievalScreenRunner(
        store=second_store,
        inventory=_inventory(),
        development_authorization=_authorization(),
    ).run(_config(), _snapshot(), _bound(retriever))

    assert retriever.calls == ["첫 질문", "둘째 질문", "둘째 질문"]
    assert second_store.get(result.run_id).status == "complete"


@pytest.mark.asyncio
async def test_screen_refuses_retriever_or_rrf_binding_mismatch_before_search() -> None:
    retriever = FakeRetriever()
    runner = RetrievalScreenRunner(
        store=MemoryScreeningStore(),
        inventory=_inventory(),
        development_authorization=_authorization(),
    )

    with pytest.raises(ValueError, match="retriever binding"):
        await runner.run(
            _config(),
            _snapshot(),
            BoundRetriever(name="dense", retriever=retriever, rrf=None),
        )
    assert _config().rrf is not None
    wrong_rrf = _config().rrf.model_copy(update={"rank_constant": 99})
    with pytest.raises(ValueError, match="RRF binding"):
        await runner.run(
            _config(),
            _snapshot(),
            BoundRetriever(name="hybrid", retriever=retriever, rrf=wrong_rrf),
        )
    assert retriever.calls == []
