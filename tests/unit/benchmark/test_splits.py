"""Contracts for human review, leakage-safe splits, and sealed gold access."""

from __future__ import annotations

import hashlib
import json
import runpy
import stat
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from ragbench.benchmark.splits import (
    REVIEW_COLUMNS,
    BenchmarkItem,
    GoldAccessError,
    GoldItem,
    GoldMetadata,
    ImmutableSnapshotError,
    ReviewCandidate,
    ReviewDecision,
    ReviewRecord,
    SnapshotName,
    adjudicate_reviews,
    authorize_gold_access,
    build_split_snapshots,
    calculate_review_agreement,
    load_sealed_gold,
    plan_review_sample,
    public_gold_metadata,
    seal_gold,
)
from ragbench.cli import build_app
from ragbench.core.hashing import canonical_json_hash

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_HOOKS = runpy.run_path(PROJECT_ROOT / "tests" / "conftest.py")
gold_test_skip_reason = TEST_HOOKS["gold_test_skip_reason"]
RUNNER = CliRunner()


def _review_candidates(count: int = 360) -> tuple[ReviewCandidate, ...]:
    kinds = ("fact", "numeric_table", "comparison", "multihop", "unanswerable", "summary")
    difficulties = ("easy", "medium", "hard")
    return tuple(
        ReviewCandidate(
            candidate_id=f"candidate-{index:03d}",
            natural_question=f"합성 검수 질문 {index}?",
            question_type=kinds[index % len(kinds)],
            difficulty=difficulties[(index // len(kinds)) % len(difficulties)],
            document_ids=(f"doc-{index % 12}",),
            parse_sensitive=(index % 4 == 0),
            answerable=(index % 5 != 0),
            generator_confidence=(index % 100) / 100,
        )
        for index in range(count)
    )


def test_review_template_columns_are_complete_and_stable() -> None:
    """Catch dropping a required human decision or provenance field from review exports."""
    assert REVIEW_COLUMNS == (
        "natural_question",
        "answer_exists",
        "evidence_correct",
        "page_correct",
        "answer_unambiguous",
        "answerable_label_correct",
        "type_difficulty_correct",
        "reviewer_decision",
        "corrected_answer",
        "corrected_evidence",
        "notes",
        "reviewer_id",
        "timestamp",
    )


def test_normal_pytest_cannot_load_gold_even_when_environment_gate_is_set() -> None:
    """Catch an inherited ALLOW_GOLD_ACCESS flag enabling gold tests without selection."""
    environment = {"ALLOW_GOLD_ACCESS": "1"}

    assert gold_test_skip_reason("", environment) == (
        "gold tests require explicit pytest -m gold selection"
    )
    assert gold_test_skip_reason("not live and not gold", environment) == (
        "gold tests require explicit pytest -m gold selection"
    )
    assert gold_test_skip_reason("gold", {}) == "gold tests require ALLOW_GOLD_ACCESS=1"
    assert gold_test_skip_reason("gold", environment) is None


def test_review_sample_is_seeded_stratified_and_hides_generator_confidence() -> None:
    """Catch an easy-page-biased or confidence-biased review queue that cannot be replayed."""
    candidates = _review_candidates()

    first = plan_review_sample(candidates, sample_size=300, seed=73)
    replay = plan_review_sample(tuple(reversed(candidates)), sample_size=300, seed=73)
    different_seed = plan_review_sample(candidates, sample_size=300, seed=74)

    assert [item.candidate_id for item in first] == [item.candidate_id for item in replay]
    assert [item.candidate_id for item in first] != [item.candidate_id for item in different_seed]
    assert len(first) == 300
    assert {item.question_type for item in first} == {
        "fact",
        "numeric_table",
        "comparison",
        "multihop",
        "unanswerable",
        "summary",
    }
    assert {item.difficulty for item in first} == {"easy", "medium", "hard"}
    assert {item.parse_sensitive for item in first} == {False, True}
    assert {item.answerable for item in first} == {False, True}
    assert len({item.document_ids[0] for item in first}) == 12
    assert all("generator_confidence" not in item.model_dump() for item in first)


def test_review_sample_refuses_to_weaken_the_300_item_protocol() -> None:
    """Catch silently treating a sub-300 queue as completion-ready manual review."""
    with pytest.raises(ValueError, match="at least 300"):
        plan_review_sample(_review_candidates(299), sample_size=299, seed=1)


def _benchmark_items() -> tuple[BenchmarkItem, ...]:
    return (
        BenchmarkItem(
            item_id="a1",
            natural_question="매출은 얼마인가?",
            document_ids=("doc-a",),
            question_family_id="family-sales",
            paraphrase_group_id="para-sales",
        ),
        BenchmarkItem(
            item_id="a2",
            natural_question="매출액을 알려줘",
            document_ids=("doc-b",),
            question_family_id="family-sales",
            paraphrase_group_id="para-other",
        ),
        BenchmarkItem(
            item_id="a3",
            natural_question="영업 수익은?",
            document_ids=("doc-c",),
            question_family_id="family-other",
            paraphrase_group_id="para-sales",
        ),
        BenchmarkItem(
            item_id="b1",
            natural_question="직원 수는?",
            document_ids=("doc-d",),
            question_family_id="family-staff",
            paraphrase_group_id="para-staff",
        ),
        BenchmarkItem(
            item_id="b2",
            natural_question="인원은 몇 명인가?",
            document_ids=("doc-d", "doc-e"),
            question_family_id="family-headcount",
            paraphrase_group_id="para-headcount",
        ),
        BenchmarkItem(
            item_id="c1",
            natural_question="설립일은?",
            document_ids=("doc-f",),
            question_family_id="family-founded",
            paraphrase_group_id="para-founded",
        ),
        BenchmarkItem(
            item_id="d1",
            natural_question="본사는 어디인가?",
            document_ids=("doc-g",),
            question_family_id="family-hq",
            paraphrase_group_id="para-hq",
        ),
        BenchmarkItem(
            item_id="e1",
            natural_question="CEO는 누구인가?",
            document_ids=("doc-h",),
            question_family_id="family-ceo",
            paraphrase_group_id="para-ceo",
        ),
        BenchmarkItem(
            item_id="f1",
            natural_question="자산은 얼마인가?",
            document_ids=("doc-i",),
            question_family_id="family-assets",
            paraphrase_group_id="para-assets",
        ),
    )


def test_split_components_prevent_document_family_and_paraphrase_leakage() -> None:
    """Catch related evidence or question variants crossing evaluation boundaries."""
    snapshots = build_split_snapshots(
        _benchmark_items(),
        version="v2026-08-14",
        seed=19,
    )
    assignment = {
        item_id: name for name, snapshot in snapshots.items() for item_id in snapshot.item_ids
    }

    assert set(snapshots) == {SnapshotName.DEV_AUTO, SnapshotName.TEST_GOLD, SnapshotName.STRESS}
    assert len(set(assignment.values())) == 3
    assert assignment["a1"] == assignment["a2"] == assignment["a3"]
    assert assignment["b1"] == assignment["b2"]
    assert all(snapshot.version == "v2026-08-14" for snapshot in snapshots.values())
    assert len({snapshot.snapshot_id for snapshot in snapshots.values()}) == 3
    assert all("item_ids" not in snapshot.model_dump() for snapshot in snapshots.values())
    assert all(
        snapshot.model_dump()["item_count"] == len(snapshot.item_ids)
        for snapshot in snapshots.values()
    )
    assert all(
        len(str(snapshot.model_dump()["membership_hash"])) == 64 for snapshot in snapshots.values()
    )
    with pytest.raises(ValidationError, match="frozen"):
        snapshots[SnapshotName.DEV_AUTO].seed = 20  # type: ignore[misc]


def test_split_normalizes_document_ids_before_leakage_grouping() -> None:
    """Catch whitespace variants of one document being assigned across split boundaries."""
    items = list(_benchmark_items())
    first_payload = items[0].model_dump()
    first_payload["document_ids"] = (" doc-shared ",)
    second_payload = items[3].model_dump()
    second_payload["document_ids"] = ("doc-shared",)
    first = BenchmarkItem.model_validate(first_payload)
    second = BenchmarkItem.model_validate(second_payload)
    items[0], items[3] = first, second

    snapshots = build_split_snapshots(items, version="v-normalized", seed=31)
    assignment = {
        item_id: name for name, snapshot in snapshots.items() for item_id in snapshot.item_ids
    }

    assert assignment[first.item_id] == assignment[second.item_id]


def _review(
    index: int,
    reviewer: str,
    decision: ReviewDecision,
    *,
    notes: str = "",
) -> ReviewRecord:
    return ReviewRecord(
        natural_question=f"이중 검수 합성 질문 {index}?",
        answer_exists=True,
        evidence_correct=True,
        page_correct=True,
        answer_unambiguous=True,
        answerable_label_correct=True,
        type_difficulty_correct=True,
        reviewer_decision=decision,
        corrected_answer="",
        corrected_evidence="",
        notes=notes,
        reviewer_id=reviewer,
        timestamp=datetime(2026, 8, 14, 9, tzinfo=UTC) + timedelta(minutes=index),
    )


def test_double_review_reports_raw_agreement_and_cohens_kappa() -> None:
    """Catch reporting raw agreement alone or accepting fewer than 50 double-reviewed items."""
    reviews: list[ReviewRecord] = []
    for index in range(50):
        first = ReviewDecision.ACCEPT if index < 30 else ReviewDecision.REJECT
        if index < 26 or 30 <= index < 49:
            second = first
        else:
            second = (
                ReviewDecision.REJECT if first is ReviewDecision.ACCEPT else ReviewDecision.ACCEPT
            )
        reviews.extend((_review(index, "reviewer-a", first), _review(index, "reviewer-b", second)))

    metrics = calculate_review_agreement(reviews)

    assert metrics.item_count == 50
    assert metrics.raw_agreement == pytest.approx(0.9)
    assert metrics.cohens_kappa == pytest.approx(0.7967479675)
    with pytest.raises(ValueError, match="at least 50"):
        calculate_review_agreement(reviews[:-2])


def test_review_records_fail_closed_and_disagreement_requires_written_adjudication() -> None:
    """Catch anonymous, naive-time, or silently resolved human decisions."""
    payload = _review(1, "reviewer-a", ReviewDecision.ACCEPT).model_dump()
    payload["timestamp"] = datetime(2026, 8, 14, 9)
    with pytest.raises(ValidationError, match="timezone"):
        ReviewRecord.model_validate(payload)

    left = _review(2, "reviewer-a", ReviewDecision.ACCEPT)
    right = _review(2, "reviewer-b", ReviewDecision.REJECT)
    with pytest.raises(ValueError, match="written adjudication"):
        adjudicate_reviews(left, right, adjudicator_id="reviewer-c", notes="")
    result = adjudicate_reviews(
        left,
        right,
        adjudicator_id="reviewer-c",
        notes="원문 PDF와 파싱 결과를 대조해 근거 불충분으로 기각",
        decision=ReviewDecision.REJECT,
    )
    assert result.reviewer_decision is ReviewDecision.REJECT
    assert result.reviewer_id == "reviewer-c"

    corrected = adjudicate_reviews(
        left,
        right,
        adjudicator_id="reviewer-c",
        notes="원문 대조 후 답과 근거를 정정",
        decision=ReviewDecision.CORRECT,
        corrected_answer="정정 답변",
        corrected_evidence="정정 근거",
    )
    assert corrected.reviewer_decision is ReviewDecision.CORRECT
    assert corrected.corrected_answer == "정정 답변"
    assert corrected.corrected_evidence == "정정 근거"


def _gold_items(count: int) -> tuple[GoldItem, ...]:
    return tuple(
        GoldItem(
            item_id=f"gold-{index:03d}",
            natural_question=f"봉인 합성 질문 {index}?",
            expected_answer=f"합성 답변 {index}",
            evidence=(f"synthetic-chunk-{index}",),
            question_type="fact",
            difficulty="medium",
            answerable=True,
            document_cluster_id=f"document-{index // 10}",
        )
        for index in range(count)
    )


def _metadata_snapshot_id(metadata: GoldMetadata, content_hash: str) -> str:
    return canonical_json_hash(
        {
            "version": metadata.version,
            "content_sha256": content_hash,
            "item_count": metadata.item_count,
            "scope_status": metadata.scope_status,
        }
    )


def test_gold_seal_is_atomic_0600_hashed_immutable_and_public_metadata_is_safe(
    tmp_path: Path,
) -> None:
    """Catch mutable or publicly revealing gold artifacts and incomplete quality claims."""
    gold_path = tmp_path / "gold-v1.jsonl"
    metadata = seal_gold(
        _gold_items(300),
        gold_path,
        version="v1",
        quality_threshold_met=True,
        sealed_at=datetime(2026, 8, 14, 10, tzinfo=UTC),
    )
    payload = gold_path.read_bytes()

    assert stat.S_IMODE(gold_path.stat().st_mode) == 0o600
    assert metadata.content_sha256 == hashlib.sha256(payload).hexdigest()
    assert metadata.item_count == 300
    assert metadata.scope_status == "full"
    public = public_gold_metadata(metadata)
    assert public == {
        "content_sha256": metadata.content_sha256,
        "file_name": "gold-v1.jsonl",
        "item_count": 300,
        "scope_status": "full",
        "sealed_at": "2026-08-14T10:00:00Z",
        "snapshot_id": metadata.snapshot_id,
        "version": "v1",
    }
    serialized = json.dumps(public, ensure_ascii=False)
    assert "gold-000" not in serialized
    assert "봉인 합성 질문" not in serialized
    with pytest.raises(ImmutableSnapshotError, match="already exists"):
        seal_gold(
            _gold_items(300),
            gold_path,
            version="v1",
            quality_threshold_met=True,
        )


def test_reduced_gold_requires_exact_150_item_scope_floor(tmp_path: Path) -> None:
    """Catch a reduced benchmark being mislabeled full or falling below the approved floor."""
    metadata = seal_gold(
        _gold_items(150),
        tmp_path / "gold-reduced.jsonl",
        version="v-reduced",
        quality_threshold_met=False,
    )
    assert metadata.item_count == 150
    assert metadata.scope_status == "reduced"
    with pytest.raises(ValueError, match="exactly 150"):
        seal_gold(
            _gold_items(149),
            tmp_path / "too-small.jsonl",
            version="v-too-small",
            quality_threshold_met=False,
        )


def test_gold_loading_needs_environment_and_explicit_command_and_never_previews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch ALLOW_GOLD_ACCESS alone making sealed answers visible to normal code paths."""
    items = _gold_items(150)
    gold_path = tmp_path / "gold.jsonl"
    metadata = seal_gold(
        items,
        gold_path,
        version="v1",
        quality_threshold_met=False,
    )

    monkeypatch.setenv("ALLOW_GOLD_ACCESS", "1")
    with pytest.raises(GoldAccessError, match="explicit gold command"):
        authorize_gold_access(command="evaluate-gold", explicit=False)
    monkeypatch.delenv("ALLOW_GOLD_ACCESS")
    with pytest.raises(GoldAccessError, match="ALLOW_GOLD_ACCESS=1"):
        authorize_gold_access(command="evaluate-gold", explicit=True)
    monkeypatch.setenv("ALLOW_GOLD_ACCESS", "1")
    with pytest.raises(GoldAccessError, match="not permitted"):
        authorize_gold_access(command="preview-gold", explicit=True)

    authorization = authorize_gold_access(command="evaluate-gold", explicit=True)
    loaded = load_sealed_gold(gold_path, metadata=metadata, authorization=authorization)
    assert loaded == items


def test_gold_loader_rejects_symlinks_tampering_and_unsafe_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch no-follow, integrity, or confidentiality checks being weakened after sealing."""
    gold_path = tmp_path / "gold.jsonl"
    metadata = seal_gold(
        _gold_items(150),
        gold_path,
        version="v1",
        quality_threshold_met=False,
    )
    monkeypatch.setenv("ALLOW_GOLD_ACCESS", "1")
    authorization = authorize_gold_access(command="evaluate-gold", explicit=True)
    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()
    alias = alias_dir / "gold.jsonl"
    alias.symlink_to(gold_path)
    with pytest.raises(ImmutableSnapshotError, match="regular private file"):
        load_sealed_gold(alias, metadata=metadata, authorization=authorization)

    gold_path.chmod(0o644)
    with pytest.raises(ImmutableSnapshotError, match="mode 0600"):
        load_sealed_gold(gold_path, metadata=metadata, authorization=authorization)
    gold_path.chmod(0o600)
    raw = gold_path.read_text(encoding="utf-8")
    gold_path.write_text(raw.replace("합성 답변 0", "변조된 답변", 1), encoding="utf-8")
    gold_path.chmod(0o600)
    with pytest.raises(ImmutableSnapshotError, match="hash mismatch"):
        load_sealed_gold(gold_path, metadata=metadata, authorization=authorization)


def test_gold_loader_revalidates_metadata_identity_and_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch coordinated content-hash metadata edits bypassing the sealed snapshot identity."""
    gold_path = tmp_path / "gold.jsonl"
    metadata = seal_gold(_gold_items(150), gold_path, version="v1", quality_threshold_met=False)
    raw = gold_path.read_text(encoding="utf-8").replace("합성 답변 0", "변조된 답변", 1)
    gold_path.write_text(raw, encoding="utf-8")
    gold_path.chmod(0o600)
    forged = metadata.model_copy(
        update={"content_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    )
    monkeypatch.setenv("ALLOW_GOLD_ACCESS", "1")
    authorization = authorize_gold_access(command="evaluate-gold", explicit=True)

    with pytest.raises(ImmutableSnapshotError, match="metadata identity"):
        load_sealed_gold(gold_path, metadata=forged, authorization=authorization)

    invalid_scope = metadata.model_copy(update={"item_count": 151})
    with pytest.raises(ImmutableSnapshotError, match="scope metadata"):
        load_sealed_gold(gold_path, metadata=invalid_scope, authorization=authorization)


def test_invalid_gold_schema_does_not_chain_restricted_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch validation tracebacks retaining restricted question or item input."""
    gold_path = tmp_path / "gold.jsonl"
    metadata = seal_gold(_gold_items(150), gold_path, version="v1", quality_threshold_met=False)
    malformed = gold_path.read_text(encoding="utf-8").replace(
        '"item_id":"gold-000"', '"item_id":123', 1
    )
    gold_path.write_text(malformed, encoding="utf-8")
    gold_path.chmod(0o600)
    forged_hash = hashlib.sha256(malformed.encode()).hexdigest()
    forged = metadata.model_copy(
        update={
            "content_sha256": forged_hash,
            "snapshot_id": _metadata_snapshot_id(metadata, forged_hash),
        }
    )
    monkeypatch.setenv("ALLOW_GOLD_ACCESS", "1")
    authorization = authorize_gold_access(command="evaluate-gold", explicit=True)

    with pytest.raises(ImmutableSnapshotError) as raised:
        load_sealed_gold(gold_path, metadata=forged, authorization=authorization)

    assert raised.value.__cause__ is None
    assert "gold-000" not in str(raised.value)
    assert "봉인 합성 질문" not in str(raised.value)


def test_gold_seal_rejects_symlink_in_any_parent_component(tmp_path: Path) -> None:
    """Catch an intermediate symlink redirecting a supposedly no-follow restricted write."""
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(ImmutableSnapshotError, match="safe directory"):
        seal_gold(
            _gold_items(150),
            alias / "gold.jsonl",
            version="v1",
            quality_threshold_met=False,
        )


def test_gold_item_ids_are_unique_before_sealing(tmp_path: Path) -> None:
    """Catch an ambiguous gold snapshot whose duplicate identity corrupts paired evaluation."""
    duplicated = (*_gold_items(149), _gold_items(1)[0])
    assert Counter(item.item_id for item in duplicated)["gold-000"] == 2
    with pytest.raises(ValueError, match="unique"):
        seal_gold(
            duplicated,
            tmp_path / "duplicate.jsonl",
            version="v-duplicate",
            quality_threshold_met=False,
        )


def test_gold_cli_requires_execute_and_environment_without_reading_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch CLI path or inherited environment bypassing either explicit gold gate."""
    gold_path = tmp_path / "sealed.jsonl"
    metadata = seal_gold(_gold_items(150), gold_path, version="v1", quality_threshold_met=False)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(metadata.model_dump_json(), encoding="utf-8")
    monkeypatch.delenv("ALLOW_GOLD_ACCESS", raising=False)
    app = build_app()

    no_flag = RUNNER.invoke(app, ["gold", "verify", str(gold_path), str(metadata_path), "--json"])
    no_environment = RUNNER.invoke(
        app,
        ["gold", "verify", str(gold_path), str(metadata_path), "--execute", "--json"],
    )

    assert no_flag.exit_code == 1
    assert no_environment.exit_code == 1
    assert "gold-000" not in no_flag.output + no_environment.output
    assert "봉인 합성 질문" not in no_flag.output + no_environment.output


def test_gold_cli_success_emits_only_public_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch a successful integrity check previewing IDs, questions, answers, or evidence."""
    gold_path = tmp_path / "sealed.jsonl"
    metadata = seal_gold(_gold_items(150), gold_path, version="v1", quality_threshold_met=False)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(metadata.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("ALLOW_GOLD_ACCESS", "1")

    result = RUNNER.invoke(
        build_app(),
        ["gold", "verify", str(gold_path), str(metadata_path), "--execute", "--json"],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {"ok": True, **public_gold_metadata(metadata)}
    assert "gold-000" not in result.output
    assert "봉인 합성 질문" not in result.output
    assert "합성 답변" not in result.output
