"""Public-safe human review planning and explicitly gated gold sealing."""

from __future__ import annotations

import hashlib
import json
import os
import random
import secrets
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragbench.core.hashing import canonical_json_hash

REVIEW_COLUMNS = (
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

_GOLD_COMMANDS = frozenset({"evaluate-gold", "sealed-gold-test"})
_AUTHORIZATION_CAPABILITY = object()


class GoldAccessError(PermissionError):
    """Raised without benchmark content when either gold access gate is closed."""


class ImmutableSnapshotError(RuntimeError):
    """Raised when a sealed snapshot is unsafe, changed, or would be overwritten."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ReviewDecision(StrEnum):
    ACCEPT = "accept"
    CORRECT = "correct"
    REJECT = "reject"


class SnapshotName(StrEnum):
    DEV_AUTO = "dev_auto"
    TEST_GOLD = "test_gold"
    STRESS = "stress"


class ReviewCandidate(_FrozenModel):
    candidate_id: str = Field(min_length=1)
    natural_question: str = Field(min_length=1)
    question_type: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    document_ids: tuple[str, ...] = Field(min_length=1)
    parse_sensitive: bool
    answerable: bool
    generator_confidence: float | None = Field(default=None, ge=0, le=1)

    @field_validator("candidate_id", "natural_question", "question_type", "difficulty")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("review candidate fields cannot be blank")
        return value.strip()

    @field_validator("document_ids")
    @classmethod
    def _valid_documents(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value) or len(value) != len(set(value)):
            raise ValueError("document IDs must be nonblank and unique")
        return value


class ReviewQueueItem(_FrozenModel):
    """Candidate view exported to reviewers; generator confidence is intentionally absent."""

    candidate_id: str
    natural_question: str
    question_type: str
    difficulty: str
    document_ids: tuple[str, ...]
    parse_sensitive: bool
    answerable: bool


class BenchmarkItem(_FrozenModel):
    item_id: str = Field(min_length=1)
    natural_question: str = Field(min_length=1)
    document_ids: tuple[str, ...] = Field(min_length=1)
    question_family_id: str = Field(min_length=1)
    paraphrase_group_id: str = Field(min_length=1)

    @field_validator(
        "item_id", "natural_question", "question_family_id", "paraphrase_group_id"
    )
    @classmethod
    def _benchmark_field_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("benchmark identity fields cannot be blank")
        return value.strip()

    @field_validator("document_ids")
    @classmethod
    def _benchmark_documents_valid(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value) or len(value) != len(set(value)):
            raise ValueError("document IDs must be nonblank and unique")
        return value


class SplitSnapshot(_FrozenModel):
    name: SnapshotName
    version: str = Field(min_length=1)
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    item_ids: tuple[str, ...]


class ReviewRecord(_FrozenModel):
    natural_question: str = Field(min_length=1)
    answer_exists: bool
    evidence_correct: bool
    page_correct: bool
    answer_unambiguous: bool
    answerable_label_correct: bool
    type_difficulty_correct: bool
    reviewer_decision: ReviewDecision
    corrected_answer: str
    corrected_evidence: str
    notes: str
    reviewer_id: str = Field(min_length=1)
    timestamp: datetime

    @field_validator("natural_question", "reviewer_id")
    @classmethod
    def _review_identity_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("review identity fields cannot be blank")
        return value.strip()

    @field_validator("timestamp")
    @classmethod
    def _timestamp_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("review timestamp must include timezone")
        return value

    @model_validator(mode="after")
    def _correction_is_written(self) -> Self:
        if self.reviewer_decision is ReviewDecision.CORRECT and not (
            self.corrected_answer.strip() or self.corrected_evidence.strip()
        ):
            raise ValueError("correct decision requires a corrected answer or evidence")
        return self


class ReviewAgreement(_FrozenModel):
    item_count: int = Field(ge=50)
    raw_agreement: float = Field(ge=0, le=1)
    cohens_kappa: float = Field(ge=-1, le=1)


class GoldItem(_FrozenModel):
    item_id: str = Field(min_length=1)
    natural_question: str = Field(min_length=1)
    expected_answer: str | None
    evidence: tuple[str, ...]
    question_type: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)
    answerable: bool

    @model_validator(mode="after")
    def _answerability_is_consistent(self) -> Self:
        if self.answerable and (not self.expected_answer or not self.evidence):
            raise ValueError("answerable gold item requires answer and evidence")
        if not self.answerable and (self.expected_answer is not None or self.evidence):
            raise ValueError("unanswerable gold item cannot contain answer or evidence")
        return self


class GoldMetadata(_FrozenModel):
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    version: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_count: int = Field(gt=0)
    scope_status: str = Field(pattern=r"^(full|reduced)$")
    sealed_at: datetime

    @field_validator("sealed_at")
    @classmethod
    def _sealed_at_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("sealed timestamp must include timezone")
        return value


@dataclass(frozen=True, slots=True)
class GoldAuthorization:
    command: str
    _capability: object


def plan_review_sample(
    candidates: Sequence[ReviewCandidate], *, sample_size: int = 300, seed: int
) -> tuple[ReviewQueueItem, ...]:
    """Build a replayable queue across every declared review-risk stratum."""
    if sample_size < 300:
        raise ValueError("human validation requires at least 300 candidates")
    if len(candidates) < sample_size:
        raise ValueError("candidate pool is smaller than the requested review sample")
    identifiers = [item.candidate_id for item in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate IDs must be unique")

    rng = random.Random(seed)
    strata: dict[tuple[object, ...], list[ReviewCandidate]] = defaultdict(list)
    for item in sorted(candidates, key=lambda candidate: candidate.candidate_id):
        strata[
            (
                item.question_type,
                item.difficulty,
                item.document_ids,
                item.parse_sensitive,
                item.answerable,
            )
        ].append(item)
    for rows in strata.values():
        rng.shuffle(rows)
    keys = sorted(strata, key=lambda value: repr(value))
    rng.shuffle(keys)
    selected: list[ReviewCandidate] = []
    while len(selected) < sample_size:
        progressed = False
        for key in keys:
            if strata[key]:
                selected.append(strata[key].pop())
                progressed = True
                if len(selected) == sample_size:
                    break
        if not progressed:  # guarded by the size check; retained as a fail-closed invariant
            raise ValueError("stratified sample could not satisfy requested size")
    rng.shuffle(selected)
    return tuple(
        ReviewQueueItem(
            candidate_id=item.candidate_id,
            natural_question=item.natural_question,
            question_type=item.question_type,
            difficulty=item.difficulty,
            document_ids=item.document_ids,
            parse_sensitive=item.parse_sensitive,
            answerable=item.answerable,
        )
        for item in selected
    )


def build_split_snapshots(
    items: Sequence[BenchmarkItem], *, version: str, seed: int
) -> dict[SnapshotName, SplitSnapshot]:
    """Assign connected leakage components, never individual questions, to splits."""
    if not version.strip() or len(items) < len(SnapshotName):
        raise ValueError("version and at least three benchmark items are required")
    by_id = {item.item_id: item for item in items}
    if len(by_id) != len(items):
        raise ValueError("benchmark item IDs must be unique")

    parent = {item.item_id: item.item_id for item in items}

    def find(item_id: str) -> str:
        while parent[item_id] != item_id:
            parent[item_id] = parent[parent[item_id]]
            item_id = parent[item_id]
        return item_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    indexes: dict[tuple[str, str], str] = {}
    for item in sorted(items, key=lambda value: value.item_id):
        tokens = [
            *(("document", document_id) for document_id in item.document_ids),
            ("family", item.question_family_id),
            ("paraphrase", item.paraphrase_group_id),
        ]
        for token in tokens:
            previous = indexes.setdefault(token, item.item_id)
            union(previous, item.item_id)

    components: dict[str, list[str]] = defaultdict(list)
    for item_id in sorted(by_id):
        components[find(item_id)].append(item_id)
    groups = list(components.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    split_order = list(SnapshotName)
    allocations: dict[SnapshotName, list[str]] = {name: [] for name in split_order}
    for group in groups:
        minimum = min(len(allocations[name]) for name in split_order)
        eligible = [name for name in split_order if len(allocations[name]) == minimum]
        chosen = rng.choice(eligible)
        allocations[chosen].extend(group)

    snapshots: dict[SnapshotName, SplitSnapshot] = {}
    for name in split_order:
        item_ids = tuple(sorted(allocations[name]))
        if not item_ids:
            raise ValueError("leakage components cannot populate every requested split")
        snapshot_id = canonical_json_hash(
            {
                "name": name.value,
                "version": version,
                "seed": seed,
                "items": [by_id[item_id].model_dump(mode="json") for item_id in item_ids],
            }
        )
        snapshots[name] = SplitSnapshot(
            name=name,
            version=version,
            snapshot_id=snapshot_id,
            seed=seed,
            item_ids=item_ids,
        )
    return snapshots


def calculate_review_agreement(reviews: Sequence[ReviewRecord]) -> ReviewAgreement:
    """Calculate categorical raw agreement and Cohen's kappa for paired reviews."""
    grouped: dict[str, list[ReviewRecord]] = defaultdict(list)
    for review in reviews:
        grouped[review.natural_question].append(review)
    if len(grouped) < 50:
        raise ValueError("double review requires at least 50 items")
    pairs: list[tuple[ReviewDecision, ReviewDecision]] = []
    for question, rows in sorted(grouped.items()):
        if len(rows) != 2 or rows[0].reviewer_id == rows[1].reviewer_id:
            raise ValueError(f"each item requires exactly two distinct reviewers: {question}")
        ordered = sorted(rows, key=lambda row: row.reviewer_id)
        pairs.append((ordered[0].reviewer_decision, ordered[1].reviewer_decision))
    observed = sum(left is right for left, right in pairs) / len(pairs)
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(
        left_counts[decision] / len(pairs) * right_counts[decision] / len(pairs)
        for decision in ReviewDecision
    )
    kappa = 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)
    return ReviewAgreement(item_count=len(pairs), raw_agreement=observed, cohens_kappa=kappa)


def adjudicate_reviews(
    left: ReviewRecord,
    right: ReviewRecord,
    *,
    adjudicator_id: str,
    notes: str,
    decision: ReviewDecision | None = None,
    timestamp: datetime | None = None,
) -> ReviewRecord:
    """Resolve paired review decisions while preserving a written rule application."""
    if left.natural_question != right.natural_question:
        raise ValueError("adjudication reviews must identify the same natural question")
    if left.reviewer_id == right.reviewer_id:
        raise ValueError("adjudication requires two distinct initial reviewers")
    if not adjudicator_id.strip() or adjudicator_id in {left.reviewer_id, right.reviewer_id}:
        raise ValueError("adjudicator must be an identified independent reviewer")
    if left.reviewer_decision is not right.reviewer_decision and not notes.strip():
        raise ValueError("disagreement requires written adjudication notes")
    resolved = decision or left.reviewer_decision
    if left.reviewer_decision is not right.reviewer_decision and decision is None:
        raise ValueError("disagreement requires an explicit adjudication decision")
    source = left if resolved is left.reviewer_decision else right
    payload = source.model_dump()
    payload.update(
        reviewer_decision=resolved,
        reviewer_id=adjudicator_id.strip(),
        notes=notes.strip(),
        timestamp=timestamp or datetime.now(UTC),
    )
    return ReviewRecord.model_validate(payload)


def authorize_gold_access(
    *, command: str, explicit: bool, environment: Mapping[str, str] | None = None
) -> GoldAuthorization:
    """Issue an in-process capability only when both independent gold gates are open."""
    active_environment = os.environ if environment is None else environment
    if active_environment.get("ALLOW_GOLD_ACCESS") != "1":
        raise GoldAccessError("gold access requires ALLOW_GOLD_ACCESS=1")
    if not explicit:
        raise GoldAccessError("gold access requires an explicit gold command or flag")
    if command not in _GOLD_COMMANDS:
        raise GoldAccessError("requested gold command is not permitted")
    return GoldAuthorization(command=command, _capability=_AUTHORIZATION_CAPABILITY)


def seal_gold(
    items: Sequence[GoldItem],
    path: Path,
    *,
    version: str,
    quality_threshold_met: bool,
    sealed_at: datetime | None = None,
) -> GoldMetadata:
    """Seal synthetic-or-restricted items via a no-follow, atomic, no-replace publish."""
    if not version.strip():
        raise ValueError("gold version cannot be blank")
    required = 300 if quality_threshold_met else 150
    if len(items) != required:
        label = "300" if quality_threshold_met else "exactly 150"
        raise ValueError(f"gold sealing requires {label} reviewed items")
    identities = [item.item_id for item in items]
    if len(identities) != len(set(identities)):
        raise ValueError("gold item IDs must be unique")
    ordered = tuple(sorted(items, key=lambda item: item.item_id))
    payload = b"".join(
        (
            json.dumps(
                item.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        for item in ordered
    )
    content_hash = hashlib.sha256(payload).hexdigest()
    scope = "full" if quality_threshold_met else "reduced"
    timestamp = sealed_at or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("sealed timestamp must include timezone")
    snapshot_id = canonical_json_hash(
        {
            "version": version,
            "content_sha256": content_hash,
            "item_count": len(items),
            "scope_status": scope,
        }
    )
    _publish_private_immutable(path, payload)
    return GoldMetadata(
        snapshot_id=snapshot_id,
        version=version,
        file_name=path.name,
        content_sha256=content_hash,
        item_count=len(items),
        scope_status=scope,
        sealed_at=timestamp,
    )


def public_gold_metadata(metadata: GoldMetadata) -> dict[str, str | int]:
    """Return the complete set of gold fields permitted in ordinary logs."""
    return {
        "content_sha256": metadata.content_sha256,
        "file_name": metadata.file_name,
        "item_count": metadata.item_count,
        "scope_status": metadata.scope_status,
        "sealed_at": metadata.sealed_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "snapshot_id": metadata.snapshot_id,
        "version": metadata.version,
    }


def load_sealed_gold(
    path: Path, *, metadata: GoldMetadata, authorization: GoldAuthorization
) -> tuple[GoldItem, ...]:
    """Read and verify sealed content only through a valid explicit capability."""
    if authorization._capability is not _AUTHORIZATION_CAPABILITY:
        raise GoldAccessError("invalid gold authorization")
    if authorization.command not in _GOLD_COMMANDS:
        raise GoldAccessError("requested gold command is not permitted")
    if path.name != metadata.file_name:
        raise ImmutableSnapshotError("sealed gold file name does not match metadata")
    payload = _read_private_regular(path)
    if hashlib.sha256(payload).hexdigest() != metadata.content_sha256:
        raise ImmutableSnapshotError("sealed gold hash mismatch")
    try:
        parsed = tuple(
            GoldItem.model_validate_json(line)
            for line in payload.splitlines()
            if line.strip()
        )
    except (ValueError, UnicodeError) as error:
        raise ImmutableSnapshotError("sealed gold payload is invalid") from error
    if len(parsed) != metadata.item_count:
        raise ImmutableSnapshotError("sealed gold item count mismatch")
    return parsed


def _open_private_parent(path: Path) -> int:
    if not path.name or path.name in {".", ".."}:
        raise ImmutableSnapshotError("sealed gold path requires a direct file name")
    flags = os.O_RDONLY | os.O_DIRECTORY
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ImmutableSnapshotError("safe no-follow file operations are unavailable")
    try:
        descriptor = os.open(path.parent, flags | no_follow)
    except OSError as error:
        raise ImmutableSnapshotError("sealed gold parent is not a safe directory") from error
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode) or opened.st_uid != os.geteuid():
        os.close(descriptor)
        raise ImmutableSnapshotError("sealed gold parent must be an EUID-owned directory")
    return descriptor


def _publish_private_immutable(path: Path, payload: bytes) -> None:
    directory_fd = _open_private_parent(path)
    temporary = f".gold-seal-{secrets.token_hex(16)}.tmp"
    descriptor: int | None = None
    linked = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ImmutableSnapshotError("sealed gold write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(
                temporary,
                path.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise ImmutableSnapshotError("sealed gold path already exists") from error
        linked = True
        os.fsync(directory_fd)
    except ImmutableSnapshotError:
        raise
    except OSError as error:
        raise ImmutableSnapshotError("sealed gold could not be published safely") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        finally:
            if linked:
                os.fsync(directory_fd)
            os.close(directory_fd)


def _read_private_regular(path: Path) -> bytes:
    directory_fd = _open_private_parent(path)
    descriptor: int | None = None
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.geteuid():
            raise ImmutableSnapshotError("sealed gold must be an EUID-owned regular private file")
        if stat.S_IMODE(opened.st_mode) != 0o600:
            raise ImmutableSnapshotError("sealed gold must have mode 0600")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except ImmutableSnapshotError:
        raise
    except OSError as error:
        raise ImmutableSnapshotError(
            "sealed gold must be an EUID-owned regular private file"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)
