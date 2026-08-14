"""Fail-closed preregistration and sealed-gold evaluation contracts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ragbench.benchmark.splits import (
    GoldItem,
    GoldMetadata,
    authorize_gold_access,
    load_sealed_gold,
)
from ragbench.core.hashing import canonical_json_hash
from ragbench.evaluation.bootstrap import PairedObservation, paired_bootstrap
from ragbench.experiments.runner import ExperimentConfig


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class GoldMetric(StrEnum):
    CORRECTNESS = "correctness"
    FAITHFULNESS = "faithfulness"
    CITATION = "citation"
    ABSTENTION = "abstention"
    LATENCY = "latency"
    COST = "cost"
    HIT = "hit"
    RECALL = "recall"
    MRR = "mrr"


class PrimaryComparison(_FrozenModel):
    left_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    right_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric: GoldMetric

    @model_validator(mode="after")
    def _different_configs(self) -> Self:
        if self.left_config_hash == self.right_config_hash:
            raise ValueError("primary comparison requires two distinct configs")
        return self


class BootstrapSpec(_FrozenModel):
    method: Literal["document-cluster-paired-bootstrap"]
    resamples: int = Field(ge=10_000)
    seed: int = Field(ge=0)
    confidence_level: float = Field(ge=0.95, le=0.95, allow_inf_nan=False)


class GoldCohort(_FrozenModel):
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_count: Literal[150, 300]
    ordered_membership_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExecutorSpec(_FrozenModel):
    entrypoint: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SolarExploratorySpec(_FrozenModel):
    """Optional comparison that cannot run until the core gold result is complete."""

    model_ids: tuple[str, str]
    subset_membership_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    subset_size: int = Field(gt=0)
    budget_cap_usd: Decimal = Field(gt=0)
    after_core_results: bool
    preregistered: bool

    @field_validator("model_ids")
    @classmethod
    def _exact_model_order(cls, value: tuple[str, str]) -> tuple[str, str]:
        if not value[0].startswith("solar-pro3-") or not value[1].startswith("solar-pro4-"):
            raise ValueError("comparison requires Solar Pro 3 then Solar Pro 4 model IDs")
        return value

    @field_validator("after_core_results")
    @classmethod
    def _after_core_only(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("Solar comparison must run after core results")
        return value

    @property
    def result_label(self) -> Literal["exploratory", "preregistered"]:
        return "preregistered" if self.preregistered else "exploratory"


class GoldPreregistration(_FrozenModel):
    schema_version: Literal["sealed-gold-preregistration-v1"]
    config_hashes: tuple[str, ...]
    metrics: tuple[GoldMetric, ...]
    primary_comparison: PrimaryComparison
    bootstrap: BootstrapSpec
    exclusions: tuple[str, ...]
    stopping_rule: str = Field(min_length=1)
    cohort: GoldCohort
    code_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    executor: ExecutorSpec
    exploratory_comparison: SolarExploratorySpec | None = None

    @field_validator("config_hashes")
    @classmethod
    def _exact_top_three(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != 3:
            raise ValueError("preregistration requires exactly three config hashes")
        if len(set(value)) != 3:
            raise ValueError("preregistered config hashes must be unique")
        if any(
            len(item) != 64 or any(char not in "0123456789abcdef" for char in item)
            for item in value
        ):
            raise ValueError("config hashes must be lowercase SHA-256 values")
        return value

    @field_validator("metrics")
    @classmethod
    def _all_frozen_metrics(cls, value: tuple[GoldMetric, ...]) -> tuple[GoldMetric, ...]:
        if value != tuple(GoldMetric):
            raise ValueError("metrics must be the complete frozen metric set in declared order")
        return value

    @field_validator("exclusions")
    @classmethod
    def _valid_exclusions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if (
            not normalized
            or any(not item for item in normalized)
            or len(set(normalized)) != len(normalized)
        ):
            raise ValueError("exclusions must be nonempty, nonblank, and unique")
        return normalized

    @field_validator("stopping_rule")
    @classmethod
    def _stopping_rule_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("stopping rule cannot be blank")
        return value.strip()

    @model_validator(mode="after")
    def _comparison_is_frozen(self) -> Self:
        selected = {
            self.primary_comparison.left_config_hash,
            self.primary_comparison.right_config_hash,
        }
        if not selected.issubset(set(self.config_hashes)):
            raise ValueError("primary comparison must reference preregistered configs")
        if self.primary_comparison.metric not in self.metrics:
            raise ValueError("primary comparison metric must be preregistered")
        return self


class PreregistrationEnvelope(_FrozenModel):
    preregistration: GoldPreregistration
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    signed_by: str = Field(min_length=1)
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    signed_at: datetime

    @field_validator("signed_by", "signature")
    @classmethod
    def _signature_fields_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("signature metadata cannot be blank")
        return value.strip()

    @field_validator("signed_at")
    @classmethod
    def _signed_at_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("signature timestamp must include timezone")
        return value

    @classmethod
    def sign(
        cls,
        preregistration: GoldPreregistration,
        *,
        signed_by: str,
        signing_key: bytes,
        signed_at: datetime,
    ) -> Self:
        if not signing_key:
            raise ValueError("preregistration signing key cannot be empty")
        artifact_hash = canonical_json_hash(preregistration.model_dump(mode="json"))
        signature = hmac.new(signing_key, artifact_hash.encode("ascii"), hashlib.sha256).hexdigest()
        return cls(
            preregistration=preregistration,
            artifact_sha256=artifact_hash,
            signed_by=signed_by,
            signature=signature,
            signed_at=signed_at,
        )


class VerifiedGoldInputs(_FrozenModel):
    preregistration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_hashes: tuple[str, str, str]
    cohort: GoldCohort


class GoldEvaluationResult(_FrozenModel):
    """One protected scored result; never serialize this model to public outputs."""

    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_id: str = Field(min_length=1)
    question_type: str = Field(min_length=1)
    document_cluster_id: str = Field(min_length=1)
    correctness: float = Field(ge=0, le=1, allow_inf_nan=False)
    faithfulness: float = Field(ge=0, le=1, allow_inf_nan=False)
    citation: float = Field(ge=0, le=1, allow_inf_nan=False)
    abstention: float = Field(ge=0, le=1, allow_inf_nan=False)
    latency: float = Field(ge=0, allow_inf_nan=False)
    cost: float = Field(ge=0, allow_inf_nan=False)
    hit: float = Field(ge=0, le=1, allow_inf_nan=False)
    recall: float = Field(ge=0, le=1, allow_inf_nan=False)
    mrr: float = Field(ge=0, le=1, allow_inf_nan=False)
    output_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("item_id", "question_type", "document_cluster_id")
    @classmethod
    def _identity_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("gold result identity fields cannot be blank")
        return value.strip()


class GoldDryRunPlan(_FrozenModel):
    mode: Literal["dry-run"] = "dry-run"
    executed: Literal[False] = False
    preregistration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_count: Literal[3]
    item_count: Literal[150, 300]
    metric_count: int = Field(gt=0)
    bootstrap_method: str
    bootstrap_resamples: int = Field(ge=10_000)


class GoldRunSummary(_FrozenModel):
    preregistration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed: int = Field(gt=0)
    expected: int = Field(gt=0)


class GoldAggregate(_FrozenModel):
    public_config_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric: GoldMetric
    mean: float
    sample_count: int = Field(gt=0)


class GoldTypeAggregate(GoldAggregate):
    question_type: str


class GoldPairedComparison(_FrozenModel):
    left_public_config_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    right_public_config_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric: GoldMetric
    question_type: str | None
    effect: float
    ci_low: float
    ci_high: float
    confidence: float = Field(ge=0.95, le=0.95, allow_inf_nan=False)
    resamples: int = Field(ge=10_000)
    method: Literal["document-cluster-paired-bootstrap"]
    sample_count: int = Field(gt=0)
    cluster_count: int = Field(gt=1)


class PublicGoldReport(_FrozenModel):
    schema_version: Literal["public-gold-report-v1"] = "public-gold-report-v1"
    preregistration_public_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    cohort_public_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    overall: tuple[GoldAggregate, ...]
    by_type: tuple[GoldTypeAggregate, ...]
    comparisons: tuple[GoldPairedComparison, ...]


class InvalidationRecord(_FrozenModel):
    sequence: int = Field(gt=0)
    preregistration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str = Field(min_length=1)
    replacement_code_commit: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    invalidated_at: datetime

    @field_validator("reason")
    @classmethod
    def _reason_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("invalidation reason cannot be blank")
        return value.strip()

    @field_validator("invalidated_at")
    @classmethod
    def _invalidation_time_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("invalidation timestamp must include timezone")
        return value


class _ProtectedManifest(_FrozenModel):
    schema_version: Literal["protected-gold-run-v1"]
    preregistration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_hashes: tuple[str, str, str]
    ordered_membership_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_count: Literal[150, 300]


def build_gold_dry_run(envelope: PreregistrationEnvelope) -> GoldDryRunPlan:
    """Describe only frozen public-safe metadata; no gold path or loader is accepted."""
    _verify_envelope_hash(envelope)
    prereg = envelope.preregistration
    return GoldDryRunPlan(
        preregistration_hash=envelope.artifact_sha256,
        config_count=3,
        item_count=prereg.cohort.item_count,
        metric_count=len(prereg.metrics),
        bootstrap_method=prereg.bootstrap.method,
        bootstrap_resamples=prereg.bootstrap.resamples,
    )


class GoldResultRepository:
    """Private, hash-bound, append-only checkpoint store for one preregistered run."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def initialize(self, envelope: PreregistrationEnvelope, items: Sequence[object]) -> None:
        _verify_envelope_hash(envelope)
        item_ids = _ordered_item_ids(items)
        prereg = envelope.preregistration
        if (
            len(item_ids) != prereg.cohort.item_count
            or canonical_json_hash(item_ids) != prereg.cohort.ordered_membership_hash
        ):
            raise ValueError("gold items differ from the preregistered ordered cohort")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        manifest = _ProtectedManifest(
            schema_version="protected-gold-run-v1",
            preregistration_hash=envelope.artifact_sha256,
            config_hashes=(
                prereg.config_hashes[0],
                prereg.config_hashes[1],
                prereg.config_hashes[2],
            ),
            ordered_membership_hash=prereg.cohort.ordered_membership_hash,
            item_count=prereg.cohort.item_count,
        )
        path = self.root / "manifest.json"
        if path.exists():
            try:
                existing = _ProtectedManifest.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValidationError) as error:
                raise ValueError("protected gold manifest is invalid") from error
            if existing != manifest:
                raise ValueError("protected gold manifest does not match preregistration")
            return
        _write_private_atomic_exclusive_json(path, manifest.model_dump(mode="json"))

    def save(self, result: GoldEvaluationResult) -> None:
        manifest = self._manifest()
        if result.config_hash not in manifest.config_hashes:
            raise ValueError("result belongs to an unregistered config")
        path = self._result_path(result.config_hash, result.item_id)
        _write_private_atomic_exclusive_json(path, self._encode_result(result, manifest))

    def load(self, config_hash: str, item_id: str) -> GoldEvaluationResult | None:
        path = self._result_path(config_hash, item_id)
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file() or stat_mode(path) != 0o600:
            raise ValueError("protected gold checkpoint is unsafe")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = self._decode_result(payload, self._manifest())
        except (OSError, json.JSONDecodeError, ValidationError, TypeError) as error:
            raise ValueError("protected gold checkpoint is invalid") from error
        if result.config_hash != config_hash or result.item_id != item_id:
            raise ValueError("protected gold checkpoint identity mismatch")
        return result

    def all_results(self) -> tuple[GoldEvaluationResult, ...]:
        manifest = self._manifest()
        results: list[GoldEvaluationResult] = []
        for config_hash in manifest.config_hashes:
            root = self.root / "results" / canonical_json_hash(config_hash)
            for path in sorted(root.glob("*.json")):
                if path.is_symlink() or not path.is_file() or stat_mode(path) != 0o600:
                    raise ValueError("protected gold checkpoint is unsafe")
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    result = self._decode_result(payload, manifest)
                except (OSError, json.JSONDecodeError, ValidationError, TypeError) as error:
                    raise ValueError("protected gold checkpoint is invalid") from error
                if result.config_hash != config_hash:
                    raise ValueError("protected gold checkpoint identity mismatch")
                if path.name != f"{canonical_json_hash(result.item_id)}.json":
                    raise ValueError("protected gold checkpoint filename identity mismatch")
                results.append(result)
        return tuple(results)

    def invalidations(self) -> tuple[InvalidationRecord, ...]:
        records: list[InvalidationRecord] = []
        for path in sorted((self.root / "invalidations").glob("*.json")):
            if path.is_symlink() or not path.is_file() or stat_mode(path) != 0o600:
                raise ValueError("protected gold invalidation is unsafe")
            try:
                record = InvalidationRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValidationError) as error:
                raise ValueError("protected gold invalidation is invalid") from error
            if path.name != f"{record.sequence:04d}.json":
                raise ValueError("protected gold invalidation sequence mismatch")
            records.append(record)
        if tuple(record.sequence for record in records) != tuple(range(1, len(records) + 1)):
            raise ValueError("protected gold invalidation lineage is not contiguous")
        return tuple(records)

    @property
    def preregistration_hash(self) -> str:
        return self._manifest().preregistration_hash

    def write_invalidation(self, record: InvalidationRecord) -> None:
        path = self.root / "invalidations" / f"{record.sequence:04d}.json"
        if path.exists():
            raise FileExistsError(f"immutable invalidation already exists: {record.sequence}")
        if record.sequence != len(self.invalidations()) + 1:
            raise ValueError("invalidation sequence must be contiguous and append-only")
        if record.preregistration_hash != self._manifest().preregistration_hash:
            raise ValueError("invalidation does not belong to this preregistration")
        _write_private_atomic_exclusive_json(
            path,
            record.model_dump(mode="json"),
        )

    def assert_bound(self, envelope: PreregistrationEnvelope) -> None:
        _verify_envelope_hash(envelope)
        manifest = self._manifest()
        prereg = envelope.preregistration
        if (
            manifest.preregistration_hash != envelope.artifact_sha256
            or manifest.config_hashes != prereg.config_hashes
            or manifest.ordered_membership_hash != prereg.cohort.ordered_membership_hash
            or manifest.item_count != prereg.cohort.item_count
        ):
            raise ValueError("protected gold repository is not bound to preregistration")

    def _manifest(self) -> _ProtectedManifest:
        try:
            return _ProtectedManifest.model_validate_json(
                (self.root / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise ValueError("protected gold manifest is missing or invalid") from error

    def _result_path(self, config_hash: str, item_id: str) -> Path:
        return (
            self.root
            / "results"
            / canonical_json_hash(config_hash)
            / f"{canonical_json_hash(item_id)}.json"
        )

    def _encode_result(
        self, result: GoldEvaluationResult, manifest: _ProtectedManifest
    ) -> dict[str, object]:
        binding = {
            "preregistration_hash": manifest.preregistration_hash,
            "result": result.model_dump(mode="json"),
        }
        return {**binding, "artifact_sha256": canonical_json_hash(binding)}

    def _decode_result(self, payload: object, manifest: _ProtectedManifest) -> GoldEvaluationResult:
        if not isinstance(payload, dict) or set(payload) != {
            "preregistration_hash",
            "result",
            "artifact_sha256",
        }:
            raise ValueError("protected gold checkpoint artifact is invalid")
        binding = {
            "preregistration_hash": payload["preregistration_hash"],
            "result": payload["result"],
        }
        if payload["preregistration_hash"] != manifest.preregistration_hash or payload[
            "artifact_sha256"
        ] != canonical_json_hash(binding):
            raise ValueError("protected gold checkpoint artifact hash mismatch")
        return GoldEvaluationResult.model_validate(payload["result"])


GoldExecutor = Callable[[ExperimentConfig, GoldItem], Awaitable[GoldEvaluationResult]]


class GoldRunner:
    """Deterministic three-config runner over an already authorized in-memory cohort."""

    def __init__(self, repository: GoldResultRepository, execute: GoldExecutor) -> None:
        self._repository = repository
        self._execute = execute

    async def run(
        self,
        envelope: PreregistrationEnvelope,
        configs: Sequence[ExperimentConfig],
        items: Sequence[GoldItem],
        *,
        resume: bool = False,
    ) -> GoldRunSummary:
        ordered = tuple(sorted(items, key=lambda item: _item_id(item)))
        config_by_hash = {config.semantic_hash: config for config in configs}
        if tuple(config_by_hash) != envelope.preregistration.config_hashes:
            raise ValueError("gold runner configs differ from preregistered order")
        if any(config.code_commit != envelope.preregistration.code_commit for config in configs):
            raise ValueError("gold runner code commit differs from preregistration")
        self._repository.initialize(envelope, ordered)
        completed = 0
        for config_hash in envelope.preregistration.config_hashes:
            for item in ordered:
                item_id = _item_id(item)
                cached = self._repository.load(config_hash, item_id)
                if cached is not None:
                    if not resume:
                        raise FileExistsError("gold checkpoint exists; explicit resume is required")
                    completed += 1
                    continue
                result = await self._execute(config_by_hash[config_hash], item)
                if result.config_hash != config_hash or result.item_id != item_id:
                    raise ValueError("gold executor returned the wrong identity")
                if (
                    result.question_type != item.question_type
                    or result.document_cluster_id != item.document_cluster_id
                ):
                    raise ValueError("gold executor returned a mismatched immutable stratum")
                self._repository.save(result)
                completed += 1
        expected = envelope.preregistration.cohort.item_count * 3
        return GoldRunSummary(
            preregistration_hash=envelope.artifact_sha256,
            completed=completed,
            expected=expected,
        )


def aggregate_gold_results(
    repository: GoldResultRepository,
    envelope: PreregistrationEnvelope,
    *,
    public_export_salt: str,
) -> PublicGoldReport:
    """Produce aggregate-only, salted public inputs with preregistered paired CIs."""
    if not public_export_salt.strip():
        raise ValueError("public export salt cannot be blank")
    repository.assert_bound(envelope)
    if repository.invalidations():
        raise ValueError("gold run was invalidated and cannot be aggregated")
    rows = repository.all_results()
    prereg = envelope.preregistration
    expected = prereg.cohort.item_count * 3
    if len(rows) != expected:
        raise ValueError("gold run is incomplete")
    by_config = {
        config: [row for row in rows if row.config_hash == config]
        for config in prereg.config_hashes
    }
    item_sets = [{row.item_id for row in config_rows} for config_rows in by_config.values()]
    if any(
        len(config_rows) != prereg.cohort.item_count for config_rows in by_config.values()
    ) or any(item_set != item_sets[0] for item_set in item_sets[1:]):
        raise ValueError("gold configurations do not share the exact cohort")

    public_ids = {
        config: canonical_json_hash({"salt": public_export_salt, "config": config})
        for config in prereg.config_hashes
    }
    overall: list[GoldAggregate] = []
    by_type: list[GoldTypeAggregate] = []
    types = sorted({row.question_type for row in rows})
    for config_hash in prereg.config_hashes:
        config_rows = by_config[config_hash]
        for metric in prereg.metrics:
            values = [_metric_value(row, metric) for row in config_rows]
            overall.append(
                GoldAggregate(
                    public_config_id=public_ids[config_hash],
                    metric=metric,
                    mean=sum(values) / len(values),
                    sample_count=len(values),
                )
            )
            for question_type in types:
                selected = [row for row in config_rows if row.question_type == question_type]
                values = [_metric_value(row, metric) for row in selected]
                by_type.append(
                    GoldTypeAggregate(
                        public_config_id=public_ids[config_hash],
                        metric=metric,
                        mean=sum(values) / len(values),
                        sample_count=len(values),
                        question_type=question_type,
                    )
                )

    comparison = prereg.primary_comparison
    comparisons = [
        _compare(
            by_config[comparison.left_config_hash],
            by_config[comparison.right_config_hash],
            metric=metric,
            question_type=question_type,
            prereg=prereg,
            left_public_id=public_ids[comparison.left_config_hash],
            right_public_id=public_ids[comparison.right_config_hash],
        )
        for metric in prereg.metrics
        for question_type in (None, *types)
    ]
    return PublicGoldReport(
        preregistration_public_id=canonical_json_hash(
            {"salt": public_export_salt, "preregistration": envelope.artifact_sha256}
        ),
        cohort_public_id=canonical_json_hash(
            {
                "salt": public_export_salt,
                "cohort": prereg.cohort.snapshot_id,
                "content": prereg.cohort.content_sha256,
            }
        ),
        overall=tuple(overall),
        by_type=tuple(by_type),
        comparisons=tuple(comparisons),
    )


def append_invalidation(
    repository: GoldResultRepository,
    *,
    reason: str,
    replacement_code_commit: str,
    invalidated_at: datetime,
) -> InvalidationRecord:
    record = InvalidationRecord(
        sequence=len(repository.invalidations()) + 1,
        preregistration_hash=repository.preregistration_hash,
        reason=reason,
        replacement_code_commit=replacement_code_commit,
        invalidated_at=invalidated_at,
    )
    repository.write_invalidation(record)
    return record


def write_public_gold_report(path: Path, report: PublicGoldReport) -> None:
    """Atomically publish aggregate-only output without overwrite semantics."""
    _write_private_atomic_exclusive_json(path, report.model_dump(mode="json"))


def _compare(
    left: Sequence[GoldEvaluationResult],
    right: Sequence[GoldEvaluationResult],
    *,
    metric: GoldMetric,
    question_type: str | None,
    prereg: GoldPreregistration,
    left_public_id: str,
    right_public_id: str,
) -> GoldPairedComparison:
    left_rows = {
        row.item_id: row
        for row in left
        if question_type is None or row.question_type == question_type
    }
    right_rows = {
        row.item_id: row
        for row in right
        if question_type is None or row.question_type == question_type
    }
    if left_rows.keys() != right_rows.keys():
        raise ValueError("paired gold comparison cohorts differ")
    observations = tuple(
        PairedObservation(
            observation_id=item_id,
            left=_metric_value(left_rows[item_id], metric),
            right=_metric_value(right_rows[item_id], metric),
            document_cluster_id=left_rows[item_id].document_cluster_id,
        )
        for item_id in sorted(left_rows)
    )
    if any(
        left_rows[item_id].document_cluster_id != right_rows[item_id].document_cluster_id
        for item_id in left_rows
    ):
        raise ValueError("paired gold document clusters differ")
    interval = paired_bootstrap(
        observations,
        seed=prereg.bootstrap.seed,
        resamples=prereg.bootstrap.resamples,
        final=True,
        confidence=prereg.bootstrap.confidence_level,
    )
    if interval.cluster_count is None:
        raise AssertionError("final paired bootstrap must report cluster count")
    return GoldPairedComparison(
        left_public_config_id=left_public_id,
        right_public_config_id=right_public_id,
        metric=metric,
        question_type=question_type,
        effect=interval.effect,
        ci_low=interval.ci_low,
        ci_high=interval.ci_high,
        confidence=0.95,
        resamples=interval.resamples,
        method="document-cluster-paired-bootstrap",
        sample_count=interval.sample_count,
        cluster_count=interval.cluster_count,
    )


def _metric_value(result: GoldEvaluationResult, metric: GoldMetric) -> float:
    return float(getattr(result, metric.value))


def _verify_envelope_hash(envelope: PreregistrationEnvelope) -> None:
    actual = canonical_json_hash(envelope.preregistration.model_dump(mode="json"))
    if actual != envelope.artifact_sha256:
        raise ValueError("preregistration artifact hash mismatch")


def _item_id(item: object) -> str:
    value = getattr(item, "item_id", None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("gold cohort item identity is invalid")
    return value


def _ordered_item_ids(items: Sequence[object]) -> tuple[str, ...]:
    values = tuple(_item_id(item) for item in items)
    if tuple(sorted(values)) != values or len(values) != len(set(values)):
        raise ValueError("gold items differ from the preregistered ordered cohort")
    return values


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def _write_private_atomic_exclusive_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"immutable gold checkpoint already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def write_preregistration(
    path: Path,
    preregistration: GoldPreregistration,
    *,
    signed_by: str,
    signing_key: bytes,
    signed_at: datetime,
) -> PreregistrationEnvelope:
    """Publish a signed preregistration once; mutation is detected on every load."""
    envelope = PreregistrationEnvelope.sign(
        preregistration,
        signed_by=signed_by,
        signing_key=signing_key,
        signed_at=signed_at,
    )
    _write_private_atomic_exclusive_json(path, envelope.model_dump(mode="json"))
    return envelope


def load_preregistration(path: Path, *, signing_key: bytes) -> PreregistrationEnvelope:
    try:
        envelope = PreregistrationEnvelope.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise ValueError("preregistration artifact is invalid") from error
    actual = canonical_json_hash(envelope.preregistration.model_dump(mode="json"))
    if actual != envelope.artifact_sha256:
        raise ValueError("preregistration artifact hash mismatch")
    if not signing_key:
        raise ValueError("preregistration signing key cannot be empty")
    expected_signature = hmac.new(
        signing_key, envelope.artifact_sha256.encode("ascii"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(envelope.signature, expected_signature):
        raise ValueError("preregistration signature verification failed")
    return envelope


def verify_frozen_inputs(
    envelope: PreregistrationEnvelope,
    *,
    configs: tuple[ExperimentConfig, ...],
    metadata: GoldMetadata,
) -> VerifiedGoldInputs:
    """Verify configs and public-safe cohort metadata without reading sealed content."""
    expected = envelope.preregistration.config_hashes
    if any(config.code_commit != envelope.preregistration.code_commit for config in configs):
        raise ValueError("config code commit differs from preregistration")
    actual = tuple(config.semantic_hash for config in configs)
    if actual != expected:
        if len(actual) == 3 and set(actual) == set(expected):
            raise ValueError("config order differs from preregistration")
        raise ValueError("configs do not match the exact frozen top three")
    cohort = envelope.preregistration.cohort
    if (
        metadata.snapshot_id != cohort.snapshot_id
        or metadata.content_sha256 != cohort.content_sha256
        or metadata.item_count != cohort.item_count
    ):
        raise ValueError("sealed gold cohort metadata differs from preregistration")
    return VerifiedGoldInputs(
        preregistration_hash=envelope.artifact_sha256,
        config_hashes=(expected[0], expected[1], expected[2]),
        cohort=cohort,
    )


def verify_runtime_code_commit(project_root: Path, preregistration: GoldPreregistration) -> None:
    """Bind a live unseal to the exact clean Git commit frozen before gold access."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot verify preregistered runtime code commit") from error
    if completed.stdout:
        raise RuntimeError("gold execution requires a clean tracked worktree")
    if not commit.startswith(preregistration.code_commit):
        raise RuntimeError("runtime code commit differs from preregistration")


def load_authorized_gold_cohort(
    path: Path,
    *,
    metadata: GoldMetadata,
    envelope: PreregistrationEnvelope,
    configs: tuple[ExperimentConfig, ...],
    explicit: bool,
) -> tuple[GoldItem, ...]:
    """Open sealed content only after preregistration, config, metadata, env, and command gates."""
    verify_frozen_inputs(envelope, configs=configs, metadata=metadata)
    authorization = authorize_gold_access(command="sealed-gold-test", explicit=explicit)
    loaded = load_sealed_gold(path, metadata=metadata, authorization=authorization)
    ordered = tuple(sorted(loaded, key=lambda item: item.item_id))
    if (
        canonical_json_hash(tuple(item.item_id for item in ordered))
        != envelope.preregistration.cohort.ordered_membership_hash
    ):
        raise ValueError("loaded gold items differ from the preregistered ordered cohort")
    return ordered
