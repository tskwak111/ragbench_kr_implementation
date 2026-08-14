"""Deterministic, aggregate-only analysis exports from immutable experiment evidence."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal, Self

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragbench.core.hashing import canonical_json_hash
from ragbench.evaluation.taxonomy import FailureCategory, FailureLabels

INSPECTION_ORDER = (
    "original_pdf",
    "standard_parse",
    "enhanced_parse",
    "chunks",
    "rankings",
    "answer",
    "citations",
)
MONEY_QUANTUM = Decimal("0.000001")


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class AnalysisConfig(_FrozenModel):
    experiment_id: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parse_mode: Literal["standard", "enhanced"]
    chunk_strategy: str = Field(min_length=1)
    retriever: Literal["dense", "bm25", "hybrid"]
    top_k: int = Field(gt=0)
    prompt_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _immutable_id_is_bound_to_config(self) -> Self:
        if not self.experiment_id.startswith(f"{self.config_hash}-"):
            raise ValueError("experiment ID is not bound to its immutable config hash")
        return self


class AnalysisResult(_FrozenModel):
    experiment_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    question_type: str = Field(min_length=1)
    correctness: Decimal = Field(ge=0, le=1)
    faithfulness: Decimal = Field(ge=0, le=1)
    citation: Decimal = Field(ge=0, le=1)
    abstention: Decimal = Field(ge=0, le=1)
    hit: Decimal = Field(ge=0, le=1)
    recall: Decimal = Field(ge=0, le=1)
    mrr: Decimal = Field(ge=0, le=1)
    latency_ms: int = Field(ge=0)

    @field_validator(
        "correctness", "faithfulness", "citation", "abstention", "hit", "recall", "mrr",
        mode="before",
    )
    @classmethod
    def _decimal_from_json(cls, value: object) -> Decimal:
        if isinstance(value, (str, int, Decimal)) and not isinstance(value, bool):
            return Decimal(value)
        raise ValueError("metric must be an exact decimal")


class AnalysisUsage(_FrozenModel):
    experiment_id: str = Field(min_length=1)
    question_id: str | None = None
    question_type: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    estimated_cost_usd: Decimal = Field(ge=0)
    cached: bool

    @field_validator("estimated_cost_usd", mode="before")
    @classmethod
    def _decimal_from_json(cls, value: object) -> Decimal:
        if isinstance(value, (str, int, Decimal)) and not isinstance(value, bool):
            return Decimal(value)
        raise ValueError("estimated cost must be an exact decimal")


class AnalysisFailure(_FrozenModel):
    experiment_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    question_type: str = Field(min_length=1)
    primary: FailureCategory
    secondary: FailureCategory | None = None

    @field_validator("primary", "secondary", mode="before")
    @classmethod
    def _category_from_json(cls, value: object) -> FailureCategory | None:
        if value is None or isinstance(value, FailureCategory):
            return value
        if isinstance(value, str):
            return FailureCategory(value)
        raise ValueError("failure category must use the frozen taxonomy")

    @model_validator(mode="after")
    def _taxonomy_contract(self) -> Self:
        FailureLabels(primary=self.primary, secondary=self.secondary)
        return self


class AnalysisBundle(_FrozenModel):
    """Clean-snapshot adapter contract; DB implementations only need to emit this shape."""

    schema_version: Literal["analysis-input-v1"]
    cohort_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_version: str = Field(min_length=1)
    code_version: str = Field(pattern=r"^[0-9a-f]{7,40}$")
    configs: tuple[AnalysisConfig, ...]
    results: tuple[AnalysisResult, ...]
    usage: tuple[AnalysisUsage, ...]
    failures: tuple[AnalysisFailure, ...]

    @field_validator("configs", "results", "usage", "failures", mode="before")
    @classmethod
    def _sequence_from_json(cls, value: object) -> tuple[object, ...]:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        raise ValueError("analysis bundle collections must be sequences")

    @model_validator(mode="after")
    def _validate_relations(self) -> Self:
        experiment_ids = [row.experiment_id for row in self.configs]
        if not experiment_ids or len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("bundle requires unique immutable experiments")
        known = set(experiment_ids)
        referenced_experiments = (
            *(row.experiment_id for row in self.results),
            *(row.experiment_id for row in self.usage),
            *(row.experiment_id for row in self.failures),
        )
        for experiment_id in referenced_experiments:
            if experiment_id not in known:
                raise ValueError("row references an unknown immutable experiment")
        result_keys = {(row.experiment_id, row.question_id) for row in self.results}
        if len(result_keys) != len(self.results):
            raise ValueError("duplicate experiment-question results are not allowed")
        for failure in self.failures:
            if (failure.experiment_id, failure.question_id) not in result_keys:
                raise ValueError("failure is not bound to a result in the immutable cohort")
        return self


class ExportRequest(_FrozenModel):
    bundles: tuple[AnalysisBundle, ...] = Field(min_length=1)
    public_salt: str = Field(min_length=1)
    failure_sample_size: int = Field(default=50, ge=50, le=100)
    vat_multiplier: Decimal = Field(default=Decimal("1.10"), ge=1)
    console_gross_usd: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _same_lineage(self) -> Self:
        identities = {
            (bundle.cohort_hash, bundle.data_version, bundle.code_version)
            for bundle in self.bundles
        }
        if len(identities) != 1:
            raise ValueError("all exports must use the same cohort and lineage")
        experiment_ids = [
            config.experiment_id for bundle in self.bundles for config in bundle.configs
        ]
        if len(experiment_ids) != len(set(experiment_ids)):
            raise ValueError("immutable experiment IDs cannot be repeated across bundles")
        if not self.public_salt.strip():
            raise ValueError("public salt cannot be blank")
        return self


class FailureSampleItem(_FrozenModel):
    public_failure_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_config_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_type: str
    primary: FailureCategory
    secondary: FailureCategory | None
    inspection_order: tuple[str, ...]


class CostRow(_FrozenModel):
    public_config_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation: str
    model_id: str
    question_type: str
    cache_status: Literal["cached", "new"]
    call_count: int = Field(ge=0)
    net_cost_usd: Decimal = Field(ge=0)
    gross_cost_usd: Decimal = Field(ge=0)
    parse_amortized_cost_usd: Decimal = Field(ge=0)


class ReconciliationRow(_FrozenModel):
    status: Literal["not_reconciled", "reconciled"]
    local_gross_usd: Decimal
    console_gross_usd: Decimal | None
    delta_usd: Decimal | None


class ExportManifest(_FrozenModel):
    schema_version: Literal["analysis-export-manifest-v1"] = "analysis-export-manifest-v1"
    cohort_hash: str
    data_version: str
    code_version: str
    experiment_ids: tuple[str, ...]
    config_hashes: tuple[str, ...]
    tables: tuple[str, ...]
    figures: tuple[str, ...]
    files: dict[str, str]
    claim_status: Literal["PENDING_EVIDENCE"] = "PENDING_EVIDENCE"


def _public_config_id(salt: str, config_hash: str) -> str:
    return canonical_json_hash({"public_salt": salt, "config_hash": config_hash})


def plan_failure_sample(
    bundle: AnalysisBundle,
    *,
    sample_size: int,
    public_salt: str,
) -> tuple[FailureSampleItem, ...]:
    """Allocate a stable round-robin sample across system/type strata."""
    if not 50 <= sample_size <= 100:
        raise ValueError("failure sample size must be 50 through 100")
    if len(bundle.failures) < sample_size:
        raise ValueError("not enough failures for the requested stratified sample")
    configs = {config.experiment_id: config for config in bundle.configs}
    strata: dict[tuple[str, str], list[AnalysisFailure]] = defaultdict(list)
    for failure in bundle.failures:
        strata[(failure.experiment_id, failure.question_type)].append(failure)
    for rows in strata.values():
        rows.sort(
            key=lambda row: canonical_json_hash(
                {"salt": public_salt, "experiment": row.experiment_id, "question": row.question_id}
            )
        )
    selected: list[AnalysisFailure] = []
    ordered_strata = sorted(strata)
    while len(selected) < sample_size:
        progressed = False
        for stratum in ordered_strata:
            if strata[stratum] and len(selected) < sample_size:
                selected.append(strata[stratum].pop(0))
                progressed = True
        if not progressed:
            raise ValueError("not enough failures for the requested stratified sample")
    return tuple(
        FailureSampleItem(
            public_failure_id=canonical_json_hash(
                {
                    "salt": public_salt,
                    "experiment": row.experiment_id,
                    "question": row.question_id,
                }
            ),
            public_config_id=_public_config_id(
                public_salt, configs[row.experiment_id].config_hash
            ),
            question_type=row.question_type,
            primary=row.primary,
            secondary=row.secondary,
            inspection_order=INSPECTION_ORDER,
        )
        for row in selected
    )


def build_cost_rows(
    bundle: AnalysisBundle,
    *,
    vat_multiplier: Decimal,
    console_gross_usd: Decimal | None = None,
    public_salt: str = "analysis-cost-v1",
) -> tuple[tuple[CostRow, ...], ReconciliationRow]:
    """Aggregate provider estimates without treating them as reconciled billing truth."""
    if vat_multiplier < 1:
        raise ValueError("VAT multiplier must be at least one")
    configs = {config.experiment_id: config for config in bundle.configs}
    grouped: dict[tuple[str, str, str, str, bool], list[AnalysisUsage]] = defaultdict(list)
    parse_totals: dict[str, Decimal] = defaultdict(Decimal)
    question_counts = Counter(row.experiment_id for row in bundle.results)
    for usage in bundle.usage:
        grouped[
            (
                usage.experiment_id,
                usage.operation,
                usage.model_id,
                usage.question_type,
                usage.cached,
            )
        ].append(usage)
        if usage.operation == "parse" and not usage.cached:
            parse_totals[usage.experiment_id] += usage.estimated_cost_usd
    rows: list[CostRow] = []
    for key, usage_rows in sorted(grouped.items()):
        experiment_id, operation, model_id, question_type, cached = key
        net = sum((row.estimated_cost_usd for row in usage_rows), Decimal())
        net = net.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
        denominator = question_counts[experiment_id]
        amortized = (
            parse_totals[experiment_id] / denominator if denominator else Decimal()
        ).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)
        rows.append(
            CostRow(
                public_config_id=_public_config_id(
                    public_salt, configs[experiment_id].config_hash
                ),
                operation=operation,
                model_id=model_id,
                question_type=question_type,
                cache_status="cached" if cached else "new",
                call_count=len(usage_rows),
                net_cost_usd=net,
                gross_cost_usd=(net * vat_multiplier).quantize(
                    MONEY_QUANTUM, rounding=ROUND_HALF_UP
                ),
                parse_amortized_cost_usd=amortized,
            )
        )
    local_gross = sum((row.gross_cost_usd for row in rows), Decimal()).quantize(
        MONEY_QUANTUM, rounding=ROUND_HALF_UP
    )
    reconciliation = ReconciliationRow(
        status="not_reconciled" if console_gross_usd is None else "reconciled",
        local_gross_usd=local_gross,
        console_gross_usd=console_gross_usd,
        delta_usd=(
            None
            if console_gross_usd is None
            else (console_gross_usd - local_gross).quantize(
                MONEY_QUANTUM, rounding=ROUND_HALF_UP
            )
        ),
    )
    return tuple(rows), reconciliation


def export_analysis(request: ExportRequest, output: Path) -> ExportManifest:
    """Generate a deterministic aggregate export and publish its manifest last."""
    _validate_new_output(output)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=parent))
    try:
        tables_dir = staging / "tables"
        figures_dir = staging / "figures"
        tables_dir.mkdir()
        figures_dir.mkdir()
        bundle = _merge_bundles(request.bundles)
        tables = _build_tables(bundle, request)
        for name, rows in tables.items():
            _write_csv(tables_dir / f"{name}.csv", rows)
            _write_parquet(tables_dir / f"{name}.parquet", rows)
        figure_names = tuple(
            name
            for name in tables
            if name
            in {
                "leaderboard",
                "parse_paired_difference",
                "chunk_heatmap",
                "retriever_by_type",
                "top_k_tradeoff",
                "prompt_abstention",
                "pareto_frontier",
                "latency_distribution",
                "failure_taxonomy",
            }
        )
        for name in figure_names:
            _write_svg(figures_dir / f"{name}.svg", name, tables[name])
        file_hashes = _hash_files(staging)
        configs = sorted(bundle.configs, key=lambda row: row.experiment_id)
        manifest = ExportManifest(
            cohort_hash=bundle.cohort_hash,
            data_version=bundle.data_version,
            code_version=bundle.code_version,
            experiment_ids=tuple(row.experiment_id for row in configs),
            config_hashes=tuple(row.config_hash for row in configs),
            tables=tuple(tables),
            figures=figure_names,
            files=file_hashes,
        )
        _write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            for path in sorted(staging.rglob("*"), reverse=True):
                if path.is_symlink() or path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            staging.rmdir()


def _merge_bundles(bundles: Sequence[AnalysisBundle]) -> AnalysisBundle:
    first = bundles[0]
    return AnalysisBundle(
        schema_version="analysis-input-v1",
        cohort_hash=first.cohort_hash,
        data_version=first.data_version,
        code_version=first.code_version,
        configs=tuple(config for bundle in bundles for config in bundle.configs),
        results=tuple(result for bundle in bundles for result in bundle.results),
        usage=tuple(usage for bundle in bundles for usage in bundle.usage),
        failures=tuple(failure for bundle in bundles for failure in bundle.failures),
    )


def _build_tables(
    bundle: AnalysisBundle, request: ExportRequest
) -> dict[str, list[dict[str, Any]]]:
    result_groups: dict[str, list[AnalysisResult]] = defaultdict(list)
    for result in bundle.results:
        result_groups[result.experiment_id].append(result)
    leaderboard: list[dict[str, Any]] = []
    for config in sorted(bundle.configs, key=lambda row: row.config_hash):
        rows = result_groups[config.experiment_id]
        leaderboard.append(
            {
                "public_config_id": _public_config_id(request.public_salt, config.config_hash),
                "parse_mode": config.parse_mode,
                "chunk_strategy": config.chunk_strategy,
                "retriever": config.retriever,
                "top_k": config.top_k,
                "prompt_version": config.prompt_version,
                "model_id": config.model_id,
                "question_count": len(rows),
                **{
                    f"mean_{metric}": _mean(getattr(row, metric) for row in rows)
                    for metric in (
                        "correctness",
                        "faithfulness",
                        "citation",
                        "abstention",
                        "hit",
                        "recall",
                        "mrr",
                    )
                },
                "mean_latency_ms": _mean(Decimal(row.latency_ms) for row in rows),
            }
        )
    parse_pairs = _paired_parse_rows(bundle, request.public_salt)
    chunk_heatmap = _aggregate_axes(bundle, request.public_salt, ("chunk_strategy", "parse_mode"))
    retriever_by_type = _aggregate_axes(
        bundle, request.public_salt, ("retriever", "question_type")
    )
    top_k = _aggregate_axes(bundle, request.public_salt, ("top_k",))
    prompt = _aggregate_axes(bundle, request.public_salt, ("prompt_version",))
    latency = _latency_rows(bundle, request.public_salt)
    failure_taxonomy = _failure_taxonomy_rows(bundle, request.public_salt)
    sample = plan_failure_sample(
        bundle, sample_size=request.failure_sample_size, public_salt=request.public_salt
    )
    cost_rows, reconciliation = build_cost_rows(
        bundle,
        vat_multiplier=request.vat_multiplier,
        console_gross_usd=request.console_gross_usd,
        public_salt=request.public_salt,
    )
    cost_by_config: dict[str, Decimal] = defaultdict(Decimal)
    for cost_row in cost_rows:
        cost_by_config[cost_row.public_config_id] += cost_row.gross_cost_usd
    pareto = []
    marginal = []
    for leaderboard_row in leaderboard:
        public_id = str(leaderboard_row["public_config_id"])
        cost = cost_by_config[public_id]
        quality = Decimal(str(leaderboard_row["mean_correctness"]))
        pareto.append(
            {
                "public_config_id": public_id,
                "quality": quality,
                "gross_cost_usd": cost,
                "pareto": _is_pareto(public_id, quality, cost, leaderboard, cost_by_config),
            }
        )
        marginal.append(
            {
                "public_config_id": public_id,
                "gross_cost_usd": cost,
                "quality": quality,
                "cost_per_quality_point_usd": (
                    None
                    if quality == 0
                    else (cost / (quality * 100)).quantize(
                        MONEY_QUANTUM, rounding=ROUND_HALF_UP
                    )
                ),
            }
        )
    return {
        "leaderboard": leaderboard,
        "parse_paired_difference": parse_pairs,
        "chunk_heatmap": chunk_heatmap,
        "retriever_by_type": retriever_by_type,
        "top_k_tradeoff": top_k,
        "prompt_abstention": prompt,
        "pareto_frontier": pareto,
        "latency_distribution": latency,
        "failure_taxonomy": failure_taxonomy,
        "failure_sample_plan": [row.model_dump(mode="json") for row in sample],
        "cost_breakdown": [row.model_dump(mode="json") for row in cost_rows],
        "marginal_cost_quality": marginal,
        "reconciliation": [reconciliation.model_dump(mode="json")],
    }


def _paired_parse_rows(bundle: AnalysisBundle, salt: str) -> list[dict[str, Any]]:
    configs = {row.experiment_id: row for row in bundle.configs}
    grouped: dict[tuple[str, str, int, str, str], dict[str, str]] = defaultdict(dict)
    for config in bundle.configs:
        binding_key = (
            config.chunk_strategy,
            config.retriever,
            config.top_k,
            config.prompt_version,
            config.model_id,
        )
        grouped[binding_key][config.parse_mode] = config.experiment_id
    result_lookup = {
        (row.experiment_id, row.question_id): row for row in bundle.results
    }
    output: list[dict[str, Any]] = []
    for pair_key, pair in sorted(grouped.items()):
        if set(pair) != {"standard", "enhanced"}:
            continue
        standard_ids = {
            row.question_id for row in bundle.results if row.experiment_id == pair["standard"]
        }
        enhanced_ids = {
            row.question_id for row in bundle.results if row.experiment_id == pair["enhanced"]
        }
        if standard_ids != enhanced_ids:
            raise ValueError("Standard and Enhanced paired cohorts differ")
        differences = [
            result_lookup[(pair["enhanced"], question)].correctness
            - result_lookup[(pair["standard"], question)].correctness
            for question in sorted(standard_ids)
        ]
        mean = _mean(differences)
        low, high = _normal_interval(differences)
        output.append(
            {
                "pair_id": canonical_json_hash(
                    {
                        "salt": salt,
                        "standard": configs[pair["standard"]].config_hash,
                        "enhanced": configs[pair["enhanced"]].config_hash,
                    }
                ),
                "chunk_strategy": pair_key[0],
                "retriever": pair_key[1],
                "top_k": pair_key[2],
                "prompt_version": pair_key[3],
                "model_id": pair_key[4],
                "sample_count": len(differences),
                "correctness_effect": mean,
                "ci_low": low,
                "ci_high": high,
                "confidence": Decimal("0.95"),
            }
        )
    return output or [{"status": "PENDING_EVIDENCE", "reason": "no matched parse pair"}]


def _aggregate_axes(
    bundle: AnalysisBundle, salt: str, axes: tuple[str, ...]
) -> list[dict[str, Any]]:
    configs = {row.experiment_id: row for row in bundle.configs}
    grouped: dict[tuple[Any, ...], list[AnalysisResult]] = defaultdict(list)
    for row in bundle.results:
        config = configs[row.experiment_id]
        values = tuple(
            row.question_type if axis == "question_type" else getattr(config, axis)
            for axis in axes
        )
        grouped[values].append(row)
    return [
        {
            **dict(zip(axes, key, strict=True)),
            "question_count": len(rows),
            "mean_correctness": _mean(row.correctness for row in rows),
            "mean_recall": _mean(row.recall for row in rows),
            "mean_latency_ms": _mean(Decimal(row.latency_ms) for row in rows),
            "cohort_public_id": canonical_json_hash({"salt": salt, "cohort": bundle.cohort_hash}),
        }
        for key, rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0])))
    ]


def _latency_rows(bundle: AnalysisBundle, salt: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    configs = {row.experiment_id: row for row in bundle.configs}
    for row in bundle.results:
        grouped[row.experiment_id].append(row.latency_ms)
    return [
        {
            "public_config_id": _public_config_id(salt, configs[experiment].config_hash),
            "count": len(values),
            "p50_ms": _percentile(values, Decimal("0.50")),
            "p95_ms": _percentile(values, Decimal("0.95")),
            "p99_ms": _percentile(values, Decimal("0.99")),
        }
        for experiment, values in sorted(grouped.items())
    ]


def _failure_taxonomy_rows(bundle: AnalysisBundle, salt: str) -> list[dict[str, Any]]:
    configs = {row.experiment_id: row for row in bundle.configs}
    counts = Counter(
        (
            _public_config_id(salt, configs[row.experiment_id].config_hash),
            row.question_type,
            row.primary.value,
            row.secondary.value if row.secondary else "",
        )
        for row in bundle.failures
    )
    return [
        {
            "public_config_id": key[0],
            "question_type": key[1],
            "primary": key[2],
            "secondary": key[3],
            "count": count,
        }
        for key, count in sorted(counts.items())
    ]


def _is_pareto(
    public_id: str,
    quality: Decimal,
    cost: Decimal,
    leaderboard: Sequence[Mapping[str, Any]],
    costs: Mapping[str, Decimal],
) -> bool:
    for candidate in leaderboard:
        candidate_id = str(candidate["public_config_id"])
        if candidate_id == public_id:
            continue
        candidate_quality = Decimal(str(candidate["mean_correctness"]))
        candidate_cost = costs[candidate_id]
        if (
            candidate_quality >= quality
            and candidate_cost <= cost
            and (candidate_quality > quality or candidate_cost < cost)
        ):
            return False
    return True


def _mean(values: Iterable[Decimal]) -> Decimal:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot aggregate an empty cohort")
    return (sum(materialized, Decimal()) / len(materialized)).quantize(
        MONEY_QUANTUM, rounding=ROUND_HALF_UP
    )


def _normal_interval(values: Sequence[Decimal]) -> tuple[Decimal, Decimal]:
    if len(values) < 2:
        return values[0], values[0]
    floats = [float(value) for value in values]
    mean = sum(floats) / len(floats)
    variance = sum((value - mean) ** 2 for value in floats) / (len(floats) - 1)
    margin = Decimal(str(1.96 * math.sqrt(variance / len(floats))))
    center = Decimal(str(mean))
    return (
        (center - margin).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP),
        (center + margin).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP),
    )


def _percentile(values: Sequence[int], quantile: Decimal) -> Decimal:
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    value = ordered[lower] if lower == upper else ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * (position - lower)
    return Decimal(str(value)).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def _normalize_cell(value: Any) -> str | int | bool | None:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, FailureCategory):
        return value.value
    if isinstance(value, (tuple, list)):
        return json.dumps(list(value), ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"table {path.stem} cannot be empty")
    fields = tuple(rows[0])
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _normalize_cell(row.get(key)) for key in fields})


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    normalized = [
        {key: _normalize_cell(value) for key, value in row.items()} for row in rows
    ]
    table = pa.Table.from_pylist(normalized)
    pq.write_table(table, path, compression="zstd", write_statistics=True)


def _write_svg(path: Path, title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    labels = [str(next(iter(row.values())))[:24] for row in rows[:12]]
    values: list[float] = []
    for row in rows[:12]:
        numeric = next(
            (
                float(value)
                for value in row.values()
                if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)
            ),
            0.0,
        )
        values.append(max(0.0, numeric))
    maximum = max(values, default=1.0) or 1.0
    bars = []
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = 58 + index * 26
        width = 460 * value / maximum
        bars.append(
            f'<text x="10" y="{y + 14}" font-size="11">{html.escape(label)}</text>'
            f'<rect x="190" y="{y}" width="{width:.2f}" height="17" fill="#355c7d"/>'
            f'<text x="{195 + width:.2f}" y="{y + 13}" font-size="10">{value:.4g}</text>'
        )
    height = 90 + 26 * len(bars)
    payload = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height}" '
        'viewBox="0 0 760 '
        f'{height}"><rect width="100%" height="100%" fill="white"/>'
        f'<text x="10" y="28" font-size="20" font-family="sans-serif">{html.escape(title)}</text>'
        f'<text x="10" y="45" font-size="10" fill="#555">aggregate public-safe fixture</text>'
        + "".join(bars)
        + "</svg>\n"
    )
    path.write_text(payload, encoding="utf-8")


def _hash_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_new_output(output: Path) -> None:
    if output.is_symlink():
        raise ValueError("analysis output cannot be a symlink")
    if output.exists():
        raise FileExistsError(f"immutable analysis output already exists: {output}")
    current = output.parent
    while current != current.parent:
        if current.exists() and current.is_symlink():
            raise ValueError("analysis output parent cannot contain a symlink")
        current = current.parent
