"""Tiny deterministic benchmark fixture requiring neither network, database, nor provider key."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ragbench.core.hashing import canonical_json_hash
from ragbench.evaluation.generation import (
    GenerationCase,
    MaterialClaim,
    aggregate_generation,
)
from ragbench.evaluation.retrieval import RetrievalCase, aggregate_retrieval
from ragbench.retrieval.base import SearchFilter
from ragbench.retrieval.bm25 import BM25Document, BM25IndexSnapshot, BM25Retriever

_RESTRICTED = re.compile(r"(?:gold|sealed|restricted)", re.I)


class _FixtureModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class OfflineSnapshots(_FixtureModel):
    corpus: str = Field(min_length=1)
    parse: str = Field(min_length=1)
    chunks: str = Field(min_length=1)
    embeddings: Literal["none"]
    questions: str = Field(min_length=1)

    @model_validator(mode="after")
    def _public_only(self) -> Self:
        if any(_RESTRICTED.search(value) for value in self.model_dump().values()):
            raise ValueError("gold or restricted snapshots are forbidden in offline fixtures")
        return self


class OfflineDocument(_FixtureModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class OfflineQuestion(_FixtureModel):
    question_id: str = Field(min_length=1)
    question_type: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    expected_answer: str | None
    answerable: bool
    evidence_chunk_ids: tuple[str, ...]

    @field_validator("evidence_chunk_ids", mode="before")
    @classmethod
    def _evidence_sequence(cls, value: object) -> tuple[object, ...]:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        raise ValueError("evidence IDs must be a sequence")

    @model_validator(mode="after")
    def _answer_contract(self) -> Self:
        if self.answerable and (not self.expected_answer or not self.evidence_chunk_ids):
            raise ValueError("answerable fixtures require an answer and evidence")
        if not self.answerable and (self.expected_answer is not None or self.evidence_chunk_ids):
            raise ValueError("unanswerable fixtures cannot contain an answer or evidence")
        return self


class OfflineFixture(_FixtureModel):
    schema_version: Literal["offline-fixture-v1"]
    name: str = Field(min_length=1)
    snapshots: OfflineSnapshots
    retrieval: dict[Literal["top_k"], int]
    documents: tuple[OfflineDocument, ...] = Field(min_length=1)
    questions: tuple[OfflineQuestion, ...] = Field(min_length=1)

    @field_validator("documents", "questions", mode="before")
    @classmethod
    def _sequence(cls, value: object) -> tuple[object, ...]:
        if isinstance(value, (list, tuple)):
            return tuple(value)
        raise ValueError("fixture collections must be sequences")

    @model_validator(mode="after")
    def _relations(self) -> Self:
        top_k = self.retrieval.get("top_k")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("offline top_k must be a positive integer")
        chunk_ids = [row.chunk_id for row in self.documents]
        question_ids = [row.question_id for row in self.questions]
        if len(chunk_ids) != len(set(chunk_ids)) or len(question_ids) != len(set(question_ids)):
            raise ValueError("offline fixture identities must be unique")
        known = set(chunk_ids)
        if any(not set(question.evidence_chunk_ids) <= known for question in self.questions):
            raise ValueError("question evidence is outside the fixture corpus")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        if not path.is_file() or path.is_symlink():
            raise ValueError("offline fixture must be a regular non-symlink file")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("offline fixture root must be a mapping")
        return cls.model_validate(raw)

    @property
    def fixture_hash(self) -> str:
        return canonical_json_hash(self.model_dump(mode="json"))


async def _retrieve(
    fixture: OfflineFixture,
) -> tuple[tuple[OfflineQuestion, tuple[str, ...]], ...]:
    search_filter = SearchFilter(
        fixture.snapshots.corpus,
        fixture.snapshots.parse,
        "offline-bm25",
        "none",
    )
    retriever = BM25Retriever(
        BM25IndexSnapshot(
            search_filter,
            tuple(
                BM25Document(row.chunk_id, row.document_id, row.content)
                for row in fixture.documents
            ),
        )
    )
    output: list[tuple[OfflineQuestion, tuple[str, ...]]] = []
    for question in fixture.questions:
        hits = await retriever.search(
            question.prompt,
            top_k=fixture.retrieval["top_k"],
            filter=search_filter,
        )
        output.append((question, tuple(hit.chunk_id for hit in hits)))
    return tuple(output)


def _retrieval_payload(fixture: OfflineFixture) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rankings = asyncio.run(_retrieve(fixture))
    cases = tuple(
        RetrievalCase(
            question.question_id,
            question.question_type,
            ranked,
            question.evidence_chunk_ids,
            0.0,
        )
        for question, ranked in rankings
    )
    evaluated = aggregate_retrieval(cases, k=fixture.retrieval["top_k"])
    rows = [
        {
            "question_id": question.question_id,
            "ranked_chunk_ids": list(ranked),
            "evidence_chunk_ids": list(question.evidence_chunk_ids),
        }
        for question, ranked in rankings
    ]
    metrics = {
        "hit_at_k": float(evaluated.overall.macro_hit_at_k or 0.0),
        "evidence_recall_at_k": float(evaluated.overall.macro_evidence_recall_at_k or 0.0),
        "mrr": float(evaluated.overall.macro_mrr or 0.0),
    }
    return rows, metrics


def run_offline_screen(config: Path, *, output_root: Path | None = None) -> dict[str, Any]:
    fixture = OfflineFixture.from_yaml(config)
    rows, metrics = _retrieval_payload(fixture)
    run_id = canonical_json_hash({"kind": "offline-screen-v1", "fixture": fixture.fixture_hash})
    result: dict[str, Any] = {
        "schema_version": "offline-screen-result-v1",
        "run_id": run_id,
        "fixture_hash": fixture.fixture_hash,
        "question_count": len(fixture.questions),
        "provider_calls": 0,
        "retriever": "bm25",
        "metrics": metrics,
        "rankings": rows,
    }
    _write_idempotent(output_root, run_id, result)
    return result


def run_offline_experiment(config: Path, *, output_root: Path | None = None) -> dict[str, Any]:
    fixture = OfflineFixture.from_yaml(config)
    rows, retrieval_metrics = _retrieval_payload(fixture)
    ranking_by_id = {row["question_id"]: row["ranked_chunk_ids"] for row in rows}
    cases: list[GenerationCase] = []
    responses: list[dict[str, Any]] = []
    for question in fixture.questions:
        ranking = ranking_by_id[question.question_id]
        selected = [item for item in ranking if item in question.evidence_chunk_ids]
        abstained = not question.answerable
        answer = question.expected_answer if question.answerable else "근거가 부족합니다."
        assert answer is not None
        claims = (
            (MaterialClaim("claim-1", True, frozenset(selected)),)
            if question.answerable and selected
            else ()
        )
        citations = {"claim-1": tuple(selected)} if claims else {}
        cases.append(
            GenerationCase(
                question.question_id,
                question.question_type,
                question.expected_answer,
                (),
                question.answerable,
                answer,
                abstained,
                claims,
                citations,
            )
        )
        responses.append(
            {
                "question_id": question.question_id,
                "answer": answer,
                "abstained": abstained,
                "citations": selected,
                "cached": True,
            }
        )
    generation = aggregate_generation(cases)
    metrics = {
        "deterministic_correctness": float(generation.deterministic_correctness or 0.0),
        "faithfulness": float(generation.faithfulness or 0.0),
        "citation_precision": float(generation.citation_precision or 0.0),
        "citation_recall": float(generation.citation_recall or 0.0),
        "abstention_accuracy": generation.abstention_accuracy,
    }
    run_id = canonical_json_hash({"kind": "offline-experiment-v1", "fixture": fixture.fixture_hash})
    result: dict[str, Any] = {
        "schema_version": "offline-experiment-result-v1",
        "run_id": run_id,
        "fixture_hash": fixture.fixture_hash,
        "snapshot_versions": fixture.snapshots.model_dump(mode="json"),
        "metric_versions": {"retrieval": "retrieval-v1", "generation": "deterministic-v1"},
        "question_count": len(fixture.questions),
        "response_count": len(responses),
        "new_responses": len(responses),
        "provider_calls": 0,
        "retrieval_metrics": retrieval_metrics,
        "metrics": metrics,
        "responses": responses,
    }
    created = _write_idempotent(output_root, run_id, result)
    execution_result = dict(result)
    execution_result["cache_reused"] = not created
    if not created:
        execution_result["new_responses"] = 0
    return execution_result


def _write_idempotent(output_root: Path | None, run_id: str, result: dict[str, Any]) -> bool:
    if output_root is None:
        return True
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{run_id}.json"
    serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(serialized)
        created = True
    except FileExistsError:
        created = False
    # An existing content-addressed artifact must be byte-equivalent in meaning.
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if loaded != result:
        raise ValueError("offline fixture artifact conflicts with its content identity")
    return created
