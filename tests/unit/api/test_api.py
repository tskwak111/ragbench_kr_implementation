from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from ragbench.api.app import AppServices, create_app
from ragbench.api.dependencies import migration_state
from ragbench.api.schemas import (
    CitationResponse,
    DocumentResponse,
    ExperimentMetricsResponse,
    ExperimentResponse,
    QueryResponse,
    SearchHitResponse,
)


@dataclass
class FakeDocuments:
    rows: dict[str, DocumentResponse] = field(default_factory=dict)

    async def create(
        self, *, title: str, source_uri: str, metadata: dict[str, Any]
    ) -> DocumentResponse:
        row = DocumentResponse(
            id="doc-1", title=title, source_uri=source_uri, metadata=metadata, status="registered"
        )
        self.rows[row.id] = row
        return row

    async def get(self, document_id: str) -> DocumentResponse | None:
        return self.rows.get(document_id)


@dataclass
class FakeSearch:
    last_filter: dict[str, Any] | None = None

    async def search(
        self, *, query: str, top_k: int, search_filter: dict[str, Any]
    ) -> list[SearchHitResponse]:
        self.last_filter = search_filter
        return [SearchHitResponse(chunk_id="chunk-1", score=0.75, rank=1, retriever="hybrid-rrf")][
            :top_k
        ]


class FakeQueries:
    async def answer(self, *, question: str, config_id: str) -> QueryResponse:
        return QueryResponse(
            question=question,
            answer="근거가 확인됩니다.",
            abstained=False,
            citations=(
                CitationResponse(
                    citation_id="C1",
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    document_title="보고서",
                    page_start=2,
                    page_end=2,
                    section_path=("요약",),
                ),
            ),
            experiment_id="fixture-experiment",
            config_id=config_id,
            cached=True,
        )


@dataclass
class FakeExperiments:
    rows: dict[str, ExperimentResponse] = field(default_factory=dict)

    async def queue(
        self, *, config: dict[str, Any], question_ids: tuple[str, ...]
    ) -> ExperimentResponse:
        row = ExperimentResponse(
            id="experiment-1",
            config_hash="a" * 64,
            status="queued",
            question_count=len(question_ids),
        )
        self.rows[row.id] = row
        return row

    async def list(self) -> tuple[ExperimentResponse, ...]:
        return tuple(self.rows.values())

    async def get(self, experiment_id: str) -> ExperimentResponse | None:
        return self.rows.get(experiment_id)

    async def metrics(self, experiment_id: str) -> ExperimentMetricsResponse | None:
        if experiment_id not in self.rows:
            return None
        return ExperimentMetricsResponse(experiment_id=experiment_id, metrics={"hit_at_1": 1.0})


class ApiClient:
    def __init__(self, app: FastAPI, *, raise_app_exceptions: bool = True) -> None:
        self.app = app
        self.raise_app_exceptions = raise_app_exceptions

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(
                app=self.app, raise_app_exceptions=self.raise_app_exceptions
            )
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, **kwargs)


def _client() -> tuple[ApiClient, FakeSearch]:
    search = FakeSearch()
    app = create_app(
        AppServices(
            documents=FakeDocuments(),
            search=search,
            queries=FakeQueries(),
            experiments=FakeExperiments(),
            database_check=lambda: (True, "reachable"),
            migration_check=lambda: (True, "head"),
            domain_check=lambda: (True, "ready"),
        )
    )
    return ApiClient(app), search


def test_health_reports_component_readiness_and_version() -> None:
    client, _ = _client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "ready": True,
        "components": {
            "application": {"ready": True, "detail": "ready"},
            "database": {"ready": True, "detail": "reachable"},
            "migration": {"ready": True, "detail": "head"},
            "domain_services": {"ready": True, "detail": "ready"},
        },
        "version": "0.1.0",
    }
    assert response.headers["x-correlation-id"]


def test_health_sanitizes_probe_details() -> None:
    services = AppServices(
        documents=FakeDocuments(),
        search=FakeSearch(),
        queries=FakeQueries(),
        experiments=FakeExperiments(),
        database_check=lambda: (False, "postgresql://user:private-secret@db/internal"),
        migration_check=lambda: (False, "provider raw-response private-secret"),
        domain_check=lambda: (True, "ready"),
    )

    response = ApiClient(create_app(services)).get("/health")

    assert response.status_code == 200
    assert "private-secret" not in response.text
    assert "raw-response" not in response.text
    assert response.json()["components"]["database"]["detail"] == "unavailable"


def test_migration_readiness_requires_the_exact_repository_head() -> None:
    assert migration_state("revision-a", "revision-a") == (True, "revision:revision-a")
    assert migration_state("revision-a", "revision-b") == (
        False,
        "revision:revision-a; expected:revision-b",
    )


def test_default_app_imports_without_key_or_database_and_reports_degraded() -> None:
    from ragbench.api.app import app

    response = ApiClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["ready"] is False
    assert response.json()["components"]["database"]["ready"] is False


def test_document_registration_accepts_metadata_not_local_paths() -> None:
    client, _ = _client()

    response = client.post(
        "/documents",
        json={
            "title": "공개 보고서",
            "source_uri": "https://example.invalid/report.pdf",
            "metadata": {"license": "public"},
        },
    )

    assert response.status_code == 201
    assert response.json()["id"] == "doc-1"
    rejected = client.post(
        "/documents",
        json={
            "title": "private",
            "source_uri": "https://example.invalid/report.pdf",
            "metadata": {},
            "local_path": "/etc/passwd",
        },
    )
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "/etc/passwd" not in rejected.text
    metadata_path = client.post(
        "/documents",
        json={
            "title": "private",
            "source_uri": "https://example.invalid/report.pdf",
            "metadata": {"local_path": "/etc/passwd"},
        },
    )
    assert metadata_path.status_code == 422
    assert "/etc/passwd" not in metadata_path.text
    disguised_path = client.post(
        "/documents",
        json={
            "title": "private",
            "source_uri": "https://example.invalid/report.pdf",
            "metadata": {"note": "/etc/passwd"},
        },
    )
    assert disguised_path.status_code == 422
    assert "/etc/passwd" not in disguised_path.text
    credentialed = client.post(
        "/documents",
        json={
            "title": "unsafe",
            "source_uri": "https://user:private-secret@example.invalid/report.pdf",
            "metadata": {},
        },
    )
    assert credentialed.status_code == 422
    assert "private-secret" not in credentialed.text


@pytest.mark.parametrize(
    ("source_uri", "metadata"),
    [
        ("/etc/passwd", {}),
        ("https://user:secret@example.invalid/report.pdf", {}),
        ("https://example.invalid/report.pdf", {"api_key": "secret"}),
        ("https://example.invalid/report.pdf", {"local_path": "/etc/passwd"}),
        ("https://example.invalid/report.pdf", {"source_path": "relative.pdf"}),
        ("https://example.invalid/report.pdf", {"note": "/etc/passwd"}),
        ("https://example.invalid/report.pdf", {"gold_question": "restricted"}),
    ],
)
def test_document_responses_reject_nonpublic_sources_and_metadata(
    source_uri: str, metadata: dict[str, str]
) -> None:
    with pytest.raises(ValidationError):
        DocumentResponse(
            id="doc-unsafe",
            title="unsafe",
            source_uri=source_uri,
            metadata=metadata,
            status="registered",
        )


def test_unknown_document_uses_sanitized_error_envelope() -> None:
    client, _ = _client()

    response = client.get("/documents/missing", headers={"X-Correlation-ID": "request-42"})

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "NOT_FOUND",
            "message": "resource not found",
            "correlation_id": "request-42",
        }
    }


def test_search_passes_exact_snapshot_and_document_filters() -> None:
    client, search = _client()
    payload = {
        "query": "2025년 매출",
        "top_k": 3,
        "filter": {
            "corpus_snapshot_id": "corpus-a",
            "parse_snapshot_id": "parse-a",
            "chunk_strategy": "fixed-600-100",
            "embedding_snapshot_id": "embedding-a",
            "document_ids": ["doc-b", "doc-a", "doc-a"],
        },
    }

    response = client.post("/search", json=payload)

    assert response.status_code == 200
    assert response.json()["hits"][0]["chunk_id"] == "chunk-1"
    assert search.last_filter == {
        "corpus_snapshot_id": "corpus-a",
        "parse_snapshot_id": "parse-a",
        "chunk_strategy": "fixed-600-100",
        "embedding_snapshot_id": "embedding-a",
        "document_ids": ("doc-a", "doc-b"),
    }


def test_query_returns_only_server_resolved_citations() -> None:
    client, _ = _client()

    response = client.post(
        "/query", json={"question": "무엇인가요?", "config_id": "public-fixture-v1"}
    )

    assert response.status_code == 200
    assert response.json()["citations"] == [
        {
            "citation_id": "C1",
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "document_title": "보고서",
            "page_start": 2,
            "page_end": 2,
            "section_path": ["요약"],
        }
    ]


def test_large_experiment_request_is_queued_and_never_runs_inline() -> None:
    client, _ = _client()

    response = client.post(
        "/experiments",
        json={
            "config": {"schema_version": "generation-experiment-v1"},
            "question_ids": [f"q-{i}" for i in range(1000)],
        },
    )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["question_count"] == 1000
    assert response.headers["location"] == "/experiments/experiment-1"
    assert client.get("/experiments").json()["experiments"][0]["id"] == "experiment-1"
    assert client.get("/experiments/experiment-1").status_code == 200
    assert client.get("/experiments/experiment-1/metrics").json()["metrics"] == {"hit_at_1": 1.0}


def test_raw_service_error_and_secret_are_never_exposed() -> None:
    class ExplodingDocuments(FakeDocuments):
        async def get(self, document_id: str) -> DocumentResponse | None:
            raise RuntimeError("provider 401 UPSTAGE_API_KEY=private-secret raw-response")

    services = AppServices(
        documents=ExplodingDocuments(),
        search=FakeSearch(),
        queries=FakeQueries(),
        experiments=FakeExperiments(),
        database_check=lambda: (True, "reachable"),
        migration_check=lambda: (True, "head"),
        domain_check=lambda: (True, "ready"),
    )

    response = ApiClient(create_app(services), raise_app_exceptions=False).get("/documents/doc-1")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "private-secret" not in response.text
    assert "raw-response" not in response.text


def test_gold_and_restricted_routes_do_not_exist() -> None:
    client, _ = _client()

    assert client.get("/gold").status_code == 404
    assert client.get("/questions").status_code == 404
    restricted = client.post(
        "/experiments",
        json={
            "config": {"questions_snapshot": "sealed-gold-v1"},
            "question_ids": ["sealed-gold-q1"],
        },
    )
    assert restricted.status_code == 422
