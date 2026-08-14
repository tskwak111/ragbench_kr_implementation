"""Thin HTTP routes delegating all domain work to injected async services."""

from __future__ import annotations

import inspect
import re

from fastapi import APIRouter, Request, Response, status

from ragbench.api.dependencies import AppServices
from ragbench.api.schemas import (
    ComponentHealth,
    DocumentCreate,
    DocumentResponse,
    ExperimentCreate,
    ExperimentListResponse,
    ExperimentMetricsResponse,
    ExperimentResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
)

router = APIRouter()
_PUBLIC_REVISION = re.compile(r"^revision:[0-9A-Za-z_.-]+(?:; expected:[0-9A-Za-z_.-]+)?$")
_PUBLIC_DETAILS = {
    "ready",
    "reachable",
    "head",
    "not migrated",
    "migration head unavailable",
    "not configured",
    "unavailable",
}


class NotFoundError(RuntimeError):
    pass


def _services(request: Request) -> AppServices:
    services = request.app.state.services
    if not isinstance(services, AppServices):
        raise RuntimeError("application services are not configured")
    return services


async def _probe(call: object) -> ComponentHealth:
    try:
        result = call()  # type: ignore[operator]
        if inspect.isawaitable(result):
            result = await result
        ready, detail = result
    except Exception:
        return ComponentHealth(ready=False, detail="unavailable")
    raw_detail = str(detail)
    public_detail = (
        raw_detail
        if raw_detail in _PUBLIC_DETAILS or _PUBLIC_REVISION.fullmatch(raw_detail)
        else ("ready" if ready else "unavailable")
    )
    return ComponentHealth(ready=bool(ready), detail=public_detail)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    services = _services(request)
    components = {
        "application": ComponentHealth(ready=True, detail="ready"),
        "database": await _probe(services.database_check),
        "migration": await _probe(services.migration_check),
        "domain_services": await _probe(services.domain_check),
    }
    ready = all(component.ready for component in components.values())
    return HealthResponse(
        status="ready" if ready else "degraded",
        ready=ready,
        components=components,
        version=request.app.version,
    )


@router.post("/documents", status_code=status.HTTP_201_CREATED, response_model=DocumentResponse)
async def create_document(payload: DocumentCreate, request: Request) -> DocumentResponse:
    return await _services(request).documents.create(
        title=payload.title,
        source_uri=str(payload.source_uri),
        metadata=payload.metadata,
    )


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str, request: Request) -> DocumentResponse:
    row = await _services(request).documents.get(document_id)
    if row is None:
        raise NotFoundError
    return row


@router.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest, request: Request) -> SearchResponse:
    hits = await _services(request).search.search(
        query=payload.query,
        top_k=payload.top_k,
        search_filter=payload.filter.model_dump(mode="python"),
    )
    return SearchResponse(hits=tuple(hits))


@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest, request: Request) -> QueryResponse:
    return await _services(request).queries.answer(
        question=payload.question, config_id=payload.config_id
    )


@router.post(
    "/experiments",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ExperimentResponse,
)
async def create_experiment(
    payload: ExperimentCreate, request: Request, response: Response
) -> ExperimentResponse:
    queued = await _services(request).experiments.queue(
        config=payload.config, question_ids=payload.question_ids
    )
    response.headers["Location"] = f"/experiments/{queued.id}"
    return queued


@router.get("/experiments", response_model=ExperimentListResponse)
async def list_experiments(request: Request) -> ExperimentListResponse:
    return ExperimentListResponse(experiments=await _services(request).experiments.list())


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str, request: Request) -> ExperimentResponse:
    row = await _services(request).experiments.get(experiment_id)
    if row is None:
        raise NotFoundError
    return row


@router.get("/experiments/{experiment_id}/metrics", response_model=ExperimentMetricsResponse)
async def get_experiment_metrics(experiment_id: str, request: Request) -> ExperimentMetricsResponse:
    row = await _services(request).experiments.metrics(experiment_id)
    if row is None:
        raise NotFoundError
    return row
