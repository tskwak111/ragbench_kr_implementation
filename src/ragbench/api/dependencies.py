"""Injectable transport dependencies; provider and gold access are intentionally absent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import asyncpg  # type: ignore[import-untyped]
from alembic.config import Config
from alembic.script import ScriptDirectory

from ragbench.api.schemas import (
    DocumentResponse,
    ExperimentMetricsResponse,
    ExperimentResponse,
    QueryResponse,
    SearchHitResponse,
)
from ragbench.core.config import Settings


class DocumentService(Protocol):
    async def create(
        self, *, title: str, source_uri: str, metadata: dict[str, Any]
    ) -> DocumentResponse: ...

    async def get(self, document_id: str) -> DocumentResponse | None: ...


class SearchService(Protocol):
    async def search(
        self, *, query: str, top_k: int, search_filter: dict[str, Any]
    ) -> list[SearchHitResponse]: ...


class QueryService(Protocol):
    async def answer(self, *, question: str, config_id: str) -> QueryResponse: ...


class ExperimentService(Protocol):
    async def queue(
        self, *, config: dict[str, Any], question_ids: tuple[str, ...]
    ) -> ExperimentResponse: ...

    async def list(self) -> tuple[ExperimentResponse, ...]: ...

    async def get(self, experiment_id: str) -> ExperimentResponse | None: ...

    async def metrics(self, experiment_id: str) -> ExperimentMetricsResponse | None: ...


class _Unavailable:
    """Common marker for fail-closed default services."""


class UnavailableDocuments(_Unavailable):
    async def create(self, **_: Any) -> DocumentResponse:
        raise ServiceUnavailableError

    async def get(self, _: str) -> DocumentResponse | None:
        raise ServiceUnavailableError


class UnavailableSearch(_Unavailable):
    async def search(self, **_: Any) -> list[SearchHitResponse]:
        raise ServiceUnavailableError


class UnavailableQueries(_Unavailable):
    async def answer(self, **_: Any) -> QueryResponse:
        raise ServiceUnavailableError


class UnavailableExperiments(_Unavailable):
    """Fail-closed default used until a queue and public aggregate store are installed."""

    async def queue(self, **_: Any) -> ExperimentResponse:
        raise ServiceUnavailableError

    async def list(self) -> tuple[ExperimentResponse, ...]:
        raise ServiceUnavailableError

    async def get(self, _: str) -> ExperimentResponse | None:
        raise ServiceUnavailableError

    async def metrics(self, _: str) -> ExperimentMetricsResponse | None:
        raise ServiceUnavailableError


class ServiceUnavailableError(RuntimeError):
    """A required local component is unavailable; details stay server-side."""


ReadinessProbe = Callable[[], tuple[bool, str] | Awaitable[tuple[bool, str]]]


@dataclass(slots=True)
class AppServices:
    documents: DocumentService
    search: SearchService
    queries: QueryService
    experiments: ExperimentService
    database_check: ReadinessProbe
    migration_check: ReadinessProbe
    domain_check: ReadinessProbe


def default_services() -> AppServices:
    settings = Settings()

    async def database_check() -> tuple[bool, str]:
        connection = await asyncpg.connect(_asyncpg_dsn(settings.database_url), timeout=0.5)
        try:
            await connection.fetchval("SELECT 1")
        finally:
            await connection.close(timeout=0.5)
        return True, "reachable"

    async def migration_check() -> tuple[bool, str]:
        connection = await asyncpg.connect(_asyncpg_dsn(settings.database_url), timeout=0.5)
        try:
            revision = await connection.fetchval("SELECT version_num FROM alembic_version")
        finally:
            await connection.close(timeout=0.5)
        if not isinstance(revision, str) or not revision:
            return False, "not migrated"
        config = Config(str(Path(__file__).resolve().parents[3] / "alembic.ini"))
        expected = ScriptDirectory.from_config(config).get_current_head()
        if expected is None:
            return False, "migration head unavailable"
        return migration_state(revision, expected)

    return AppServices(
        documents=UnavailableDocuments(),
        search=UnavailableSearch(),
        queries=UnavailableQueries(),
        experiments=UnavailableExperiments(),
        database_check=database_check,
        migration_check=migration_check,
        domain_check=lambda: (False, "not configured"),
    )


def _asyncpg_dsn(database_url: str) -> str:
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def migration_state(revision: str, expected: str) -> tuple[bool, str]:
    """Report ready only when the database is at the repository's exact migration head."""
    if revision == expected:
        return True, f"revision:{revision}"
    return False, f"revision:{revision}; expected:{expected}"
