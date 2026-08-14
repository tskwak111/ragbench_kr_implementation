"""FastAPI application factory with fail-closed defaults and sanitized errors."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ragbench.api.dependencies import (
    AppServices,
    ServiceUnavailableError,
    default_services,
)
from ragbench.api.routes import NotFoundError, router

_CORRELATION_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _version() -> str:
    try:
        return version("ragbench-kr")
    except PackageNotFoundError:
        return "0.1.0"


def _error(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    correlation_id = getattr(request.state, "correlation_id", str(uuid4()))
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "correlation_id": correlation_id,
            }
        },
        headers={"X-Correlation-ID": correlation_id},
    )


def create_app(services: AppServices | None = None) -> FastAPI:
    application = FastAPI(
        title="RAGBench-KR API",
        version=_version(),
        docs_url="/docs",
        redoc_url=None,
    )
    application.state.services = services or default_services()

    @application.middleware("http")
    async def correlation_id(request: Request, call_next: object) -> JSONResponse:
        supplied = request.headers.get("X-Correlation-ID", "")
        request.state.correlation_id = (
            supplied if _CORRELATION_ID.fullmatch(supplied) else str(uuid4())
        )
        response = await call_next(request)  # type: ignore[operator]
        response.headers["X-Correlation-ID"] = request.state.correlation_id
        return response  # type: ignore[no-any-return]

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, _: RequestValidationError) -> JSONResponse:
        return _error(request, 422, "VALIDATION_ERROR", "request validation failed")

    @application.exception_handler(NotFoundError)
    async def not_found(request: Request, _: NotFoundError) -> JSONResponse:
        return _error(request, 404, "NOT_FOUND", "resource not found")

    @application.exception_handler(ServiceUnavailableError)
    async def unavailable(request: Request, _: ServiceUnavailableError) -> JSONResponse:
        return _error(request, 503, "SERVICE_UNAVAILABLE", "service unavailable")

    @application.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        if error.status_code == 404:
            return _error(request, 404, "NOT_FOUND", "resource not found")
        return _error(request, error.status_code, "HTTP_ERROR", "request failed")

    @application.exception_handler(Exception)
    async def internal_error(request: Request, _: Exception) -> JSONResponse:
        return _error(request, 500, "INTERNAL_ERROR", "internal server error")

    application.include_router(router)
    return application


app = create_app()

__all__ = ["AppServices", "app", "create_app"]
