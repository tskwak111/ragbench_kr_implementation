"""Provider-neutral request, response, and gateway contracts."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    model_id: str
    prompt: str
    context: tuple[str, ...] = ()
    provider_params: dict[str, Any] = field(default_factory=dict)
    input_tokens: int = 0
    max_output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class GenerateResponse:
    content: str
    raw_response: dict[str, Any]
    correlation_id: str | None = None
    cache_hit: bool | None = None


@dataclass(frozen=True, slots=True)
class EmbedRequest:
    model_id: str
    texts: tuple[str, ...]
    input_tokens: int
    provider_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbedResponse:
    embeddings: tuple[tuple[float, ...], ...]
    raw_response: dict[str, Any]
    correlation_id: str | None = None
    model_id: str | None = None


@dataclass(frozen=True, slots=True)
class ParseRequest:
    model_id: str
    document_sha256: str
    content: bytes
    billable_pages: int
    mode: str = "standard"
    provider_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    raw_response: dict[str, Any]
    correlation_id: str | None = None


class ProviderGateway(Protocol):
    async def parse(self, request: ParseRequest) -> ParsedDocument: ...

    async def embed(self, request: EmbedRequest) -> EmbedResponse: ...

    async def generate(self, request: GenerateRequest) -> GenerateResponse: ...
