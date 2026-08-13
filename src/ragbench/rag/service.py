"""Grounded RAG orchestration through the metered provider gateway only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from types import MappingProxyType
from typing import Any, Protocol

from ragbench.chunking.tokenizer import encoding
from ragbench.providers.base import GenerateRequest, GenerateResponse, ProviderGateway
from ragbench.rag.citations import (
    ResolvedCitation,
    parse_generated_answer,
    resolve_citations,
    validate_answer_policy,
)
from ragbench.rag.context import ContextBuilder, ContextItem, RetrievedChunk
from ragbench.rag.prompts import PromptVersion, render_prompt
from ragbench.retrieval.base import Retriever, SearchFilter, SearchHit

# The vendored tokenizer counts the exact rendered text. The gateway additionally needs a
# conservative upper bound for provider-specific chat framing that is not part of that text.
PROVIDER_CHAT_FRAMING_TOKEN_MARGIN = 16


class EvidenceSource(Protocol):
    """Load immutable chunk text/provenance for IDs returned by one retriever call."""

    async def resolve(
        self, hits: list[SearchHit], *, filter: SearchFilter
    ) -> list[RetrievedChunk]: ...


@dataclass(frozen=True, slots=True)
class RagConfig:
    """Immutable generation identity and bounded runtime parameters."""

    search_filter: SearchFilter
    model_id: str
    prompt_version: PromptVersion
    top_k: int
    context_token_budget: int
    max_output_tokens: int
    experiment_id: str
    config_id: str
    provider_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_version, PromptVersion):
            raise ValueError("unknown prompt version")
        if any(
            not value.strip()
            for value in (self.model_id, self.experiment_id, self.config_id)
        ):
            raise ValueError("model, experiment, and config identities cannot be blank")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if (
            isinstance(self.context_token_budget, bool)
            or not isinstance(self.context_token_budget, int)
            or self.context_token_budget < 0
        ):
            raise ValueError("context_token_budget must be a nonnegative integer")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        frozen = _freeze_json(dict(self.provider_params))
        if not isinstance(frozen, Mapping):  # pragma: no cover - dict input guarantees mapping
            raise TypeError("provider_params must be a JSON object")
        object.__setattr__(self, "provider_params", frozen)


@dataclass(frozen=True, slots=True)
class GenerationUsage:
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """Complete public response contract with model claims and server-owned evidence."""

    question: str
    answer: str
    abstained: bool
    evidence: tuple[ContextItem, ...]
    citations: tuple[ResolvedCitation, ...]
    latency_ms: int
    usage: GenerationUsage
    experiment_id: str
    config_id: str
    model_id: str
    prompt_version: PromptVersion
    cached: bool | None
    correlation_id: str | None
    provider_response_id: str | None


class RagService:
    """Retrieve, serialize, generate, parse, and cite without any direct provider transport."""

    def __init__(
        self,
        retriever: Retriever,
        evidence_source: EvidenceSource,
        gateway: ProviderGateway,
        *,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self._retriever = retriever
        self._evidence_source = evidence_source
        self._gateway = gateway
        self._context_builder = context_builder or ContextBuilder()

    async def answer(self, question: str, config: RagConfig) -> RagAnswer:
        if not question.strip():
            raise ValueError("question cannot be blank")
        started = perf_counter()
        hits = await self._retriever.search(
            question, top_k=config.top_k, filter=config.search_filter
        )
        chunks = await self._evidence_source.resolve(hits, filter=config.search_filter)
        _validate_resolved_chunks(hits, chunks)
        context = self._context_builder.build(chunks, config.context_token_budget)
        prompt = render_prompt(config.prompt_version, question=question, context=context)
        exact_prompt_tokens = len(encoding().encode(prompt))
        response = await self._gateway.generate(
            GenerateRequest(
                model_id=config.model_id,
                prompt=prompt,
                context=(),
                provider_params=_provider_params_dict(config.provider_params),
                input_tokens=exact_prompt_tokens + PROVIDER_CHAT_FRAMING_TOKEN_MARGIN,
                max_output_tokens=config.max_output_tokens,
            )
        )
        generated = parse_generated_answer(response.content)
        validate_answer_policy(generated, prompt_version=config.prompt_version.value)
        citations = resolve_citations(generated.citations, context)
        usage = _generation_usage(response)
        raw_id = response.raw_response.get("id")
        return RagAnswer(
            question=question,
            answer=generated.answer,
            abstained=generated.abstained,
            evidence=context.items,
            citations=citations,
            latency_ms=max(0, round((perf_counter() - started) * 1000)),
            usage=usage,
            experiment_id=config.experiment_id,
            config_id=config.config_id,
            model_id=config.model_id,
            prompt_version=config.prompt_version,
            cached=response.cache_hit,
            correlation_id=response.correlation_id,
            provider_response_id=raw_id if isinstance(raw_id, str) else None,
        )


def _validate_resolved_chunks(
    hits: Sequence[SearchHit], chunks: Sequence[RetrievedChunk]
) -> None:
    by_id: dict[str, SearchHit] = {}
    for hit in hits:
        if hit.chunk_id in by_id:
            raise ValueError("retriever returned duplicate chunk IDs")
        by_id[hit.chunk_id] = hit
    resolved: set[str] = set()
    for chunk in chunks:
        if chunk.hit.chunk_id in resolved:
            raise ValueError("evidence source returned duplicate chunk IDs")
        resolved.add(chunk.hit.chunk_id)
        canonical = by_id.get(chunk.hit.chunk_id)
        if canonical is None:
            raise ValueError("evidence source returned a chunk not requested from retrieval")
        if chunk.hit != canonical:
            raise ValueError("evidence source changed canonical retrieval evidence")
    if resolved != set(by_id):
        raise ValueError("resolved evidence IDs must exactly match retrieval hit IDs")


def _generation_usage(response: GenerateResponse) -> GenerationUsage:
    raw = response.raw_response.get("usage")
    if not isinstance(raw, dict):
        return GenerationUsage(None, None)
    return GenerationUsage(
        _nonnegative_int(raw.get("prompt_tokens")),
        _nonnegative_int(raw.get("completion_tokens")),
    )


def _nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("provider parameter keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("provider parameters must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _provider_params_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    thawed = _thaw_json(value)
    if not isinstance(thawed, dict):  # pragma: no cover - mapping input guarantees dict
        raise TypeError("provider_params must be a JSON object")
    return thawed
