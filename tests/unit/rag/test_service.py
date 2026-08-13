"""Offline end-to-end grounded RAG service contract tests."""

from dataclasses import replace
from typing import Any

import pytest

from ragbench.chunking.tokenizer import encoding
from ragbench.providers.base import GenerateRequest, GenerateResponse
from ragbench.rag.citations import CitationValidationError
from ragbench.rag.context import ContextBuilder, RetrievedChunk
from ragbench.rag.prompts import PromptVersion
from ragbench.rag.service import (
    PROVIDER_CHAT_FRAMING_TOKEN_MARGIN,
    RagConfig,
    RagService,
)
from ragbench.retrieval.base import SearchFilter, SearchHit


class FakeRetriever:
    async def search(
        self, query: str, *, top_k: int, filter: SearchFilter
    ) -> list[SearchHit]:
        assert query == "매출은 얼마인가?"
        assert top_k == 2
        assert filter.corpus_snapshot_id == "corpus"
        return [SearchHit("chunk-1", 0.9, 1, "hybrid-rrf")]


class FakeEvidenceSource:
    async def resolve(
        self, hits: list[SearchHit], *, filter: SearchFilter
    ) -> list[RetrievedChunk]:
        assert [hit.chunk_id for hit in hits] == ["chunk-1"]
        assert filter.embedding_snapshot_id == "embed"
        return [
            RetrievedChunk(
                hits[0], "doc-1", "사업보고서", 10, 11, ("실적",), "매출은 100억 원이다."
            )
        ]


class FakeGateway:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[GenerateRequest] = []

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.requests.append(request)
        return GenerateResponse(
            self.content,
            {
                "id": "fake-response",
                "usage": {"prompt_tokens": 321, "completion_tokens": 17},
            },
            "corr-1",
            cache_hit=True,
        )

    async def embed(self, request: Any) -> Any:  # pragma: no cover - protocol-only fake
        raise AssertionError("generation must not embed directly")

    async def parse(self, request: Any) -> Any:  # pragma: no cover - protocol-only fake
        raise AssertionError("generation must not parse directly")


def _config() -> RagConfig:
    return RagConfig(
        search_filter=SearchFilter("corpus", "parse", "heading", "embed"),
        model_id="solar-pro4",
        prompt_version=PromptVersion.V3,
        top_k=2,
        context_token_budget=1_000,
        max_output_tokens=200,
        experiment_id="experiment-7",
        config_id="config-11",
        provider_params={"temperature": 0},
    )


def test_rag_config_copies_and_freezes_provider_parameters() -> None:
    """Catch cache/config identity changing after the validated request is created."""
    parameters: dict[str, object] = {"temperature": 0, "response": {"format": "json"}}
    config = replace(_config(), provider_params=parameters)
    parameters["temperature"] = 1

    assert config.provider_params["temperature"] == 0
    with pytest.raises(TypeError):
        config.provider_params["temperature"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        config.provider_params["response"]["format"] = "text"  # type: ignore[index]


@pytest.mark.asyncio
async def test_answer_uses_gateway_and_returns_auditable_contract() -> None:
    """Catch direct HTTP generation or loss of usage/config/provenance evidence."""
    gateway = FakeGateway(
        '{"answer":"100억 원","citations":["C1"],"abstained":false}'
    )
    service = RagService(FakeRetriever(), FakeEvidenceSource(), gateway)

    result = await service.answer("매출은 얼마인가?", _config())

    assert result.question == "매출은 얼마인가?"
    assert result.answer == "100억 원"
    assert result.abstained is False
    assert result.experiment_id == "experiment-7"
    assert result.config_id == "config-11"
    assert result.model_id == "solar-pro4"
    assert result.correlation_id == "corr-1"
    assert result.cached is True
    assert result.usage.input_tokens == 321
    assert result.usage.output_tokens == 17
    assert result.latency_ms >= 0
    assert [citation.chunk_id for citation in result.citations] == ["chunk-1"]
    assert [evidence.chunk_id for evidence in result.evidence] == ["chunk-1"]
    assert len(gateway.requests) == 1
    request = gateway.requests[0]
    assert request.model_id == "solar-pro4"
    assert request.context == ()
    assert request.provider_params == {"temperature": 0}
    assert request.max_output_tokens == 200
    assert request.input_tokens == (
        len(encoding().encode(request.prompt)) + PROVIDER_CHAT_FRAMING_TOKEN_MARGIN
    )
    assert "<UNTRUSTED_CONTEXT>" in request.prompt


@pytest.mark.asyncio
async def test_unanswerable_v3_response_returns_clean_abstention() -> None:
    """Catch forced hallucination when retrieved evidence is insufficient."""
    gateway = FakeGateway(
        '{"answer":"제공된 근거로 답할 수 없습니다.","citations":[],"abstained":true}'
    )

    result = await RagService(FakeRetriever(), FakeEvidenceSource(), gateway).answer(
        "매출은 얼마인가?", _config()
    )

    assert result.abstained is True
    assert result.citations == ()


@pytest.mark.asyncio
async def test_missing_resolved_evidence_fails_closed_before_generation() -> None:
    """Catch generating from an incomplete or silently dropped retrieval evidence set."""
    class MissingEvidenceSource:
        async def resolve(
            self, hits: list[SearchHit], *, filter: SearchFilter
        ) -> list[RetrievedChunk]:
            return []

    gateway = FakeGateway(
        '{"answer":"답","citations":["C1"],"abstained":false}'
    )

    with pytest.raises(ValueError, match="exactly match"):
        await RagService(FakeRetriever(), MissingEvidenceSource(), gateway).answer(
            "매출은 얼마인가?", _config()
        )

    assert gateway.requests == []


@pytest.mark.asyncio
async def test_model_cannot_cite_retrieved_but_budget_excluded_chunk() -> None:
    """Catch validating against raw retrieval hits instead of actually included context."""
    class TwoHitRetriever:
        async def search(
            self, query: str, *, top_k: int, filter: SearchFilter
        ) -> list[SearchHit]:
            return [
                SearchHit("chunk-1", 0.9, 1, "dense"),
                SearchHit("chunk-2", 0.8, 2, "dense"),
            ]

    class TwoChunkSource:
        async def resolve(
            self, hits: list[SearchHit], *, filter: SearchFilter
        ) -> list[RetrievedChunk]:
            return [
                RetrievedChunk(
                    hits[0], "doc-1", "문서", 1, 1, ("첫째",), "첫 번째 근거"
                ),
                RetrievedChunk(
                    hits[1], "doc-1", "문서", 2, 2, ("둘째",), "두 번째 근거"
                ),
            ]

    gateway = FakeGateway(
        '{"answer":"답","citations":["C2"],"abstained":false}'
    )
    first = RetrievedChunk(
        SearchHit("chunk-1", 0.9, 1, "dense"),
        "doc-1",
        "문서",
        1,
        1,
        ("첫째",),
        "첫 번째 근거",
    )
    exact_first_budget = ContextBuilder().build((first,), 2_000).token_count
    config = replace(_config(), context_token_budget=exact_first_budget)

    with pytest.raises(CitationValidationError, match="not in included context"):
        await RagService(TwoHitRetriever(), TwoChunkSource(), gateway).answer(
            "매출은 얼마인가?", config
        )
