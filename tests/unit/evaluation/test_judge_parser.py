from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from ragbench.evaluation.judge import (
    CalibrationCandidate,
    CalibrationPair,
    EvidenceUnit,
    JudgeConfig,
    JudgeInput,
    JudgeParseError,
    JudgeRunner,
    calibrate_judge,
    parse_judge_response,
    plan_human_calibration,
    render_judge_prompt,
)
from ragbench.providers.base import GenerateRequest, GenerateResponse


def _judge_payload() -> dict[str, object]:
    return {
        "correctness": 0.75,
        "correctness_evidence_ids": ["e1"],
        "claims": [
            {
                "claim_id": "c1",
                "supported": True,
                "evidence_ids": ["e1"],
                "rationale": "e1이 주장을 직접 뒷받침한다.",
            }
        ],
        "citations": [
            {
                "citation_id": "e1",
                "claim_ids": ["c1"],
                "supported": True,
                "evidence_ids": ["e1"],
                "rationale": "e1 인용 단위가 일치한다.",
            }
        ],
        "benchmark_defect": False,
        "benchmark_defect_evidence_ids": [],
        "rationale": "제공된 e1만 사용해 판정했다.",
    }


def _judge_input() -> JudgeInput:
    return JudgeInput(
        question="질문",
        gold_answer="정답",
        gold_evidence=(EvidenceUnit(evidence_id="g1", text="정답 근거"),),
        model_answer="답변",
        answer_claims=("c1: 주장",),
        model_citation_ids=("e1",),
        retrieved_context=(EvidenceUnit(evidence_id="e1", text="주장 근거"),),
    )


def test_judge_parser_accepts_only_known_evidence_and_citations() -> None:
    record = parse_judge_response(json.dumps(_judge_payload()), _judge_input())
    assert record.correctness == 0.75
    assert record.claims[0].supported


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(extra="forbidden"),
        lambda payload: payload.update(correctness=2),
        lambda payload: payload["claims"][0].update(evidence_ids=["invented"]),  # type: ignore[index,union-attr]
        lambda payload: payload["citations"][0].update(citation_id="invented"),  # type: ignore[index,union-attr]
    ],
)
def test_judge_parser_rejects_schema_drift_and_invented_provenance(mutate: object) -> None:
    payload = _judge_payload()
    mutate(payload)  # type: ignore[operator]
    with pytest.raises(JudgeParseError):
        parse_judge_response(json.dumps(payload), _judge_input())


def test_judge_parser_does_not_repair_markdown_or_semantics() -> None:
    raw = "```json\n" + json.dumps(_judge_payload()) + "\n```"
    with pytest.raises(JudgeParseError):
        parse_judge_response(raw, _judge_input())


def test_judge_parser_requires_cited_support_for_positive_decisions() -> None:
    payload = _judge_payload()
    payload["claims"][0]["evidence_ids"] = []  # type: ignore[index]
    with pytest.raises(JudgeParseError, match="supported"):
        parse_judge_response(json.dumps(payload), _judge_input())


def test_judge_parser_requires_rationale_to_name_its_evidence() -> None:
    payload = _judge_payload()
    payload["claims"][0]["rationale"] = "외부 지식상 맞다."  # type: ignore[index]
    with pytest.raises(JudgeParseError, match="rationale"):
        parse_judge_response(json.dumps(payload), _judge_input())


def test_judge_rationale_does_not_match_an_evidence_id_prefix() -> None:
    payload = _judge_payload()
    payload["claims"][0]["rationale"] = "e10이 근거다."  # type: ignore[index]
    with pytest.raises(JudgeParseError, match="rationale"):
        parse_judge_response(json.dumps(payload), _judge_input())


def test_blind_prompt_excludes_system_and_configuration_identity() -> None:
    rendered = render_judge_prompt(_judge_input())
    assert "질문" in rendered
    assert "e1" in rendered
    assert "generator-config-secret" not in rendered
    assert "system_id" not in rendered


@dataclass
class RecordingGateway:
    request: GenerateRequest | None = None

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.request = request
        return GenerateResponse(json.dumps(_judge_payload()), {"usage": {}}, "corr", True)

    async def parse(self, request: object) -> object:
        raise AssertionError("judge must not parse")

    async def embed(self, request: object) -> object:
        raise AssertionError("judge must not embed")


@pytest.mark.asyncio
async def test_judge_runner_uses_gateway_and_records_exact_identity() -> None:
    gateway = RecordingGateway()
    config = JudgeConfig(
        model_id="judge-model-v2",
        generator_model_id="generator-model-v1",
        rubric_version="judge-v1",
        temperature=0.0,
        max_output_tokens=800,
    )
    result = await JudgeRunner(gateway).evaluate(_judge_input(), config)
    assert gateway.request is not None
    assert gateway.request.model_id == "judge-model-v2"
    assert gateway.request.provider_params == {"temperature": 0.0}
    assert result.model_id == "judge-model-v2"
    assert len(result.rubric_hash) == 64
    assert result.temperature == 0.0
    assert result.cached is True


def test_same_judge_and_generator_requires_unavailability_reason() -> None:
    with pytest.raises(ValueError, match="distinct"):
        JudgeConfig(
            model_id="same",
            generator_model_id="same",
            rubric_version="judge-v1",
            temperature=0.0,
            max_output_tokens=800,
        )


def test_human_calibration_plan_is_balanced_and_includes_hard_cases() -> None:
    candidates = tuple(
        CalibrationCandidate(
            response_id=f"r{index:03d}",
            system_id=f"s{(index // 4) % 2}",
            question_type=f"t{index % 4}",
            judge_human_disagreement=index in {0, 1, 2},
            known_failure=index in {3, 4, 5},
        )
        for index in range(160)
    )
    plan = plan_human_calibration(candidates, sample_size=120, seed=17)
    assert len(plan.responses) == 120
    assert any(row.judge_human_disagreement for row in plan.responses)
    assert any(row.known_failure for row in plan.responses)
    counts = list(plan.stratum_counts.values())
    assert max(counts) - min(counts) <= 1
    assert plan.requires_real_human_labels


def test_human_calibration_plan_rejects_missing_system_type_strata() -> None:
    candidates = tuple(
        CalibrationCandidate(
            response_id=f"r{index:03d}",
            system_id="s0" if index < 100 else "s1",
            question_type="fact" if index < 100 else "summary",
            judge_human_disagreement=index == 0,
            known_failure=index == 100,
        )
        for index in range(200)
    )
    with pytest.raises(ValueError, match="system/type"):
        plan_human_calibration(candidates, sample_size=100, seed=17)


def test_human_calibration_plan_rejects_exhausted_imbalanced_stratum() -> None:
    candidates = tuple(
        CalibrationCandidate(
            response_id=f"r{index:03d}",
            system_id="s0" if index in {0, 101} else "s1",
            question_type="fact" if index <= 100 else "summary",
            judge_human_disagreement=index == 0,
            known_failure=index == 101,
        )
        for index in range(201)
    )
    with pytest.raises(ValueError, match="balanced"):
        plan_human_calibration(candidates, sample_size=100, seed=17)


def test_calibration_reports_rank_binary_agreement_and_type_bias_not_authority() -> None:
    pairs = tuple(
        CalibrationPair(
            response_id=f"r{index:03d}",
            question_type="fact" if index % 2 == 0 else "summary",
            judge_score=index / 99,
            human_score=(index / 99) * (0.9 if index % 2 else 1.0),
            reviewer_id=f"human-{index % 3}",
            human_attested=True,
        )
        for index in range(100)
    )
    report = calibrate_judge(pairs, binary_threshold=0.5)
    assert report.sample_size == 100
    assert report.spearman_correlation > 0.99
    assert report.threshold_agreement > 0.9
    assert report.threshold_f1 > 0.9
    assert set(report.mean_bias_by_question_type) == {"fact", "summary"}
    assert report.status == "calibrated-assistant-only"
    assert not report.is_final_authority


def test_calibration_rejects_too_few_or_nonhuman_labels() -> None:
    pair = CalibrationPair("r1", "fact", 1.0, 1.0, "human-a", True)
    with pytest.raises(ValueError, match="100"):
        calibrate_judge((pair,), binary_threshold=0.5)
    fake = tuple(
        CalibrationPair(f"r{i}", "fact", 1.0, 1.0, "synthetic", False)
        for i in range(100)
    )
    with pytest.raises(ValueError, match="real human"):
        calibrate_judge(fake, binary_threshold=0.5)


def test_calibration_rejects_non_boolean_human_attestation() -> None:
    with pytest.raises(ValueError, match="attestation"):
        CalibrationPair("r1", "fact", 1.0, 1.0, "human-a", 1)  # type: ignore[arg-type]
