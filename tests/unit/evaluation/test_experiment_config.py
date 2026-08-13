from pathlib import Path

import pytest
from pydantic import ValidationError

from ragbench.experiments.config import RetrievalExperimentConfig
from ragbench.experiments.planner import generate_core_retrieval_configs, require_unique_configs


def _payload() -> dict[str, object]:
    return {
        "schema_version": "retrieval-screen-v1",
        "corpus_snapshot_id": "corpus-a",
        "parse_snapshot_id": "parse-standard-a",
        "parse_mode": "standard",
        "chunk_snapshot_id": "chunks-a",
        "chunk_strategy": "fixed-300-0",
        "embedding_snapshot_id": "embed-a",
        "retriever": "hybrid",
        "rrf": {"rank_constant": 60, "dense_weight": 1.0, "sparse_weight": 1.0},
        "top_k": 5,
        "question_snapshot_id": "dev-a",
        "question_split": "dev_auto",
        "random_seed": 17,
        "code_commit": "0bce46e",
        "metric_version": "retrieval-v1",
    }


def test_yaml_round_trip_is_strict_immutable_and_semantically_hashed(tmp_path: Path) -> None:
    config = RetrievalExperimentConfig.model_validate(_payload())
    path = tmp_path / "screen.yaml"
    config.to_yaml(path)

    loaded = RetrievalExperimentConfig.from_yaml(path)

    assert loaded == config
    assert loaded.semantic_hash == (
        "d0552e48f820ced3f33e4228cb54823f04e3eb396998f89ceff75931aafc0c48"
    )
    with pytest.raises(ValidationError, match="frozen"):
        loaded.top_k = 10  # type: ignore[misc]


def test_config_rejects_unknown_fields_gold_split_and_semantic_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RetrievalExperimentConfig.model_validate({**_payload(), "temperature": 0})
    with pytest.raises(ValidationError, match="dev_auto"):
        RetrievalExperimentConfig.model_validate({**_payload(), "question_split": "test_gold"})
    with pytest.raises(ValidationError, match="hybrid"):
        RetrievalExperimentConfig.model_validate(
            {**_payload(), "retriever": "dense", "rrf": _payload()["rrf"]}
        )
    path = tmp_path / "bad.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        RetrievalExperimentConfig.from_yaml(path)

    with pytest.raises(ValidationError):
        RetrievalExperimentConfig.model_validate({**_payload(), "random_seed": "17"})
    with pytest.raises(ValidationError, match="blank"):
        RetrievalExperimentConfig.model_validate({**_payload(), "corpus_snapshot_id": "  "})


def test_yaml_export_refuses_to_overwrite_an_existing_config(tmp_path: Path) -> None:
    path = tmp_path / "screen.yaml"
    path.write_text("owner: user\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        RetrievalExperimentConfig.model_validate(_payload()).to_yaml(path)

    assert path.read_text(encoding="utf-8") == "owner: user\n"


def test_core_planner_generates_exactly_126_unique_configs() -> None:
    configs = generate_core_retrieval_configs(
        corpus_snapshot_id="corpus-a",
        question_snapshot_id="dev-a",
        code_commit="0bce46e",
        random_seed=17,
    )

    assert len(configs) == 2 * 7 * 3 * 3 == 126
    assert len({config.semantic_hash for config in configs}) == 126
    assert {config.parse_mode for config in configs} == {"standard", "enhanced"}
    assert {config.top_k for config in configs} == {3, 5, 10}


def test_planner_rejects_duplicate_semantics_even_when_objects_are_repeated() -> None:
    config = RetrievalExperimentConfig.model_validate(_payload())

    with pytest.raises(ValueError, match="duplicate semantic"):
        require_unique_configs((config, config.model_copy()))
