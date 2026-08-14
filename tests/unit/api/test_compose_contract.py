from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_compose_starts_migrated_api_with_liveness_and_named_volumes() -> None:
    payload = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    assert set(payload["services"]) == {"api", "db"}
    assert set(payload["volumes"]) == {"ragbench_cache", "ragbench_pgdata"}
    api = payload["services"]["api"]
    assert api["depends_on"]["db"]["condition"] == "service_healthy"
    assert "alembic upgrade head" in api["command"][2]
    assert api["command"][2].index("alembic upgrade head") < api["command"][2].index("uvicorn")
    probe = api["healthcheck"]["test"][3]
    assert "urlopen" in probe
    assert ".get('ready')" not in probe
    assert api["environment"]["RUN_LIVE_UPSTAGE_TESTS"] == "0"
    assert api["environment"]["ALLOW_GOLD_ACCESS"] == "0"
