from decimal import Decimal

import pytest
from pydantic import ValidationError

from ragbench.core.config import Settings
from ragbench.core.versions import VersionBundle


def test_offline_settings_allow_missing_upstage_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UPSTAGE_API_KEY", raising=False)
    monkeypatch.delenv("RUN_LIVE_UPSTAGE_TESTS", raising=False)

    assert Settings(_env_file=None).upstage_api_key is None


def test_live_settings_require_upstage_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UPSTAGE_API_KEY", raising=False)
    monkeypatch.setenv("RUN_LIVE_UPSTAGE_TESTS", "1")

    with pytest.raises(ValidationError, match="UPSTAGE_API_KEY is required"):
        Settings(_env_file=None)


def test_settings_use_safe_runtime_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "UPSTAGE_API_KEY",
        "RUN_LIVE_UPSTAGE_TESTS",
        "MAX_PROJECT_BUDGET_USD",
        "MAX_CONCURRENCY",
        "MAX_LOCK_CONNECTIONS",
        "MAX_RETRIES",
        "ALLOW_GOLD_ACCESS",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.max_project_budget_usd == Decimal("135.00")
    assert settings.max_concurrency == 5
    assert settings.max_lock_connections == 2
    assert settings.max_retries == 5
    assert settings.allow_gold_access is False


@pytest.mark.parametrize("max_lock_connections", [0, -1])
def test_settings_reject_nonpositive_lock_connection_bound(
    max_lock_connections: int,
) -> None:
    """Catch runtime configuration disabling the PostgreSQL lock-session bound."""
    with pytest.raises(ValidationError, match="max_lock_connections"):
        Settings(max_lock_connections=max_lock_connections)


def test_version_bundle_preserves_reproducibility_identifiers() -> None:
    bundle = VersionBundle(
        code_commit="abc123",
        config_hash="config456",
        data_snapshot="corpus789",
    )

    assert bundle.as_dict() == {
        "code_commit": "abc123",
        "config_hash": "config456",
        "data_snapshot": "corpus789",
    }
