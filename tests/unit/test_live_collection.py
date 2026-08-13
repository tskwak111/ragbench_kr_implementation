"""Pure contracts for the collection-time live-test safety gate."""

from __future__ import annotations

import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOKS = runpy.run_path(PROJECT_ROOT / "tests" / "conftest.py")
live_tests_requested = HOOKS["live_tests_requested"]
live_test_skip_reason = HOOKS["live_test_skip_reason"]


def test_live_tests_require_an_explicit_live_marker_selection() -> None:
    """Catch ordinary pytest runs becoming paid runs because a credential is present."""
    assert live_tests_requested("") is False
    assert live_tests_requested("not live and not gold") is False
    assert live_tests_requested("live") is True


def test_live_collection_requires_marker_environment_flag_and_key() -> None:
    """Catch a configured credential making normal pytest runs eligible for live tests."""
    environment = {"RUN_LIVE_UPSTAGE_TESTS": "1", "UPSTAGE_API_KEY": "redacted"}

    assert (
        live_test_skip_reason("", environment)
        == "live tests require explicit pytest -m live selection"
    )
    assert live_test_skip_reason("live", {}) == "live tests require RUN_LIVE_UPSTAGE_TESTS=1"
    assert live_test_skip_reason("live", {"RUN_LIVE_UPSTAGE_TESTS": "1"}) == (
        "live tests require UPSTAGE_API_KEY"
    )
    assert live_test_skip_reason("live", environment) is None
