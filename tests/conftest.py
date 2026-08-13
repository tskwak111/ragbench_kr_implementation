"""Collection-time safety controls for tests that could contact paid providers."""

from __future__ import annotations

import os

import pytest


def live_tests_requested(marker_expression: str) -> bool:
    """Return true only for an explicit positive ``-m live`` selection."""
    expression = marker_expression.strip()
    return (
        expression == "live"
        or expression.startswith("live and ")
        or expression.startswith("live or ")
    )


def live_test_skip_reason(marker_expression: str, environment: dict[str, str]) -> str | None:
    """Explain why a marked test must remain skipped without exposing a credential."""
    if not live_tests_requested(marker_expression):
        return "live tests require explicit pytest -m live selection"
    if environment.get("RUN_LIVE_UPSTAGE_TESTS") != "1":
        return "live tests require RUN_LIVE_UPSTAGE_TESTS=1"
    if not environment.get("UPSTAGE_API_KEY"):
        return "live tests require UPSTAGE_API_KEY"
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep normal pytest runs offline even if an operator shell has credentials."""
    reason = live_test_skip_reason(config.option.markexpr, dict(os.environ))
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker("live") is not None:
            item.add_marker(skip)
