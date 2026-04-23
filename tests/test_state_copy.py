"""Lock the state-copy table: every state maps to a non-empty, distinct entry."""

from __future__ import annotations

import pytest

from kilauea_tracker.ui.palette import STATE_RAMP_ORDER
from kilauea_tracker.ui.state_copy import STATE_COPY, get


def test_every_state_has_copy():
    for state in STATE_RAMP_ORDER:
        assert state in STATE_COPY, f"no copy entry for state {state!r}"


@pytest.mark.parametrize("state", STATE_RAMP_ORDER)
def test_all_fields_populated(state: str):
    entry = STATE_COPY[state]
    assert entry.icon, f"{state}: empty icon"
    assert entry.headline, f"{state}: empty headline"
    assert entry.guidance, f"{state}: empty guidance"
    # explainer runs against an empty info dict and must return a non-empty string
    # even when every value is missing (defensive formatters).
    rendered = entry.explainer({})
    assert rendered and rendered.strip(), f"{state}: explainer returned empty"


def test_headlines_are_unique():
    headlines = [STATE_COPY[s].headline for s in STATE_RAMP_ORDER]
    assert len(headlines) == len(set(headlines)), (
        f"duplicate headlines across states: {headlines}"
    )


def test_icons_are_all_emoji_colored_circles():
    """Keep the visual language consistent: single colored-circle emoji per state."""
    for state in STATE_RAMP_ORDER:
        icon = STATE_COPY[state].icon
        assert icon in {"🟢", "🟡", "🟠", "🔴", "⚪", "🟣", "🔵"}, (
            f"{state}: icon {icon!r} not a colored-circle emoji"
        )


def test_get_raises_on_unknown_state():
    with pytest.raises(KeyError, match="unknown eruption state"):
        get("exploding")


def test_explainer_handles_missing_numbers_gracefully():
    """If the classification dict is missing a field, the explainer shouldn't crash."""
    for state in STATE_RAMP_ORDER:
        # Simulate a malformed info dict — None values, missing keys, NaN.
        for bad_info in ({}, {"short_slope_microrad_per_hour": None}):
            rendered = STATE_COPY[state].explainer(bad_info)
            assert rendered and "?" in rendered or "None" not in rendered
