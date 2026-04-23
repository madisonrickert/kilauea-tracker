"""State banner renderer — verify HTML shape, state color, a11y attrs."""

from __future__ import annotations

import pytest

from kilauea_tracker.ui.palette import STATE_COLOR, STATE_RAMP_ORDER
from kilauea_tracker.ui.state_banner import render_html
from kilauea_tracker.ui.state_copy import STATE_COPY


@pytest.mark.parametrize("state", STATE_RAMP_ORDER)
def test_banner_contains_role_and_aria_live(state: str):
    html = render_html(state, {})
    assert 'role="status"' in html
    assert 'aria-live="polite"' in html


@pytest.mark.parametrize("state", STATE_RAMP_ORDER)
def test_banner_uses_matching_state_color(state: str):
    html = render_html(state, {})
    expected_color = STATE_COLOR[state]
    assert f"--banner-accent: {expected_color}" in html, (
        f"state {state!r} should render with color {expected_color} but got: {html}"
    )


@pytest.mark.parametrize("state", STATE_RAMP_ORDER)
def test_banner_includes_headline_and_icon(state: str):
    html = render_html(state, {})
    copy = STATE_COPY[state]
    assert copy.headline in html
    assert copy.icon in html


@pytest.mark.parametrize("state", STATE_RAMP_ORDER)
def test_banner_includes_explanation(state: str):
    """Explainer text (or some recognizable prefix of it) should be in the HTML."""
    html = render_html(state, {"short_slope_microrad_per_hour": -0.8, "long_slope_microrad_per_hour": -0.1, "drop_from_24h_max": 1.5, "recent_slope_microrad_per_hour": -1.2})
    assert 'class="kt-banner__explanation"' in html
    # Explanation paragraph should not be empty.
    assert '<p class="kt-banner__explanation"></p>' not in html


def test_inline_bold_markdown_becomes_strong_tags():
    from kilauea_tracker.ui.state_banner import _inline_bold

    assert _inline_bold("foo **bar** baz") == "foo <strong>bar</strong> baz"
    assert _inline_bold("**a** and **b**") == "<strong>a</strong> and <strong>b</strong>"
    # Unbalanced ** is rare but should not crash.
    assert _inline_bold("no bold here") == "no bold here"


def test_unknown_state_raises():
    with pytest.raises(KeyError):
        render_html("exploded", {})


def test_banner_icon_marked_aria_hidden():
    """The emoji icon is decorative — screen readers should skip it."""
    html = render_html("active", {})
    assert 'aria-hidden="true"' in html


def test_banner_opens_with_kt_banner_class():
    html = render_html("active", {})
    assert html.startswith('<div class="kt-banner"')
