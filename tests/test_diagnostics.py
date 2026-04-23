"""Teaching-tool diagnostic classifiers — cover each threshold branch.

The classifiers are the source of truth for what the Pipeline tab tells
a visitor is "typical" vs "elevated" vs "extreme." A regression that
moves a threshold would silently change what the app teaches, so we lock
each branch with a parametrized test.
"""

from __future__ import annotations

import pytest

from kilauea_tracker.ui.diagnostics import (
    Tint,
    episode_samples_tint,
    exp_amplitude_tint,
    exp_k_tint,
    render_chip_html,
    trendline_slope_tint,
)
from kilauea_tracker.ui.palette import ASH, EMBER, FLAME, STEAM


@pytest.mark.parametrize(
    "slope, expected_color, expected_label",
    [
        (0.0, ASH, "flat"),
        (0.04, ASH, "flat"),
        (0.10, STEAM, "trending"),
        (-0.20, STEAM, "trending"),
        (0.50, EMBER, "strong trend"),
        (-0.80, EMBER, "strong trend"),
        (1.50, FLAME, "extreme trend"),
    ],
)
def test_trendline_slope_branches(slope, expected_color, expected_label):
    tint = trendline_slope_tint(slope)
    assert tint.color == expected_color
    assert tint.label == expected_label


@pytest.mark.parametrize(
    "n, expected_color",
    [
        (0, FLAME),
        (3, FLAME),
        (4, EMBER),
        (11, EMBER),
        (12, STEAM),
        (100, STEAM),
    ],
)
def test_episode_samples_branches(n, expected_color):
    assert episode_samples_tint(n).color == expected_color


@pytest.mark.parametrize(
    "k, expected_color, expected_label",
    [
        (0.0, FLAME, "invalid"),
        (-0.1, FLAME, "invalid"),
        (0.01, EMBER, "slow rise"),
        (0.1, STEAM, "typical"),
        (0.25, STEAM, "typical"),
        (0.5, EMBER, "rapid rise"),
    ],
)
def test_exp_k_branches(k, expected_color, expected_label):
    tint = exp_k_tint(k)
    assert tint.color == expected_color
    assert tint.label == expected_label


@pytest.mark.parametrize(
    "A, expected_color, expected_label",
    [
        (-1.0, FLAME, "invalid"),
        (0.0, FLAME, "invalid"),
        (3.0, EMBER, "small"),
        (20.0, STEAM, "typical"),
        (35.0, STEAM, "typical"),
        (60.0, EMBER, "large"),
    ],
)
def test_exp_amplitude_branches(A, expected_color, expected_label):
    tint = exp_amplitude_tint(A)
    assert tint.color == expected_color
    assert tint.label == expected_label


def test_render_chip_html_shape():
    """A chip must include all four pieces: label, value + unit, verdict, note."""
    html = render_chip_html(
        label="A (amplitude)",
        value="20.0",
        unit="µrad",
        tint=Tint(color=STEAM, label="typical", note="within normal range"),
    )
    assert 'class="kt-diag-chip"' in html
    assert "A (amplitude)" in html
    assert "20.0" in html
    assert "µrad" in html
    assert "typical" in html
    assert "within normal range" in html


def test_render_chip_html_applies_tint_color():
    html = render_chip_html(
        label="k",
        value="0.05",
        unit="/day",
        tint=Tint(color=FLAME, label="extreme", note="out of band"),
    )
    assert f"color: {FLAME}" in html
