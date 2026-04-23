"""Hero formatter — cover every prediction branch + render shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pytest

from kilauea_tracker.ui.hero import compose, render_html
from kilauea_tracker.ui.palette import STATE_COLOR


@dataclass
class _StubPrediction:
    """Minimal duck-type of ``model.Prediction`` — only the fields hero reads."""
    next_event_date: Optional[pd.Timestamp]
    confidence_band: Optional[tuple[pd.Timestamp, pd.Timestamp]]


NOW = pd.Timestamp("2026-04-22 00:00:00")


# ─────────────────────────────────────────────────────────────────────────────
# Branch coverage
# ─────────────────────────────────────────────────────────────────────────────


def test_calm_with_future_prediction_and_band():
    pred = _StubPrediction(
        next_event_date=pd.Timestamp("2026-04-27"),
        confidence_band=(pd.Timestamp("2026-04-25"), pd.Timestamp("2026-04-29")),
    )
    copy = compose("calm", pred, now=NOW)
    assert copy.headline == "5 DAYS"  # 2026-04-22 → 2026-04-27
    assert "±2 days" in copy.subhead
    assert "Apr 25–29" in copy.subhead
    assert copy.state == "calm"


def test_starting_same_shape_as_calm():
    pred = _StubPrediction(
        next_event_date=pd.Timestamp("2026-04-24"),
        confidence_band=(pd.Timestamp("2026-04-23"), pd.Timestamp("2026-04-25")),
    )
    copy = compose("starting", pred, now=NOW)
    assert copy.headline == "2 DAYS"
    assert copy.state == "starting"


def test_imminent_reads_any_time_now():
    pred = _StubPrediction(
        next_event_date=pd.Timestamp("2026-04-23"),
        confidence_band=(pd.Timestamp("2026-04-21"), pd.Timestamp("2026-04-25")),
    )
    copy = compose("imminent", pred, now=NOW)
    assert copy.headline == "Any time now"
    assert "Apr 21–25" in copy.subhead


def test_overdue_uses_overdue_headline():
    pred = _StubPrediction(
        next_event_date=pd.Timestamp("2026-04-18"),
        confidence_band=(pd.Timestamp("2026-04-16"), pd.Timestamp("2026-04-20")),
    )
    copy = compose("overdue", pred, now=NOW)
    assert copy.eyebrow == "Overdue by"
    # 2026-04-20 → 2026-04-22 = 2 days
    assert copy.headline == "2 days"


def test_active_shows_right_now():
    # Even with a prediction, active overrides everything.
    pred = _StubPrediction(
        next_event_date=pd.Timestamp("2026-04-27"),
        confidence_band=(pd.Timestamp("2026-04-25"), pd.Timestamp("2026-04-29")),
    )
    copy = compose("active", pred, now=NOW)
    assert copy.headline == "Right now"
    assert copy.state_label == "ACTIVE"


def test_no_prediction_shows_dash():
    copy = compose("calm", None, now=NOW)
    assert copy.headline == "—"
    assert "no prediction" in copy.subhead.lower()


def test_prediction_object_with_all_none_fields_shows_dash():
    pred = _StubPrediction(next_event_date=None, confidence_band=None)
    copy = compose("calm", pred, now=NOW)
    assert copy.headline == "—"


def test_today_special_case():
    pred = _StubPrediction(
        next_event_date=NOW,
        confidence_band=(NOW - pd.Timedelta(days=1), NOW + pd.Timedelta(days=1)),
    )
    copy = compose("calm", pred, now=NOW)
    assert copy.headline.lower().strip() == "today"


def test_tomorrow_special_case():
    pred = _StubPrediction(
        next_event_date=NOW + pd.Timedelta(days=1),
        confidence_band=(NOW, NOW + pd.Timedelta(days=2)),
    )
    copy = compose("calm", pred, now=NOW)
    assert copy.headline.lower().strip() == "tomorrow"


def test_band_only_prediction():
    pred = _StubPrediction(
        next_event_date=None,
        confidence_band=(pd.Timestamp("2026-04-25"), pd.Timestamp("2026-04-29")),
    )
    copy = compose("calm", pred, now=NOW)
    assert "Apr 25–29" in copy.headline


def test_cross_month_band_renders_correctly():
    pred = _StubPrediction(
        next_event_date=pd.Timestamp("2026-04-30"),
        confidence_band=(pd.Timestamp("2026-04-29"), pd.Timestamp("2026-05-03")),
    )
    copy = compose("calm", pred, now=NOW)
    # The band has both month names.
    assert "Apr" in copy.subhead and "May" in copy.subhead


# ─────────────────────────────────────────────────────────────────────────────
# render_html — structural shape + a11y
# ─────────────────────────────────────────────────────────────────────────────


def test_render_html_wraps_in_kt_hero():
    pred = _StubPrediction(
        next_event_date=pd.Timestamp("2026-04-27"),
        confidence_band=(pd.Timestamp("2026-04-25"), pd.Timestamp("2026-04-29")),
    )
    html = render_html(compose("calm", pred, now=NOW))
    assert '<div class="kt-hero">' in html
    assert '<h1 class="kt-hero__headline">' in html


def test_render_html_uses_correct_state_color():
    pred = _StubPrediction(
        next_event_date=pd.Timestamp("2026-04-27"),
        confidence_band=(pd.Timestamp("2026-04-25"), pd.Timestamp("2026-04-29")),
    )
    for state in ("calm", "starting", "imminent", "overdue", "active"):
        html = render_html(compose(state, pred, now=NOW))
        expected = STATE_COLOR[state]
        assert f"--chip-bg: {expected}" in html, f"state {state} missing color {expected}"


def test_chip_has_role_status():
    html = render_html(compose("starting", None, now=NOW))
    assert 'role="status"' in html


def test_headline_appears_inside_h1():
    pred = _StubPrediction(
        next_event_date=pd.Timestamp("2026-04-27"),
        confidence_band=None,
    )
    copy = compose("calm", pred, now=NOW)
    html = render_html(copy)
    assert f'<h1 class="kt-hero__headline">{copy.headline}</h1>' in html
