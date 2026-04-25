"""Hero formatter — cover every prediction branch + render shape."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.ui.hero import build_sparkline, compose, render_html
from kilauea_tracker.ui.palette import STATE_COLOR


@dataclass
class _StubPrediction:
    """Minimal duck-type of ``model.Prediction`` — only the fields hero reads."""
    next_event_date: pd.Timestamp | None
    confidence_band: tuple[pd.Timestamp, pd.Timestamp] | None


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


# ─────────────────────────────────────────────────────────────────────────────
# Sparkline — recent-activity fingerprint rendered inside the hero card
# ─────────────────────────────────────────────────────────────────────────────


def _make_tilt_df(n_days: int = 30, points_per_day: int = 4) -> pd.DataFrame:
    """Build a synthetic tilt history with a gentle rising trend."""
    start = NOW - pd.Timedelta(days=n_days)
    dates = pd.date_range(start, NOW, periods=n_days * points_per_day)
    # A simple rising signal from 0 → 10 µrad so plotly has something to draw.
    values = [i * (10.0 / len(dates)) for i in range(len(dates))]
    return pd.DataFrame({DATE_COL: dates, TILT_COL: values})


def test_sparkline_returns_none_for_empty_history():
    empty = pd.DataFrame({DATE_COL: [], TILT_COL: []})
    assert build_sparkline(empty, "calm") is None


def test_sparkline_returns_none_for_single_sample():
    one = pd.DataFrame({DATE_COL: [NOW], TILT_COL: [1.0]})
    assert build_sparkline(one, "calm") is None


def test_sparkline_returns_figure_when_history_is_present():
    fig = build_sparkline(_make_tilt_df(), "calm")
    assert fig is not None
    # No axes, no legend — reads as a glyph, not a chart.
    assert fig.layout.xaxis.visible is False
    assert fig.layout.yaxis.visible is False
    # Sparkline is the hero visual now, so it's taller than the v1 120px.
    assert fig.layout.height >= 180


def test_sparkline_uses_state_color_for_line():
    fig_active = build_sparkline(_make_tilt_df(), "active")
    fig_calm = build_sparkline(_make_tilt_df(), "calm")
    # Line color must change with the state so the hero chip, banner, and
    # sparkline speak the same signal.
    assert fig_active.data[0].line.color == STATE_COLOR["active"]
    assert fig_calm.data[0].line.color == STATE_COLOR["calm"]
    assert fig_active.data[0].line.color != fig_calm.data[0].line.color


def test_sparkline_respects_n_days_window():
    # Build 90 days of history and ask for 14; only the tail should appear.
    df = _make_tilt_df(n_days=90, points_per_day=4)
    fig = build_sparkline(df, "calm", n_days=14)
    x = fig.data[0].x
    earliest = pd.Timestamp(x[0])
    assert earliest >= NOW - pd.Timedelta(days=15)  # generous slack for points_per_day boundaries


def test_sparkline_draws_prediction_star_when_provided():
    """Passing a prediction must add a star marker trace + extend the x-axis
    past the last sample so the star has visible headroom."""
    pred = _StubPrediction(
        next_event_date=pd.Timestamp("2026-04-27"),
        confidence_band=(pd.Timestamp("2026-04-25"), pd.Timestamp("2026-04-29")),
    )
    fig = build_sparkline(_make_tilt_df(), "calm", pred)
    star = [t for t in fig.data if getattr(t, "mode", None) == "markers"]
    assert len(star) == 1
    assert star[0].marker.symbol == "star"
    # The x-axis range extends past the last sample to include the
    # predicted event (otherwise the star would be clipped).
    x_end = pd.Timestamp(fig.layout.xaxis.range[1])
    assert x_end >= pd.Timestamp("2026-04-27")


def test_sparkline_draws_tapered_confidence_lens():
    """The confidence band renders as a tapered lens (Scatter fill=toself),
    NOT a flat full-height rect — the shape conveys "most likely here, less
    likely at the edges"."""
    pred = _StubPrediction(
        next_event_date=pd.Timestamp("2026-04-27"),
        confidence_band=(pd.Timestamp("2026-04-25"), pd.Timestamp("2026-04-29")),
    )
    fig = build_sparkline(_make_tilt_df(), "calm", pred)
    lens_traces = [t for t in fig.data if getattr(t, "fill", None) == "toself"]
    # Two concentric lenses (inner + outer) form the soft-edged band.
    assert len(lens_traces) >= 2, (
        "expected at least two fill=toself lens polygons for the confidence band"
    )
    # Legacy assertion lock: we must NOT regress back to a flat vrect
    # rectangle covering the whole band horizontally.
    full_height_rects = [
        s for s in (fig.layout.shapes or ())
        if getattr(s, "type", None) == "rect"
    ]
    assert full_height_rects == [], (
        "sparkline must not draw a flat full-height confidence rect"
    )


def test_sparkline_draws_dashed_trajectory_line():
    """A dashed trajectory line connects the last sample to the predicted
    pulse — communicates 'the model expects the rise to continue.'"""
    pred = _StubPrediction(
        next_event_date=pd.Timestamp("2026-04-27"),
        confidence_band=(pd.Timestamp("2026-04-25"), pd.Timestamp("2026-04-29")),
    )
    fig = build_sparkline(_make_tilt_df(), "calm", pred)
    dashed = [
        t for t in fig.data
        if getattr(t, "mode", None) == "lines"
        and getattr(getattr(t, "line", None), "dash", None) in ("dot", "dash", "dashdot")
    ]
    assert len(dashed) >= 1, "expected at least one dashed trajectory line"


def test_sparkline_suppresses_prediction_when_active():
    """While a fountain is ACTIVE we don't draw a 'next event' marker — the
    eruption is happening on screen, not in the future."""
    pred = _StubPrediction(
        next_event_date=pd.Timestamp("2026-04-27"),
        confidence_band=(pd.Timestamp("2026-04-25"), pd.Timestamp("2026-04-29")),
    )
    fig = build_sparkline(_make_tilt_df(), "active", pred)
    star = [t for t in fig.data if getattr(t, "mode", None) == "markers"]
    assert star == []


def test_sparkline_no_prediction_still_renders():
    """Passing ``None`` for prediction is fine — the sparkline still renders
    without the star/band."""
    fig = build_sparkline(_make_tilt_df(), "calm", None)
    assert fig is not None
    star = [t for t in fig.data if getattr(t, "mode", None) == "markers"]
    assert star == []
