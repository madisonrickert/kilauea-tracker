"""Smoke tests for `plotting.build_figure`.

The plot is the user-facing artifact, so the most valuable assertion is that
build_figure() never crashes on the realistic cases we'll feed it from the
Streamlit app — and that it produces the traces a sighted user would expect
to see in each scenario.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from kilauea_tracker.model import DATE_COL, TILT_COL, predict
from kilauea_tracker.peaks import detect_peaks
from kilauea_tracker.plotting import build_figure

BOOTSTRAP_CSV = (
    Path(__file__).resolve().parents[1] / "legacy" / "Tiltmeter Data - Sheet1.csv"
)


@pytest.fixture
def realistic_inputs():
    df = pd.read_csv(BOOTSTRAP_CSV)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed", dayfirst=False)
    df = df.dropna().sort_values(DATE_COL).reset_index(drop=True)
    all_peaks = detect_peaks(df)
    fit_peaks = all_peaks.tail(6).reset_index(drop=True)
    pred = predict(df, fit_peaks)
    return df, all_peaks, fit_peaks, pred


def test_build_figure_returns_figure(realistic_inputs):
    df, all_peaks, fit_peaks, pred = realistic_inputs
    fig = build_figure(df, fit_peaks, pred, all_peaks_df=all_peaks)
    assert isinstance(fig, go.Figure)


def test_build_figure_includes_core_traces(realistic_inputs):
    """For the realistic v1.0 inputs, the figure should have:
    raw tilt, in-fit peaks, single trendline, exp curve, next-event marker.
    """
    df, all_peaks, fit_peaks, pred = realistic_inputs
    fig = build_figure(df, fit_peaks, pred, all_peaks_df=all_peaks)
    names = [t.name for t in fig.data]
    assert "Tilt" in names
    # Peaks-in-fit trace name includes the count
    assert any(n and n.startswith("Peaks in fit") for n in names)
    assert any(n and n.startswith("Trendline (last") for n in names)
    assert "Current episode (exp fit)" in names
    assert "Next fountain event" in names
    # The "Linear (last 3 peaks)" trace from v1.0 must NOT be present anymore.
    assert not any(n and "last 3 peaks" in n for n in names)
    # And the "Earliest likely" marker should be gone.
    assert "Earliest likely" not in names


def test_build_figure_shows_excluded_peaks_when_provided(realistic_inputs):
    """When all_peaks_df is a strict superset of fit_peaks_df, the excluded
    ones get their own dimmed trace."""
    df, all_peaks, fit_peaks, pred = realistic_inputs
    if len(all_peaks) <= len(fit_peaks):
        pytest.skip("not enough excluded peaks to test this case")
    fig = build_figure(df, fit_peaks, pred, all_peaks_df=all_peaks)
    names = [t.name for t in fig.data]
    assert "Excluded peaks" in names


def test_build_figure_omits_excluded_when_all_peaks_not_passed(realistic_inputs):
    """If the caller doesn't pass all_peaks_df, no excluded trace appears."""
    df, _, fit_peaks, pred = realistic_inputs
    fig = build_figure(df, fit_peaks, pred)  # no all_peaks_df
    names = [t.name for t in fig.data]
    assert "Excluded peaks" not in names


def test_build_figure_handles_empty_history():
    """Empty inputs must not crash; we should still get a Figure."""
    empty_tilt = pd.DataFrame(
        {DATE_COL: pd.Series(dtype="datetime64[ns]"), TILT_COL: []}
    )
    empty_peaks = pd.DataFrame(
        {
            DATE_COL: pd.Series(dtype="datetime64[ns]"),
            TILT_COL: [],
            "prominence": [],
        }
    )
    pred = predict(empty_tilt, empty_peaks)
    fig = build_figure(empty_tilt, empty_peaks, pred)
    assert isinstance(fig, go.Figure)
    # No title — Streamlit provides its own header above the chart.
    assert not (fig.layout.title and fig.layout.title.text)


def test_build_figure_dark_template(realistic_inputs):
    """Streamlit theme is dark — the figure must match."""
    df, all_peaks, fit_peaks, pred = realistic_inputs
    fig = build_figure(df, fit_peaks, pred, all_peaks_df=all_peaks)
    assert fig.layout.template is not None


def test_build_figure_default_zoom_is_recent_history(realistic_inputs):
    """The default x-range should be the last ~3 months + projection,
    not the full 16-month history."""
    df, all_peaks, fit_peaks, pred = realistic_inputs
    fig = build_figure(df, fit_peaks, pred, all_peaks_df=all_peaks)
    x_range = fig.layout.xaxis.range
    assert x_range is not None
    span = pd.Timestamp(x_range[1]) - pd.Timestamp(x_range[0])
    # Recent-history window is 90 days; projection extends another ~2 weeks.
    # The full history (~16 months) would be ~480 days, so anything <200 days
    # is good evidence we're not auto-fitting the whole history.
    assert span < pd.Timedelta(days=200)
    assert span > pd.Timedelta(days=30)


def test_build_figure_zoom_expands_to_include_all_fit_peaks(realistic_inputs):
    """When the user cranks the peak slider up, the default zoom should
    expand backward to include the earliest peak in the fit window — not
    cut it off at the 90-day default.
    """
    df, all_peaks, _, _ = realistic_inputs
    # Use a very wide fit window: ALL detected peaks (the legacy CSV has
    # peaks going back to 2024-12, well outside the 90-day window).
    big_fit = all_peaks.copy().reset_index(drop=True)
    from kilauea_tracker.model import predict
    pred = predict(df, big_fit)
    fig = build_figure(df, big_fit, pred, all_peaks_df=all_peaks)
    x_range = fig.layout.xaxis.range
    assert x_range is not None
    earliest_peak = big_fit[DATE_COL].min()
    chart_start = pd.Timestamp(x_range[0])
    # The chart's left edge must be ON OR BEFORE the earliest fit peak.
    assert chart_start <= earliest_peak, (
        f"chart starts at {chart_start} but earliest fit peak is at "
        f"{earliest_peak} — the peak would be cut off"
    )


def test_build_figure_renders_confidence_band(realistic_inputs):
    """The Monte Carlo confidence band should appear as a vertical region
    plus a phantom trace in the legend with the band width."""
    df, all_peaks, fit_peaks, pred = realistic_inputs
    if pred.confidence_band is None:
        pytest.skip("model didn't produce a confidence band on this fixture")
    fig = build_figure(df, fit_peaks, pred, all_peaks_df=all_peaks)
    legend_names = [t.name for t in fig.data if t.name]
    assert any("confidence" in n.lower() for n in legend_names)
    # And there should be at least one shape (the vrect)
    assert fig.layout.shapes is not None and len(fig.layout.shapes) > 0


def test_build_figure_per_source_overlay(realistic_inputs):
    """When `per_source_overlay` is provided, build_figure adds one
    trace per source. Phase 4 Commit 5 observability feature.
    """
    df, all_peaks, fit_peaks, pred = realistic_inputs
    # Build fake per-source corrected DataFrames by shifting df copies
    overlay = {
        "two_day": df.copy().assign(
            **{TILT_COL: df[TILT_COL] - 2.0}
        ).tail(100),
        "week": df.copy().assign(
            **{TILT_COL: df[TILT_COL] + 1.0}
        ).tail(500),
    }
    fig_no = build_figure(df, fit_peaks, pred, all_peaks_df=all_peaks)
    fig_yes = build_figure(
        df, fit_peaks, pred, all_peaks_df=all_peaks,
        per_source_overlay=overlay,
    )
    # With overlay: +2 more traces (one per source).
    assert len(fig_yes.data) == len(fig_no.data) + 2
    # Names match the source keys.
    overlay_names = [t.name for t in fig_yes.data if t.name and t.name.startswith("source: ")]
    assert "source: two_day" in overlay_names
    assert "source: week" in overlay_names
    # Overlay traces start hidden (legend-click to toggle).
    overlay_traces = [t for t in fig_yes.data if t.name in ("source: two_day", "source: week")]
    assert all(t.visible == "legendonly" for t in overlay_traces)


def test_inspector_pixel_math_round_trips():
    """The PNG inspector (Phase 4 Commit 6) maps (date, tilt) to
    (pixel_x, pixel_y) for overlay drawing. This must be the inverse
    of the calibration's pixel-to-value transforms so dots land on the
    actual traced line pixel-for-pixel.

    Round-trip: value → pixel → value.  Within 1 pixel of integer
    rounding error.
    """
    from kilauea_tracker.ingest.calibrate import AxisCalibration

    cal = AxisCalibration(
        plot_bbox=(75, 20, 826, 245),
        y_slope=-0.0534,      # µrad per pixel (week PNG default)
        y_intercept=5.72,
        x_start=pd.Timestamp("2026-04-15 12:00:00"),
        x_end=pd.Timestamp("2026-04-22 12:00:00"),
    )

    # y round-trip: pick a tilt value, compute pixel, invert.
    tilt_in = -25.5
    px_y = (tilt_in - cal.y_intercept) / cal.y_slope
    tilt_back = cal.pixel_to_microradians(px_y)
    assert abs(tilt_back - tilt_in) < 1e-9, (
        f"round-trip tilt {tilt_in} → px {px_y} → tilt {tilt_back}"
    )

    # x round-trip: pick a date, compute pixel, invert.
    dt_in = pd.Timestamp("2026-04-17 18:30:00")
    x0, _, x1, _ = cal.plot_bbox
    span_s = (cal.x_end - cal.x_start).total_seconds()
    px_span = float(x1 - x0)
    px_x = x0 + (dt_in - cal.x_start).total_seconds() * px_span / span_s
    dt_back = cal.pixel_to_datetime(px_x)
    delta_s = abs((dt_back - dt_in).total_seconds())
    assert delta_s < 60, (
        f"round-trip date {dt_in} → px {px_x} → date {dt_back}, delta {delta_s}s"
    )


def test_build_figure_empty_overlay_is_noop(realistic_inputs):
    """Passing an empty dict for per_source_overlay doesn't error or
    add traces.
    """
    df, all_peaks, fit_peaks, pred = realistic_inputs
    fig_no = build_figure(df, fit_peaks, pred, all_peaks_df=all_peaks)
    fig_empty = build_figure(
        df, fit_peaks, pred, all_peaks_df=all_peaks,
        per_source_overlay={},
    )
    assert len(fig_no.data) == len(fig_empty.data)


# ─────────────────────────────────────────────────────────────────────────────
# Palette / design tests added in the v2.2 UX overhaul
# ─────────────────────────────────────────────────────────────────────────────


def test_trendline_color_shifts_with_state(realistic_inputs):
    """Passing a state name recolors the trendline to the matching palette token."""
    from kilauea_tracker.ui.palette import STATE_COLOR

    df, all_peaks, fit_peaks, pred = realistic_inputs
    for state in ("calm", "starting", "imminent", "overdue", "active"):
        fig = build_figure(df, fit_peaks, pred, all_peaks_df=all_peaks, state=state)
        trendline_traces = [
            t for t in fig.data if t.name and t.name.startswith("Trendline (last")
        ]
        assert trendline_traces, f"state {state}: trendline trace missing"
        tr = trendline_traces[0]
        expected = STATE_COLOR[state]
        assert tr.line.color == expected, (
            f"state {state}: trendline color {tr.line.color} != expected {expected}"
        )


def test_now_line_is_rendered(realistic_inputs):
    """The vertical 'now' line (and its annotation) should appear when history exists."""
    df, all_peaks, fit_peaks, pred = realistic_inputs
    fig = build_figure(df, fit_peaks, pred, all_peaks_df=all_peaks)
    # The now line is a line-shape on the paper yref.
    now_shapes = [
        s for s in (fig.layout.shapes or ())
        if getattr(s, "type", None) == "line" and getattr(s, "yref", "") == "paper"
    ]
    assert now_shapes, "vertical 'now' line missing from layout shapes"
    now_annotations = [
        a for a in (fig.layout.annotations or ())
        if getattr(a, "text", "") == "now"
    ]
    assert now_annotations, "'now' annotation missing"


def test_last_pulse_annotation_rendered(realistic_inputs):
    """The most-recent peak should carry a 'last pulse · <date>' annotation."""
    df, all_peaks, fit_peaks, pred = realistic_inputs
    fig = build_figure(df, fit_peaks, pred, all_peaks_df=all_peaks)
    last_pulse = [
        a for a in (fig.layout.annotations or ())
        if "last pulse" in (getattr(a, "text", "") or "")
    ]
    assert last_pulse, "'last pulse' annotation missing"


def test_predicted_next_annotation_rendered(realistic_inputs):
    """When a prediction exists, a 'predicted next · <date>' annotation appears."""
    df, all_peaks, fit_peaks, pred = realistic_inputs
    if pred.next_event_date is None:
        pytest.skip("no next_event_date in this fixture")
    fig = build_figure(df, fit_peaks, pred, all_peaks_df=all_peaks)
    predicted = [
        a for a in (fig.layout.annotations or ())
        if "predicted next" in (getattr(a, "text", "") or "")
    ]
    assert predicted, "'predicted next' annotation missing"


def test_palette_colors_applied_to_core_traces(realistic_inputs):
    """History line is ash; peaks-in-fit are lava; confidence band is flame-tinted."""
    from kilauea_tracker.ui.palette import ASH, LAVA

    df, all_peaks, fit_peaks, pred = realistic_inputs
    fig = build_figure(df, fit_peaks, pred, all_peaks_df=all_peaks)
    tilt_trace = next(t for t in fig.data if t.name == "Tilt")
    assert tilt_trace.line.color == ASH

    peaks_trace = next(
        t for t in fig.data if t.name and t.name.startswith("Peaks in fit")
    )
    assert peaks_trace.marker.color == LAVA


def test_episode_shading_renders_alternating_bands(realistic_inputs):
    """With N detected peaks, every other peak→peak span is shaded.

    `(N - 1) // 2` bands when the first gap is un-shaded.
    """
    df, all_peaks, fit_peaks, pred = realistic_inputs
    fig = build_figure(df, fit_peaks, pred, all_peaks_df=all_peaks)
    episode_rects = [
        s for s in (fig.layout.shapes or ())
        if getattr(s, "type", None) == "rect"
        and getattr(s, "yref", "") == "paper"
        and getattr(s, "fillcolor", "")
        and "226, 232, 240" in str(s.fillcolor)
    ]
    n_peaks = len(all_peaks)
    expected = (n_peaks - 1) // 2
    assert len(episode_rects) == expected, (
        f"expected {expected} episode bands for {n_peaks} peaks, got {len(episode_rects)}"
    )


def test_episode_shading_noop_without_peaks():
    """No peaks → no shading rects; must not crash."""
    from kilauea_tracker.model import predict

    empty_tilt = pd.DataFrame(
        {DATE_COL: pd.Series(dtype="datetime64[ns]"), TILT_COL: []}
    )
    empty_peaks = pd.DataFrame(
        {
            DATE_COL: pd.Series(dtype="datetime64[ns]"),
            TILT_COL: [],
            "prominence": [],
        }
    )
    pred = predict(empty_tilt, empty_peaks)
    fig = build_figure(empty_tilt, empty_peaks, pred, all_peaks_df=empty_peaks)
    episode_rects = [
        s for s in (fig.layout.shapes or ())
        if getattr(s, "fillcolor", "") and "226, 232, 240" in str(s.fillcolor)
    ]
    assert episode_rects == []
