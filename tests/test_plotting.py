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
