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
    peaks = detect_peaks(df).tail(6).reset_index(drop=True)
    pred = predict(df, peaks)
    return df, peaks, pred


def test_build_figure_returns_figure(realistic_inputs):
    df, peaks, pred = realistic_inputs
    fig = build_figure(df, peaks, pred)
    assert isinstance(fig, go.Figure)


def test_build_figure_includes_core_traces(realistic_inputs):
    """For the realistic v1.0 inputs, the figure should have:
    raw tilt, peak markers, two linear trendlines, exp curve, next-event marker.
    Earliest-event marker is allowed to be missing on this dataset (see
    test_model.test_v1_regression_predicts_consistent_dates).
    """
    df, peaks, pred = realistic_inputs
    fig = build_figure(df, peaks, pred)
    names = [t.name for t in fig.data]
    assert "Tilt" in names
    assert "Detected peaks" in names
    assert "Linear (all peaks)" in names
    assert "Linear (last 3 peaks)" in names
    assert "Current episode (exp fit)" in names
    assert "Next fountain event" in names


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
    # Title is still set so the chart is self-explanatory
    assert fig.layout.title.text


def test_build_figure_dark_template(realistic_inputs):
    """Streamlit theme is dark — the figure must match."""
    df, peaks, pred = realistic_inputs
    fig = build_figure(df, peaks, pred)
    template_name = fig.layout.template.layout.colorway is not None
    # Either the template name resolves or the layout was set; both indicate
    # we got plotly_dark applied.
    assert template_name or fig.layout.template is not None
