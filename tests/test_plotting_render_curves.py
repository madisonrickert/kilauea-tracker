"""Tests for `render_named_curves` — the chart-side renderer that turns
a model's ``NamedCurve`` list into Plotly traces.

The renderer is the boundary that lets prediction models stay
Plotly-free. A new model declares ``[NamedCurve, ...]``; the renderer
maps style hints to Plotly trace properties and adds them to the
figure. These tests pin the mapping so future model authors can rely
on the contract.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from kilauea_tracker.models.output import NamedCurve
from kilauea_tracker.plotting import render_named_curves


def _solid_curve(label: str = "trend") -> NamedCurve:
    return NamedCurve(
        label=label,
        days=np.linspace(20000.0, 20100.0, 50),
        values=np.linspace(0.0, 5.0, 50),
        style="solid",
        color_role="primary",
    )


def _dashed_curve() -> NamedCurve:
    return NamedCurve(
        label="dashed",
        days=np.linspace(20000.0, 20100.0, 50),
        values=np.linspace(0.0, 5.0, 50),
        style="dashed",
        color_role="primary",
    )


def _ribbon_curve() -> NamedCurve:
    days = np.linspace(20000.0, 20100.0, 50)
    return NamedCurve(
        label="ribbon",
        days=days,
        values=np.linspace(0.0, 5.0, 50),
        color_role="ribbon",
        band_lo=np.linspace(-1.0, 4.0, 50),
        band_hi=np.linspace(1.0, 6.0, 50),
    )


def test_empty_curve_list_adds_no_traces():
    fig = go.Figure()
    render_named_curves(fig, [])
    assert len(fig.data) == 0


def test_solid_curve_adds_one_line_trace():
    fig = go.Figure()
    render_named_curves(fig, [_solid_curve()])
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.mode == "lines"
    assert trace.name == "trend"


def test_dashed_curve_uses_plotly_dash_style():
    fig = go.Figure()
    render_named_curves(fig, [_dashed_curve()])
    assert fig.data[0].line.dash == "dash"


def test_dotted_curve_uses_plotly_dot_style():
    curve = NamedCurve(
        label="dotted",
        days=np.linspace(20000.0, 20100.0, 50),
        values=np.linspace(0.0, 5.0, 50),
        style="dotted",
        color_role="primary",
    )
    fig = go.Figure()
    render_named_curves(fig, [curve])
    assert fig.data[0].line.dash == "dot"


def test_ribbon_renders_as_filled_band():
    """A ``color_role="ribbon"`` curve with band_lo/band_hi populated
    should render as a single filled-band trace (the Plotly ``fill='toself'``
    polygon trick used by the legacy ``_add_band``)."""
    fig = go.Figure()
    render_named_curves(fig, [_ribbon_curve()])
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.fill == "toself"
    assert trace.name == "ribbon"
    # Polygon = forward along hi then back along lo → 2N points.
    assert len(trace.x) == 100


def test_mixed_curves_preserve_order():
    fig = go.Figure()
    curves = [_ribbon_curve(), _solid_curve("primary line"), _dashed_curve()]
    render_named_curves(fig, curves)
    assert len(fig.data) == 3
    assert fig.data[0].name == "ribbon"
    assert fig.data[1].name == "primary line"
    assert fig.data[2].name == "dashed"


def test_curve_x_axis_uses_dates_not_days():
    """The chart's x-axis is dates; ``days`` floats need conversion to
    Timestamps before plotting. Verify the renderer does this."""
    import pandas as pd

    fig = go.Figure()
    render_named_curves(fig, [_solid_curve()])
    # Plotly turns the x-array into a tuple of Timestamps when given them;
    # accept either pandas Timestamp or datetime-like.
    first_x = fig.data[0].x[0]
    assert isinstance(first_x, (pd.Timestamp, np.datetime64))
