"""Build the interactive Plotly figure for the Streamlit dashboard.

Replaces the matplotlib code at `legacy/eruption_projection.py:78-336`. Plotly
gives us:
  - hover tooltips with exact tilt readings,
  - pan/zoom without re-running the model,
  - no hardcoded axis limits — the user explores the chart at any time scale,
  - the same dark theme as the rest of the Streamlit app.

`build_figure` is a pure function — same inputs always give the same figure.
The Streamlit layer is responsible for embedding it via `st.plotly_chart`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .model import DATE_COL, TILT_COL, Prediction, from_days, to_days

# Visual defaults — kept here so the Streamlit theme and the chart agree.
TILT_LINE_COLOR = "#79b8ff"        # cool blue for raw tilt
PEAK_MARKER_COLOR = "#39d353"      # green X glyphs
LINEAR_COLOR = "#ff6b35"           # lava orange (matches .streamlit/config.toml)
LINEAR3_COLOR = "#ffa657"          # softer orange for the steeper trendline
EXP_COLOR = "#bc8cff"              # violet for the exponential
NEXT_EVENT_COLOR = "#ff4d4d"
EARLIEST_EVENT_COLOR = "#ffd700"
CONFIDENCE_BAND_COLOR = "rgba(255, 77, 77, 0.18)"


def build_figure(
    tilt_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    prediction: Prediction,
    *,
    title: str = "Kīlauea Summit Electronic Tilt — Projection",
) -> go.Figure:
    """Render the full prediction chart.

    Args:
        tilt_df:    DataFrame `[Date, Tilt (microradians)]` — the full history.
        peaks_df:   DataFrame `[Date, Tilt (microradians), prominence]` from
                    `peaks.detect_peaks`. May be empty.
        prediction: A `Prediction` from `model.predict`. Any field may be None.
        title:      Plot title. Pass an empty string if Streamlit is providing
                    its own header.

    Returns:
        A `plotly.graph_objects.Figure` ready for `st.plotly_chart`.
    """
    fig = go.Figure()

    # ── 1. raw tilt ─────────────────────────────────────────────────────────
    if len(tilt_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=tilt_df[DATE_COL],
                y=tilt_df[TILT_COL],
                mode="lines+markers",
                name="Tilt",
                line=dict(color=TILT_LINE_COLOR, width=1.5),
                marker=dict(size=3),
                hovertemplate=(
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    "<b>%{y:.2f}</b> µrad"
                    "<extra></extra>"
                ),
            )
        )

    # ── 2. detected peaks ───────────────────────────────────────────────────
    if len(peaks_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=peaks_df[DATE_COL],
                y=peaks_df[TILT_COL],
                mode="markers",
                name="Detected peaks",
                marker=dict(
                    symbol="x",
                    color=PEAK_MARKER_COLOR,
                    size=14,
                    line=dict(width=2, color=PEAK_MARKER_COLOR),
                ),
                hovertemplate=(
                    "Peak: %{x|%Y-%m-%d %H:%M}<br>"
                    "<b>%{y:.2f}</b> µrad"
                    "<extra></extra>"
                ),
            )
        )

    # The forward extent for trendline visualization. We extend each curve to
    # reach the latest of: end of its own domain, the predicted intersection
    # date, or the latest tilt sample + 7 days. This keeps the projected lines
    # visually anchored to the data.
    extent_end_day = _resolve_extent_end_day(tilt_df, prediction)

    # ── 3. linear trendlines ────────────────────────────────────────────────
    if prediction.linear_curve is not None:
        _add_curve(
            fig,
            curve=prediction.linear_curve,
            extent_end_day=extent_end_day,
            name="Linear (all peaks)",
            color=LINEAR_COLOR,
            dash="dash",
        )
    if prediction.linear3_curve is not None:
        _add_curve(
            fig,
            curve=prediction.linear3_curve,
            extent_end_day=extent_end_day,
            name="Linear (last 3 peaks)",
            color=LINEAR3_COLOR,
            dash="dashdot",
        )

    # ── 4. exponential saturation curve ─────────────────────────────────────
    if prediction.exp_curve is not None:
        _add_curve(
            fig,
            curve=prediction.exp_curve,
            extent_end_day=extent_end_day,
            name="Current episode (exp fit)",
            color=EXP_COLOR,
            dash="solid",
            width=2.5,
        )

    # ── 5. predicted event markers ──────────────────────────────────────────
    if prediction.next_event_date is not None and prediction.next_event_tilt is not None:
        fig.add_trace(
            go.Scatter(
                x=[prediction.next_event_date],
                y=[prediction.next_event_tilt],
                mode="markers",
                name="Next fountain event",
                marker=dict(
                    symbol="star",
                    color=NEXT_EVENT_COLOR,
                    size=22,
                    line=dict(width=2, color="black"),
                ),
                hovertemplate=(
                    f"<b>Next fountain event</b><br>"
                    f"{prediction.next_event_date.strftime('%Y-%m-%d %H:%M')}<br>"
                    f"Tilt: {prediction.next_event_tilt:.2f} µrad"
                    "<extra></extra>"
                ),
            )
        )

    if (
        prediction.earliest_event_date is not None
        and prediction.earliest_event_tilt is not None
    ):
        fig.add_trace(
            go.Scatter(
                x=[prediction.earliest_event_date],
                y=[prediction.earliest_event_tilt],
                mode="markers",
                name="Earliest likely",
                marker=dict(
                    symbol="triangle-up",
                    color=EARLIEST_EVENT_COLOR,
                    size=18,
                    line=dict(width=2, color="black"),
                ),
                hovertemplate=(
                    f"<b>Earliest likely fountain event</b><br>"
                    f"{prediction.earliest_event_date.strftime('%Y-%m-%d %H:%M')}<br>"
                    f"Tilt: {prediction.earliest_event_tilt:.2f} µrad"
                    "<extra></extra>"
                ),
            )
        )

    # ── 6. confidence band (Monte Carlo over exp covariance) ────────────────
    if prediction.confidence_band is not None:
        lo, hi = prediction.confidence_band
        fig.add_vrect(
            x0=lo,
            x1=hi,
            fillcolor=CONFIDENCE_BAND_COLOR,
            line_width=0,
            annotation_text="Confidence band",
            annotation_position="top left",
        )

    fig.update_layout(
        title=title or None,
        xaxis_title="Date",
        yaxis_title="Electronic tilt — azimuth 300 (µrad)",
        template="plotly_dark",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_extent_end_day(
    tilt_df: pd.DataFrame, prediction: Prediction
) -> Optional[float]:
    """Choose how far forward in time the trendline curves should extend."""
    candidates: list[float] = []
    if len(tilt_df) > 0:
        candidates.append(to_days(tilt_df[DATE_COL].max()) + 7.0)
    if prediction.next_event_date is not None:
        candidates.append(to_days(prediction.next_event_date) + 3.0)
    if prediction.earliest_event_date is not None:
        candidates.append(to_days(prediction.earliest_event_date) + 3.0)
    if prediction.exp_curve is not None:
        candidates.append(prediction.exp_curve.domain[1])
    return max(candidates) if candidates else None


def _add_curve(
    fig: go.Figure,
    *,
    curve,
    extent_end_day: Optional[float],
    name: str,
    color: str,
    dash: str,
    width: float = 2.0,
) -> None:
    x_min, x_max = curve.domain
    if extent_end_day is not None and extent_end_day > x_max:
        x_max = extent_end_day

    days = np.linspace(x_min, x_max, 200)
    dates = [from_days(d) for d in days]
    values = [float(curve.f(d)) for d in days]

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            name=name,
            line=dict(color=color, dash=dash, width=width),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "%{x|%Y-%m-%d %H:%M}<br>"
                "%{y:.2f} µrad"
                "<extra></extra>"
            ),
        )
    )
