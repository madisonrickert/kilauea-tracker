"""Build the interactive Plotly figure for the Streamlit dashboard.

Replaces the matplotlib code at `legacy/eruption_projection.py:78-336`. Plotly
gives us hover tooltips, pan/zoom, no hardcoded axis limits, and the same dark
theme as the rest of the Streamlit app.

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
TILT_LINE_COLOR = "#79b8ff"           # cool blue for raw tilt
PEAK_FIT_COLOR = "#39d353"            # bright green X — peaks used for the fit
PEAK_OUT_COLOR = "rgba(57, 211, 83, 0.35)"  # dimmed — peaks NOT in the fit
TRENDLINE_COLOR = "#ff6b35"           # lava orange (matches .streamlit/config.toml)
EXP_COLOR = "#bc8cff"                 # violet for the exponential
NEXT_EVENT_COLOR = "#ff4d4d"
CONFIDENCE_BAND_FILL = "rgba(255, 77, 77, 0.22)"
CONFIDENCE_BAND_LINE = "rgba(255, 77, 77, 0.55)"

# Default zoom window when no user interaction has happened yet.
DEFAULT_ZOOM_HISTORY_DAYS = 90
DEFAULT_ZOOM_FUTURE_DAYS = 14
# Extra days of padding before the earliest peak that's feeding the trendline,
# so cranking the peak slider up doesn't pin a peak to the chart's left edge.
DEFAULT_ZOOM_PEAK_PADDING_DAYS = 7


def build_figure(
    tilt_df: pd.DataFrame,
    fit_peaks_df: pd.DataFrame,
    prediction: Prediction,
    *,
    all_peaks_df: Optional[pd.DataFrame] = None,
    title: str = "",
) -> go.Figure:
    """Render the full prediction chart.

    Args:
        tilt_df:       Full tilt history (`[Date, Tilt (microradians)]`).
        fit_peaks_df:  Peaks that fed the trendline fit — drawn as bright X.
        prediction:    A `Prediction` from `model.predict`. Any field may be
                       None when the underlying fit didn't converge.
        all_peaks_df:  Optional. All detected peaks, a superset of
                       `fit_peaks_df`. Peaks NOT in the fit window are drawn
                       as dimmed X markers so the user sees what was excluded.
        title:         Plot title. Pass empty when Streamlit provides its own.
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

    # ── 2. detected peaks (split into "in fit" and "not in fit") ────────────
    fit_dates: set[pd.Timestamp] = (
        set(fit_peaks_df[DATE_COL]) if len(fit_peaks_df) > 0 else set()
    )
    if all_peaks_df is not None and len(all_peaks_df) > 0:
        excluded = all_peaks_df[~all_peaks_df[DATE_COL].isin(fit_dates)]
        if len(excluded) > 0:
            fig.add_trace(
                go.Scatter(
                    x=excluded[DATE_COL],
                    y=excluded[TILT_COL],
                    mode="markers",
                    name="Excluded peaks",
                    marker=dict(
                        symbol="x",
                        color=PEAK_OUT_COLOR,
                        size=10,
                        line=dict(width=1.5, color=PEAK_OUT_COLOR),
                    ),
                    hovertemplate=(
                        "Peak (excluded from fit): %{x|%Y-%m-%d %H:%M}<br>"
                        "<b>%{y:.2f}</b> µrad"
                        "<extra></extra>"
                    ),
                )
            )

    if len(fit_peaks_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=fit_peaks_df[DATE_COL],
                y=fit_peaks_df[TILT_COL],
                mode="markers",
                name=f"Peaks in fit ({len(fit_peaks_df)})",
                marker=dict(
                    symbol="x",
                    color=PEAK_FIT_COLOR,
                    size=14,
                    line=dict(width=2, color=PEAK_FIT_COLOR),
                ),
                hovertemplate=(
                    "Peak: %{x|%Y-%m-%d %H:%M}<br>"
                    "<b>%{y:.2f}</b> µrad"
                    "<extra></extra>"
                ),
            )
        )

    extent_end_day = _resolve_extent_end_day(tilt_df, prediction)

    # ── 3. linear trendline ─────────────────────────────────────────────────
    if prediction.trendline is not None:
        n = prediction.n_peaks_in_fit
        _add_curve(
            fig,
            curve=prediction.trendline,
            extent_end_day=extent_end_day,
            name=f"Trendline (last {n} peaks)",
            color=TRENDLINE_COLOR,
            dash="dash",
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

    # ── 5. confidence band — drawn BEFORE the event marker so the marker
    #      sits on top of the band, not under it ─────────────────────────────
    if prediction.confidence_band is not None:
        lo, hi = prediction.confidence_band
        fig.add_vrect(
            x0=lo,
            x1=hi,
            fillcolor=CONFIDENCE_BAND_FILL,
            line_width=0,
            layer="below",
        )
        for x_edge in (lo, hi):
            fig.add_vline(
                x=x_edge,
                line=dict(color=CONFIDENCE_BAND_LINE, width=1, dash="dot"),
                layer="below",
            )
        # Phantom trace so the band shows up in the legend with a width label.
        band_width_days = (hi - lo).total_seconds() / 86400.0
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[None, None],
                mode="lines",
                name=f"80% confidence ({band_width_days:.0f} days)",
                line=dict(color=CONFIDENCE_BAND_LINE, width=8),
                showlegend=True,
                hoverinfo="skip",
            )
        )

    # ── 6. predicted event marker ───────────────────────────────────────────
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

    # ── 7. default zoom: recent history + projection horizon ────────────────
    x_range = _default_x_range(tilt_df, fit_peaks_df, prediction)

    layout_kwargs = dict(
        xaxis_title="Date",
        yaxis_title="Electronic tilt — UWD station, azimuth 300° (µrad)",
        template="plotly_dark",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=30, t=40, b=60),
    )
    if x_range is not None:
        layout_kwargs["xaxis"] = dict(range=x_range)
    if title:
        layout_kwargs["title"] = title
    fig.update_layout(**layout_kwargs)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────


def _default_x_range(
    tilt_df: pd.DataFrame,
    fit_peaks_df: pd.DataFrame,
    prediction: Prediction,
) -> Optional[list]:
    """Pick a sensible default zoom: enough recent history to show every peak
    that's currently feeding the trendline, plus the projected event window.

    Baseline is "last 90 days," but if the user has cranked the peak count
    slider up so the earliest fit peak is older than that, the window
    expands backward to include it (plus a few days of padding so the peak
    isn't pinned to the edge).

    Returning None lets Plotly auto-fit (used when there's no useful anchor).
    """
    if len(tilt_df) == 0:
        return None

    end = tilt_df[DATE_COL].max()
    start = end - pd.Timedelta(days=DEFAULT_ZOOM_HISTORY_DAYS)

    # Expand backward if the earliest fit peak is older than the default window.
    if len(fit_peaks_df) > 0:
        earliest_fit_peak = fit_peaks_df[DATE_COL].min()
        padded_start = earliest_fit_peak - pd.Timedelta(days=DEFAULT_ZOOM_PEAK_PADDING_DAYS)
        if padded_start < start:
            start = padded_start

    if prediction.next_event_date is not None:
        end = max(end, prediction.next_event_date)
    if prediction.confidence_band is not None:
        end = max(end, prediction.confidence_band[1])
    end = end + pd.Timedelta(days=DEFAULT_ZOOM_FUTURE_DAYS)
    return [start, end]


def _resolve_extent_end_day(
    tilt_df: pd.DataFrame, prediction: Prediction
) -> Optional[float]:
    """Choose how far forward in time the trendline curves should extend."""
    candidates: list[float] = []
    if len(tilt_df) > 0:
        candidates.append(to_days(tilt_df[DATE_COL].max()) + 7.0)
    if prediction.next_event_date is not None:
        candidates.append(to_days(prediction.next_event_date) + 3.0)
    if prediction.confidence_band is not None:
        candidates.append(to_days(prediction.confidence_band[1]) + 3.0)
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
