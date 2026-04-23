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

from .model import DATE_COL, TILT_COL, CurveBand, Prediction, from_days, to_days
from .ui.palette import (
    FLAME,
    HALO,
    LAVA,
    STATE_COLOR,
    STEAM,
)

# Volcano palette applied to the chart. Locked to the UI palette so chip,
# banner, and chart all speak the same color language.
TILT_LINE_COLOR = "rgba(226, 232, 240, 0.85)"  # steam @ 85% — reads clean-white, not gray
PEAK_FIT_COLOR = LAVA                       # peaks-in-fit: brand accent (hot)
PEAK_OUT_COLOR = "rgba(100, 116, 139, 0.45)"  # ash, faded — peaks NOT in the fit
TRENDLINE_COLOR_DEFAULT = LAVA              # overridden per-state in build_figure
TRENDLINE_BAND_FILL = HALO                  # transparent lava gradient
EXP_COLOR = LAVA                            # dashed lava for current-episode fit
EXP_BAND_FILL = HALO
NEXT_EVENT_COLOR = FLAME                    # high-chroma red — the event itself
CONFIDENCE_BAND_FILL = "rgba(224, 55, 42, 0.18)"   # flame at low alpha
CONFIDENCE_BAND_LINE = "rgba(224, 55, 42, 0.55)"
NOW_LINE_COLOR = "rgba(226, 232, 240, 0.45)"       # steam at medium alpha
GRID_COLOR = "rgba(226, 232, 240, 0.07)"

# Per-source overlay colors (Phase 4 Commit 5). Each source traces a
# distinct translucent color on top of the merged line so the user can
# see which source contributed each region and where they disagree.
# Chosen to be distinguishable without shouting over the merged line.
SOURCE_OVERLAY_COLORS: dict[str, str] = {
    "two_day": "rgba(255, 193, 7, 0.55)",        # amber
    "week": "rgba(139, 195, 74, 0.55)",          # light green
    "month": "rgba(0, 188, 212, 0.55)",          # cyan
    "three_month": "rgba(156, 39, 176, 0.55)",   # purple
    "dec2024_to_now": "rgba(233, 30, 99, 0.55)", # pink
    "digital": "rgba(100, 221, 23, 0.75)",       # lime, slightly stronger
    "archive": "rgba(158, 158, 158, 0.45)",      # grey
}

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
    show_current_episode: bool = True,
    show_next_event_prediction: bool = True,
    per_source_overlay: Optional[dict[str, pd.DataFrame]] = None,
    state: Optional[str] = None,
) -> go.Figure:
    """Render the full prediction chart.

    Args:
        tilt_df:                    Full tilt history (`[Date, Tilt (microradians)]`).
        fit_peaks_df:               Peaks that fed the trendline fit — drawn as bright X.
        prediction:                 A `Prediction` from `model.predict`. Any
                                    field may be None when the underlying fit
                                    didn't converge.
        all_peaks_df:               Optional. All detected peaks, a superset of
                                    `fit_peaks_df`. Peaks NOT in the fit window
                                    are drawn as dimmed X markers so the user
                                    sees what was excluded.
        title:                      Plot title. Pass empty when Streamlit
                                    provides its own.
        show_current_episode:       When False, suppress the exponential
                                    "current episode" curve and its confidence
                                    band. The Streamlit layer flips this off
                                    once an eruption is actively underway —
                                    at that point the exp fit is modelling
                                    the inflation phase that just ended, so
                                    it would only mislead.
        show_next_event_prediction: When False, suppress the predicted-event
                                    star marker and its 80% confidence band
                                    rectangle. Streamlit flips this off when
                                    an eruption is confirmed to be active —
                                    at that point we shouldn't be telling
                                    the user when "the next" event is when
                                    one is happening on screen.
        per_source_overlay:         Optional `{source_name → DataFrame}` of
                                    each source's corrected tilt values. When
                                    provided, draws one translucent trace per
                                    source beneath the merged line so the
                                    user can see which source contributed
                                    each region and where they disagree.
                                    Phase 4 Commit 5 observability feature.
        state:                      Optional eruption state name (calm /
                                    starting / imminent / overdue / active).
                                    When provided, the trendline color shifts
                                    to the matching palette token so the chart
                                    echoes the hero chip's state color.
    """
    fig = go.Figure()
    trendline_color = STATE_COLOR.get(state, TRENDLINE_COLOR_DEFAULT) if state else TRENDLINE_COLOR_DEFAULT

    # ── 0a. episode shading: alternating subtle bands between consecutive
    #       detected peaks, so each eruption cycle reads as a discrete
    #       block of time. Drawn FIRST so everything else sits on top.
    _add_episode_shading(fig, all_peaks_df)

    # ── 0. per-source overlay (drawn first so merged line sits on top) ──────
    if per_source_overlay:
        for name, src_df in per_source_overlay.items():
            if src_df is None or len(src_df) == 0:
                continue
            color = SOURCE_OVERLAY_COLORS.get(name, "rgba(255, 255, 255, 0.35)")
            fig.add_trace(
                go.Scatter(
                    x=src_df[DATE_COL],
                    y=src_df[TILT_COL],
                    mode="lines",
                    name=f"source: {name}",
                    line=dict(color=color, width=1.0, dash="dot"),
                    visible="legendonly",  # opt-in via legend click; off by default
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        "%{x|%Y-%m-%d %H:%M}<br>"
                        "%{y:.2f} µrad"
                        "<extra></extra>"
                    ),
                )
            )

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

    # ── 3a. trendline 80% CI ribbon (drawn BEHIND the trendline) ────────────
    if prediction.trendline_band is not None:
        _add_band(
            fig,
            band=prediction.trendline_band,
            fillcolor=TRENDLINE_BAND_FILL,
            name="Trendline 80% CI",
        )

    # ── 3b. exp curve 80% CI ribbon (drawn BEHIND the exp curve) ────────────
    if show_current_episode and prediction.exp_band is not None:
        _add_band(
            fig,
            band=prediction.exp_band,
            fillcolor=EXP_BAND_FILL,
            name="Exp fit 80% CI",
        )

    # ── 3c. linear trendline ────────────────────────────────────────────────
    if prediction.trendline is not None:
        n = prediction.n_peaks_in_fit
        _add_curve(
            fig,
            curve=prediction.trendline,
            extent_end_day=extent_end_day,
            name=f"Trendline (last {n} peaks)",
            color=trendline_color,
            dash="dash",
        )

    # ── 4. exponential saturation curve ─────────────────────────────────────
    if show_current_episode and prediction.exp_curve is not None:
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
    if show_next_event_prediction and prediction.confidence_band is not None:
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
    if (
        show_next_event_prediction
        and prediction.next_event_date is not None
        and prediction.next_event_tilt is not None
    ):
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

    # ── 7a. vertical "now" line: strong past/future split ───────────────────
    # Use a Scatter trace rather than add_vline + annotation_text, because
    # the latter's shapeannotation helper averages two Timestamp endpoints
    # via sum()/len() — which fails with recent pandas (Timestamp + int).
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    if len(tilt_df) > 0:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=now,
            x1=now,
            y0=0,
            y1=1,
            line=dict(color=NOW_LINE_COLOR, width=1.2, dash="dot"),
            layer="below",
        )
        fig.add_annotation(
            x=now,
            y=1.0,
            xref="x",
            yref="paper",
            text="now",
            showarrow=False,
            yanchor="bottom",
            font=dict(color=STEAM, size=10),
        )

    # ── 7b. annotate the most recent peak and the predicted intersection ────
    if len(fit_peaks_df) > 0:
        last = fit_peaks_df.iloc[-1]
        fig.add_annotation(
            x=last[DATE_COL],
            y=last[TILT_COL],
            text=f"last pulse · {pd.Timestamp(last[DATE_COL]).strftime('%b %-d')}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor=LAVA,
            ax=0,
            ay=-28,
            font=dict(color=STEAM, size=11),
            bgcolor="rgba(30, 37, 55, 0.85)",  # basalt-ish
            bordercolor=LAVA,
            borderwidth=1,
            borderpad=3,
        )
    if (
        show_next_event_prediction
        and prediction.next_event_date is not None
        and prediction.next_event_tilt is not None
    ):
        fig.add_annotation(
            x=prediction.next_event_date,
            y=prediction.next_event_tilt,
            text=(
                f"predicted next · "
                f"{pd.Timestamp(prediction.next_event_date).strftime('%b %-d')}"
            ),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowcolor=NEXT_EVENT_COLOR,
            ax=0,
            ay=-36,
            font=dict(color=STEAM, size=11),
            bgcolor="rgba(30, 37, 55, 0.85)",
            bordercolor=NEXT_EVENT_COLOR,
            borderwidth=1,
            borderpad=3,
        )

    # ── 7c. default zoom: recent history + projection horizon ───────────────
    x_range = _default_x_range(tilt_df, fit_peaks_df, prediction)

    layout_kwargs = dict(
        xaxis_title="Date",
        yaxis_title="Electronic tilt — UWD station, azimuth 300° (µrad)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        # Legend lives BELOW the plot (not floating over the data). When it
        # sat top-right inside the plot it covered the rising tail of the
        # history line — exactly the region the user is trying to read.
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            bgcolor="rgba(0, 0, 0, 0)",
            font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=40, b=120),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            showgrid=True,
            tickformat="%b %-d",
            minor=dict(showgrid=True, gridcolor=GRID_COLOR),
        ),
        yaxis=dict(gridcolor=GRID_COLOR),
    )
    if x_range is not None:
        # Preserve the dtick/minor/gridcolor config when we attach the range.
        layout_kwargs["xaxis"] = {**layout_kwargs["xaxis"], "range": x_range}
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


EPISODE_SHADE_FILL = "rgba(226, 232, 240, 0.035)"  # steam @ 3.5% — very subtle


def _add_episode_shading(
    fig: go.Figure, all_peaks_df: Optional[pd.DataFrame]
) -> None:
    """Shade every-other span between consecutive detected peaks.

    An episode runs peak-to-peak (each band is one inflation+deflation
    cycle). Alternating shades delineate cycles without adding any new
    legend entries or shouting over the data.
    """
    if all_peaks_df is None or len(all_peaks_df) < 2:
        return
    peak_dates = sorted(all_peaks_df[DATE_COL].tolist())
    for i in range(len(peak_dates) - 1):
        if i % 2 == 0:
            continue  # un-shaded = even episodes; shaded = odd
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=peak_dates[i],
            x1=peak_dates[i + 1],
            y0=0,
            y1=1,
            fillcolor=EPISODE_SHADE_FILL,
            line_width=0,
            layer="below",
        )


def _add_band(
    fig: go.Figure,
    *,
    band: CurveBand,
    fillcolor: str,
    name: str,
) -> None:
    """Render a CurveBand as a filled ribbon between hi and lo curves."""
    dates = [from_days(d) for d in band.days]
    # Plotly's "fill='toself'" needs a closed polygon: forward along hi,
    # then back along lo reversed.
    fig.add_trace(
        go.Scatter(
            x=dates + dates[::-1],
            y=list(band.hi) + list(band.lo)[::-1],
            fill="toself",
            fillcolor=fillcolor,
            line=dict(width=0),
            name=name,
            hoverinfo="skip",
            showlegend=True,
        )
    )


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
