"""Build the interactive Plotly figure for the Streamlit dashboard.

Replaces the matplotlib code at `legacy/eruption_projection.py:78-336`. Plotly
gives us hover tooltips, pan/zoom, no hardcoded axis limits, and the same dark
theme as the rest of the Streamlit app.

`build_figure` is a pure function — same inputs always give the same figure.
The Streamlit layer is responsible for embedding it via `st.plotly_chart`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .model import DATE_COL, TILT_COL, from_days
from .ui.palette import (
    FLAME,
    HALO,
    LAVA,
    STATE_COLOR,
    STEAM,
)

if TYPE_CHECKING:
    from .models.output import ModelOutput, NamedCurve

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
    model_output: ModelOutput,
    *,
    all_peaks_df: pd.DataFrame | None = None,
    title: str = "",
    show_current_episode: bool = True,
    show_next_event_prediction: bool = True,
    per_source_overlay: dict[str, pd.DataFrame] | None = None,
    state: str | None = None,
) -> go.Figure:
    """Render the full prediction chart.

    Args:
        tilt_df:                    Full tilt history (`[Date, Tilt (microradians)]`).
        fit_peaks_df:               Peaks that fed the trendline fit — drawn as bright X.
        model_output:               A ``ModelOutput`` from any registered
                                    prediction model. Carries the curves to
                                    overlay, the predicted next-event date +
                                    tilt, and the 80% confidence band. Any
                                    field may be ``None`` when the underlying
                                    fit didn't converge.
        all_peaks_df:               Optional. All detected peaks, a superset of
                                    `fit_peaks_df`. Peaks NOT in the fit window
                                    are drawn as dimmed X markers so the user
                                    sees what was excluded.
        title:                      Plot title. Pass empty when Streamlit
                                    provides its own.
        show_current_episode:       When False, suppress any ribbon curves
                                    in ``model_output.curves`` whose label
                                    suggests they're the current-episode fit.
                                    The Streamlit layer flips this off once
                                    an eruption is actively underway — at
                                    that point the exp fit is modelling the
                                    inflation phase that just ended, so it
                                    would only mislead.
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

    # Drawn FIRST so every other trace sits on top.
    _add_episode_shading(fig, all_peaks_df)
    _add_per_source_overlay(fig, per_source_overlay)
    _add_tilt_trace(fig, tilt_df)
    _add_peak_markers(fig, fit_peaks_df, all_peaks_df)
    _add_curve_traces(
        fig, model_output,
        show_current_episode=show_current_episode,
        primary_color=trendline_color,
    )
    if show_next_event_prediction:
        _add_confidence_band(fig, model_output)
        _add_next_event_marker(fig, model_output, tilt_df)
    _add_now_line(fig, tilt_df)
    _add_event_annotations(
        fig, fit_peaks_df, model_output, tilt_df,
        show_next_event_prediction=show_next_event_prediction,
    )
    _apply_layout(fig, tilt_df, fit_peaks_df, model_output, title)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Internals — build_figure step helpers
#
# Each `_add_*` helper adds zero or more traces / shapes / annotations to
# `fig` and returns nothing. Splitting the build into named steps keeps
# the orchestration in `build_figure` readable and lets future per-step
# perf optimizations (e.g., caching individual traces) target a single
# function instead of carving up an inline block.
# ─────────────────────────────────────────────────────────────────────────────


def _add_per_source_overlay(
    fig: go.Figure,
    per_source_overlay: dict[str, pd.DataFrame] | None,
) -> None:
    """Translucent per-source traces, hidden behind the merged line.

    Off by default (``visible="legendonly"``) — power users opt in via the
    legend to see which source contributed each region.
    """
    if not per_source_overlay:
        return
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
                line={"color": color, "width": 1.0, "dash": "dot"},
                visible="legendonly",
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    "%{y:.2f} µrad"
                    "<extra></extra>"
                ),
            )
        )


def _add_tilt_trace(fig: go.Figure, tilt_df: pd.DataFrame) -> None:
    """Main reconciled tilt history trace."""
    if len(tilt_df) == 0:
        return
    fig.add_trace(
        go.Scatter(
            x=tilt_df[DATE_COL],
            y=tilt_df[TILT_COL],
            mode="lines+markers",
            name="Tilt",
            line={"color": TILT_LINE_COLOR, "width": 1.5},
            marker={"size": 3},
            hovertemplate=(
                "%{x|%Y-%m-%d %H:%M}<br>"
                "<b>%{y:.2f}</b> µrad"
                "<extra></extra>"
            ),
        )
    )


def _add_peak_markers(
    fig: go.Figure,
    fit_peaks_df: pd.DataFrame,
    all_peaks_df: pd.DataFrame | None,
) -> None:
    """Peaks split into "in fit" (bright X) and "excluded" (dimmed X)."""
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
                    marker={
                        "symbol": "x",
                        "color": PEAK_OUT_COLOR,
                        "size": 10,
                        "line": {"width": 1.5, "color": PEAK_OUT_COLOR},
                    },
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
                marker={
                    "symbol": "x",
                    "color": PEAK_FIT_COLOR,
                    "size": 14,
                    "line": {"width": 2, "color": PEAK_FIT_COLOR},
                },
                hovertemplate=(
                    "Peak: %{x|%Y-%m-%d %H:%M}<br>"
                    "<b>%{y:.2f}</b> µrad"
                    "<extra></extra>"
                ),
            )
        )


def _add_curve_traces(
    fig: go.Figure,
    model_output: ModelOutput,
    *,
    show_current_episode: bool,
    primary_color: str,
) -> None:
    """Model-declared overlay curves (trendline, exp, ribbons, ...).

    The model emits ``NamedCurve``s carrying style hints; ``render_named_curves``
    maps them to Plotly traces. We pre-filter "current episode" curves out
    when ``show_current_episode`` is False — once an eruption is active the
    exp fit is modelling an inflation phase that just ended, so it would
    only mislead.
    """
    visible_curves = list(model_output.curves)
    if not show_current_episode:
        visible_curves = [
            c for c in visible_curves
            if "exp" not in c.label.lower() and "episode" not in c.label.lower()
        ]
    render_named_curves(
        fig,
        visible_curves,
        primary_color=primary_color,
        ribbon_fill=TRENDLINE_BAND_FILL,
    )


def _add_confidence_band(fig: go.Figure, model_output: ModelOutput) -> None:
    """80% confidence band as a vrect + dotted edges + legend phantom trace.

    Drawn BEFORE the event marker so the marker sits on top of the band.
    Caller is responsible for the ``show_next_event_prediction`` gate.
    """
    if model_output.confidence_band is None:
        return
    lo, hi = model_output.confidence_band
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
            line={"color": CONFIDENCE_BAND_LINE, "width": 1, "dash": "dot"},
            layer="below",
        )
    band_width_days = (hi - lo).total_seconds() / 86400.0
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[None, None],
            mode="lines",
            name=f"80% confidence ({band_width_days:.0f} days)",
            line={"color": CONFIDENCE_BAND_LINE, "width": 8},
            showlegend=True,
            hoverinfo="skip",
        )
    )


def _add_next_event_marker(
    fig: go.Figure,
    model_output: ModelOutput,
    tilt_df: pd.DataFrame,
) -> None:
    """Star marker at the predicted next-event date.

    Models that only forecast a date (e.g. interval-median) leave
    ``next_event_tilt`` None — fall back via ``_resolve_marker_y`` to "the
    last observed tilt + a small upward nudge." Caller is responsible for
    the ``show_next_event_prediction`` gate.
    """
    if model_output.next_event_date is None:
        return
    marker_y = _resolve_marker_y(model_output, tilt_df)
    if marker_y is None:
        return
    fig.add_trace(
        go.Scatter(
            x=[model_output.next_event_date],
            y=[marker_y],
            mode="markers",
            name="Next fountain event",
            marker={
                "symbol": "star",
                "color": NEXT_EVENT_COLOR,
                "size": 22,
                "line": {"width": 2, "color": "black"},
            },
            hovertemplate=(
                f"<b>Next fountain event</b><br>"
                f"{model_output.next_event_date.strftime('%Y-%m-%d %H:%M')}<br>"
                f"Tilt: {marker_y:.2f} µrad"
                "<extra></extra>"
            ),
        )
    )


def _add_now_line(fig: go.Figure, tilt_df: pd.DataFrame) -> None:
    """Vertical "now" line that splits past from future.

    Uses ``add_shape`` rather than ``add_vline`` because the latter's
    shapeannotation helper averages two Timestamp endpoints via
    ``sum()/len()``, which fails with recent pandas (Timestamp + int).
    """
    if len(tilt_df) == 0:
        return
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=now,
        x1=now,
        y0=0,
        y1=1,
        line={"color": NOW_LINE_COLOR, "width": 1.2, "dash": "dot"},
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
        font={"color": STEAM, "size": 10},
    )


def _add_event_annotations(
    fig: go.Figure,
    fit_peaks_df: pd.DataFrame,
    model_output: ModelOutput,
    tilt_df: pd.DataFrame,
    *,
    show_next_event_prediction: bool,
) -> None:
    """Callout labels on the most recent peak and the predicted next event."""
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
            font={"color": STEAM, "size": 11},
            bgcolor="rgba(30, 37, 55, 0.85)",  # basalt-ish
            bordercolor=LAVA,
            borderwidth=1,
            borderpad=3,
        )
    if show_next_event_prediction and model_output.next_event_date is not None:
        annotation_y = _resolve_marker_y(model_output, tilt_df)
        if annotation_y is not None:
            fig.add_annotation(
                x=model_output.next_event_date,
                y=annotation_y,
                text=(
                    f"predicted next · "
                    f"{pd.Timestamp(model_output.next_event_date).strftime('%b %-d')}"
                ),
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowcolor=NEXT_EVENT_COLOR,
                ax=0,
                ay=-36,
                font={"color": STEAM, "size": 11},
                bgcolor="rgba(30, 37, 55, 0.85)",
                bordercolor=NEXT_EVENT_COLOR,
                borderwidth=1,
                borderpad=3,
            )


def _apply_layout(
    fig: go.Figure,
    tilt_df: pd.DataFrame,
    fit_peaks_df: pd.DataFrame,
    model_output: ModelOutput,
    title: str,
) -> None:
    """Final pass: axes, legend placement, default zoom range, optional title."""
    x_range = _default_x_range(tilt_df, fit_peaks_df, model_output)

    layout_kwargs = {
        "xaxis_title": "Date",
        "yaxis_title": "Electronic tilt — UWD station, azimuth 300° (µrad)",
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "hovermode": "closest",
        # Legend lives BELOW the plot (not floating over the data). When it
        # sat top-right inside the plot it covered the rising tail of the
        # history line — exactly the region the user is trying to read.
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0, 0, 0, 0)",
            "font": {"size": 11},
        },
        "margin": {"l": 60, "r": 30, "t": 40, "b": 120},
        "xaxis": {
            "gridcolor": GRID_COLOR,
            "showgrid": True,
            "tickformat": "%b %-d",
            "minor": {"showgrid": True, "gridcolor": GRID_COLOR},
        },
        "yaxis": {"gridcolor": GRID_COLOR},
    }
    if x_range is not None:
        # Preserve the dtick/minor/gridcolor config when we attach the range.
        layout_kwargs["xaxis"] = {**layout_kwargs["xaxis"], "range": x_range}
    if title:
        layout_kwargs["title"] = title
    fig.update_layout(**layout_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Internals — shared math + style helpers
# ─────────────────────────────────────────────────────────────────────────────


def _default_x_range(
    tilt_df: pd.DataFrame,
    fit_peaks_df: pd.DataFrame,
    model_output: ModelOutput,
) -> list | None:
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

    if model_output.next_event_date is not None:
        end = max(end, model_output.next_event_date)
    if model_output.confidence_band is not None:
        end = max(end, model_output.confidence_band[1])
    end = end + pd.Timedelta(days=DEFAULT_ZOOM_FUTURE_DAYS)
    return [start, end]


def _resolve_marker_y(
    model_output: ModelOutput, tilt_df: pd.DataFrame
) -> float | None:
    """Y-position for the predicted-event marker.

    When the model declares ``next_event_tilt`` (e.g. trendline×exp's
    intersection y-value), use it directly. Otherwise — for models like
    interval-median that only forecast a date — fall back to "the last
    observed tilt nudged 10% upward through the recent y-span," same
    convention the hero sparkline uses.
    """
    if model_output.next_event_tilt is not None:
        return model_output.next_event_tilt
    if len(tilt_df) == 0:
        return None
    last_tilt = float(tilt_df[TILT_COL].iloc[-1])
    y_min = float(tilt_df[TILT_COL].min())
    y_max = float(tilt_df[TILT_COL].max())
    y_span = max(y_max - y_min, 1.0)
    return last_tilt + 0.10 * y_span


EPISODE_SHADE_FILL = "rgba(226, 232, 240, 0.035)"  # steam @ 3.5% — very subtle


def _add_episode_shading(
    fig: go.Figure, all_peaks_df: pd.DataFrame | None
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


# ─────────────────────────────────────────────────────────────────────────────
# NamedCurve renderer — the boundary that lets ``models/`` stay Plotly-free
# ─────────────────────────────────────────────────────────────────────────────


# Plotly dash-style strings keyed by the public ``CurveStyle`` literal.
# Centralized so a future model author who sets ``style="dotted"`` doesn't
# have to know what string Plotly expects.
_DASH_STYLE_MAP: dict[str, str] = {
    "solid": "solid",
    "dashed": "dash",
    "dotted": "dot",
}

# Default colors for each ``ColorRole``. The chart already ties trendline
# color to eruption state via ``STATE_COLOR``; these are the fallbacks
# when no state-specific color applies.
_DEFAULT_PRIMARY_COLOR = LAVA
_DEFAULT_SECONDARY_COLOR = "rgba(226, 232, 240, 0.55)"
_DEFAULT_RIBBON_FILL = HALO


def render_named_curves(
    fig: go.Figure,
    curves: list[NamedCurve],
    *,
    primary_color: str | None = None,
    secondary_color: str | None = None,
    ribbon_fill: str | None = None,
) -> None:
    """Render each ``NamedCurve`` as a Plotly trace and add it to ``fig``.

    Maps the model layer's style hints to Plotly trace properties:

      - ``color_role="primary"`` line curves use ``primary_color`` (or the
        chart-default LAVA when none is provided).
      - ``color_role="secondary"`` line curves use the dimmer ``STEAM``
        token so they read as supporting context.
      - ``color_role="ribbon"`` curves expect ``band_lo`` and ``band_hi``
        populated; render as a single filled-band trace using the same
        ``fill="toself"`` polygon trick the legacy ``_add_band`` uses.
      - ``style`` ∈ {"solid", "dashed", "dotted"} → Plotly dash strings.

    Curves are added in input order; the chart-page caller is responsible
    for ordering them (ribbons typically before lines so lines draw on top).
    """
    primary = primary_color or _DEFAULT_PRIMARY_COLOR
    secondary = secondary_color or _DEFAULT_SECONDARY_COLOR
    fill = ribbon_fill or _DEFAULT_RIBBON_FILL

    for curve in curves:
        if curve.color_role == "ribbon":
            if curve.band_lo is None or curve.band_hi is None:
                # Ribbon role without bounds is a model bug — skip rather
                # than silently misrender as a line.
                continue
            dates = [from_days(d) for d in curve.days]
            fig.add_trace(
                go.Scatter(
                    x=dates + dates[::-1],
                    y=list(curve.band_hi) + list(curve.band_lo)[::-1],
                    fill="toself",
                    fillcolor=fill,
                    line={"width": 0},
                    name=curve.label,
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
            continue

        color = primary if curve.color_role == "primary" else secondary
        dates = [from_days(d) for d in curve.days]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=list(np.asarray(curve.values, dtype=float)),
                mode="lines",
                name=curve.label,
                line={
                    "color": color,
                    "dash": _DASH_STYLE_MAP[curve.style],
                    "width": 2.0,
                },
                hovertemplate=(
                    f"<b>{curve.label}</b><br>"
                    "%{x|%Y-%m-%d %H:%M}<br>"
                    "%{y:.2f} µrad"
                    "<extra></extra>"
                ),
            )
        )


