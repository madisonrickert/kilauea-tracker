"""Hero block — one dramatic answer above the fold.

Replaces the three-column metric row + banner + aviation badge stack with a
single big hero: state chip, "Next pulse in N DAYS" headline, confidence
window subhead, plain-English gloss.

The copy formatter is a pure function so it's unit-testable across every
branch (in-range, overdue, active, no-prediction). The ``show`` entry point
emits the HTML through Streamlit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from .palette import STATE_COLOR

# Default lookback for the hero sparkline. 30 days gives the visitor one full
# eruption cycle of context without needing to zoom.
SPARKLINE_DAYS = 30


@dataclass(frozen=True)
class HeroCopy:
    eyebrow: str          # small-caps label above the headline, e.g. "Next pulse in"
    headline: str         # dramatic number/phrase, e.g. "5 DAYS", "Right now", "—"
    subhead: str          # confidence window, e.g. "±2 days · Apr 25–29" (may be "")
    state: str            # canonical state name — drives the chip color
    state_label: str      # chip text — capitalized, human-readable


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)


def _fmt_date_short(ts: pd.Timestamp) -> str:
    """``Apr 25`` — compact, matches the subhead band format."""
    return ts.strftime("%b %-d") if hasattr(ts, "strftime") else str(ts)


def _band_range_str(lo: pd.Timestamp, hi: pd.Timestamp) -> str:
    """Render a confidence band as ``Apr 25–29`` or ``Apr 29–May 3``."""
    lo_month = lo.strftime("%b")
    hi_month = hi.strftime("%b")
    if lo_month == hi_month:
        return f"{lo_month} {lo.day}–{hi.day}"
    return f"{_fmt_date_short(lo)}–{_fmt_date_short(hi)}"


def _days_until(ts: pd.Timestamp, now: pd.Timestamp) -> float:
    return (ts - now).total_seconds() / 86400.0


def _days_phrase(days: float) -> str:
    """``5 days`` / ``1 day`` / ``0 days`` (today) / ``tomorrow``."""
    whole = int(round(days))
    if whole < 0:
        return f"{abs(whole)} days"
    if whole == 0:
        return "today"
    if whole == 1:
        return "tomorrow"
    return f"{whole} days"


def _band_half_width_days(band: tuple[pd.Timestamp, pd.Timestamp]) -> float:
    lo, hi = band
    return (hi - lo).total_seconds() / 86400.0 / 2.0


def compose(
    state: str,
    prediction: Optional[object],
    *,
    now: Optional[pd.Timestamp] = None,
) -> HeroCopy:
    """Pure formatter: state + prediction → (eyebrow, headline, subhead, state).

    ``prediction`` is a ``model.Prediction`` or None. We duck-type the two
    fields we need (``next_event_date``, ``confidence_band``) so tests don't
    need to import the full Prediction dataclass.
    """
    current = now if now is not None else _now_utc()
    state_label = state.upper() if state == "active" else state.capitalize()

    # "Active" — nothing to count down to.
    if state == "active":
        return HeroCopy(
            eyebrow="Status",
            headline="Right now",
            subhead="Eruption in progress — watch the live webcams.",
            state=state,
            state_label="ACTIVE",
        )

    next_event = getattr(prediction, "next_event_date", None) if prediction else None
    band = getattr(prediction, "confidence_band", None) if prediction else None

    # No model output at all — either trendline or exp fit failed to converge.
    if next_event is None and band is None:
        return HeroCopy(
            eyebrow="Next pulse",
            headline="—",
            subhead="Model has no prediction yet. Need more peaks in the window.",
            state=state,
            state_label=state_label,
        )

    # Overdue — past the high end of the band.
    if state == "overdue":
        if band is not None:
            _, hi = band
            overdue_days = _days_until(current, hi)
            return HeroCopy(
                eyebrow="Overdue by",
                headline=_days_phrase(overdue_days),
                subhead=f"Predicted window ended {_fmt_date_short(hi)}.",
                state=state,
                state_label=state_label,
            )
        if next_event is not None:
            overdue_days = _days_until(current, next_event)
            return HeroCopy(
                eyebrow="Overdue by",
                headline=_days_phrase(overdue_days),
                subhead=f"Predicted {_fmt_date_short(next_event)}.",
                state=state,
                state_label=state_label,
            )

    # Imminent — inside the band.
    if state == "imminent":
        headline = "Any time now"
        subhead = (
            f"Inside the {_band_range_str(*band)} confidence window."
            if band is not None
            else (f"Predicted {_fmt_date_short(next_event)}." if next_event else "")
        )
        return HeroCopy(
            eyebrow="Next pulse",
            headline=headline,
            subhead=subhead,
            state=state,
            state_label=state_label,
        )

    # Calm / starting — count down to the predicted date.
    if next_event is not None:
        days = _days_until(next_event, current)
        headline = _days_phrase(days).upper()
        if band is not None:
            half = _band_half_width_days(band)
            subhead = f"±{half:.0f} days · {_band_range_str(*band)}"
        else:
            subhead = _fmt_date_short(next_event)
        return HeroCopy(
            eyebrow="Next pulse in" if days >= 0 else "Next pulse was due",
            headline=headline,
            subhead=subhead,
            state=state,
            state_label=state_label,
        )

    # Band only, no point estimate — render the window.
    if band is not None:
        return HeroCopy(
            eyebrow="Next pulse window",
            headline=_band_range_str(*band),
            subhead="No point estimate — using the confidence band.",
            state=state,
            state_label=state_label,
        )

    # Unreachable, but keep the type stable.
    return HeroCopy(
        eyebrow="Next pulse",
        headline="—",
        subhead="",
        state=state,
        state_label=state_label,
    )


def render_html(copy: HeroCopy) -> str:
    """Pure HTML so tests can assert structure without Streamlit."""
    accent = STATE_COLOR.get(copy.state, STATE_COLOR["calm"])
    return (
        f'<div class="kt-hero">'
        f'<div class="kt-hero__eyebrow">'
        f'<span class="kt-chip" role="status" style="--chip-bg: {accent};">'
        f'{copy.state_label}'
        f'</span> '
        f'{copy.eyebrow}'
        f'</div>'
        f'<h1 class="kt-hero__headline">{copy.headline}</h1>'
        f'<div class="kt-hero__subhead">{copy.subhead}</div>'
        f'</div>'
    )


def build_sparkline(
    tilt_df: pd.DataFrame,
    state: str,
    prediction: Optional[object] = None,
    *,
    n_days: int = SPARKLINE_DAYS,
    height: int = 240,
) -> Optional[go.Figure]:
    """Compact last-N-days tilt line + predicted next-event marker.

    Reads as a visual fingerprint of recent activity rather than a
    readable chart: the shape of the curve conveys "flat / rising /
    just erupted" at a glance. Color picks up the current state so the
    hero chip, banner accent, and sparkline all speak the same signal.

    When ``prediction`` carries a ``next_event_date``, the x-axis extends
    past the last sample, a faint confidence band shades the predicted
    window, and a star marker pins the predicted date — so the "Next
    pulse in N DAYS" headline is visibly tied to a point on the timeline
    rather than just a number.

    Returns None when there's nothing to plot (no history, or all
    points fall outside the window).
    """
    from ..model import DATE_COL, TILT_COL

    if tilt_df is None or len(tilt_df) == 0:
        return None
    last = pd.Timestamp(tilt_df[DATE_COL].max())
    cutoff = last - pd.Timedelta(days=n_days)
    window = tilt_df[tilt_df[DATE_COL] >= cutoff]
    if len(window) < 2:
        return None

    line_color = STATE_COLOR.get(state, STATE_COLOR["calm"])
    fill_color = _rgba(line_color, alpha=0.18)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=window[DATE_COL],
                y=window[TILT_COL],
                mode="lines",
                line=dict(color=line_color, width=2),
                fill="tozeroy",
                fillcolor=fill_color,
                hoverinfo="skip",
                showlegend=False,
            )
        ]
    )

    # Predicted-event marker + confidence band, so the sparkline reads as
    # a compact preview of the main chart. The main-chart prediction logic
    # decides what to show based on eruption state; here we mirror that:
    # while a fountain is ACTIVE we hide the "next event" marker (it's
    # happening on screen right now) and the same field is also suppressed
    # by the Prediction layer in that case.
    next_date = getattr(prediction, "next_event_date", None) if prediction else None
    conf_band = getattr(prediction, "confidence_band", None) if prediction else None
    hide_prediction = state == "active"

    last_y = float(window[TILT_COL].iloc[-1])
    y_min = float(window[TILT_COL].min())
    y_max = float(window[TILT_COL].max())
    y_span = max(y_max - y_min, 1.0)

    if not hide_prediction and next_date is not None:
        last_date = pd.Timestamp(window[DATE_COL].iloc[-1])
        next_ts = pd.Timestamp(next_date)

        # Dashed "anticipated trajectory" from the last sample to the
        # predicted pulse — communicates "the model expects the line to
        # keep rising toward here." A 10% nudge upward reinforces the
        # rising feel without pretending to be a real sample.
        projected_y = last_y + 0.1 * y_span
        fig.add_trace(
            go.Scatter(
                x=[last_date, next_ts],
                y=[last_y, projected_y],
                mode="lines",
                line=dict(color=line_color, width=1.5, dash="dot"),
                opacity=0.55,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Confidence band as a tapered lens (diamond) centered on the
        # predicted date and the projected y. A flat rectangle reads like
        # "it's some box" — a lens reads like "most likely here, less
        # likely at the edges" which is the shape of the actual posterior.
        if conf_band is not None:
            lo = pd.Timestamp(conf_band[0])
            hi = pd.Timestamp(conf_band[1])
            bulge = 0.12 * y_span  # vertical extent at the lens center
            # Polygon: (lo, mid_y) → (mid, top) → (hi, mid_y) → (mid, bottom) → close
            # Two concentric lenses at different alpha simulate a soft edge.
            for scale, alpha in ((1.0, 0.14), (0.6, 0.26)):
                b = bulge * scale
                x_poly = [lo, next_ts, hi, next_ts, lo]
                y_poly = [
                    projected_y,
                    projected_y + b,
                    projected_y,
                    projected_y - b,
                    projected_y,
                ]
                fig.add_trace(
                    go.Scatter(
                        x=x_poly,
                        y=y_poly,
                        mode="lines",
                        line=dict(color=line_color, width=0),
                        fill="toself",
                        fillcolor=_rgba(line_color, alpha=alpha),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        # Star marker last so it sits on top of the lens + trajectory.
        fig.add_trace(
            go.Scatter(
                x=[next_ts],
                y=[projected_y],
                mode="markers",
                marker=dict(
                    symbol="star",
                    color=line_color,
                    size=15,
                    line=dict(color="rgba(15, 20, 25, 0.6)", width=1),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Extend the x-axis forward so the prediction marker has room to sit
    # past the last sample — otherwise Plotly's autorange clips it tight
    # against the rising tail of the history line.
    x_start = cutoff
    x_end = last + pd.Timedelta(days=2)
    if not hide_prediction:
        if conf_band is not None:
            x_end = max(x_end, pd.Timestamp(conf_band[1]) + pd.Timedelta(days=1))
        if next_date is not None:
            x_end = max(x_end, pd.Timestamp(next_date) + pd.Timedelta(days=1))

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, fixedrange=True, range=[x_start, x_end]),
        yaxis=dict(visible=False, fixedrange=True),
    )
    return fig


def render_caption_html(copy: HeroCopy) -> str:
    """Compact single-card caption rendered BELOW the sparkline hero.

    Before: a 5rem headline dominated the page and the sparkline sat
    below it as a subordinate. After the sparkline-as-hero redesign, the
    visual is the curve; the text summarizes it on one line with a chip
    for state color, an eyebrow label, and a condensed headline+subhead.
    """
    accent = STATE_COLOR.get(copy.state, STATE_COLOR["calm"])
    return (
        f'<div class="kt-hero-caption" role="status">'
        f'<span class="kt-chip" style="--chip-bg: {accent};">{copy.state_label}</span>'
        f'<span class="kt-hero-caption__eyebrow">{copy.eyebrow}</span>'
        f'<span class="kt-hero-caption__headline">{copy.headline}</span>'
        f'<span class="kt-hero-caption__subhead">{copy.subhead}</span>'
        f'</div>'
    )


def _rgba(color: str, *, alpha: float) -> str:
    """Convert a ``#rrggbb`` palette token into an ``rgba(r,g,b,a)`` string."""
    c = color.lstrip("#")
    if len(c) != 6:
        return color  # already rgba(...) or unknown form — let Plotly handle it
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"


def show(
    state: str,
    prediction: Optional[object],
    tilt_df: Optional[pd.DataFrame] = None,
    *,
    sparkline_days: int = SPARKLINE_DAYS,
) -> None:
    """Render the hero — sparkline on top, compact caption beneath.

    The sparkline is the visual hero now: ~240 px tall, state-colored line
    with a tapered confidence lens + star marker + dashed trajectory. The
    text is a one-line supporting caption underneath, with the state chip,
    eyebrow label, condensed headline and subhead.
    """
    import streamlit as st

    copy = compose(state, prediction)

    # Sparkline first — it's the hero visual.
    if tilt_df is not None:
        spark = build_sparkline(tilt_df, state, prediction, n_days=sparkline_days)
        if spark is not None:
            st.plotly_chart(
                spark,
                width="stretch",
                config={"displayModeBar": False, "staticPlot": True},
                key="hero_sparkline",
            )

    # Supporting caption: chip + eyebrow + headline + subhead on one line.
    st.markdown(render_caption_html(copy), unsafe_allow_html=True)
