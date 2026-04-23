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

from .palette import STATE_COLOR


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


def show(state: str, prediction: Optional[object]) -> None:
    """Render the hero inside the current Streamlit container."""
    import streamlit as st

    copy = compose(state, prediction)
    st.markdown(render_html(copy), unsafe_allow_html=True)
