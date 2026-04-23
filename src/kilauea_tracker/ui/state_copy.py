"""Per-state headline / explanation / guidance copy.

Centralizes what had been bespoke prose scattered across ``streamlit_app.py``
for each of the five eruption states. The state banner reads from this table
so every state renders the same three-part shape (icon + headline, plain
explainer, collapsible "what this means for you").

All fields are callables taking ``info`` (the diagnostics dict from
``_eruption_state``) and returning a rendered string, so numeric values from
the classification can land inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping


@dataclass(frozen=True)
class StateCopy:
    icon: str                              # single-char emoji or unicode
    headline: str                          # one-line verdict, bold in the banner
    explainer: Callable[[Mapping], str]    # 1–2 sentences, numbers formatted in
    guidance: str                          # "what this means for you", plain text


def _fmt(value, spec: str, fallback: str = "?") -> str:
    """Format a value through ``spec``; return fallback if value is None/NaN."""
    if value is None:
        return fallback
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return fallback


STATE_COPY: dict[str, StateCopy] = {
    "calm": StateCopy(
        icon="🟢",
        headline="Tilt rebuilding — no eruption signal",
        explainer=lambda info: (
            "Uēkahuna tilt is climbing steadily between pulses. The model is "
            "watching for the next peak-then-deflation cycle but nothing is "
            "imminent right now."
        ),
        guidance=(
            "No action needed. Check back in a day or two, or glance at the "
            "chart to see how the current rise compares to previous episodes."
        ),
    ),
    "starting": StateCopy(
        icon="🟠",
        headline="Possible deflation onset — watching",
        explainer=lambda info: (
            "Tilt has steepened to "
            f"**{_fmt(info.get('short_slope_microrad_per_hour'), '+.2f')} µrad/hour** "
            "over the last 30 minutes, up from "
            f"**{_fmt(info.get('long_slope_microrad_per_hour'), '+.2f')} µrad/hour** "
            "over 6 hours. Consistent with the very earliest stages of a "
            "fountain event, but small enough it could still fizzle."
        ),
        guidance=(
            "The live webcams are the best corroboration right now. Status "
            "will escalate to *Eruption active* if the deflation continues."
        ),
    ),
    "imminent": StateCopy(
        icon="🟠",
        headline="Eruption window open — possibly imminent",
        explainer=lambda info: (
            "Current time is inside the predicted 80% confidence band. Based "
            "on the trendline and the exponential-saturation fit of the "
            "current rise, the next fountain event is expected any time now."
        ),
        guidance=(
            "Worth checking the webcams if you're planning to watch live. "
            "The prediction is a probability, not a schedule — the actual "
            "event can land anywhere in the shaded band on the chart."
        ),
    ),
    "overdue": StateCopy(
        icon="🟡",
        headline="Eruption overdue — model or volcano is behind schedule",
        explainer=lambda info: (
            "Today is past the high end of the predicted confidence band. "
            "Either the next fountain event is running late, or the model "
            "needs more recent peaks to recompute."
        ),
        guidance=(
            "Either outcome is normal. Check the chart for the exponential "
            "fit — if it's still rising, the eruption is still ahead."
        ),
    ),
    "active": StateCopy(
        icon="🔴",
        headline="Eruption active right now",
        explainer=lambda info: (
            "Tilt is dropping at "
            f"**{_fmt(info.get('recent_slope_microrad_per_hour'), '+.2f')} µrad/hour** "
            f"(**{_fmt(info.get('drop_from_24h_max'), '.1f')} µrad** below the "
            "24-hour max). The deflation signature of a fountain event is "
            "unmistakable in the live data."
        ),
        guidance=(
            "The live webcams should be lighting up. Full feeds on the "
            "[USGS webcams page](https://www.usgs.gov/volcanoes/kilauea/summit-webcams)."
        ),
    ),
}


def get(state: str) -> StateCopy:
    """Lookup with a helpful error if an unexpected state slips in."""
    if state not in STATE_COPY:
        raise KeyError(
            f"unknown eruption state {state!r}; "
            f"expected one of {sorted(STATE_COPY)}"
        )
    return STATE_COPY[state]
