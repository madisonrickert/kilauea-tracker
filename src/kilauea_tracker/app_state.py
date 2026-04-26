"""Cached compute helpers shared by every Streamlit page.

Pages run independently in Streamlit's multipage model — each page is its
own top-level script. Without a shared accessor layer, every page would
either duplicate I/O + cache wiring or import private helpers from the
entrypoint. This module is the single source of truth for "the data the
app shows": tilt history, latest run report, detected peaks, prediction,
eruption-lifecycle classification, and safety alerts.

The module owns the Streamlit cache decorators (`@st.cache_data`) so the
caching strategy lives in one place. The pure compute layers it wraps —
``model.predict``, ``peaks.detect_peaks``, ``safety_alerts.fetch_safety_alerts``
— are unchanged and remain pure (no I/O, no clock, no module state).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .cache import load_history
from .config import HISTORY_CSV, INGEST_CACHE_TTL_SECONDS
from .ingest.pipeline import IngestRunResult, load_latest_run_report
from .model import DATE_COL, TILT_COL, Prediction, predict
from .peaks import detect_peaks
from .safety_alerts import SafetyAlertSummary, fetch_safety_alerts

# ─────────────────────────────────────────────────────────────────────────────
# Tilt history
# ─────────────────────────────────────────────────────────────────────────────


# show_spinner=False on purpose. The cache invalidates whenever
# `data/tilt_history.csv` is rewritten (cron pull or manual refresh), and
# until 2026-04 the default spinner here flashed a SECOND fullscreen
# overlay on top of the inline Refresh-button indicator — one visible
# overlay per Refresh click is enough. The CSV read itself is fast
# (~tens of ms); a cold-start delay lands inside Streamlit's own page-
# render skeleton and doesn't need a separate label.
@st.cache_data(show_spinner=False)
def _read_tilt_history_cached(path_str: str, mtime: float) -> pd.DataFrame:
    """Streamlit-cache shim. Mtime is the invalidation key."""
    return load_history(Path(path_str))


def _history_mtime() -> float:
    return HISTORY_CSV.stat().st_mtime if HISTORY_CSV.exists() else 0.0


def load_tilt_df() -> pd.DataFrame:
    """Read ``data/tilt_history.csv``. Cached on file mtime so a cron-driven
    cache update on disk forces a re-read on the next rerun."""
    return _read_tilt_history_cached(str(HISTORY_CSV), _history_mtime())


# ─────────────────────────────────────────────────────────────────────────────
# Latest run report (per-source ingest diagnostics)
# ─────────────────────────────────────────────────────────────────────────────


def load_run_report() -> IngestRunResult:
    """Most-recent ingest run report, or an empty placeholder if none exists.
    Not cached — the underlying loader is already cheap (single JSON read)."""
    return load_latest_run_report() or IngestRunResult()


# ─────────────────────────────────────────────────────────────────────────────
# Peak detection + prediction
# ─────────────────────────────────────────────────────────────────────────────


def get_peaks(
    tilt_df: pd.DataFrame,
    *,
    min_prominence: float,
    min_distance_days: float,
    min_height: float,
) -> pd.DataFrame:
    """Pass-through to ``peaks.detect_peaks``. Lives here so pages have a
    single entry point and so future caching changes (NOTES #2) only have
    to touch this module."""
    return detect_peaks(
        tilt_df,
        min_prominence=min_prominence,
        min_distance_days=min_distance_days,
        min_height=min_height,
    )


def get_recent_peaks(all_peaks: pd.DataFrame, n_peaks_for_fit: int) -> pd.DataFrame:
    return all_peaks.tail(n_peaks_for_fit).reset_index(drop=True)


def get_prediction(tilt_df: pd.DataFrame, recent_peaks: pd.DataFrame) -> Prediction:
    return predict(tilt_df, recent_peaks)


# ─────────────────────────────────────────────────────────────────────────────
# Eruption lifecycle state — drives the status banner above the chart
# ─────────────────────────────────────────────────────────────────────────────


# Slope thresholds for the "active deflation" detection.
# A typical fountain event drops 10-30 µrad over 6-14 hours = -1 to -3 µrad/h.
# Background noise is well under 0.1 µrad/h. -0.5 µrad/h is a comfortable
# midpoint that won't fire on quiet inflation but catches every real
# deflation event we have in the history.
_ACTIVE_DEFLATION_SLOPE_MICRORAD_PER_HOUR = -0.5
# How much below the recent peak the current value must sit before we
# consider the eruption "active." Filters out tiny zero-crossings during
# background noise that happen to be paired with a momentary negative slope.
_ACTIVE_DEFLATION_MIN_DROP_MICRORAD = 2.0
# How far back to look for the recent peak when checking the drop.
_ACTIVE_DEFLATION_LOOKBACK_HOURS = 24
# How many of the trailing samples feed the active-state slope fit.
_RECENT_SLOPE_WINDOW_HOURS = 3.0

# "Starting" state thresholds — fires when something MIGHT be happening but
# we can't yet confirm a fountain event. The state is intentionally hedged:
# the banner will say "possible deflation onset" rather than declaring an
# eruption. Real episode 44 telemetry: at the moment USGS officially
# announced the eruption (T+0), the 0.5h slope was -0.21 µrad/h while the
# 6h slope was -0.015 µrad/h — i.e. the slope had just steepened ~13×.
# That ratio is the smoking gun for an early deflation; the absolute slope
# is still well under the active threshold.
_STARTING_SHORT_WINDOW_HOURS = 0.5
_STARTING_LONG_WINDOW_HOURS = 6.0
_STARTING_SHORT_SLOPE_MAX = -0.15  # short-window slope must be steeper than this
# Short-window slope must be at least this much MORE NEGATIVE than the
# long-window slope. Catches "the slope just steepened" without firing
# whenever both windows happen to be quietly negative.
_STARTING_SLOPE_ACCELERATION_MICRORAD_PER_HOUR = 0.10


def _recent_slope_microrad_per_hour(df: pd.DataFrame, hours: float) -> float | None:
    """Linear-fit slope of the last ``hours`` of tilt data, in µrad/hour.

    Returns None if there aren't enough samples to fit a line.
    """
    if len(df) < 3:
        return None
    end = df[DATE_COL].max()
    start = end - pd.Timedelta(hours=hours)
    window = df[df[DATE_COL] >= start]
    if len(window) < 3:
        return None
    x = (window[DATE_COL] - window[DATE_COL].min()).dt.total_seconds().to_numpy() / 3600.0
    y = window[TILT_COL].to_numpy()
    if len(np.unique(x)) < 2:
        return None
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _drop_from_recent_max(df: pd.DataFrame, lookback_hours: float) -> float | None:
    """How far below the recent max the current value sits, in µrad.

    Positive return value = current is BELOW the recent max (we're dropping).
    """
    if len(df) == 0:
        return None
    end = df[DATE_COL].max()
    start = end - pd.Timedelta(hours=lookback_hours)
    window = df[df[DATE_COL] >= start]
    if len(window) == 0:
        return None
    return float(window[TILT_COL].max() - window[TILT_COL].iloc[-1])


def get_eruption_state(
    tilt_df: pd.DataFrame,
    prediction: Prediction | None,
) -> tuple[str, dict]:
    """Classify the current point in the eruption lifecycle.

    Returns ``(state, info)`` where state is one of:
        "active"   — sharp sustained negative slope → eruption confirmed
        "starting" — short-window slope has steepened relative to a longer
                     window, suggesting deflation may be beginning. The
                     classification is intentionally hedged because tilt
                     can also dip briefly without a fountain following.
        "imminent" — current time is inside the predicted confidence band
        "overdue"  — current time is past the high end of the band
        "calm"     — none of the above; building toward the next eruption

    ``info`` carries the diagnostics that fed the classification (slopes,
    drop, predicted dates) so the banner can quote them.
    """
    info: dict = {}

    slope = _recent_slope_microrad_per_hour(tilt_df, _RECENT_SLOPE_WINDOW_HOURS)
    short_slope = _recent_slope_microrad_per_hour(
        tilt_df, _STARTING_SHORT_WINDOW_HOURS
    )
    long_slope = _recent_slope_microrad_per_hour(
        tilt_df, _STARTING_LONG_WINDOW_HOURS
    )
    drop = _drop_from_recent_max(tilt_df, _ACTIVE_DEFLATION_LOOKBACK_HOURS)
    info["recent_slope_microrad_per_hour"] = slope
    info["short_slope_microrad_per_hour"] = short_slope
    info["long_slope_microrad_per_hour"] = long_slope
    info["drop_from_24h_max"] = drop

    if (
        slope is not None
        and drop is not None
        and slope < _ACTIVE_DEFLATION_SLOPE_MICRORAD_PER_HOUR
        and drop > _ACTIVE_DEFLATION_MIN_DROP_MICRORAD
    ):
        return "active", info

    # Possible early deflation onset: short-window slope is meaningfully
    # negative AND meaningfully steeper than the longer-window slope. Both
    # conditions are needed — the short slope alone is too noisy, and the
    # acceleration alone fires when both windows are quietly positive.
    if (
        short_slope is not None
        and long_slope is not None
        and short_slope < _STARTING_SHORT_SLOPE_MAX
        and (long_slope - short_slope)
        > _STARTING_SLOPE_ACCELERATION_MICRORAD_PER_HOUR
    ):
        return "starting", info

    # Anything below depends on having a prediction at all
    band = prediction.confidence_band if prediction is not None else None
    next_event = prediction.next_event_date if prediction is not None else None
    if band is None and next_event is None:
        return "calm", info

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    info["now_utc"] = now

    if band is not None:
        lo, hi = band
        if now < lo:
            return "calm", info
        if lo <= now <= hi:
            return "imminent", info
        # now > hi
        return "overdue", info

    # No confidence band but we have a point estimate — fall back to
    # comparing the point estimate alone with a small buffer.
    if next_event is not None:
        buffer = pd.Timedelta(days=2)
        if now < next_event - buffer:
            return "calm", info
        if next_event - buffer <= now <= next_event + buffer:
            return "imminent", info
        return "overdue", info

    return "calm", info


# ─────────────────────────────────────────────────────────────────────────────
# Safety alerts (USGS HANS + NWS Hawaii)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=INGEST_CACHE_TTL_SECONDS, show_spinner=False)
def _fetch_safety_alerts_cached() -> SafetyAlertSummary:
    """Best-effort: errors from either source are recorded on
    SafetyAlertSummary.errors rather than raised, so the rest of the app
    keeps rendering if the alert APIs are slow or down."""
    return fetch_safety_alerts()


def get_safety_alerts() -> SafetyAlertSummary:
    """Public accessor for cached USGS HANS + NWS Hawaii alerts."""
    return _fetch_safety_alerts_cached()


def clear_safety_alerts_cache() -> None:
    """Bust the safety-alerts cache. Used by the manual Refresh button so
    one click also refetches the volcano alert level + advisories."""
    _fetch_safety_alerts_cached.clear()
