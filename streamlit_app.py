"""Kīlauea Fountain Event Tracker — Streamlit entrypoint.

This is the file Streamlit Community Cloud loads by default. Keep its name
exactly `streamlit_app.py` at the project root.

What the app does, top to bottom:
  1. (On demand) ingest the four USGS tilt PNGs into `data/tilt_history.csv`.
  2. Read the cached tilt history.
  3. Auto-detect peaks (with sidebar-tunable thresholds).
  4. Slice to the most recent N peaks for the trendline fit.
  5. Run the curve-fit + intersection prediction.
  6. Render the status banner, chart, and diagnostics.

Streamlit re-runs this script top-to-bottom every time a widget changes. The
expensive bits (`load_tilt`, `load_history`) are wrapped in `@st.cache_data`
so the typical interaction (e.g. moving a peak-detection slider) only re-runs
the cheap pure-function math.
"""

from __future__ import annotations

import io
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `src/kilauea_tracker/` importable WITHOUT relying on the package
# being pip-installed. Inserting src/ at the front of sys.path means imports
# resolve against the source tree in the live checkout on a cold-started
# Python process.
#
# We used to also evict any previously-imported kilauea_tracker submodules
# from sys.modules here, but that was causing two nasty failures that were
# worse than the stale-install problem it tried to fix:
#   1. On Python 3.11 (Streamlit Cloud), the freshly re-imported cache.py
#      tripped a CPython dataclass bug where `_is_type()` dereferences a
#      None return from `sys.modules.get(cls.__module__)`.
#   2. Everywhere, any IngestRunResult instance whose class was constructed
#      BEFORE re-import had a different `type()` than the post-re-import
#      class, so pickle failed with "not the same object as
#      kilauea_tracker.ingest.pipeline.IngestRunResult" — which in turn
#      made @st.cache_data raise UnserializableReturnValueError and the
#      whole app crashed.
# sys.path.insert alone is sufficient on a cold start; Streamlit Cloud
# spins a fresh process on each redeploy.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from kilauea_tracker.cache import load_history
from kilauea_tracker.config import (
    ALL_SOURCES,
    HISTORY_CSV,
    INGEST_CACHE_TTL_SECONDS,
    PEAK_DEFAULTS,
    TILT_SOURCE_NAME,
    USGS_TILT_URLS,
)
from kilauea_tracker.ingest.pipeline import IngestRunResult, ingest_all
from kilauea_tracker.model import DATE_COL, TILT_COL, predict
from kilauea_tracker.peaks import detect_peaks
from kilauea_tracker.plotting import build_figure

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Kīlauea Fountain Event Tracker",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit's st.metric refuses to wrap long values by default — they get
# truncated with an ellipsis. The pretty timestamp format ("Sat, Apr 18 ·
# 4:31 AM HST") doesn't fit on one line in a 3-column layout, so we override
# the white-space rule with a tiny CSS injection.
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        white-space: normal !important;
        overflow-wrap: anywhere;
        line-height: 1.1;
        font-size: 1.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cached ingest — at most once per 15 minutes regardless of widget activity
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=INGEST_CACHE_TTL_SECONDS, show_spinner="Fetching latest USGS tilt data…")
def cached_ingest() -> IngestRunResult:
    """Run all USGS sources through the ingest pipeline and reconcile.

    Wrapped in `st.cache_data` so the same browser session reuses results
    until the TTL expires (15 minutes by default — USGS updates these PNGs
    on roughly that cadence). Clearing the cache via `cached_ingest.clear()`
    forces a fresh fetch the next time this function runs.

    Returns the full `IngestRunResult` with per-source reports AND the
    reconciliation summary (per-source y-offsets, conflicts, warnings).
    """
    return ingest_all()


# Safety alert sources (USGS HANS aviation color code + NWS Hawaii
# alerts) live in their own module — they're independent of the tilt
# ingest pipeline and have their own cache TTL. 15 minutes matches the
# tilt cache so a single Refresh-button click busts both caches via
# `cached_safety_alerts.clear()` below.
from kilauea_tracker.safety_alerts import (  # noqa: E402
    SafetyAlertSummary,
    fetch_safety_alerts,
)


@st.cache_data(ttl=INGEST_CACHE_TTL_SECONDS, show_spinner=False)
def cached_safety_alerts() -> SafetyAlertSummary:
    """Fetch USGS HANS volcano status + filtered NWS Hawaii alerts.

    Best-effort: errors from either source are recorded on the returned
    SafetyAlertSummary.errors list rather than raised, so the rest of
    the app keeps rendering if the alert APIs are slow or down.
    """
    return fetch_safety_alerts()


# Initialize session state — used to remember whether ingestion has already
# run at least once in this Streamlit session, so we can show "Last update"
# accurately even after the cache TTL expires.
if "last_ingest_at" not in st.session_state:
    st.session_state.last_ingest_at = None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Controls")

    refresh_clicked = st.button(
        "🔄 Refresh data from USGS",
        width="stretch",
        help="Re-fetch and re-trace all four USGS tilt PNGs.",
    )
    if refresh_clicked:
        cached_ingest.clear()
        cached_safety_alerts.clear()
        st.session_state.last_ingest_at = None  # force the block below to re-run

    st.divider()
    st.subheader("Trendline window")
    n_peaks_for_fit = st.slider(
        "Number of recent peaks",
        min_value=3,
        max_value=20,
        value=6,
        step=1,
        help=(
            "How many of the most recent detected peaks to use for the linear "
            "trendline fit. Smaller = more sensitive to recent shifts; larger "
            "= smoother long-term trend."
        ),
    )

    # Peak detection lives behind an expander because the defaults work
    # across the full range of Kīlauea tilt regimes (large 8-15 µrad cycles
    # AND the recent small 2-5 µrad cycles) — users very rarely need to
    # tune these. Keep them accessible but out of the default view.
    with st.expander("⚙️ Advanced: peak detection", expanded=False):
        min_prominence = st.slider(
            "Minimum prominence (µrad)",
            min_value=1.0,
            max_value=15.0,
            value=PEAK_DEFAULTS.min_prominence,
            step=0.5,
            help=(
                "How much the peak must rise above its surrounding troughs. "
                "Higher = fewer, more confident peaks. This is the dominant "
                "filter — adjust this first if peak detection looks wrong."
            ),
        )
        min_distance_days = st.slider(
            "Minimum spacing (days)",
            min_value=1.0,
            max_value=30.0,
            value=PEAK_DEFAULTS.min_distance_days,
            step=0.5,
            help="Reject peaks that fall within this many days of a stronger one.",
        )
        # PEAK_DEFAULTS.min_height defaults to None ("no absolute floor"),
        # but a slider needs a numeric value. We use -10 µrad as the UI
        # default — well below any real tilt-cycle trough, so it's
        # effectively no constraint while still showing a number on the
        # control.
        min_height = st.slider(
            "Minimum height (µrad)",
            min_value=-20.0,
            max_value=20.0,
            value=-10.0,
            step=0.5,
            help=(
                "Absolute tilt threshold a sample must clear to count as a "
                "peak. The default of -10 effectively disables this filter "
                "(real tilt cycles never bottom out below ~-30 µrad, so a "
                "-10 floor accepts all real peaks). Raise this if you only "
                "want to see peaks above a specific tilt level."
            ),
        )

    st.divider()
    st.subheader("Display")
    timezone_choice = st.selectbox(
        "Time zone",
        options=["HST (Pacific/Honolulu)", "UTC"],
        index=0,
        help="All displayed dates use this time zone. HST is the local time at Kīlauea.",
    )
    DISPLAY_TZ = (
        "Pacific/Honolulu" if timezone_choice.startswith("HST") else "UTC"
    )
    TZ_LABEL = "HST" if DISPLAY_TZ == "Pacific/Honolulu" else "UTC"

    # Phase 4 Commit 5: per-source overlay toggle. Off by default so
    # non-technical viewers see the clean merged line. Click on →
    # overlay traces for each source appear in the legend and can be
    # toggled individually by clicking the legend entries.
    show_per_source = st.toggle(
        "🔍 Show per-source traces",
        value=False,
        help=(
            "Overlay each USGS source's calibrated trace beneath the "
            "merged line. Useful for diagnosing apparent alignment "
            "issues — you can see whether a visible step comes from "
            "calibration drift, source handoff, or transcription noise. "
            "Individual source traces are hidden by default; click the "
            "legend entries to toggle them on."
        ),
    )

    st.divider()
    st.caption("**Data source**")
    st.caption(
        "Electronic tilt at the **UWD** station (Uēkahuna, summit), "
        "**azimuth 300°**. Published by USGS Hawaiian Volcano Observatory — "
        "see the "
        "[Kīlauea monitoring data page]"
        "(https://www.usgs.gov/volcanoes/kilauea/science/monitoring-data-kilauea) "
        "for the user-friendly view."
    )

    st.divider()
    st.caption(
        "Built by [Madison Rickert](https://github.com/madisonrickert) · "
        "[source on GitHub](https://github.com/madisonrickert/kilauea-tracker)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run ingest — populates `data/tilt_history.csv`
# ─────────────────────────────────────────────────────────────────────────────

ingest_result = cached_ingest()
reports = ingest_result.per_source
reconcile_report = ingest_result.reconcile
if st.session_state.last_ingest_at is None:
    st.session_state.last_ingest_at = datetime.now(tz=timezone.utc)

# Surface ingest errors and warnings in a single collapsed "ingest pipeline
# status" panel. Per-source ingest failures are NOT fatal — the reconcile
# layer falls back to whichever sources did succeed plus the on-disk archive,
# so the user always sees a working chart. Treating a single source's
# transient OCR misread as a top-level red banner is too loud for what is
# actually a recoverable degraded state. Genuine fatal errors (no data at
# all) get caught further down where the empty-history check lives.
ingest_errors = [r for r in reports if r.error]
ingest_warnings: list[tuple[str, str]] = []
for r in reports:
    for w in r.warnings:
        ingest_warnings.append((r.source_name, w))
if reconcile_report is not None:
    for w in reconcile_report.warnings:
        ingest_warnings.append(("reconcile", w))

total_notes = len(ingest_errors) + len(ingest_warnings)
if total_notes > 0:
    n_err = len(ingest_errors)
    label_parts = []
    if n_err:
        label_parts.append(f"{n_err} error(s)")
    if ingest_warnings:
        label_parts.append(f"{len(ingest_warnings)} note(s)")
    label = "ⓘ Ingest pipeline status — " + ", ".join(label_parts)
    with st.expander(label, expanded=False):
        st.caption(
            "Non-fatal diagnostics from the ingest + reconcile pipeline. "
            "Per-source errors mean ONE USGS PNG fetch failed (transient OCR "
            "misread, network blip, etc.) — the reconcile layer falls back "
            "to the other sources and the on-disk archive, so the chart "
            "above is still using the freshest data we could get. Frame-"
            "shift corrections and proximity-gate drops are normal."
        )
        for r in ingest_errors:
            st.markdown(f"- ❌ **{r.source_name}** — {r.error}")
        for src, w in ingest_warnings:
            st.markdown(f"- **{src}** — {w}")


# ─────────────────────────────────────────────────────────────────────────────
# Load history (post-ingest)
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Loading tilt history…")
def _load_cached_tilt(path_str: str, mtime: float) -> pd.DataFrame:
    """Reads `data/tilt_history.csv`. The mtime parameter forces a re-read
    when the file changes on disk (Streamlit caches by argument values)."""
    return load_history(Path(path_str))


def _cache_mtime() -> float:
    return HISTORY_CSV.stat().st_mtime if HISTORY_CSV.exists() else 0.0


tilt_df = _load_cached_tilt(str(HISTORY_CSV), _cache_mtime())

if len(tilt_df) == 0:
    st.error(
        "No tilt history available. The ingest pipeline didn't produce any "
        "rows and `legacy/Tiltmeter Data - Sheet1.csv` is missing."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Compute prediction
# ─────────────────────────────────────────────────────────────────────────────

all_peaks = detect_peaks(
    tilt_df,
    min_prominence=min_prominence,
    min_distance_days=min_distance_days,
    min_height=min_height,
)
recent_peaks = all_peaks.tail(n_peaks_for_fit).reset_index(drop=True)
prediction = predict(tilt_df, recent_peaks)


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
    """Linear-fit slope of the last `hours` of tilt data, in µrad/hour.

    Returns None if there aren't enough samples to fit a line.
    """
    if len(df) < 3:
        return None
    end = df[DATE_COL].max()
    start = end - pd.Timedelta(hours=hours)
    window = df[df[DATE_COL] >= start]
    if len(window) < 3:
        return None
    # Convert dates to float-hours for the fit; subtracting the start avoids
    # the int64-overflow trap that bit us earlier on the to_days helper.
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


def _eruption_state(
    tilt_df: pd.DataFrame,
    prediction,
) -> tuple[str, dict]:
    """Classify the current point in the eruption lifecycle.

    Returns `(state, info)` where state is one of:
        "active"   — sharp sustained negative slope → eruption confirmed
        "starting" — short-window slope has steepened relative to a longer
                     window, suggesting deflation may be beginning. The
                     classification is intentionally hedged because tilt
                     can also dip briefly without a fountain following.
        "imminent" — current time is inside the predicted confidence band
        "overdue"  — current time is past the high end of the band
        "calm"     — none of the above; building toward the next eruption

    `info` carries the diagnostics that fed the classification (slopes,
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


eruption_state, eruption_state_info = _eruption_state(tilt_df, prediction)


# ─────────────────────────────────────────────────────────────────────────────
# Header + status banner
# ─────────────────────────────────────────────────────────────────────────────

st.title("🌋 Kīlauea Fountain Event Tracker")
st.markdown(
    "**Tiltmeter:** UWD station (Uēkahuna, summit) · azimuth 300° · "
    "data from [USGS Hawaiian Volcano Observatory](https://www.usgs.gov/volcanoes/kilauea/science/monitoring-data-kilauea)"
)
st.caption(
    "Predicts the next eruption pulse at Kīlauea by fitting an exponential "
    "saturation curve to the current tilt episode and intersecting it with the "
    "linear trendline through recent peaks."
)


def _to_display_tz(ts: pd.Timestamp | None) -> pd.Timestamp | None:
    """Convert a naive (assumed-UTC) timestamp into the user's chosen tz."""
    if ts is None:
        return None
    aware = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    return aware.tz_convert(DISPLAY_TZ)


def _fmt_date(ts: pd.Timestamp | None) -> str:
    """Pretty long-form date for the big metric tiles."""
    converted = _to_display_tz(ts)
    if converted is None:
        return "—"
    # e.g. "Sat, Apr 18 · 3:23 PM HST"
    return f"{converted.strftime('%a, %b %-d · %-I:%M %p')} {TZ_LABEL}"


def _fmt_short(ts: pd.Timestamp | None) -> str:
    """Short month/day for the confidence band delta."""
    converted = _to_display_tz(ts)
    if converted is None:
        return "—"
    return converted.strftime("%b %-d")


def _ago(ts: datetime) -> str:
    delta = datetime.now(tz=timezone.utc) - ts
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _fmt_band(band: tuple[pd.Timestamp, pd.Timestamp] | None) -> str:
    if band is None:
        return "—"
    lo, hi = band
    return f"{_fmt_short(lo)} → {_fmt_short(hi)}"


col1, col2, col3 = st.columns(3)
with col1:
    if eruption_state == "active":
        # Don't display a "next fountain event" forecast while one is
        # actively underway — at best the number is stale (still pointing
        # at the event that just started), at worst it implies the user
        # should be waiting for something other than what's happening on
        # the live webcam right now. Replace the prediction tile with a
        # plain "currently erupting" status so the column doesn't go blank.
        st.metric(
            "Status",
            "🔴 Erupting now",
            delta="forecast paused",
            delta_color="off",
            help=(
                "An active fountain event is in progress. The next-event "
                "forecast is hidden until deflation completes and the "
                "next inflation cycle establishes a new exponential fit."
            ),
        )
    else:
        st.metric(
            "Next fountain event (model)",
            _fmt_date(prediction.next_event_date),
            delta=_fmt_band(prediction.confidence_band),
            delta_color="off",
            help=(
                "Date where the rising exponential fit on the current episode "
                "crosses the trendline through recent peaks. The delta line is "
                "the 10th–90th percentile Monte Carlo confidence band derived "
                "from the exponential fit's covariance matrix."
            ),
        )

    # Independent baseline forecast: median time between detected peaks
    # added to the most recent peak. Doesn't use the trendline or the exp
    # curve at all — useful as a sanity check on the model. If both
    # predictions agree, confidence is higher; if they diverge, the model
    # is struggling with the current regime.
    #
    # Suppressed during an active eruption for the same reason as the
    # primary forecast above — we don't want to imply we're waiting for
    # something while the volcano is in the middle of doing it.
    if eruption_state != "active" and prediction.interval_based_next_event_date is not None:
        ib_date = prediction.interval_based_next_event_date
        ib_band = prediction.interval_based_band
        median_days = prediction.median_peak_interval_days or 0.0

        # Is the interval-based estimate already in the past? That's a
        # meaningful "we're overdue by typical-cycle reckoning" signal.
        ib_aware = (
            ib_date.tz_localize("UTC") if ib_date.tzinfo is None else ib_date
        )
        now_utc = pd.Timestamp.now(tz="UTC")
        delta_days = (now_utc - ib_aware).total_seconds() / 86400
        if delta_days > 0.5:
            overdue_str = f" · ⚠️ **{delta_days:.0f}d overdue** by this metric"
        else:
            overdue_str = ""

        band_str = ""
        if ib_band is not None:
            band_str = f" &nbsp;·&nbsp; IQR {_fmt_band(ib_band)}"

        st.markdown(
            f"📊 **Interval baseline:** {_fmt_short(ib_date)}"
            f"{band_str}"
            f"{overdue_str}",
            help=(
                "Independent forecast based purely on the median time "
                f"between detected peaks ({median_days:.1f} days), added "
                "to the most recent detected peak. Doesn't use the curve-"
                "fit model at all — useful as a sanity check. The IQR "
                "shows the 25th-75th percentile of historical peak "
                "intervals so you can see how variable the cycle length "
                "has been."
            ),
        )
with col2:
    last_data = tilt_df[DATE_COL].max()
    # Compute the age of the latest sample, treating its naive timestamp as
    # UTC. This is the freshness number the user actually cares about: how
    # old is the most recent data point USGS has published, regardless of
    # when we last polled. We compute the delta directly via pandas to
    # preserve nanosecond precision (going through datetime.to_pydatetime()
    # would emit a "Discarding nonzero nanoseconds" warning).
    if last_data is not None and pd.notna(last_data):
        last_data_aware = (
            last_data.tz_localize("UTC") if last_data.tzinfo is None else last_data
        )
        delta = pd.Timestamp.now(tz="UTC") - last_data_aware
        seconds = int(delta.total_seconds())
        if seconds < 60:
            sample_age = f"{seconds}s ago"
        elif seconds < 3600:
            sample_age = f"{seconds // 60}m ago"
        elif seconds < 86400:
            sample_age = f"{seconds // 3600}h ago"
        else:
            sample_age = f"{seconds // 86400}d ago"
    else:
        sample_age = "—"
    st.metric(
        "Latest tilt sample",
        _fmt_date(last_data),
        delta=sample_age,
        delta_color="off",
        help=(
            "Timestamp of the most recent data point USGS has published. "
            "USGS updates roughly every 15-30 minutes; the age below shows "
            "how long ago that point was recorded."
        ),
    )
with col3:
    if st.session_state.last_ingest_at is not None:
        successful = sum(1 for r in reports if r.error is None)
        total = len(reports)
        if successful == total:
            indicator = "🟢"
        elif successful > 0:
            indicator = "🟡"
        else:
            indicator = "🔴"
        # Show the absolute timestamp instead of "X ago" — Streamlit only
        # reruns this expression on widget interaction, so an "ago" value
        # would be frozen at the moment of the most recent rerun (always
        # ~0s after a refresh button click). The absolute timestamp is
        # honest: it shows when we last polled, and the user can compare
        # against the freshness delta on the "Latest tilt sample" tile to
        # see whether the poll actually pulled new data.
        poll_ts = pd.Timestamp(st.session_state.last_ingest_at)
        st.metric(
            "Last poll attempt",
            f"{indicator} {_fmt_date(poll_ts)}",
            delta=f"{successful}/{total} sources",
            delta_color="off",
            help=(
                "Wall-clock time of the most recent USGS poll. A recent "
                "poll doesn't guarantee newer data — if USGS hasn't "
                "published anything new since the last poll, the 'Latest "
                "tilt sample' above stays the same. Click the refresh "
                "button in the sidebar to poll again."
            ),
        )
    else:
        st.metric("Last poll attempt", "—")


# ─────────────────────────────────────────────────────────────────────────────
# Eruption lifecycle status banner
# ─────────────────────────────────────────────────────────────────────────────

# When the banner is "active" or "starting" we tuck a small live webcam
# thumbnail next to the alert text — V1cam looks straight at the west wall
# of Halemaʻumaʻu, where the recent fountain events have been venting.
_BANNER_WEBCAM_URL = (
    "https://volcanoes.usgs.gov/observatories/hvo/cams/V1cam/images/M.jpg"
)
_BANNER_WEBCAM_CAPTION = "USGS V1cam · Halemaʻumaʻu west wall · live"


def _render_alert_with_webcam(render_alert) -> None:
    """Render the alert in a left column and the V1cam thumbnail on the right.

    `render_alert` is a no-arg callable that emits the alert (`st.error` or
    `st.warning`). The webcam image is loaded fresh from USGS on each
    rerun, so the thumbnail in the banner stays as live as the rest of the
    USGS source plots panel below.
    """
    col_text, col_img = st.columns([4, 1])
    with col_text:
        render_alert()
    with col_img:
        try:
            st.image(_BANNER_WEBCAM_URL, caption=_BANNER_WEBCAM_CAPTION)
        except Exception:
            # If the webcam fails to load, the alert still renders cleanly.
            pass


if eruption_state == "active":
    slope = eruption_state_info.get("recent_slope_microrad_per_hour")
    drop = eruption_state_info.get("drop_from_24h_max")

    def _active_alert() -> None:
        st.error(
            f"### 🔴 Eruption active right now\n\n"
            f"Tilt is dropping at **{slope:+.2f} µrad/hour** "
            f"(**{drop:.1f} µrad** below the 24-hour max). The deflation "
            f"signature of a fountain event is unmistakable in the live data — "
            f"the live webcam alongside should be lighting up. More feeds at "
            f"the [USGS webcams page]"
            f"(https://www.usgs.gov/volcanoes/kilauea/summit-webcams)."
        )

    _render_alert_with_webcam(_active_alert)
elif eruption_state == "starting":
    short_slope = eruption_state_info.get("short_slope_microrad_per_hour")
    long_slope = eruption_state_info.get("long_slope_microrad_per_hour")
    drop = eruption_state_info.get("drop_from_24h_max") or 0.0

    def _starting_alert() -> None:
        st.warning(
            f"### 🟠 Possible deflation onset — watching\n\n"
            f"Tilt slope has steepened to **{short_slope:+.2f} µrad/hour** over "
            f"the last 30 minutes, up from **{long_slope:+.2f} µrad/hour** over "
            f"the last 6 hours (drop from 24h max: **{drop:.2f} µrad**). This "
            f"is consistent with the very early stages of a fountain event, "
            f"but the signal is small enough that it could also be a brief "
            f"pressure release that doesn't develop into a full eruption. "
            f"The status will escalate to *Eruption active* if the deflation "
            f"continues. The live webcam alongside is the best place to "
            f"corroborate visually — more feeds at the "
            f"[USGS webcams page]"
            f"(https://www.usgs.gov/volcanoes/kilauea/summit-webcams)."
        )

    _render_alert_with_webcam(_starting_alert)
elif eruption_state == "imminent":
    band = prediction.confidence_band
    if band is not None:
        lo, hi = band
        days_open = (eruption_state_info["now_utc"] - lo).total_seconds() / 86400
        days_remaining = (hi - eruption_state_info["now_utc"]).total_seconds() / 86400
        st.warning(
            f"### 🟠 Eruption window open — possibly imminent\n\n"
            f"We're inside the predicted 80% confidence band, which opened "
            f"**{days_open:.1f} days ago** and runs for another "
            f"**{days_remaining:.1f} days**. The next fountain event is "
            f"expected around **{_fmt_date(prediction.next_event_date)}**."
        )
elif eruption_state == "overdue":
    band = prediction.confidence_band
    if band is not None:
        _lo, hi = band
        days_overdue = (eruption_state_info["now_utc"] - hi).total_seconds() / 86400
        st.warning(
            f"### 🟡 Eruption overdue\n\n"
            f"Today is **{days_overdue:.1f} days past** the high end of the "
            f"predicted confidence band. Either the next fountain event is "
            f"running late or the model needs more recent peaks to recompute. "
            f"Predicted point estimate was "
            f"**{_fmt_date(prediction.next_event_date)}**."
        )
# `calm` shows no banner — the metric tiles above already convey the
# upcoming-event countdown without adding visual noise.


# ─────────────────────────────────────────────────────────────────────────────
# Public safety alerts (USGS HANS aviation color code + NWS Hawaii alerts)
# ─────────────────────────────────────────────────────────────────────────────
#
# Independent of the tilt model — these come from official channels (USGS
# HANS for the volcano alert level / aviation color code, NWS for tephra/
# wind/SO2 advisories). Rendered here, immediately below the eruption
# lifecycle banner, because that's where the user is already looking when
# the volcano is doing something interesting.

safety = cached_safety_alerts()


def _render_usgs_color_badge(status) -> None:
    """Render the USGS aviation color code as a colored markdown badge."""
    color_map = {
        "GREEN": "#1e8e3e",
        "YELLOW": "#f9ab00",
        "ORANGE": "#e8710a",
        "RED": "#d93025",
    }
    bg = color_map.get(status.color_code, "#5f6368")
    sent_str = ""
    if status.sent_utc is not None:
        sent_local = pd.Timestamp(status.sent_utc).tz_convert(DISPLAY_TZ)
        sent_str = f" · issued {sent_local.strftime('%b %-d, %-I:%M %p')} {TZ_LABEL}"
    notice_link = (
        f" &nbsp;[full notice ↗]({status.notice_url})"
        if status.notice_url
        else ""
    )
    st.markdown(
        f"<div style='padding: 0.6rem 0.9rem; border-radius: 6px; "
        f"background: {bg}; color: white; font-weight: 600; "
        f"margin-bottom: 0.5rem;'>"
        f"USGS volcano alert: <b>{status.color_code}</b> aviation color · "
        f"<b>{status.alert_level}</b> ground alert"
        f"<span style='font-weight: 400; opacity: 0.9;'>{sent_str}</span>"
        f"</div>"
        f"<div style='font-size: 0.85rem; color: #aaa; margin-bottom: 1rem;'>"
        f"{status.observatory}{notice_link}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_nws_alert(alert) -> None:
    """Render one filtered NWS alert as a compact info card."""
    severity_icon = {
        "Extreme": "🛑",
        "Severe": "⚠️",
        "Moderate": "🟠",
        "Minor": "🟡",
    }.get(alert.severity, "ⓘ")
    expires_str = ""
    if alert.expires is not None:
        exp_local = pd.Timestamp(alert.expires).tz_convert(DISPLAY_TZ)
        expires_str = f" · expires {exp_local.strftime('%b %-d, %-I:%M %p')} {TZ_LABEL}"
    st.markdown(
        f"**{severity_icon} {alert.event}** &nbsp; "
        f"<span style='color:#888'>· {alert.area_desc}{expires_str}</span>",
        unsafe_allow_html=True,
    )
    if alert.headline and alert.headline != alert.event:
        st.caption(alert.headline)
    if alert.description:
        with st.expander("Full advisory text"):
            st.text(alert.description)


if safety.usgs_status is not None or safety.nws_alerts:
    _render_usgs_color_badge(safety.usgs_status) if safety.usgs_status else None
    if safety.nws_alerts:
        st.markdown(
            f"**Active NWS advisories ({len(safety.nws_alerts)})** "
            f"<span style='color:#888; font-size:0.85rem'>"
            f"· filtered for Kīlauea / Big Island relevance</span>",
            unsafe_allow_html=True,
        )
        for alert in safety.nws_alerts:
            _render_nws_alert(alert)
        st.caption(
            "Source: USGS Hazard Alert Notification System + NWS Honolulu. "
            "Updated every 15 minutes; click the sidebar Refresh button to "
            "force a fresh fetch."
        )
elif safety.errors:
    # Both sources failed. Show a single quiet caption rather than a
    # loud red banner — alerts are auxiliary, not load-bearing.
    st.caption(
        "Safety alerts unavailable right now "
        f"({len(safety.errors)} source(s) returned an error)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main chart
# ─────────────────────────────────────────────────────────────────────────────

# Once an eruption is underway (or we've detected the early-deflation
# "starting" signal), the exponential "current episode" curve is fitted to
# the inflation phase that just ended — extending it forward would just
# project a saturation level the volcano is in the middle of releasing.
# Hide it so the chart doesn't tell two contradictory stories at once.
_episode_curve_visible = eruption_state not in ("active", "starting")

# Once an eruption is CONFIRMED active, also suppress the next-event
# prediction (the star marker and the 80% confidence band rectangle).
# We don't want to show "next eruption: someday" while one is visibly in
# progress on the live webcam — that's the chart equivalent of telling
# someone to look out for the bus they're already riding.
_next_event_visible = eruption_state != "active"

def _load_per_source_corrected_for_overlay() -> dict[str, pd.DataFrame]:
    """Reload each per-source CSV and apply the run's (a, b) correction.

    Reads:
      - anchor_fits on ingest_result to replicate Phase 1c correction
        for dec2024_to_now (and any other anchor-corrected source)
      - reconcile_report.sources for each source's (a, b) — applied as
        `(y - b) / a` per `_apply_ab_corrections`

    Returns `{source_name → DataFrame}` ready to hand to build_figure.
    Best-effort: a missing CSV or malformed row report returns an
    empty dict so overlay mode silently falls back to no overlay.
    """
    from kilauea_tracker.config import source_csv_path, ALL_SOURCES, TILT_SOURCE_NAME

    if reconcile_report is None:
        return {}

    out: dict[str, pd.DataFrame] = {}
    anchor_fits_by_source = {
        f.source_name: f for f in (ingest_result.anchor_fits or [])
    }
    alignments_by_source = {s.name: s for s in reconcile_report.sources}

    for s in ALL_SOURCES:
        name = TILT_SOURCE_NAME[s]
        path = source_csv_path(name)
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, parse_dates=[DATE_COL])
        except Exception:
            continue
        df = df[[DATE_COL, TILT_COL]].dropna().sort_values(DATE_COL)
        # 1. Phase 1c anchor correction (if warning fired)
        af = anchor_fits_by_source.get(name)
        if af is not None and af.ran and af.warning is not None:
            df[TILT_COL] = af.a * df[TILT_COL] + af.b
        # 2. Phase 2 pairwise (a, b) correction
        align = alignments_by_source.get(name)
        if align is not None and align.a and abs(align.a) > 1e-9:
            df[TILT_COL] = (df[TILT_COL] - align.b) / align.a
        out[name] = df.reset_index(drop=True)
    return out


fig = build_figure(
    tilt_df,
    recent_peaks,
    prediction,
    all_peaks_df=all_peaks,
    title="",
    show_current_episode=_episode_curve_visible,
    show_next_event_prediction=_next_event_visible,
    per_source_overlay=_load_per_source_corrected_for_overlay() if show_per_source else None,
)
# Phase 4 Commit 5: enable box-select on the chart so the user can drag
# a rectangle over a region to populate the CSV export date range.
chart_selection = st.plotly_chart(
    fig,
    width="stretch",
    on_select="rerun",
    selection_mode="box",
    key="main_chart",
)

# Hover-to-clipboard: press ⌘C / Ctrl+C while hovering a datapoint on the
# chart to copy "YYYY-MM-DD HH:MM | X.XX µrad" to the clipboard. Injected
# as a zero-height components.v1.html block so it can reach up through the
# Streamlit iframe sandbox and bind to the parent document's Plotly
# instance. The polling loop handles the race where Plotly mounts a moment
# after the component; the 10-second timeout keeps this quiet on static
# page loads where the chart never renders.
components.html(
    """
    <script>
    (function() {
      const parentDoc = window.parent.document;
      let current = null;
      const bind = (gd) => {
        if (!gd || gd._copyBound) return;
        gd._copyBound = true;
        gd.on('plotly_hover', (ev) => {
          if (!ev || !ev.points || !ev.points.length) return;
          const p = ev.points[0];
          const t = typeof p.x === 'string' ? p.x.slice(0, 16).replace('T', ' ') : p.x;
          const y = typeof p.y === 'number' ? p.y.toFixed(2) : p.y;
          current = t + ' | ' + y + ' \u00b5rad';
        });
        gd.on('plotly_unhover', () => { current = null; });
      };
      const onKey = (ev) => {
        if (!current) return;
        if (!(ev.metaKey || ev.ctrlKey) || ev.key !== 'c') return;
        // Don't intercept copies originating from a text selection.
        const sel = window.parent.getSelection && window.parent.getSelection();
        if (sel && sel.toString().length > 0) return;
        navigator.clipboard && navigator.clipboard.writeText(current);
      };
      parentDoc.addEventListener('keydown', onKey);
      const iv = setInterval(() => {
        parentDoc.querySelectorAll('.js-plotly-plot').forEach(bind);
      }, 400);
      setTimeout(() => clearInterval(iv), 10000);
    })();
    </script>
    """,
    height=0,
)
st.caption(
    "Tip: hover a point and press ⌘C / Ctrl+C to copy it. "
    "Or drag a box over the chart to set the CSV export range below."
)


# ─────────────────────────────────────────────────────────────────────────────
# CSV export (Phase 4 Commit 5)
# ─────────────────────────────────────────────────────────────────────────────
#
# Two modes. Simple: date + merged tilt + winning source + configurable
# per-source corrected values. Debug: every column, everything, as a zip
# bundle including pair fits + per-bucket MAD diagnostics.
#
# Range selection works three ways that all feed the same date pickers:
#   1. Box-select on the chart (see above) pre-populates the range.
#   2. Manual date pickers in the sidebar-style expander.
#   3. "Export full history" button ignores the range entirely.

with st.expander("📤 Export data as CSV"):
    # Resolve initial range from box-selection on the chart, falling
    # back to the last 7 days of data.
    default_to = pd.Timestamp(tilt_df[DATE_COL].max())
    default_from = default_to - pd.Timedelta(days=7)
    if chart_selection and "box" in (chart_selection or {}).get("selection", {}):
        boxes = chart_selection["selection"]["box"]
        if boxes:
            # Plotly selection box has x0/x1 as datetime strings when the
            # x-axis is a datetime axis. Convert and order.
            try:
                x0 = pd.Timestamp(boxes[0]["x"][0]) if "x" in boxes[0] else None
                x1 = pd.Timestamp(boxes[0]["x"][1]) if "x" in boxes[0] else None
                if x0 is not None and x1 is not None:
                    default_from = min(x0, x1)
                    default_to = max(x0, x1)
            except (KeyError, ValueError, IndexError):
                pass

    col_from, col_to = st.columns(2)
    with col_from:
        range_from = st.date_input(
            "From",
            value=default_from.date(),
            help="Start of the export window (inclusive, UTC).",
        )
    with col_to:
        range_to = st.date_input(
            "To",
            value=default_to.date(),
            help="End of the export window (inclusive, UTC).",
        )

    st.caption("**Simple mode** — pick which extra columns to include:")
    col_opts_a, col_opts_b, col_opts_c = st.columns(3)
    with col_opts_a:
        inc_corrected = st.checkbox(
            "Per-source corrected values",
            value=False,
            help=(
                "One column per source with the pairwise-calibrated tilt "
                "value at each bucket. Makes it easy to see which source "
                "disagreed where."
            ),
        )
    with col_opts_b:
        inc_raw = st.checkbox(
            "Per-source raw values",
            value=False,
            help=(
                "One column per source with the UNCORRECTED traced value. "
                "Useful for distinguishing transcription errors from "
                "calibration drift."
            ),
        )
    with col_opts_c:
        inc_mad = st.checkbox(
            "MAD rejection flags",
            value=False,
            help="Column listing sources MAD-flagged as outliers per bucket.",
        )

    # Build the simple export on demand.
    def _build_simple_export() -> pd.DataFrame:
        start_ts = pd.Timestamp(range_from)
        end_ts = pd.Timestamp(range_to) + pd.Timedelta(days=1)  # inclusive
        out = tilt_df[
            (tilt_df[DATE_COL] >= start_ts) & (tilt_df[DATE_COL] < end_ts)
        ].copy()
        out = out.rename(columns={
            DATE_COL: "date_utc",
            TILT_COL: "tilt_merged_microrad",
        })
        # Add MAD rejections (grouped by bucket)
        if inc_mad and reconcile_report is not None:
            rej_by_bucket: dict = {}
            for f in reconcile_report.transcription_failures:
                key = pd.Timestamp(f.bucket).round("15min")
                rej_by_bucket.setdefault(key, []).append(f.source)
            out["mad_rejected_sources"] = out["date_utc"].dt.round("15min").map(
                lambda b: ",".join(rej_by_bucket.get(b, []))
            )
        # Add per-source columns (corrected and/or raw)
        if inc_corrected or inc_raw:
            corrected_map = (
                _load_per_source_corrected_for_overlay() if inc_corrected else {}
            )
            from kilauea_tracker.config import source_csv_path, ALL_SOURCES, TILT_SOURCE_NAME
            for s in ALL_SOURCES:
                name = TILT_SOURCE_NAME[s]
                if inc_corrected and name in corrected_map:
                    cdf = corrected_map[name].copy()
                    cdf["_bucket"] = cdf[DATE_COL].dt.round("15min")
                    series = cdf.drop_duplicates("_bucket", keep="last").set_index("_bucket")[TILT_COL]
                    out[f"{name}_corrected_microrad"] = out["date_utc"].dt.round("15min").map(series)
                if inc_raw:
                    p = source_csv_path(name)
                    if p.exists():
                        try:
                            rdf = pd.read_csv(p, parse_dates=[DATE_COL])
                            rdf["_bucket"] = rdf[DATE_COL].dt.round("15min")
                            r_series = rdf.drop_duplicates("_bucket", keep="last").set_index("_bucket")[TILT_COL]
                            out[f"{name}_raw_microrad"] = out["date_utc"].dt.round("15min").map(r_series)
                        except Exception:
                            pass
        return out.reset_index(drop=True)

    if pd.Timestamp(range_from) > pd.Timestamp(range_to):
        st.error("'From' date must be on or before 'To' date.")
    else:
        col_simple, col_debug, col_full = st.columns(3)
        with col_simple:
            simple_df = _build_simple_export()
            st.download_button(
                "⬇ Simple CSV",
                data=simple_df.to_csv(index=False),
                file_name=(
                    f"tilt_{range_from.isoformat()}_to_{range_to.isoformat()}.csv"
                ),
                mime="text/csv",
                help=f"{len(simple_df)} rows in the selected range.",
            )

        with col_debug:
            # Debug bundle: simple CSV + pair fits + alignments + full run
            # report JSON, zipped in-memory.
            import io, zipfile, json as _json_dbg

            def _build_debug_zip() -> bytes:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    # 1. merged CSV (full range)
                    full = tilt_df.rename(columns={
                        DATE_COL: "date_utc",
                        TILT_COL: "tilt_merged_microrad",
                    })
                    zf.writestr("tilt_history_merged.csv", full.to_csv(index=False))
                    # 2. simple selected-range CSV
                    zf.writestr("selected_range.csv", simple_df.to_csv(index=False))
                    # 3. pair fits, alignments, MAD rejections as JSON
                    if reconcile_report is not None:
                        debug = {
                            "rows_out": reconcile_report.rows_out,
                            "winner_counts": dict(
                                getattr(reconcile_report, "winner_counts", {}) or {}
                            ),
                            "pairs": [
                                {
                                    "source_i": p.source_i,
                                    "source_j": p.source_j,
                                    "alpha": p.alpha,
                                    "beta": p.beta,
                                    "overlap_buckets": p.overlap_buckets,
                                    "residual_std_microrad": p.residual_std_microrad,
                                }
                                for p in reconcile_report.pairs
                            ],
                            "alignments": [
                                {
                                    "name": s.name,
                                    "a": s.a,
                                    "b": s.b,
                                    "pairs_used": s.pairs_used,
                                    "effective_resolution": (
                                        s.effective_resolution_microrad_per_pixel
                                    ),
                                    "rows_mad_rejected": s.rows_mad_rejected,
                                    "is_anchor": s.is_anchor,
                                    "note": s.note,
                                }
                                for s in reconcile_report.sources
                            ],
                            "transcription_failures": [
                                {
                                    "bucket": str(f.bucket),
                                    "source": f.source,
                                    "value_corrected": f.value_corrected,
                                    "bucket_median": f.bucket_median,
                                    "delta_microrad": f.delta_microrad,
                                }
                                for f in reconcile_report.transcription_failures
                            ],
                            "continuity_violations": [
                                {
                                    "bucket_before": str(v.bucket_before),
                                    "bucket_after": str(v.bucket_after),
                                    "tilt_before": v.tilt_before,
                                    "tilt_after": v.tilt_after,
                                    "delta_microrad": v.delta_microrad,
                                }
                                for v in reconcile_report.continuity_violations
                            ],
                            "warnings": list(reconcile_report.warnings),
                        }
                        zf.writestr(
                            "reconcile_diagnostics.json",
                            _json_dbg.dumps(debug, indent=2, default=str),
                        )
                    # 4. run report if available on disk
                    if ingest_result.run_report_path and ingest_result.run_report_path.exists():
                        zf.writestr(
                            f"run_report_{ingest_result.run_report_path.name}",
                            ingest_result.run_report_path.read_text(),
                        )
                return buf.getvalue()

            st.download_button(
                "⬇ Debug ZIP",
                data=_build_debug_zip(),
                file_name=(
                    f"tilt_debug_{range_from.isoformat()}_to_{range_to.isoformat()}.zip"
                ),
                mime="application/zip",
                help=(
                    "Zipped bundle: merged history, selected-range CSV, per-"
                    "source alignments, pair fits, MAD rejections, continuity "
                    "violations, and the raw run report. This is the bug-"
                    "report payload."
                ),
            )

        with col_full:
            full_df = tilt_df.rename(columns={
                DATE_COL: "date_utc",
                TILT_COL: "tilt_merged_microrad",
            })
            st.download_button(
                "⬇ Full history CSV",
                data=full_df.to_csv(index=False),
                file_name="tilt_history_full.csv",
                mime="text/csv",
                help=f"Complete {len(full_df)}-row merged tilt history (all time).",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Reconcile diagnostics (Phase 4 Commit 5)
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("🔎 Reconcile diagnostics"):
    st.caption(
        "Deep view into the per-source calibration and merge decisions "
        "for the current run. If a visible step or disagreement in the "
        "chart surprises you, the answer is almost always in one of "
        "these three tables."
    )
    if reconcile_report is None:
        st.info("Reconcile didn't run this session.")
    else:
        # ── Per-source alignments ──────────────────────────────────────────
        st.markdown("**Per-source alignments** — the (a, b) applied to each source")
        align_rows = []
        for s in reconcile_report.sources:
            align_rows.append({
                "source": s.name,
                "a (slope)": round(s.a, 4),
                "b (µrad)": round(s.b, 3),
                "pairs used": s.pairs_used,
                "effective resolution (µrad/px)": round(
                    s.effective_resolution_microrad_per_pixel, 4
                ),
                "MAD flags": s.rows_mad_rejected,
                "anchor": "🎯" if s.is_anchor else "",
                "note": s.note or "",
            })
        if align_rows:
            st.dataframe(pd.DataFrame(align_rows), hide_index=True, width="stretch")

        # ── Bucket winner distribution ─────────────────────────────────────
        st.markdown("**Bucket winner distribution** — which source contributed each region")
        winner_counts = getattr(reconcile_report, "winner_counts", None) or {}
        if winner_counts:
            total = sum(winner_counts.values()) or 1
            winner_rows = [
                {
                    "source": name,
                    "buckets won": count,
                    "share %": round(100 * count / total, 1),
                }
                for name, count in sorted(
                    winner_counts.items(), key=lambda kv: -kv[1]
                )
            ]
            st.dataframe(pd.DataFrame(winner_rows), hide_index=True, width="stretch")
        else:
            st.caption("No winner counts recorded.")

        # ── Pairwise fits ──────────────────────────────────────────────────
        st.markdown(
            "**Pairwise fits** — each source pair's Huber-robust `y_i = α · y_j + β` regression"
        )
        if reconcile_report.pairs:
            pair_rows = [
                {
                    "i": p.source_i,
                    "j": p.source_j,
                    "α": round(p.alpha, 4),
                    "β (µrad)": round(p.beta, 3),
                    "overlap": p.overlap_buckets,
                    "σ(residual) µrad": round(p.residual_std_microrad, 3),
                }
                for p in sorted(
                    reconcile_report.pairs,
                    key=lambda p: (p.source_i, p.source_j),
                )
            ]
            st.dataframe(pd.DataFrame(pair_rows), hide_index=True, width="stretch")
        else:
            st.caption("No pairs with sufficient overlap this run.")

        # ── MAD outlier rejections ─────────────────────────────────────────
        if reconcile_report.transcription_failures:
            st.markdown(
                f"**MAD outlier rejections** — {len(reconcile_report.transcription_failures)} "
                "bucket(s) where a source disagreed with the consensus beyond the "
                "MAD threshold (diagnostic only; winner is still best-effective-"
                "resolution per Phase 4 Commit 4)"
            )
            worst = sorted(
                reconcile_report.transcription_failures,
                key=lambda f: abs(f.delta_microrad),
                reverse=True,
            )[:15]
            mad_rows = [
                {
                    "bucket": pd.Timestamp(f.bucket).strftime("%Y-%m-%d %H:%M"),
                    "source": f.source,
                    "value (µrad)": round(f.value_corrected, 2),
                    "bucket median": round(f.bucket_median, 2),
                    "delta (µrad)": round(f.delta_microrad, 2),
                }
                for f in worst
            ]
            st.dataframe(pd.DataFrame(mad_rows), hide_index=True, width="stretch")
            if len(reconcile_report.transcription_failures) > 15:
                st.caption(
                    f"…and {len(reconcile_report.transcription_failures) - 15} more "
                    "(smaller deltas)."
                )


# ─────────────────────────────────────────────────────────────────────────────
# PNG transcription inspector (Phase 4 Commit 6)
# ─────────────────────────────────────────────────────────────────────────────
#
# Side-by-side renderer for each rolling source: raw USGS PNG on the
# left, same PNG with traced-sample dots overlaid on the right. The
# traced dots come from the per-source CSV this run produced, mapped
# back to pixel coordinates via the AxisCalibration stored on each
# IngestReport. Any visible drift between the dots and the USGS line
# is a transcription defect — pixel-level ground truth for bug reports.

with st.expander("🔬 Transcription quality inspector"):
    st.caption(
        "Each USGS source is rendered once with the selected overlay "
        "layers stacked on top. Toggle layers below to isolate what "
        "you're diagnosing — bbox/ticks to verify axis OCR, blue-mask "
        "to verify the tracer, dropped/outlier markers to see where "
        "the pipeline punted. All layers off = the plain USGS PNG."
    )

    # Per-layer toggles, grouped by what they catch.
    tier1_col, tier2_col, tier3_col = st.columns(3)
    with tier1_col:
        st.markdown("**Axis calibration**")
        layer_dots = st.checkbox("Re-traced dots", value=True, key="ovl_dots")
        layer_bbox = st.checkbox("Plot bbox outline", value=False, key="ovl_bbox")
        layer_yticks = st.checkbox("Y-tick OCR markers", value=False, key="ovl_yticks")
        layer_ygrid = st.checkbox("Integer-µrad gridlines", value=False, key="ovl_ygrid")
        layer_corners = st.checkbox("Corner (date, µrad) labels", value=False, key="ovl_corners")
    with tier2_col:
        st.markdown("**Tracer pixels**")
        layer_blue = st.checkbox("Blue-curve HSV mask", value=False, key="ovl_blue")
        layer_legend = st.checkbox("Legend-exclusion zone", value=False, key="ovl_legend")
        layer_dropcols = st.checkbox("Dropped-width columns", value=False, key="ovl_dropcols")
        layer_outliers = st.checkbox("Rolling-median outlier drops", value=False, key="ovl_outliers")
    with tier3_col:
        st.markdown("**World state**")
        layer_now = st.checkbox("'Now' UTC vertical line", value=False, key="ovl_now")
        layer_green = st.checkbox("Green Az 30° HSV mask", value=False, key="ovl_green")
        layer_csv = st.checkbox("Accumulated CSV samples", value=False, key="ovl_csv")

    layers = {
        "dots": layer_dots,
        "bbox": layer_bbox,
        "yticks": layer_yticks,
        "ygrid": layer_ygrid,
        "corners": layer_corners,
        "blue": layer_blue,
        "legend": layer_legend,
        "dropcols": layer_dropcols,
        "outliers": layer_outliers,
        "now": layer_now,
        "green": layer_green,
        "csv": layer_csv,
    }

    @st.cache_data(ttl=INGEST_CACHE_TTL_SECONDS, show_spinner=False)
    def _fetch_png_bytes(url: str) -> bytes | None:
        """Fetch a USGS PNG and return the raw bytes. None on failure."""
        from kilauea_tracker.ingest.fetch import fetch_tilt_png
        try:
            result = fetch_tilt_png(url, None)
            return result.body
        except Exception:
            return None

    def _build_overlay_png(
        raw_bytes: bytes,
        calibration,
        report,
        layers: dict,
    ) -> tuple[bytes | None, dict]:
        """Render the PNG with every enabled layer drawn on top.

        Returns (png_bytes, stats) where stats has counters for each layer
        so the caption can surface what was actually drawn.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return None, {}
        if calibration is None or calibration.y_slope == 0:
            return None, {}
        try:
            import cv2
            import numpy as np
            from kilauea_tracker.ingest.trace import (
                BLUE_HUE_MIN, BLUE_HUE_MAX,
                SATURATION_FLOOR, VALUE_FLOOR,
                LEGEND_EXCLUSION_PLOT_RELATIVE,
            )
            from kilauea_tracker.ingest.trace import trace_curve
            from kilauea_tracker.config import CURVE_MAX_COLUMN_WIDTH_PIXELS
        except ImportError:
            return None, {}

        stats: dict = {}
        try:
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return None, {}
        except Exception:
            return None, {}

        x0, y0, x1, y1 = calibration.plot_bbox
        span_seconds = (calibration.x_end - calibration.x_start).total_seconds()
        px_span = max(1.0, float(x1 - x0))

        def date_to_px(ts) -> float:
            return x0 + (ts - calibration.x_start).total_seconds() * px_span / span_seconds

        def val_to_py(val) -> float:
            return (val - calibration.y_intercept) / calibration.y_slope

        img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
        W, H = img.size
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # ── Tier 2 masks (need the HSV-converted plot crop) ───────────
        need_masks = layers.get("blue") or layers.get("green") or layers.get("dropcols")
        blue_mask = green_mask = None
        if need_masks:
            plot = img_bgr[y0:y1, x0:x1]
            if plot.size > 0:
                hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)
                blue_mask = (
                    (hsv[:, :, 0] >= BLUE_HUE_MIN)
                    & (hsv[:, :, 0] <= BLUE_HUE_MAX)
                    & (hsv[:, :, 1] >= SATURATION_FLOOR)
                    & (hsv[:, :, 2] >= VALUE_FLOOR)
                )
                # Green Az 30° curve — H≈60 in OpenCV HSV.
                green_mask = (
                    (hsv[:, :, 0] >= 35)
                    & (hsv[:, :, 0] <= 85)
                    & (hsv[:, :, 1] >= SATURATION_FLOOR)
                    & (hsv[:, :, 2] >= VALUE_FLOOR)
                )

        # ── Blue-mask highlight (cyan) ────────────────────────────────
        if layers.get("blue") and blue_mask is not None:
            ys, xs = np.where(blue_mask)
            stats["blue_px"] = int(ys.size)
            for py, px in zip(ys, xs):
                overlay.putpixel((int(px + x0), int(py + y0)), (0, 200, 255, 140))

        # ── Green-mask highlight (yellow, contrasts with green) ───────
        if layers.get("green") and green_mask is not None:
            ys, xs = np.where(green_mask)
            stats["green_px"] = int(ys.size)
            for py, px in zip(ys, xs):
                overlay.putpixel((int(px + x0), int(py + y0)), (255, 230, 0, 140))

        # ── Legend-exclusion rectangle ────────────────────────────────
        if layers.get("legend"):
            lx0, ly0, lx1, ly1 = LEGEND_EXCLUSION_PLOT_RELATIVE
            draw.rectangle(
                (x0 + lx0, y0 + ly0, x0 + lx1, y0 + ly1),
                fill=(255, 100, 0, 60),
                outline=(255, 100, 0, 200),
                width=1,
            )

        # ── Dropped-width columns (orange ticks at the top) ───────────
        if layers.get("dropcols") and blue_mask is not None:
            col_counts = blue_mask.sum(axis=0)
            dropped = np.where(col_counts > CURVE_MAX_COLUMN_WIDTH_PIXELS)[0]
            stats["dropped_cols"] = int(dropped.size)
            for col_offset in dropped:
                px = int(col_offset + x0)
                draw.line((px, y0, px, y0 + 6), fill=(255, 140, 0, 220), width=1)

        # ── Integer-µrad gridlines (dashed, cyan) ─────────────────────
        if layers.get("ygrid"):
            y_top = calibration.pixel_to_microradians(y0)
            y_bot = calibration.pixel_to_microradians(y1)
            lo, hi = sorted((y_top, y_bot))
            for val in range(int(np.floor(lo)), int(np.ceil(hi)) + 1):
                py = val_to_py(val)
                if not (y0 <= py <= y1):
                    continue
                for seg_x in range(x0, x1, 6):
                    draw.line(
                        (seg_x, py, min(seg_x + 3, x1), py),
                        fill=(100, 220, 255, 150),
                        width=1,
                    )
                if font is not None:
                    draw.text(
                        (x1 - 22, py - 6),
                        f"{val:+d}",
                        fill=(100, 220, 255, 220),
                        font=font,
                    )

        # ── Y-tick OCR markers (horizontal line + label) ──────────────
        if layers.get("yticks"):
            for py, val in (calibration.y_labels_found or []):
                if not (y0 - 3 <= py <= y1 + 3):
                    continue
                draw.line((x0, py, x0 + 18, py), fill=(255, 0, 255, 220), width=2)
                if font is not None:
                    draw.text(
                        (x0 + 20, py - 6),
                        f"OCR {val:+.1f}",
                        fill=(255, 0, 255, 230),
                        font=font,
                    )

        # ── 'Now' vertical line ───────────────────────────────────────
        if layers.get("now"):
            from datetime import datetime as _dt, timezone as _tz
            now = pd.Timestamp(_dt.now(tz=_tz.utc)).tz_localize(None)
            px_now = date_to_px(now)
            if x0 <= px_now <= x1:
                draw.line(
                    (px_now, y0, px_now, y1),
                    fill=(0, 255, 120, 200),
                    width=1,
                )
                if font is not None:
                    draw.text(
                        (px_now + 2, y0 + 2),
                        "now",
                        fill=(0, 255, 120, 230),
                        font=font,
                    )

        # ── Outlier drops (yellow X markers at dropped samples) ───────
        if layers.get("outliers"):
            dropped_samples = getattr(report, "dropped_outlier_samples", []) or []
            n_drawn = 0
            for ts, val, _median in dropped_samples:
                px = date_to_px(ts)
                py = val_to_py(val)
                if not (x0 <= px <= x1 and y0 <= py <= y1):
                    continue
                r = 4
                draw.line((px - r, py - r, px + r, py + r), fill=(255, 230, 0, 230), width=2)
                draw.line((px - r, py + r, px + r, py - r), fill=(255, 230, 0, 230), width=2)
                n_drawn += 1
            stats["outliers"] = n_drawn

        # ── Plot bbox outline (yellow rectangle) ──────────────────────
        if layers.get("bbox"):
            draw.rectangle(
                (x0, y0, x1, y1),
                outline=(255, 255, 0, 230),
                width=1,
            )

        # ── Corner (date, µrad) labels ────────────────────────────────
        if layers.get("corners") and font is not None:
            def _fmt_dt(ts):
                try:
                    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    return str(ts)
            y_top = calibration.pixel_to_microradians(y0)
            y_bot = calibration.pixel_to_microradians(y1)
            corner_labels = [
                (x0 + 2, y0 + 2, f"TL {_fmt_dt(calibration.x_start)} {y_top:+.1f}"),
                (max(x0, x1 - 220), y0 + 2, f"TR {_fmt_dt(calibration.x_end)} {y_top:+.1f}"),
                (x0 + 2, y1 - 14, f"BL {_fmt_dt(calibration.x_start)} {y_bot:+.1f}"),
                (max(x0, x1 - 220), y1 - 14, f"BR {_fmt_dt(calibration.x_end)} {y_bot:+.1f}"),
            ]
            for tx, ty, txt in corner_labels:
                # Draw a dark rectangle behind text for legibility.
                if font is not None:
                    w = draw.textlength(txt, font=font) if hasattr(draw, "textlength") else 6 * len(txt)
                    draw.rectangle((tx - 1, ty - 1, tx + w + 1, ty + 11), fill=(0, 0, 0, 170))
                    draw.text((tx, ty), txt, fill=(255, 255, 0, 230), font=font)

        # ── Accumulated CSV samples (purple) ──────────────────────────
        # Reads the per-source CSV and plots each sample at (pixel_x, pixel_y)
        # using the CURRENT PNG's calibration. Offsets vs the blue line
        # reveal the delta between (a) the CSV's cumulative reference frame
        # (frame-aligned by append_history across fetches) and (b) the raw
        # frame USGS labelled on this PNG. Uniform vertical offset = USGS
        # rebaseline (expected); parallel tracks or sawtooth = alignment bug.
        if layers.get("csv"):
            try:
                from kilauea_tracker.config import source_csv_path
                csv_path = source_csv_path(report.source_name)
            except Exception:
                csv_path = None
            n_csv_drawn = 0
            n_csv_clipped = 0
            if csv_path is not None and csv_path.exists():
                try:
                    df_csv = pd.read_csv(csv_path, parse_dates=[DATE_COL])
                except Exception:
                    df_csv = None
                if df_csv is not None and len(df_csv) > 0:
                    in_x = (df_csv[DATE_COL] >= calibration.x_start) & (
                        df_csv[DATE_COL] <= calibration.x_end
                    )
                    df_win = df_csv[in_x]
                    dot_r = 2
                    for _, row in df_win.iterrows():
                        px = date_to_px(row[DATE_COL])
                        py = val_to_py(row[TILT_COL])
                        if not (x0 <= px <= x1 and y0 <= py <= y1):
                            n_csv_clipped += 1
                            continue
                        draw.ellipse(
                            (px - dot_r, py - dot_r, px + dot_r, py + dot_r),
                            fill=(170, 60, 230, 90),
                            outline=(130, 0, 200, 160),
                        )
                        n_csv_drawn += 1
            stats["csv_drawn"] = n_csv_drawn
            stats["csv_clipped_y"] = n_csv_clipped

        # ── Re-traced dots (red, translucent) ─────────────────────────
        n_dots = 0
        if layers.get("dots"):
            try:
                traced = trace_curve(img_bgr, calibration)
            except Exception:
                traced = None
            if traced is not None:
                dot_r = 2
                for _, row in traced.iterrows():
                    px = date_to_px(row[DATE_COL])
                    py = val_to_py(row[TILT_COL])
                    if not (x0 - dot_r <= px <= x1 + dot_r and y0 - dot_r <= py <= y1 + dot_r):
                        continue
                    draw.ellipse(
                        (px - dot_r, py - dot_r, px + dot_r, py + dot_r),
                        fill=(255, 40, 40, 80),
                        outline=(200, 0, 0, 140),
                    )
                    n_dots += 1
        stats["dots"] = n_dots

        blended = Image.alpha_composite(img, overlay)
        buf = io.BytesIO()
        blended.save(buf, format="PNG", optimize=False)
        return buf.getvalue(), stats

    # Map source_name → URL; replicated here because the original
    # mapping in the USGS-source-plots expander is built further down.
    _inspector_url_map = {TILT_SOURCE_NAME[s]: USGS_TILT_URLS[s] for s in ALL_SOURCES}

    for r in reports:
        if r.source_name not in _inspector_url_map:
            continue
        if r.calibration is None:
            continue
        raw = _fetch_png_bytes(_inspector_url_map[r.source_name])
        if raw is None:
            st.caption(f"⚠️ {r.source_name}: could not fetch PNG")
            continue
        overlay_bytes, stats = _build_overlay_png(raw, r.calibration, r, layers)

        cal = r.calibration
        bx0, by0, bx1, by1 = cal.plot_bbox
        y_top = cal.pixel_to_microradians(by0)
        y_bot = cal.pixel_to_microradians(by1)
        y_span = abs(y_top - y_bot)
        x_span_hrs = (cal.x_end - cal.x_start).total_seconds() / 3600.0

        caption_bits = [f"**{r.source_name}**"]
        if stats.get("dots"):
            caption_bits.append(f"{stats['dots']} re-traced dots")
        if stats.get("blue_px"):
            caption_bits.append(f"{stats['blue_px']:,} blue px")
        if stats.get("green_px"):
            caption_bits.append(f"{stats['green_px']:,} green px")
        if stats.get("dropped_cols"):
            caption_bits.append(f"{stats['dropped_cols']} width-dropped cols")
        if stats.get("outliers"):
            caption_bits.append(f"{stats['outliers']} outlier drops")
        if stats.get("csv_drawn") is not None and (layers.get("csv")):
            csv_drawn = stats.get("csv_drawn", 0)
            csv_clipped = stats.get("csv_clipped_y", 0)
            csv_bit = f"{csv_drawn} CSV samples in-plot"
            if csv_clipped:
                csv_bit += f" ({csv_clipped} clipped off-range → frame drift)"
            caption_bits.append(csv_bit)
        st.markdown(" · ".join(caption_bits))

        if overlay_bytes is not None:
            st.image(
                overlay_bytes,
                width="stretch",
                output_format="PNG",
            )
        else:
            st.caption("(overlay unavailable — cv2/PIL missing)")

        # Debug metadata — the numbers the rest of the pipeline trusts
        # for this PNG. First place to check when overlays disagree.
        with st.expander(f"🔧 {r.source_name} calibration diagnostics", expanded=False):
            dbg_cols = st.columns(2)
            with dbg_cols[0]:
                st.markdown("**Time axis**")
                st.caption(
                    f"x-start (UTC): `{cal.x_start}`  \n"
                    f"x-end (UTC): `{cal.x_end}`  \n"
                    f"window: `{x_span_hrs:.1f}` hrs  \n"
                    f"title PSM: `{cal.title_psm_used or '—'}`  \n"
                    f"title OCR text:  \n`{(cal.title_raw_text or '').strip() or '(empty)'}`"
                )
            with dbg_cols[1]:
                st.markdown("**Y axis**")
                y_labels_str = ", ".join(
                    f"{val:+.1f}@px{py}"
                    for py, val in (cal.y_labels_found or [])
                )
                fallback_note = (
                    f"(history fallback median={cal.y_slope_history_median:.4f})"
                    if cal.y_slope_fallback_used
                    else "(fit from this run's OCR labels)"
                )
                st.caption(
                    f"y-slope: `{cal.y_slope:+.4f}` µrad/px {fallback_note}  \n"
                    f"y-intercept: `{cal.y_intercept:+.3f}` µrad  \n"
                    f"plot top → bottom: `{y_top:+.2f}` → `{y_bot:+.2f}` µrad "
                    f"(span `{y_span:.1f}` µrad)  \n"
                    f"µrad/px: `{cal.microradians_per_pixel():.4f}`  \n"
                    f"plot_bbox (x0,y0,x1,y1): `({bx0}, {by0}, {bx1}, {by1})`  \n"
                    f"y labels OCR'd: {y_labels_str or '(none)'}"
                )
        st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# USGS Kīlauea summit webcams
# ─────────────────────────────────────────────────────────────────────────────
#
# Eight live webcams from USGS HVO. Each one's "M.jpg" endpoint returns
# the latest still capture. URLs were extracted from
# https://www.usgs.gov/volcanoes/kilauea/summit-webcams and HEAD-checked
# 2026-04-09. The user-friendly USGS page above also has the same
# images plus camera location maps and refresh information.
USGS_WEBCAMS: list[tuple[str, str, str]] = [
    (
        "K2cam",
        "Caldera view from Uēkahuna bluff observation tower",
        "https://volcanoes.usgs.gov/observatories/hvo/cams/K2cam/images/M.jpg",
    ),
    (
        "V1cam",
        "West Halemaʻumaʻu crater from northwest rim",
        "https://volcanoes.usgs.gov/observatories/hvo/cams/V1cam/images/M.jpg",
    ),
    (
        "V2cam",
        "East Halemaʻumaʻu crater from northeast rim",
        "https://volcanoes.usgs.gov/cams/V2cam/images/M.jpg",
    ),
    (
        "V3cam",
        "South Halemaʻumaʻu crater from south rim",
        "https://volcanoes.usgs.gov/cams/V3cam/images/M.jpg",
    ),
    (
        "B1cam",
        "Caldera down-dropped block from east rim",
        "https://volcanoes.usgs.gov/observatories/hvo/cams/B1cam/images/M.jpg",
    ),
    (
        "KWcam",
        "Halemaʻumaʻu panorama from west rim",
        "https://volcanoes.usgs.gov/observatories/hvo/cams/KWcam/images/M.jpg",
    ),
    (
        "F1cam",
        "Thermal imagery from west rim",
        "https://volcanoes.usgs.gov/observatories/hvo/cams/F1cam/images/M.jpg",
    ),
    (
        "KPcam",
        "Summit view from Mauna Loa Strip Road",
        "https://volcanoes.usgs.gov/cams/KPcam/images/M.jpg",
    ),
]

with st.expander("📷 USGS Kīlauea summit webcams"):
    st.caption(
        "Live still captures from the eight USGS HVO webcams that look at "
        "Kīlauea's summit caldera and Halemaʻumaʻu crater. Click any "
        "thumbnail to open the full-resolution image. Visit the "
        "[USGS webcams page]"
        "(https://www.usgs.gov/volcanoes/kilauea/summit-webcams) for the "
        "live time-lapse feeds and map of camera locations."
    )
    # 2-column grid of webcams.
    for i in range(0, len(USGS_WEBCAMS), 2):
        pair = USGS_WEBCAMS[i : i + 2]
        cols = st.columns(2)
        for col, (name, desc, url) in zip(cols, pair):
            with col:
                try:
                    st.image(url, width="stretch")
                except Exception as e:
                    st.caption(f"⚠️ could not load {name}: {e}")
                st.markdown(f"**[{name}]({url})** &nbsp;·&nbsp; {desc}")


# ─────────────────────────────────────────────────────────────────────────────
# Detail expanders
# ─────────────────────────────────────────────────────────────────────────────

with st.expander(
    f"📍 Detected peaks ({len(all_peaks)} total, {len(recent_peaks)} used for fit)"
):
    if len(all_peaks) == 0:
        st.info(
            "No peaks detected at the current sensitivity. Try lowering the "
            "prominence threshold."
        )
    else:
        # Tag the rows that fed the trendline fit BEFORE sorting, so the flag
        # survives the reverse-by-date sort below.
        fit_dates = set(recent_peaks[DATE_COL])
        display_peaks = all_peaks.copy()
        display_peaks["used_for_fit"] = display_peaks[DATE_COL].isin(fit_dates)
        display_peaks = display_peaks.sort_values(DATE_COL, ascending=False).reset_index(drop=True)
        display_peaks = display_peaks.rename(
            columns={
                DATE_COL: "Date",
                TILT_COL: "Tilt (µrad)",
                "prominence": "Prominence",
                "used_for_fit": "Used for fit",
            }
        )
        st.dataframe(display_peaks, width="stretch", hide_index=True)

with st.expander("🔬 Model diagnostics"):
    diag = prediction.fit_diagnostics
    if not diag:
        st.write("No diagnostics available.")
    else:
        # ─── Trendline slope ──────────────────────────────────────────────
        slope = diag.get("trendline_slope_per_day")
        if slope is not None:
            direction = "rising" if slope > 0 else "falling"
            st.markdown(
                f"**Trendline slope** &nbsp;·&nbsp; "
                f"`{slope:+.4f} µrad/day` &nbsp;·&nbsp; "
                f"{direction} ~`{abs(slope) * 7:.2f}` µrad/week"
            )
            st.caption(
                "How fast the peak heights are changing over time. Positive "
                "means the deformation episodes are getting more intense (more "
                "magma pressure builds before each release); negative means "
                "they're tapering off. The trendline is the linear regression "
                "through the last N peaks (slider-controlled)."
            )
            st.markdown("&nbsp;")

        # ─── Current episode sample count ────────────────────────────────
        n_episode = diag.get("current_episode_n")
        if n_episode is not None:
            st.markdown(
                f"**Current episode samples** &nbsp;·&nbsp; "
                f"`{n_episode}` tilt readings since the last detected peak"
            )
            st.caption(
                "How many tilt samples fed the exponential saturation fit. "
                "The fit needs at least 4 to estimate its 3 parameters; more "
                "samples (and more variation across them) give a tighter fit "
                "and a narrower confidence band."
            )
            st.markdown("&nbsp;")

        # ─── Exponential fit parameters ──────────────────────────────────
        if prediction.exp_params:
            A, k, C = prediction.exp_params
            asymptote = A + C
            tau_days = 1.0 / k if k > 0 else float("inf")
            half_life = math.log(2) / k if k > 0 else float("inf")
            st.markdown(
                f"**Exponential saturation fit** &nbsp;·&nbsp; "
                f"`tilt = A·(1 − exp(−k·t)) + C`"
            )
            col_a, col_k, col_c, col_asym = st.columns(4)
            col_a.metric("A (amplitude)", f"{A:.2f} µrad")
            col_k.metric("k (rise rate)", f"{k:.4f} /day")
            col_c.metric("C (baseline)", f"{C:.2f} µrad")
            col_asym.metric("A + C (asymptote)", f"{asymptote:.2f} µrad")
            st.caption(
                f"Each parameter has a job. **A** is the total rise amplitude "
                f"this episode will gain if it's allowed to fully saturate. "
                f"**k** is how fast it rises — the time constant 1/k is "
                f"~{tau_days:.1f} days to reach 63% of A, and the half-time "
                f"ln(2)/k is ~{half_life:.1f} days to reach 50% of A. "
                f"**C** is the starting tilt offset where the episode began "
                f"(the trough after the previous eruption). **A + C** is the "
                f"asymptote — where tilt would settle if no eruption "
                f"interrupted the rise. The next fountain event is predicted "
                f"to happen well *before* this asymptote, when the rising "
                f"exp curve crosses the linear trendline through recent peaks."
            )
            st.markdown("&nbsp;")

        # ─── Warnings + errors ───────────────────────────────────────────
        if "warning" in diag:
            st.warning(f"⚠️ {diag['warning']}")
        if "error" in diag:
            st.error(f"❌ {diag['error']}")
        if "exp_fit_error" in diag:
            st.error(f"❌ Exp fit error: {diag['exp_fit_error']}")

        # ─── Footer: peaks in fit ────────────────────────────────────────
        st.markdown(
            f"**Peaks in fit** &nbsp;·&nbsp; `{prediction.n_peaks_in_fit}` "
            "(controlled by the sidebar's *Number of recent peaks* slider)"
        )

with st.expander("📡 Ingest pipeline status"):
    if not reports:
        st.write("No ingest reports — pipeline hasn't run yet.")
    else:
        st.markdown("**Per-source fetch & trace**")
        st.caption(
            "Each USGS source is fetched, traced, and appended to its own "
            "raw CSV under `data/sources/`. The merged history is rebuilt "
            "from these files plus the digital and legacy reference data "
            "by the reconciliation step below."
        )
        for r in reports:
            status_icon = "✅" if r.error is None else "❌"
            with st.container():
                st.markdown(
                    f"{status_icon} **{r.source_name}** — "
                    f"`{r.rows_traced}` rows traced, "
                    f"`{r.rows_appended}` appended to source CSV"
                )
                if r.last_modified:
                    st.caption(f"Last-Modified: {r.last_modified}")
                if r.error:
                    st.caption(f"Error: {r.error}")
                if r.warnings:
                    for w in r.warnings:
                        st.caption(f"⚠️ {w}")
                if r.calibration:
                    cal = r.calibration
                    st.caption(
                        f"y-axis ticks recovered: {len(cal.y_labels_found)}, "
                        f"y-fit residual: {cal.fit_residual_per_axis.get('y_max_residual_microrad', 0):.3f} µrad"
                    )

        if reconcile_report is not None:
            st.divider()
            st.markdown("**Reconciliation** (anchor-based alignment + priority merge)")
            st.caption(
                "Every source is shifted into a single y-frame anchored on "
                "the highest-confidence source it can transitively reach. "
                "Higher-priority sources win per 15-minute bucket in the "
                "merge. The merged history has "
                f"`{reconcile_report.rows_out}` rows."
            )
            for s in reconcile_report.sources:
                if s.is_anchor:
                    detail = "🎯 **anchor** (defines y-frame, offset 0)"
                elif s.offset_microrad is not None:
                    detail = (
                        f"offset `{s.offset_microrad:+.3f}` µrad "
                        f"(median over `{s.overlap_buckets}` overlap buckets)"
                    )
                else:
                    detail = f"⚠️ **unaligned** — {s.note}"
                st.markdown(
                    f"- `{s.name}` &nbsp;·&nbsp; `{s.rows_in}` rows &nbsp;·&nbsp; {detail}"
                )

            if reconcile_report.conflicts:
                st.markdown(
                    f"\n**{len(reconcile_report.conflicts)} bucket conflict(s)** — "
                    "the higher-priority source's value won, but the disagreement "
                    "is recorded here for audit:"
                )
                # Show the 5 worst (largest |delta|) conflicts to keep the UI brief
                worst = sorted(
                    reconcile_report.conflicts,
                    key=lambda c: abs(c.delta),
                    reverse=True,
                )[:5]
                for c in worst:
                    bucket_str = pd.Timestamp(c.bucket).strftime("%Y-%m-%d %H:%M")
                    st.caption(
                        f"`{bucket_str}` — "
                        f"`{c.winning_source}` ({c.winning_tilt:+.2f}) "
                        f"vs `{c.losing_source}` ({c.losing_tilt:+.2f}) "
                        f"→ Δ {c.delta:+.2f} µrad"
                    )
                if len(reconcile_report.conflicts) > 5:
                    st.caption(
                        f"…and {len(reconcile_report.conflicts) - 5} more "
                        "(smaller deltas)"
                    )

with st.expander("🛰 USGS source plots"):
    st.caption(
        "These are the live PNG plots we fetch from USGS Hawaiian Volcano "
        "Observatory. Each one covers a different time window — they're "
        "displayed here at thumbnail size; click any title to open the "
        "full-resolution original on the USGS site."
    )

    # Map source_name (e.g. "two_day") back to the upstream URL via the
    # TiltSource enum so we can render each PNG inline alongside its stats.
    _source_to_url = {TILT_SOURCE_NAME[s]: USGS_TILT_URLS[s] for s in ALL_SOURCES}
    _png_reports = [r for r in reports if r.source_name in _source_to_url]
    # Render in the priority order from ALL_SOURCES (THREE_MONTH first, then
    # MONTH, WEEK, TWO_DAY, DEC2024_TO_NOW) so the longest-window plot leads.
    _order = {TILT_SOURCE_NAME[s]: i for i, s in enumerate(ALL_SOURCES)}
    _png_reports.sort(key=lambda r: _order.get(r.source_name, 999))

    for r in _png_reports:
        url = _source_to_url[r.source_name]
        col_img, col_meta = st.columns([3, 2])
        with col_img:
            try:
                st.image(url, width="stretch")
            except Exception as e:
                st.caption(f"⚠️ could not load preview: {e}")
        with col_meta:
            st.markdown(f"**[{r.source_name}.png]({url})**")
            if r.last_modified:
                st.caption(f"USGS Last-Modified: {r.last_modified}")
            else:
                st.caption("USGS Last-Modified: —")
            st.caption(f"Rows traced this run: `{r.rows_traced}`")
            st.caption(f"Appended to source CSV: `{r.rows_appended}`")
            if r.calibration:
                cal = r.calibration
                st.caption(
                    f"Calibration: {len(cal.y_labels_found)} y-ticks recovered, "
                    f"residual {cal.fit_residual_per_axis.get('y_max_residual_microrad', 0):.2f} µrad"
                )
            if r.error:
                st.caption(f"❌ {r.error}")
            elif not r.fetched:
                st.caption("ℹ️ Not changed since last poll (304)")
        st.markdown("---")

    # ── Non-PNG reference data sources ────────────────────────────────────
    st.markdown("**Other reference data**")
    st.caption(
        "Beyond the live PNG plots above, the reconciliation layer also "
        "consumes a one-shot **digital tiltmeter** dataset from a USGS "
        "research release. It's not live but it's the most accurate source "
        "we have for the period it covers, so the alignment math anchors "
        "every other source against it."
    )
    st.markdown(
        "- **`digital`** &nbsp;·&nbsp; raw 1-minute samples of (X, Y) "
        "tilt from the UWD LILY borehole tiltmeter, projected onto "
        "azimuth 300° and resampled to 30-min means. Covers **Jan–Jun "
        "2025** in 6 segments split at instrument relevelings. Used as "
        "the global y-frame anchor for all other sources. Processed "
        "locally by `scripts/import_digital_data.py`. Source: "
        "[USGS ScienceBase research release]"
        "(https://www.sciencebase.gov/catalog/item/67ead922d34ed02007f83585)."
    )
    st.caption(
        "ℹ️ A `legacy` hand-traced PlotDigitizer CSV from the v1 "
        "prototype was previously merged in here too, but it was removed "
        "in 2026-04 because its samples didn't reliably match "
        "`dec2024_to_now`'s auto-traced frame and were creating systemic "
        "~6 µrad offsets in the Jul-Aug 2025 region. `dec2024_to_now` "
        "covers the same range with one consistent y-frame."
    )


with st.expander("📜 Recent refresh runs"):
    st.caption(
        "Structured diagnostic reports written by the ingest pipeline on "
        "every cron-triggered refresh. Each file captures the OCR'd title "
        "text and PSM mode per source, the calibration, frame-alignment "
        "offsets, per-row outliers that were filtered, reconciliation "
        "offsets, archive promotion counts, and any warnings. The full "
        "history is committed to `data/run_reports/` (capped at the last "
        "90 runs)."
    )
    _RUN_REPORTS_DIR = Path(HISTORY_CSV).resolve().parent / "run_reports"
    if not _RUN_REPORTS_DIR.exists():
        st.info(
            "No run reports yet — the first report will be written on the "
            "next ingest pipeline run."
        )
    else:
        _report_files = sorted(
            _RUN_REPORTS_DIR.glob("*.json"), reverse=True
        )[:7]
        if not _report_files:
            st.info("No run reports yet.")
        for _rp in _report_files:
            with st.expander(f"`{_rp.name}`"):
                try:
                    import json as _json

                    _doc = _json.loads(_rp.read_text())
                except Exception as e:
                    st.caption(f"⚠️ could not parse: {e}")
                    continue
                _rows = []
                for _src in _doc.get("per_source", []):
                    _rows.append(
                        {
                            "source": _src["source_name"],
                            "fetched": _src.get("fetched"),
                            "rows_traced": _src.get("rows_traced"),
                            "outliers": _src.get("rows_outlier_dropped", 0),
                            "frame_offset": round(
                                _src.get("frame_offset_microrad") or 0.0, 3
                            ),
                            "psm": _src.get("title_psm_used"),
                            "error": bool(_src.get("error")),
                        }
                    )
                if _rows:
                    st.dataframe(pd.DataFrame(_rows), hide_index=True)
                _arc = _doc.get("archive", {})
                if _arc:
                    st.caption(
                        f"archive: before {_arc.get('rows_in_archive_before')} → "
                        f"after {_arc.get('rows_in_archive_after')} "
                        f"(promoted {_arc.get('rows_promoted')}, "
                        f"deferred by quorum {_arc.get('rows_deferred_by_quorum')})"
                    )


with st.expander("ℹ️ How does this work?"):
    st.markdown(
        """
        **The model in three sentences.**
        Each Kīlauea eruption episode shows tilt rising along a saturating
        exponential curve until pressure builds enough to trigger a fountain
        event, which releases the pressure and resets tilt sharply downward.
        The "next fountain event" prediction is the date where the rising
        exponential of the *current* episode crosses the linear trendline
        through the *peaks* of recent episodes. The "earliest likely" prediction
        does the same thing but only with the last 3 peaks, whose steeper trend
        gives an aggressive (earlier) bound.

        **Tunable parameters live in the sidebar.** Peak sensitivity controls
        which local maxima count as "real" episodic peaks. The trendline window
        controls how many of those peaks feed the linear fit.

        **Data source.** Electronic tilt at the **UWD** station (Uēkahuna,
        on the summit caldera rim), **azimuth 300°**, published by USGS
        Hawaiian Volcano Observatory. The "Refresh" button re-fetches the
        five tilt plots (2-day, week, month, 3-month, and the long-history
        Dec 2024 → now) and merges new samples into the local history
        cache.

        **About azimuth 300°.** Tilt is a vector quantity, so each
        measurement is projected onto a chosen compass direction. Per USGS:
        *"On July 9, 2025, tilt azimuths for SDH and UWD plots have been
        updated from 320 to 300 degrees to optimize displaying maximum
        magnitudes of deformation consistent with the current activity at
        the summit of Kīlauea."* This means historical tilt readings from
        before that date were originally projected at 320° rather than 300°
        — but the USGS plots themselves have been re-rendered onto the new
        300° projection across the full historical range, so all data this
        app ingests is consistent.
        """
    )
