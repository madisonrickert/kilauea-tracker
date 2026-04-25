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
import logging
import math
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

# Emit pipeline warnings/errors to stderr so Streamlit Cloud's log viewer
# captures them. The in-UI IngestReport mechanism surfaces failures to the
# user browsing the app; this layer makes "did this fail overnight?" a
# grep-able question in the Cloud admin console. Kept at WARNING so only
# actionable signals flow through — INFO-level traffic (normal fetch /
# trace counts) would drown out the important bits.
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
    force=False,
)

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

from kilauea_tracker.cache import load_history
from kilauea_tracker.config import (
    ALL_SOURCES,
    HISTORY_CSV,
    INGEST_CACHE_TTL_SECONDS,
    PEAK_DEFAULTS,
    TILT_SOURCE_NAME,
    USGS_TILT_URLS,
)
from kilauea_tracker.ingest.pipeline import (
    IngestRunResult,
    data_age_seconds,
    ingest_all,
    load_latest_run_report,
    try_acquire_refresh_slot,
)
from kilauea_tracker.model import DATE_COL, TILT_COL, predict
from kilauea_tracker.peaks import detect_peaks
from kilauea_tracker.plotting import build_figure
from kilauea_tracker.ui import (
    about_tab,
    cameras,
    hero,
    state_banner,
)
from kilauea_tracker.ui.diagnostics import (
    episode_samples_tint,
    exp_amplitude_tint,
    exp_k_tint,
    render_chip_html,
    trendline_slope_tint,
)
from kilauea_tracker.ui.styles import build_style_block

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Kīlauea Fountain Event Tracker",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="collapsed",  # sidebar retired; controls live in the top bar + tabs
)

# Inject the full volcano palette + typography + hero/chip/banner styles.
# Replaces the one-off metric-wrap override with a full design-system block.
st.markdown(build_style_block(), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Ingest — non-blocking. Page renders from on-disk CSVs; a background thread
# refreshes if data is stale; the manual Refresh button is the only path that
# shows a spinner.
# ─────────────────────────────────────────────────────────────────────────────

# Re-poll USGS if on-disk data is older than this. Background thread fires
# on page load when this threshold is crossed. 10 min matches USGS's typical
# publish cadence for the rolling-window PNGs.
STALE_THRESHOLD_SECONDS = 10 * 60

# Don't fire the background refresh more than once per this window —
# coordinates concurrent users via the file lock so N users → 1 fetch.
BACKGROUND_REFRESH_COOLDOWN_SECONDS = 5 * 60

# Tighter cooldown for the manual Refresh button — explicit clicks should
# feel responsive, but spam-clicks still get gated.
USER_REFRESH_COOLDOWN_SECONDS = 30


def _run_ingest_with_spinner() -> None:
    """Synchronous ingest with a staged-status spinner overlay.

    Used only by the manual Refresh button. The spinner walks through the
    pipeline's natural stage boundaries via the `on_stage` callback so the
    user sees "Fetching week…", "Reconciling sources…", "Updating archive…"
    instead of a single static label. The placeholder is drained on
    completion so it doesn't leave a persistent collapsed panel above the
    rest of the page.

    The returned data isn't used directly — `ingest_all()` writes fresh
    CSVs and a new run report to disk, and the next script rerun (via
    `st.rerun()` after this returns) picks them up via
    `load_latest_run_report()` at module scope.
    """
    placeholder = st.empty()
    with placeholder.container():
        with st.status("Fetching latest USGS tilt data…", expanded=False) as status:
            ingest_all(on_stage=lambda msg: status.update(label=msg))
            status.update(label="Up to date", state="complete")
    placeholder.empty()


def _maybe_kick_background_refresh() -> None:
    """Fire a background ingest if on-disk data is stale AND no other
    session/thread already refreshed within the cooldown window.

    Silent: never blocks the page render, never raises, never touches
    Streamlit widgets or session_state. The thread writes to disk only;
    the chart fragment (running every 30s) picks up the new mtime and
    re-renders automatically.
    """
    if data_age_seconds() < STALE_THRESHOLD_SECONDS:
        return
    acquired, _ = try_acquire_refresh_slot(BACKGROUND_REFRESH_COOLDOWN_SECONDS)
    if not acquired:
        return  # someone else is on it

    def _worker() -> None:
        try:
            ingest_all()
        except Exception:
            logging.getLogger(__name__).exception("background refresh failed")

    threading.Thread(target=_worker, daemon=True).start()


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
# Session-state defaults for every widget in the app.
# ─────────────────────────────────────────────────────────────────────────────
# The sidebar was retired in favor of a compact top bar + per-tab controls
# (chart-only controls live inside the Chart tab, overlay checkboxes inside
# the Pipeline tab, etc.). Streamlit recomputes the entire script top-to-
# bottom on every interaction, and the math that happens at module scope —
# peak detection, prediction, figure construction — has to run BEFORE any
# tab body renders. So we seed session_state with widget defaults here and
# read the values back immediately; later when the user drags a slider or
# clicks a toggle inside a tab, Streamlit writes the new value into the
# same session_state key and reruns, and this block picks it up.
#
# `tw_*` — top-bar / chart-widget keys (tz, trendline window, per-source)
# `adv_*` — advanced peak-detection sliders (Chart tab)
# `ovl_*` — PNG inspector overlay layers (Pipeline tab)

st.session_state.setdefault("tw_n_peaks_for_fit", 6)
st.session_state.setdefault("tw_show_per_source", False)
st.session_state.setdefault("tw_timezone_choice", "HST (Pacific/Honolulu)")
n_peaks_for_fit = st.session_state["tw_n_peaks_for_fit"]
show_per_source = st.session_state["tw_show_per_source"]
timezone_choice = st.session_state["tw_timezone_choice"]
DISPLAY_TZ = (
    "Pacific/Honolulu" if timezone_choice.startswith("HST") else "UTC"
)
TZ_LABEL = "HST" if DISPLAY_TZ == "Pacific/Honolulu" else "UTC"

st.session_state.setdefault("adv_min_prominence", PEAK_DEFAULTS.min_prominence)
st.session_state.setdefault("adv_min_distance_days", PEAK_DEFAULTS.min_distance_days)
st.session_state.setdefault("adv_min_height", -10.0)
min_prominence = st.session_state["adv_min_prominence"]
min_distance_days = st.session_state["adv_min_distance_days"]
min_height = st.session_state["adv_min_height"]

for _ovl_key, _default in (
    ("ovl_dots", True),
    ("ovl_bbox", False),
    ("ovl_yticks", False),
    ("ovl_ygrid", False),
    ("ovl_corners", False),
    ("ovl_blue", False),
    ("ovl_legend", False),
    ("ovl_dropcols", False),
    ("ovl_outliers", False),
    ("ovl_now", False),
    ("ovl_green", False),
    ("ovl_csv", False),
):
    st.session_state.setdefault(_ovl_key, _default)
layer_dots = st.session_state["ovl_dots"]
layer_bbox = st.session_state["ovl_bbox"]
layer_yticks = st.session_state["ovl_yticks"]
layer_ygrid = st.session_state["ovl_ygrid"]
layer_corners = st.session_state["ovl_corners"]
layer_blue = st.session_state["ovl_blue"]
layer_legend = st.session_state["ovl_legend"]
layer_dropcols = st.session_state["ovl_dropcols"]
layer_outliers = st.session_state["ovl_outliers"]
layer_now = st.session_state["ovl_now"]
layer_green = st.session_state["ovl_green"]
layer_csv = st.session_state["ovl_csv"]


# ─────────────────────────────────────────────────────────────────────────────
# Always-fresh data: page renders from on-disk CSVs immediately. A background
# thread refreshes if the data is stale (gated by the file-lock cooldown so
# concurrent users don't multiply USGS calls). The chart fragment polls
# every 30s for new file mtimes so users see fresh data without clicking.
# ─────────────────────────────────────────────────────────────────────────────

_maybe_kick_background_refresh()

ingest_result = load_latest_run_report() or IngestRunResult()
reports = ingest_result.per_source
reconcile_report = ingest_result.reconcile
if st.session_state.last_ingest_at is None:
    st.session_state.last_ingest_at = (
        ingest_result.run_finished_at_utc
        if ingest_result.run_finished_at_utc is not None
        else datetime.now(tz=timezone.utc)
    )

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

# Diagnostics are collected here but rendered inside the Pipeline tab, not at
# the top of the front page — they're a maintainer concern, not a visitor one.
# The top-bar freshness caption conveys the "N/M sources fetched" summary that
# casual visitors care about.


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


@st.fragment(run_every="30s")
def _watch_for_fresh_data() -> None:
    """Polling fragment: every 30s, check whether the tilt history CSV has
    been refreshed by the background ingest thread. If yes, trigger a full
    app rerun so the chart and "last update" caption reflect the fresh data
    without any user click.

    Renders no visible output — fragments are normally tied to a UI region,
    but here we use it purely for its `run_every` ticker. The mtime
    comparison + early-return ensures we only call `st.rerun(scope="app")`
    when the file actually changed, so we're not thrashing the whole page
    on every 30s tick.
    """
    current = _cache_mtime()
    last_seen = st.session_state.get("_last_data_mtime")
    if last_seen is None:
        st.session_state["_last_data_mtime"] = current
        return
    if current > last_seen:
        st.session_state["_last_data_mtime"] = current
        st.rerun(scope="app")


_watch_for_fresh_data()


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
# Header + top bar
# ─────────────────────────────────────────────────────────────────────────────

st.title("🌋 Kīlauea Fountain Event Tracker")
st.caption(
    "Predicting the next eruption pulse at Kīlauea from UWD tiltmeter data "
    "(Uēkahuna, Az 300°) published by the "
    "[USGS Hawaiian Volcano Observatory](https://www.usgs.gov/volcanoes/kilauea/science/monitoring-data-kilauea)."
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


# ── Compact top bar — replaces the old sidebar. ──────────────────────────
# Refresh button + tz selector + freshness caption, in a single row just
# above the primary navigation. The refresh button clears both ingest and
# safety-alert caches and calls st.rerun() so the next pass starts with an
# empty cache and the user sees the fresh data without a second click.
_tb_refresh, _tb_tz, _tb_meta = st.columns([1, 1.3, 2.5])
with _tb_refresh:
    if st.button(
        "🔄 Refresh",
        width="stretch",
        help="Re-fetch and re-trace all USGS tilt PNGs + safety alerts.",
    ):
        _acquired, _retry_in = try_acquire_refresh_slot(USER_REFRESH_COOLDOWN_SECONDS)
        if _acquired:
            _run_ingest_with_spinner()
            cached_safety_alerts.clear()
            st.session_state.last_ingest_at = None
            st.rerun()
        else:
            st.toast(f"Just refreshed — try again in {int(_retry_in)}s")
with _tb_tz:
    st.selectbox(
        "Time zone",
        options=["HST (Pacific/Honolulu)", "UTC"],
        key="tw_timezone_choice",
        label_visibility="collapsed",
        help="All displayed dates use this time zone. HST is local at Kīlauea.",
    )
with _tb_meta:
    _last_data = tilt_df[DATE_COL].max()
    if _last_data is not None and pd.notna(_last_data):
        _last_aware = (
            _last_data.tz_localize("UTC") if _last_data.tzinfo is None else _last_data
        )
        _age_seconds = int((pd.Timestamp.now(tz="UTC") - _last_aware).total_seconds())
        if _age_seconds < 3600:
            _age = f"{_age_seconds // 60}m ago"
        elif _age_seconds < 86400:
            _age = f"{_age_seconds // 3600}h ago"
        else:
            _age = f"{_age_seconds // 86400}d ago"
        _sample_meta = f"latest sample {_fmt_date(_last_data)} ({_age})"
    else:
        _sample_meta = "latest sample —"

    if st.session_state.last_ingest_at is not None:
        _successful = sum(1 for r in reports if r.error is None)
        _total = len(reports)
        _indicator = "🟢" if _successful == _total else ("🟡" if _successful else "🔴")
        _poll_ts = pd.Timestamp(st.session_state.last_ingest_at)
        _poll_meta = (
            f"{_indicator} {_successful}/{_total} sources · "
            f"last poll {_fmt_date(_poll_ts)}"
        )
    else:
        _poll_meta = "no polls yet this session"

    st.markdown(
        f'<div class="kt-topbar__meta">'
        f'<div>{_sample_meta}</div>'
        f'<div>{_poll_meta}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tabs — Now / Chart / Cameras / Data / Pipeline / About
# ─────────────────────────────────────────────────────────────────────────────

tab_now, tab_chart, tab_cameras, tab_pipeline, tab_about = st.tabs([
    "Now",
    "Chart",
    "Cameras",
    "Pipeline",
    "About",
])

# Tab ↔ URL sync. Streamlit's st.tabs has no Python-level "active tab"
# API — the active panel is decided in the browser. So we bridge with a
# tiny JS snippet:
#   • On page load, read ``?tab=chart`` and click the matching tab button.
#   • On any tab click, write ``?tab=<label>`` via history.pushState so the
#     URL reflects the current view (shareable link, browser back/forward).
# The polling loop handles the race where Streamlit mounts the tab buttons
# a moment after this <script> runs; the 8-second timeout keeps us quiet
# on the normal case where the tabs are already present.
st.html(
    """
    <script>
    (function() {
      const TABS = ["now", "chart", "cameras", "pipeline", "about"];

      function normalize(label) {
        return (label || "").trim().toLowerCase();
      }

      function activateFromUrl() {
        const params = new URLSearchParams(window.location.search);
        const desired = normalize(params.get("tab"));
        if (!desired || !TABS.includes(desired)) return false;
        const buttons = document.querySelectorAll('button[role="tab"]');
        for (const btn of buttons) {
          if (normalize(btn.textContent) === desired) {
            if (btn.getAttribute("aria-selected") !== "true") {
              btn.click();
            }
            return true;
          }
        }
        return false;
      }

      // Poll for up to 8 seconds waiting for tabs to render, then give up.
      const iv = setInterval(() => {
        if (activateFromUrl()) clearInterval(iv);
      }, 120);
      setTimeout(() => clearInterval(iv), 8000);

      // Delegated click listener — capture ALL tab clicks and push the new
      // tab name into ?tab=… so the URL stays in sync with what the user
      // is looking at.
      document.addEventListener("click", (ev) => {
        const btn = ev.target.closest('button[role="tab"]');
        if (!btn) return;
        const label = normalize(btn.textContent);
        if (!TABS.includes(label)) return;
        const url = new URL(window.location);
        if (url.searchParams.get("tab") === label) return;
        url.searchParams.set("tab", label);
        history.pushState({tab: label}, "", url);
      }, true);

      // In-page CTAs: any <a href="?tab=X"> anchor should activate that
      // tab instead of reloading. This lets Now-tab buttons like "View
      // full prediction model" jump straight to the Chart tab without
      // a full page round-trip.
      document.addEventListener("click", (ev) => {
        const anchor = ev.target.closest('a[href^="?tab="]');
        if (!anchor) return;
        ev.preventDefault();
        const href = anchor.getAttribute("href");
        const target = new URLSearchParams(href.slice(1)).get("tab");
        const label = normalize(target);
        if (!TABS.includes(label)) return;
        const url = new URL(window.location);
        url.searchParams.set("tab", label);
        history.pushState({tab: label}, "", url);
        activateFromUrl();
      }, true);

      // Browser back/forward: re-run the URL-driven activation.
      window.addEventListener("popstate", activateFromUrl);

      // Inspector-overlay dock — tag the "Inspector overlay layers"
      // expander with .kt-overlay-dock so CSS can float it on desktop
      // (see styles.py). We identify the expander by its summary text
      // because Streamlit doesn't expose a Python-level class hook on
      // st.expander. Idempotent: skip if the class is already there.
      function tagOverlayDock() {
        document.querySelectorAll('[data-testid="stExpander"]').forEach(exp => {
          if (exp.classList.contains("kt-overlay-dock")) return;
          const summaryText = (exp.querySelector("summary") || exp).textContent || "";
          if (summaryText.includes("Inspector overlay layers")) {
            exp.classList.add("kt-overlay-dock");
          }
        });
      }
      tagOverlayDock();
      const overlayObs = new MutationObserver(tagOverlayDock);
      overlayObs.observe(document.body, {childList: true, subtree: true});
    })();
    </script>
    """,
    unsafe_allow_javascript=True,
)

with tab_now:
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
        # Inside a raw-HTML <div>, Streamlit doesn't re-parse markdown, so
        # `[label](url)` would render as literal text. Emit an <a> tag.
        notice_link = (
            f' &nbsp;<a href="{status.notice_url}" target="_blank" rel="noopener"'
            f' style="color: inherit; text-decoration: underline;">full notice ↗</a>'
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
                "Updated every 15 minutes; click the top-bar Refresh button to "
                "force a fresh fetch."
            )
    elif safety.errors:
        # Both sources failed. Show a single quiet caption rather than a
        # loud red banner — alerts are auxiliary, not load-bearing.
        st.caption(
            "Safety alerts unavailable right now "
            f"({len(safety.errors)} source(s) returned an error)."
        )

    # ── Hero block: one dramatic answer + last-30-days sparkline ───────────
    # Moved from above-the-tabs into tab_now so each tab is self-contained.
    # The sparkline reads as a visual fingerprint of recent activity rather
    # than a readable chart — axes are suppressed, hover disabled.
    hero.show(eruption_state, prediction, tilt_df)

    # CTA: jump to the full prediction model chart. Rendered as an anchor
    # with ``?tab=chart`` so the JS tab router (injected near st.tabs below)
    # intercepts the click and activates the Chart tab.
    st.markdown(
        '<div class="kt-cta-row">'
        '<a class="kt-cta" href="?tab=chart">📈 View full prediction model →</a>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── State banner ──────────────────────────────────────────────────────
    # Consistent 3-part banner (icon + headline, plain explainer, guidance)
    # rendered from the state_copy table. `calm` skips rendering.
    state_banner.show(eruption_state, eruption_state_info)

    # ── Live camera strip ─────────────────────────────────────────────────
    # Four-camera strip below the hero — seeing the volcano is the second-
    # most emotionally valuable thing on the page after the prediction.
    # Full 8-camera grid stays available on the Cameras tab.
    st.markdown("#### 📷 Live cameras")
    cameras.show_strip()

    # CTA: jump to the full camera grid.
    st.markdown(
        '<div class="kt-cta-row">'
        '<a class="kt-cta" href="?tab=cameras">📷 View all cameras →</a>'
        '</div>',
        unsafe_allow_html=True,
    )


with tab_chart:
    # Chart options — controls that only make sense next to the chart.
    # `tw_*` session-state keys are seeded at module scope so the eager
    # prediction compute above can read them before this expander renders.
    _chart_opts_cols = st.columns([1.2, 1])
    with _chart_opts_cols[0]:
        st.slider(
            "Trendline window — number of recent peaks",
            min_value=3, max_value=20, step=1,
            key="tw_n_peaks_for_fit",
            help=(
                "How many of the most recent detected peaks feed the linear "
                "trendline fit. Smaller = more sensitive to recent shifts; "
                "larger = smoother long-term trend."
            ),
        )
    with _chart_opts_cols[1]:
        st.toggle(
            "🔍 Show per-source traces",
            key="tw_show_per_source",
            help=(
                "Overlay each USGS source's calibrated trace beneath the "
                "merged line. Useful for spotting calibration drift or "
                "source handoff. Traces are off by default; click legend "
                "entries to toggle individual sources."
            ),
        )

    # Advanced model tuning — sliders that tune peak detection sensitivity.
    # Writes directly into `adv_*` session_state keys that the top-of-module
    # setup block reads on the next rerun.
    with st.expander("⚙️ Advanced model tuning", expanded=False):
        st.caption(
            "Nudge these if you want to see how peak detection and the "
            "trendline react to different sensitivities. Defaults are "
            "tuned for UWD Az 300°."
        )
        # Sliders read their initial value from session_state via ``key=``.
        # Passing ``value=`` alongside ``key=`` when the key is already seeded
        # triggers Streamlit's widget-key-collision warning in 1.45+.
        st.slider(
            "Minimum prominence (µrad)",
            min_value=1.0, max_value=15.0, step=0.5,
            key="adv_min_prominence",
            help="How much a peak must rise above surrounding troughs.",
        )
        st.slider(
            "Minimum spacing (days)",
            min_value=1.0, max_value=30.0, step=0.5,
            key="adv_min_distance_days",
            help="Reject peaks within this many days of a stronger one.",
        )
        st.slider(
            "Minimum height (µrad)",
            min_value=-20.0, max_value=20.0, step=0.5,
            key="adv_min_height",
            help="Absolute tilt floor for peaks. -10 effectively disables.",
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
    # Plotly's native double-click-to-deselect is fiddly to hit on the chart
    # area; give the user an explicit button that clears the box-select
    # state in session_state so the CSV export date pickers fall back to
    # their defaults. Only render when a selection is actually live to keep
    # the UI quiet otherwise.
    _has_chart_selection = (
        bool(chart_selection)
        and isinstance(chart_selection, dict)
        and chart_selection.get("selection", {}).get("box")
    )
    if _has_chart_selection:
        if st.button("✕ Clear chart selection", key="clear_chart_selection"):
            st.session_state.pop("main_chart", None)
            st.rerun()

    # Hover-to-clipboard: press ⌘C / Ctrl+C while hovering a datapoint on the
    # chart to copy "YYYY-MM-DD HH:MM | X.XX µrad" to the clipboard. Uses
    # st.html(unsafe_allow_javascript=True) which renders the <script> inline
    # in the main Streamlit document — not in a sandboxed iframe — so the JS
    # reaches Plotly's `.js-plotly-plot` node directly via `document`. We
    # migrated here from st.components.v1.html because that API was deprecated
    # in Streamlit 1.56; st.iframe can't replace it for JS injection (null-
    # origin iframes can't touch the parent DOM), but st.html can.
    # The polling loop handles the race where Plotly mounts a moment after
    # the component; the 10-second timeout keeps this quiet on static page
    # loads where the chart never renders.
    st.html(
        """
        <script>
        (function() {
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
            const sel = window.getSelection && window.getSelection();
            if (sel && sel.toString().length > 0) return;
            navigator.clipboard && navigator.clipboard.writeText(current);
          };
          document.addEventListener('keydown', onKey);
          const iv = setInterval(() => {
            document.querySelectorAll('.js-plotly-plot').forEach(bind);
          }, 400);
          setTimeout(() => clearInterval(iv), 10000);
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )
    st.caption(
        "Tip: hover a point and press ⌘C / Ctrl+C to copy it. "
        "Or drag a box over the chart to set the CSV export range just below."
    )

    # ── Interval-based sanity check (secondary forecast) ───────────────────
    # Independent from the exp-fit prediction above. Uses the median gap
    # between detected peaks as a crude "when was the last one + typical
    # cycle length?" baseline. Lives here (with the chart) rather than above
    # the fold — it's a technical cross-check, not the headline answer.
    if (
        eruption_state != "active"
        and prediction.interval_based_next_event_date is not None
    ):
        ib_date = prediction.interval_based_next_event_date
        ib_band = prediction.interval_based_band
        median_days = prediction.median_peak_interval_days or 0.0
        ib_aware = ib_date.tz_localize("UTC") if ib_date.tzinfo is None else ib_date
        delta_days = (pd.Timestamp.now(tz="UTC") - ib_aware).total_seconds() / 86400
        overdue_str = (
            f" · ⚠️ **{delta_days:.0f}d overdue** by this metric"
            if delta_days > 0.5
            else ""
        )
        band_str = f" &nbsp;·&nbsp; IQR {_fmt_band(ib_band)}" if ib_band is not None else ""
        st.caption(
            f"📊 Interval baseline (independent sanity check): "
            f"{_fmt_short(ib_date)}{band_str}{overdue_str} — "
            f"median cycle {median_days:.1f} days."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CSV export — co-located with the chart so the box-select → export
    # workflow is one tab, not two.
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Two modes. Simple: date + merged tilt + winning source + configurable
    # per-source corrected values. Debug: every column, everything, as a zip
    # bundle including pair fits + per-bucket MAD diagnostics.
    #
    # Range selection works three ways that all feed the same date pickers:
    #   1. Box-select on the chart above pre-populates the range.
    #   2. Manual date pickers inside the expander.
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

    # ── Detected peaks ────────────────────────────────────────────────────
    # Peaks depend on the prominence / spacing / height sliders in the
    # Advanced model tuning expander above, so the table belongs here next
    # to its controls — not on a separate "Data" tab.
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


with tab_cameras:
    cameras.show_grid()


# The former "Data" tab was merged into the Pipeline tab. Reconcile + model
# diagnostics are behind-the-scenes instrumentation, so they live alongside
# the ingest/transcription inspector; detected peaks moved to the Chart tab.
# Streamlit's tab containers can be re-entered, so this ``with tab_pipeline:``
# simply streams the migrated expanders into the same Pipeline panel as the
# main pipeline block further down.
with tab_pipeline:

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
            st.caption(
                "These are the internals of the prediction model, shown as "
                "tinted chips so you can see which readings are typical and "
                "which are outside the normal range for the current eruptive "
                "phase. The text below each chip explains the expected range."
            )

            # ─── Trendline slope + current-episode sample count ──────────────
            slope = diag.get("trendline_slope_per_day")
            n_episode = diag.get("current_episode_n")
            col_slope, col_n = st.columns(2)
            if slope is not None:
                direction = "rising" if slope > 0 else "falling"
                col_slope.markdown(
                    render_chip_html(
                        label="Trendline slope",
                        value=f"{slope:+.3f}",
                        unit="µrad/day",
                        tint=trendline_slope_tint(slope),
                    ),
                    unsafe_allow_html=True,
                )
                col_slope.caption(
                    f"{direction.capitalize()} at ~{abs(slope)*7:.2f} µrad/week. "
                    "How fast the peak heights are shifting. Positive = episodes "
                    "getting more intense (more magma pressure per cycle); "
                    "negative = tapering off. Linear regression through the last "
                    "N peaks (Chart tab's Trendline window slider)."
                )
            if n_episode is not None:
                col_n.markdown(
                    render_chip_html(
                        label="Current episode samples",
                        value=str(n_episode),
                        unit="readings",
                        tint=episode_samples_tint(n_episode),
                    ),
                    unsafe_allow_html=True,
                )
                col_n.caption(
                    "Tilt samples feeding the exponential saturation fit since "
                    "the last detected peak. The fit needs ≥ 4 to estimate its "
                    "three parameters; more samples with more variation across "
                    "them tighten the confidence band."
                )
            st.markdown("&nbsp;")

            # ─── Exponential fit parameters ──────────────────────────────────
            if prediction.exp_params:
                A, k, C = prediction.exp_params
                asymptote = A + C
                tau_days = 1.0 / k if k > 0 else float("inf")
                half_life = math.log(2) / k if k > 0 else float("inf")
                st.markdown(
                    "**Exponential saturation fit** &nbsp;·&nbsp; "
                    "`tilt = A·(1 − exp(−k·t)) + C`"
                )
                col_a, col_k, col_tau, col_asym = st.columns(4)
                col_a.markdown(
                    render_chip_html(
                        label="A (amplitude)",
                        value=f"{A:.1f}",
                        unit="µrad",
                        tint=exp_amplitude_tint(A),
                    ),
                    unsafe_allow_html=True,
                )
                col_k.markdown(
                    render_chip_html(
                        label="k (rise rate)",
                        value=f"{k:.3f}",
                        unit="/day",
                        tint=exp_k_tint(k),
                    ),
                    unsafe_allow_html=True,
                )
                # Time-constant chip reuses the k tint — they're the same signal
                # in different units (τ = 1/k). A second chip lets the user
                # see the human-readable "days to 63% saturation" number directly.
                col_tau.markdown(
                    render_chip_html(
                        label="τ (time const)",
                        value=f"{tau_days:.1f}" if tau_days != float("inf") else "—",
                        unit="days",
                        tint=exp_k_tint(k),
                    ),
                    unsafe_allow_html=True,
                )
                # The asymptote isn't a teaching signal in itself — just show
                # as a neutral metric.
                col_asym.metric("A + C (asymptote)", f"{asymptote:.2f} µrad")
                st.caption(
                    f"**A** = total rise amplitude the episode would gain at "
                    f"full saturation. **k** = rise rate; the time constant "
                    f"1/k (~{tau_days:.1f} days) is how long to reach 63% of A, "
                    f"half-time ln(2)/k (~{half_life:.1f} days) is 50% of A. "
                    f"**C** = starting tilt offset (the trough after the last "
                    f"eruption, at {C:+.2f} µrad). **A + C** = asymptote — "
                    f"where tilt would settle if no eruption interrupted the "
                    f"rise. The next fountain event is predicted *before* this "
                    f"asymptote, where the rising exp curve crosses the "
                    f"linear trendline through recent peaks."
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
                "(controlled by the Chart tab's *Trendline window* slider)"
            )


with tab_pipeline:
    # Inspector-overlay checkboxes — moved from the sidebar so they sit
    # next to the PNG overlays they control. Write into the `ovl_*`
    # session_state keys that the top-of-module block reads back into
    # `layer_*` locals.
    with st.expander("🔬 Inspector overlay layers", expanded=False):
        st.caption(
            "Toggle visualization layers on top of each source's raw USGS "
            "PNG. Overlays render below in the transcription inspector."
        )
        _c1, _c2 = st.columns(2)
        with _c1:
            st.markdown("**Axis**")
            st.checkbox("Re-traced dots", key="ovl_dots")
            st.checkbox("Plot bbox outline", key="ovl_bbox")
            st.checkbox("Y-tick OCR markers", key="ovl_yticks")
            st.checkbox("Integer-µrad gridlines", key="ovl_ygrid")
            st.checkbox("Corner (date, µrad) labels", key="ovl_corners")
        with _c2:
            st.markdown("**Pixels**")
            st.checkbox("Blue-curve HSV mask", key="ovl_blue")
            st.checkbox("Legend-exclusion zone", key="ovl_legend")
            st.checkbox("Dropped-width columns", key="ovl_dropcols")
            st.checkbox("Outlier drops", key="ovl_outliers")
        st.markdown("**World / frame**")
        st.checkbox("'Now' UTC vertical line", key="ovl_now")
        st.checkbox("Green Az 30° HSV mask", key="ovl_green")
        st.checkbox(
            "Accumulated CSV samples (frame-corrected)",
            key="ovl_csv",
        )
        st.caption(
            "Layer changes take effect on the next rerun. Changing a box "
            "below triggers a rerun immediately."
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
            "layers stacked on top. On desktop the **🔬 Inspector overlay "
            "layers** panel docks to the top-right so toggles stay visible "
            "as you scroll through the PNGs; on mobile it sits above the "
            "inspector in-flow."
        )

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

        _PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

        def _is_valid_png(b: bytes | None) -> bool:
            """True iff b starts with the PNG file-signature magic bytes.

            Streamlit's st.image() passes bytes through PIL.Image.open which
            raises UnidentifiedImageError on anything non-PNG (HTML error
            pages, 304-with-body, empty responses). Validate before rendering
            to show a readable caption instead of crashing the whole page.
            """
            return isinstance(b, (bytes, bytearray)) and b.startswith(_PNG_MAGIC)

        @st.cache_data(ttl=INGEST_CACHE_TTL_SECONDS, show_spinner=False)
        def _fetch_png_bytes(url: str) -> bytes | None:
            """Fetch a USGS PNG and return the raw bytes. None on failure or
            on any non-PNG response body."""
            from kilauea_tracker.ingest.fetch import fetch_tilt_png
            try:
                result = fetch_tilt_png(url, None)
            except Exception:
                return None
            body = result.body
            if not _is_valid_png(body):
                return None
            return body

        def _build_overlay_png(
            raw_bytes: bytes,
            calibration,
            report,
            layers: dict,
            display_tz: str = "UTC",
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

            def fmt_dt_tz(ts) -> str:
                """Format a naive-UTC timestamp in the user's display timezone."""
                try:
                    return (
                        pd.Timestamp(ts)
                        .tz_localize("UTC")
                        .tz_convert(display_tz)
                        .strftime("%Y-%m-%d %H:%M:%S")
                    )
                except Exception:
                    return str(ts)

            # Run trace_curve once if anything needs it (dots or csv layer).
            traced = None
            if layers.get("dots") or layers.get("csv"):
                try:
                    traced = trace_curve(img_bgr, calibration)
                except Exception:
                    traced = None

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

            # ── Green-mask highlight (magenta for high contrast vs green) ─
            if layers.get("green") and green_mask is not None:
                ys, xs = np.where(green_mask)
                stats["green_px"] = int(ys.size)
                for py, px in zip(ys, xs):
                    overlay.putpixel((int(px + x0), int(py + y0)), (255, 20, 220, 180))

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
            # Step is inferred from the OCR'd labels themselves (minimum
            # adjacent difference == USGS's true tick interval) so our
            # gridlines match USGS's 1:1. OCR gaps (missing '0' on week,
            # etc.) show up as integer multiples of the base step, and
            # `min(diffs)` correctly recovers the base. Falls back to a
            # span-based heuristic only when OCR returned <2 labels.
            # Range is clipped to the OCR'd extrema so we don't draw lines
            # in axis-margin whitespace past the outermost printed ticks.
            if layers.get("ygrid"):
                labels_sorted = sorted(
                    (v for _, v in (calibration.y_labels_found or []))
                )
                if len(labels_sorted) >= 2:
                    lo, hi = float(labels_sorted[0]), float(labels_sorted[-1])
                    diffs = [
                        labels_sorted[i + 1] - labels_sorted[i]
                        for i in range(len(labels_sorted) - 1)
                    ]
                    step = float(min(d for d in diffs if d > 1e-6))
                else:
                    y_top = calibration.pixel_to_microradians(y0)
                    y_bot = calibration.pixel_to_microradians(y1)
                    lo, hi = sorted((y_top, y_bot))
                    span = hi - lo
                    step = 1.0 if span < 10 else 5.0 if span < 60 else 20.0

                # Generate gridline values at `step` increments inside [lo, hi].
                # Build as a list of floats since step may be fractional (e.g. 0.5
                # on two_day). Align to `step` so our lines pass through USGS's
                # printed ticks exactly.
                start = np.ceil(lo / step) * step
                end = np.floor(hi / step) * step
                n_lines = int(round((end - start) / step)) + 1
                grid_values = [start + i * step for i in range(max(0, n_lines))]

                for val in grid_values:
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
                        # Format integer steps as "+N"/"-N" and fractional
                        # steps as "+N.N"/"-N.N" so two_day's "0.5" doesn't
                        # render as "+0".
                        if abs(step - round(step)) < 1e-6:
                            label = f"{int(round(val)):+d}"
                        else:
                            label = f"{val:+.1f}"
                        lw = (
                            draw.textlength(label, font=font)
                            if hasattr(draw, "textlength")
                            else 6 * len(label)
                        )
                        draw.rectangle(
                            (x1 - lw - 4, py - 6, x1 - 2, py + 6),
                            fill=(0, 0, 0, 140),
                        )
                        draw.text(
                            (x1 - lw - 2, py - 6),
                            label,
                            fill=(100, 220, 255, 230),
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
            # Clamp to the plot's right edge if 'now' is past x_end (common
            # on rolling windows: USGS PNGs lag real time by a few minutes).
            # Visualize as bright if inside the window, dimmer if clamped —
            # so you can tell whether the window's right edge = now or
            # whether x_end has drifted significantly from wall-clock.
            if layers.get("now"):
                from datetime import datetime as _dt, timezone as _tz
                now = pd.Timestamp(_dt.now(tz=_tz.utc)).tz_localize(None)
                px_now_raw = date_to_px(now)
                clamped = not (x0 <= px_now_raw <= x1)
                px_now = int(max(x0, min(px_now_raw, x1)))
                line_color = (150, 255, 150, 180) if clamped else (0, 255, 120, 220)
                draw.line((px_now, y0, px_now, y1), fill=line_color, width=2)
                if font is not None:
                    label = "now ↩" if clamped else "now"
                    lw = (
                        draw.textlength(label, font=font)
                        if hasattr(draw, "textlength")
                        else 6 * len(label)
                    )
                    # Position label to the LEFT of the line if it's at x1,
                    # otherwise right.
                    tx = px_now - lw - 2 if clamped and px_now >= x1 - 10 else px_now + 2
                    draw.rectangle((tx - 1, y0 + 1, tx + lw + 1, y0 + 13), fill=(0, 0, 0, 170))
                    draw.text((tx, y0 + 2), label, fill=line_color, font=font)

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
            # Y values use the OCR'd label range rather than the bbox
            # extrapolation, because the plot bbox extends a few pixels
            # past the outermost tick into axis-margin whitespace. Using
            # the labelled range matches the user's visual intuition
            # ("the top of this plot says +20"). If OCR didn't land any
            # labels, we fall back to the bbox extrapolation.
            if layers.get("corners") and font is not None:
                labels = calibration.y_labels_found or []
                if labels:
                    label_vals = [v for _, v in labels]
                    y_top_disp = max(label_vals)
                    y_bot_disp = min(label_vals)
                    suffix = ""
                else:
                    y_top_disp = calibration.pixel_to_microradians(y0)
                    y_bot_disp = calibration.pixel_to_microradians(y1)
                    suffix = " (bbox-extrap)"

                corner_texts = [
                    ("TL", x0 + 2, y0 + 2, False,
                     f"TL {fmt_dt_tz(calibration.x_start)} {y_top_disp:+.1f}{suffix}"),
                    ("TR", x1 - 2, y0 + 2, True,
                     f"{fmt_dt_tz(calibration.x_end)} {y_top_disp:+.1f}{suffix} TR"),
                    ("BL", x0 + 2, y1 - 14, False,
                     f"BL {fmt_dt_tz(calibration.x_start)} {y_bot_disp:+.1f}{suffix}"),
                    ("BR", x1 - 2, y1 - 14, True,
                     f"{fmt_dt_tz(calibration.x_end)} {y_bot_disp:+.1f}{suffix} BR"),
                ]
                for _id, anchor_x, ty, right_aligned, txt in corner_texts:
                    w = (
                        draw.textlength(txt, font=font)
                        if hasattr(draw, "textlength")
                        else 6 * len(txt)
                    )
                    tx = anchor_x - w if right_aligned else anchor_x
                    draw.rectangle(
                        (tx - 2, ty - 1, tx + w + 2, ty + 11),
                        fill=(0, 0, 0, 180),
                    )
                    draw.text((tx, ty), txt, fill=(255, 255, 0, 235), font=font)

            # ── Accumulated CSV samples (purple, frame-corrected) ─────────
            # The per-source CSV lives in the cumulative frame that
            # append_history aligns fetches to. That frame is offset from
            # the CURRENT PNG's raw y-axis whenever USGS rebaselines the
            # plot (two_day/week/month are relative-zero; they drift over
            # time). Without correction the dots all fall off-plot.
            #
            # We subtract the median offset between CSV values and re-
            # traced values at overlapping timestamps — so the "rebaseline
            # offset" is zeroed out. Any REMAINING offset between dots and
            # the blue line = a real per-sample alignment defect worth
            # investigating (parallel tracks, sawtooth splits).
            if layers.get("csv"):
                try:
                    from kilauea_tracker.config import source_csv_path
                    csv_path = source_csv_path(report.source_name)
                except Exception:
                    csv_path = None
                n_csv_drawn = 0
                n_csv_clipped = 0
                applied_offset = 0.0
                if csv_path is not None and csv_path.exists():
                    try:
                        df_csv = pd.read_csv(csv_path, parse_dates=[DATE_COL])
                    except Exception:
                        df_csv = None
                    if df_csv is not None and len(df_csv) > 0:
                        in_x = (df_csv[DATE_COL] >= calibration.x_start) & (
                            df_csv[DATE_COL] <= calibration.x_end
                        )
                        df_win = df_csv[in_x].copy()
                        # Compute frame offset from re-trace ∩ CSV at
                        # matching 15-min buckets.
                        if traced is not None and len(traced) > 0 and len(df_win) > 0:
                            bucket_csv = (
                                df_win.assign(
                                    _b=df_win[DATE_COL].dt.floor("15min")
                                )
                                .groupby("_b")[TILT_COL]
                                .median()
                            )
                            bucket_trace = (
                                traced.assign(
                                    _b=traced[DATE_COL].dt.floor("15min")
                                )
                                .groupby("_b")[TILT_COL]
                                .median()
                            )
                            common = bucket_csv.index.intersection(bucket_trace.index)
                            if len(common) >= 3:
                                deltas = (
                                    bucket_trace.loc[common] - bucket_csv.loc[common]
                                ).values
                                applied_offset = float(np.median(deltas))
                        dot_r = 2
                        for _, row in df_win.iterrows():
                            px = date_to_px(row[DATE_COL])
                            py = val_to_py(row[TILT_COL] + applied_offset)
                            if not (x0 <= px <= x1 and y0 <= py <= y1):
                                n_csv_clipped += 1
                                continue
                            draw.ellipse(
                                (px - dot_r, py - dot_r, px + dot_r, py + dot_r),
                                fill=(170, 60, 230, 110),
                                outline=(130, 0, 200, 180),
                            )
                            n_csv_drawn += 1
                stats["csv_drawn"] = n_csv_drawn
                stats["csv_clipped_y"] = n_csv_clipped
                stats["csv_offset"] = applied_offset

            # ── Re-traced dots (red, translucent) ─────────────────────────
            n_dots = 0
            if layers.get("dots") and traced is not None:
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

        def _inspector_calibrate(raw_bytes: bytes, source_name: str):
            """Decode raw PNG bytes and run the calibration pipeline.

            The pipeline's `ingest()` skips calibration on 304 Not Modified
            (the per-source CSV is already fresh, no need), so on a re-visit
            to a warmed-cache app the IngestReports come back with
            `calibration=None`. The inspector fetches fresh PNG bytes
            unconditionally, so we can always re-calibrate from those bytes
            here and show the overlays against the current PNG's axes —
            instead of misleadingly rendering "calibration failed" for
            every source just because nothing actually *failed*.

            NOT cached via st.cache_data: @st.cache_data's hash-based cache
            invalidation only sees changes to THIS function's body, so
            pushing a fix to calibrate_axes (which this wrapper calls into)
            wouldn't invalidate the cache if raw_bytes were unchanged —
            users would keep seeing stale wrong calibrations until USGS
            updated the PNG. Calibration is ~50ms; not worth the staleness
            risk.
            """
            try:
                import cv2
                import numpy as np
                from kilauea_tracker.ingest.calibrate import calibrate_axes
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    return None, "cv2.imdecode returned None"
                return calibrate_axes(img, source_name=source_name), None
            except Exception as e:
                return None, str(e)

        for r in reports:
            if r.source_name not in _inspector_url_map:
                continue
            raw = _fetch_png_bytes(_inspector_url_map[r.source_name])
            if raw is None:
                st.caption(f"⚠️ {r.source_name}: could not fetch PNG")
                continue
            # Prefer the pipeline's calibration when present (this fetch was
            # fresh and went through the full trace), otherwise calibrate the
            # PNG ourselves so we can still render overlays on 304-Not-
            # Modified runs.
            calibration = r.calibration
            inspector_cal_error: str | None = None
            if calibration is None:
                calibration, inspector_cal_error = _inspector_calibrate(raw, r.source_name)
            if calibration is None:
                st.markdown(f"**{r.source_name}** — calibration failed")
                msg = inspector_cal_error or r.error or "unknown cause"
                st.caption(f"❌ {msg}")
                try:
                    st.image(
                        raw,
                        width="stretch",
                        output_format="PNG",
                        caption=f"Raw USGS PNG for source {r.source_name} (calibration failed)",
                    )
                except Exception as e:
                    st.caption(f"(PNG render failed: {e})")
                st.markdown("---")
                continue
            overlay_bytes, stats = _build_overlay_png(
                raw, calibration, r, layers, display_tz=DISPLAY_TZ,
            )

            cal = calibration  # may have come from inspector_calibrate on a 304
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
                csv_offset = stats.get("csv_offset", 0.0)
                csv_bit = f"{csv_drawn} CSV samples in-plot"
                if abs(csv_offset) > 0.01:
                    csv_bit += f" (frame offset {csv_offset:+.2f} µrad applied)"
                if csv_clipped:
                    csv_bit += f" · {csv_clipped} still clipped → per-sample drift"
                caption_bits.append(csv_bit)
            st.markdown(" · ".join(caption_bits))

            if overlay_bytes is not None and _is_valid_png(overlay_bytes):
                try:
                    st.image(
                        overlay_bytes,
                        width="stretch",
                        output_format="PNG",
                        caption=(
                            f"USGS {r.source_name} plot with inspector overlay layers applied"
                        ),
                    )
                except Exception as e:
                    st.caption(f"(overlay render failed: {e})")
            else:
                st.caption("(overlay unavailable — cv2/PIL missing or bad bytes)")

            # Debug metadata — the numbers the rest of the pipeline trusts
            # for this PNG. First place to check when overlays disagree.
            with st.expander(f"🔧 {r.source_name} calibration diagnostics", expanded=False):
                # X-axis tick cross-check results: the calibration pipeline
                # already ran this and stashed the stats on `cal`. Compute
                # the approximate minute-scale equivalent from sec-per-pixel
                # so the caption shows both px and minutes. If the pipeline
                # hasn't run this for the source (old cached calibration),
                # fall back to re-running the OCR locally.
                _xtick_n = getattr(cal, "x_tick_cross_check_count", 0) or 0
                _xtick_median_px = getattr(cal, "x_tick_cross_check_median_err_px", None)
                _xtick_max_px = getattr(cal, "x_tick_cross_check_max_err_px", None)
                if _xtick_n == 0 or _xtick_max_px is None:
                    # Recompute on the fly for calibrations from the
                    # inspector's own re-calibration path (which doesn't
                    # populate these fields yet).
                    try:
                        import cv2 as _cv2
                        import numpy as _np
                        from kilauea_tracker.ingest.calibrate import (
                            ocr_x_axis_ticks as _ocr_x_ticks,
                        )
                        _arr = _np.frombuffer(raw, dtype=_np.uint8)
                        _img_for_ticks = _cv2.imdecode(_arr, _cv2.IMREAD_COLOR)
                        _ticks = _ocr_x_ticks(
                            _img_for_ticks, cal.plot_bbox, r.source_name,
                            x_start_utc=cal.x_start, x_end_utc=cal.x_end,
                        )
                        _span_s = (cal.x_end - cal.x_start).total_seconds()
                        _px_span = max(1.0, float(cal.plot_bbox[2] - cal.plot_bbox[0]))
                        _sec_per_px_local = _span_s / _px_span if _px_span else 0
                        _errs_px = []
                        for _px_cx, _dt in _ticks:
                            _pred = cal.x_start + pd.Timedelta(
                                seconds=(_px_cx - cal.plot_bbox[0]) / _px_span * _span_s
                            )
                            if _sec_per_px_local:
                                _errs_px.append(
                                    abs((_dt - _pred).total_seconds()) / _sec_per_px_local
                                )
                        if _errs_px:
                            _errs_px.sort()
                            _xtick_n = len(_errs_px)
                            _xtick_median_px = _errs_px[len(_errs_px) // 2]
                            _xtick_max_px = _errs_px[-1]
                    except Exception:
                        pass

                _span_s = (cal.x_end - cal.x_start).total_seconds()
                _px_span = max(1.0, float(cal.plot_bbox[2] - cal.plot_bbox[0]))
                _sec_per_px = _span_s / _px_span if _px_span else 0
                if _xtick_n and _xtick_max_px is not None:
                    _status = "✓ consistent" if _xtick_max_px <= 2.0 else "⚠ drift"
                    _xcheck_caption = (
                        f"x-axis tick cross-check: {_status}  \n"
                        f"  {_xtick_n} ticks parsed · "
                        f"median err `{_xtick_median_px:.2f}` px "
                        f"(`{_xtick_median_px * _sec_per_px / 60:.1f}` min) · "
                        f"max err `{_xtick_max_px:.2f}` px "
                        f"(`{_xtick_max_px * _sec_per_px / 60:.1f}` min)"
                    )
                else:
                    _xcheck_caption = "x-axis tick cross-check: no ticks parsed"

                dbg_cols = st.columns(2)
                with dbg_cols[0]:
                    st.markdown("**Time axis**")
                    st.caption(
                        f"x-start (UTC): `{cal.x_start}`  \n"
                        f"x-end (UTC): `{cal.x_end}`  \n"
                        f"window: `{x_span_hrs:.1f}` hrs  \n"
                        f"time-range OCR PSM: `{cal.title_psm_used or '—'}`  \n"
                        f"time-range OCR text (from strip below plot):  \n"
                        f"`{(cal.title_raw_text or '').strip() or '(empty)'}`  \n"
                        f"{_xcheck_caption}"
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
                    st.image(
                        url,
                        width="stretch",
                        caption=f"USGS {r.source_name} tilt plot",
                    )
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



with tab_about:
    about_tab.show()


# ─────────────────────────────────────────────────────────────────────────────
# Footer — compact attribution strip below the tabs.
# ─────────────────────────────────────────────────────────────────────────────

from kilauea_tracker import __version__ as _kt_version  # noqa: E402

_GH_ICON = (
    '<svg viewBox="0 0 16 16" aria-hidden="true">'
    '<path d="M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38 '
    '0-.19-.01-.82-.01-1.49-2 .37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13'
    '-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66'
    '.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15'
    '-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 '
    '1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 '
    '1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 '
    '1.93-.01 2.2 0 .21.15.46.55.38A8 8 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>'
    '</svg>'
)

st.markdown(
    '<footer class="kt-footer">'
    '<span class="kt-footer__inner">'
    'Built by <a href="https://github.com/madisonrickert">Madison Rickert</a>'
    '<span class="kt-footer__sep">·</span>'
    f'<a href="https://github.com/madisonrickert/kilauea-tracker">{_GH_ICON} source</a>'
    '<span class="kt-footer__sep">·</span>'
    f'v{_kt_version}'
    '</span>'
    '</footer>',
    unsafe_allow_html=True,
)
