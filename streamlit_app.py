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

import logging
import sys
import threading
from datetime import UTC, datetime
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

import pandas as pd
import streamlit as st

from kilauea_tracker import app_state
from kilauea_tracker.config import (
    PEAK_DEFAULTS,
)
from kilauea_tracker.ingest.pipeline import (
    data_age_seconds,
    ingest_all,
    try_acquire_refresh_slot,
)
from kilauea_tracker.model import DATE_COL
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
    with (
        placeholder.container(),
        st.status("Fetching latest USGS tilt data…", expanded=False) as status,
    ):
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

ingest_result = app_state.load_run_report()
reports = ingest_result.per_source
reconcile_report = ingest_result.reconcile
if st.session_state.last_ingest_at is None:
    st.session_state.last_ingest_at = (
        ingest_result.run_finished_at_utc
        if ingest_result.run_finished_at_utc is not None
        else datetime.now(tz=UTC)
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
    current = app_state.history_mtime()
    last_seen = st.session_state.get("_last_data_mtime")
    if last_seen is None:
        st.session_state["_last_data_mtime"] = current
        return
    if current > last_seen:
        st.session_state["_last_data_mtime"] = current
        st.rerun(scope="app")


_watch_for_fresh_data()


tilt_df = app_state.load_tilt_df()

if len(tilt_df) == 0:
    st.error(
        "No tilt history available. The ingest pipeline didn't produce any "
        "rows and `legacy/Tiltmeter Data - Sheet1.csv` is missing."
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Compute prediction
# ─────────────────────────────────────────────────────────────────────────────

all_peaks = app_state.get_peaks(
    tilt_df,
    min_prominence=min_prominence,
    min_distance_days=min_distance_days,
    min_height=min_height,
)
recent_peaks = app_state.get_recent_peaks(all_peaks, n_peaks_for_fit)
prediction = app_state.get_prediction(tilt_df, recent_peaks)


# ─────────────────────────────────────────────────────────────────────────────
# Eruption lifecycle state — drives the status banner above the chart
# ─────────────────────────────────────────────────────────────────────────────

eruption_state, eruption_state_info = app_state.get_eruption_state(tilt_df, prediction)


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
            app_state.clear_safety_alerts_cache()
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
# Routing — st.navigation registers each page; the active page's script
# runs on every interaction. position="top" keeps the visual close to the
# previous st.tabs strip without taking over the sidebar.
# ─────────────────────────────────────────────────────────────────────────────

_pages = [
    st.Page("pages/now.py", title="Now", url_path="now", default=True),
    st.Page("pages/chart.py", title="Chart", url_path="chart"),
    st.Page("pages/cameras.py", title="Cameras", url_path="cameras"),
    st.Page("pages/pipeline.py", title="Pipeline", url_path="pipeline"),
    st.Page("pages/about.py", title="About", url_path="about"),
]

st.navigation(_pages, position="top").run()


# ─────────────────────────────────────────────────────────────────────────────
# Footer — compact attribution strip below the tabs.
# ─────────────────────────────────────────────────────────────────────────────

from kilauea_tracker import __version__ as _kt_version

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
