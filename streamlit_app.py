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
from kilauea_tracker.ingest.pipeline import data_age_seconds, ingest_all
from kilauea_tracker.model import DATE_COL
from kilauea_tracker.state import (
    RefreshStore,
    get_refresh_store,
    get_state,
    init_widget_defaults,
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
# Refresh subsystem — both manual click and the page-load background path
# go through the unified ``RefreshStore`` (see src/kilauea_tracker/state/
# refresh_store.py). The store coordinates across threads and tabs via a
# fcntl-locked JSON file; the topbar fragment subscribes to its snapshot
# and renders a disabled button while a refresh is in flight.
# ─────────────────────────────────────────────────────────────────────────────

# Re-poll USGS if on-disk data is older than this. Background daemon fires
# on page load when this threshold is crossed. 10 min matches USGS's typical
# publish cadence for the rolling-window PNGs.
STALE_THRESHOLD_SECONDS = 10 * 60


def _refresh_worker(store: RefreshStore) -> None:
    """Daemon-thread entrypoint for both manual and background refreshes.

    Single code path, regardless of who initiated. ``ingest_all``'s
    ``on_stage`` callback feeds the store so the topbar fragment can
    show stage progress. ``try/finally`` guarantees the store transitions
    out of the running state even on unexpected exceptions; the
    stale-refresh detector covers OS-level kills that bypass this.
    """
    try:
        ingest_all(on_stage=store.advance)
        store.complete()
    except Exception as e:
        store.fail(str(e))
        logging.getLogger(__name__).exception("refresh worker failed")


def _kick_background_refresh_if_stale() -> None:
    """Fire a background ingest on page load if data is older than the
    stale threshold AND the store grants a fresh refresh slot. Silent —
    never blocks page render."""
    if data_age_seconds() < STALE_THRESHOLD_SECONDS:
        return
    store = get_refresh_store()
    if not store.start("background"):
        return  # already running or in cooldown
    threading.Thread(target=_refresh_worker, args=(store,), daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Widget-state defaults — every session_state key the app relies on, seeded
# with its default in one place. Streamlit's widget→key auto-binding still
# owns the writes; views read through ``state.widgets.*`` rather than
# poking session_state directly. See src/kilauea_tracker/state/widgets.py
# for the canonical key list.
# ─────────────────────────────────────────────────────────────────────────────

init_widget_defaults()
state = get_state()


# ─────────────────────────────────────────────────────────────────────────────
# Always-fresh data: page renders from on-disk CSVs immediately. A background
# daemon refreshes if the data is stale (gated by the RefreshStore so
# concurrent users don't multiply USGS calls). The topbar fragment polls
# the store + the history-CSV mtime so users see fresh data automatically.
# ─────────────────────────────────────────────────────────────────────────────

_kick_background_refresh_if_stale()

ingest_result = app_state.load_run_report()
reports = ingest_result.per_source
reconcile_report = ingest_result.reconcile
# "Last poll" timestamp for the topbar caption: prefer the just-finished
# refresh, fall back to whatever the persisted run report knew about.
_last_poll_ts = state.refresh.finished_utc or ingest_result.run_finished_at_utc

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
    min_prominence=state.widgets.peaks.min_prominence,
    min_distance_days=state.widgets.peaks.min_distance_days,
    min_height=state.widgets.peaks.min_height,
)
recent_peaks = app_state.get_recent_peaks(all_peaks, state.widgets.chart.n_peaks_for_fit)
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


# ── Compact top bar — entire bar is rendered inside one fragment.
# Streamlit forbids fragments writing widgets into containers (columns)
# created outside themselves, so the columns *and* their content all
# live in ``_topbar_fragment``. Polling cadence flips between 1s (a
# refresh is in flight) and 30s (idle, watching for cron-driven mtime
# changes); the cadence is fixed at fragment registration and
# transitions trigger a full ``st.rerun(scope="app")``.
_REFRESH_RUN_EVERY = "1s" if state.refresh.running else "30s"


@st.fragment(run_every=_REFRESH_RUN_EVERY)
def _topbar_fragment() -> None:
    """Render the full topbar (refresh + tz + freshness caption).

    Re-reads ``get_state()`` on every tick so the indicator and caption
    reflect the freshest snapshot. Trigger ``st.rerun(scope="app")``
    on running→idle transitions and on history-CSV mtime changes so
    the rest of the page picks up new data without user action.
    """
    snap = get_state()
    last_running = st.session_state.get("_kt_last_running")
    last_mtime = st.session_state.get("_kt_last_mtime")
    st.session_state["_kt_last_running"] = snap.refresh.running
    st.session_state["_kt_last_mtime"] = snap.history_mtime

    # Local tz derivation — keeps the meta caption correct on the tick
    # where the user just changed the tz selectbox below. Pages pull
    # tz off ``state.widgets.chart.timezone_choice`` via their own
    # ``get_state()`` call on next rerun.
    local_display_tz = (
        "Pacific/Honolulu"
        if snap.widgets.chart.timezone_choice.startswith("HST")
        else "UTC"
    )
    local_tz_label = "HST" if local_display_tz == "Pacific/Honolulu" else "UTC"

    def _local_fmt(ts: pd.Timestamp | None) -> str:
        if ts is None or pd.isna(ts):
            return "—"
        aware = ts.tz_localize("UTC") if ts.tzinfo is None else ts
        converted = aware.tz_convert(local_display_tz)
        return f"{converted.strftime('%a, %b %-d · %-I:%M %p')} {local_tz_label}"

    tb_refresh, tb_tz, tb_meta = st.columns([1, 1.3, 2.5])

    with tb_refresh:
        if snap.refresh.running:
            label = snap.refresh.current_stage or "Refreshing…"
            st.button(
                f"⏳ {label}",
                disabled=True,
                width="stretch",
                # Unique key per stage label so Streamlit treats each
                # render as a fresh widget. ``started_utc`` keeps state
                # from leaking across separate refresh runs.
                key=f"kt_refresh_running_{snap.refresh.started_utc}_{label}",
            )
        else:
            if st.button(
                "🔄 Refresh",
                width="stretch",
                help="Re-fetch and re-trace all USGS tilt PNGs + safety alerts.",
                key="kt_refresh_idle",
            ):
                store = get_refresh_store()
                if store.start("manual"):
                    threading.Thread(
                        target=_refresh_worker, args=(store,), daemon=True
                    ).start()
                    app_state.clear_safety_alerts_cache()
                    st.rerun(scope="app")
                else:
                    st.toast("Refresh already in progress or cooling down")

    with tb_tz:
        st.selectbox(
            "Time zone",
            options=["HST (Pacific/Honolulu)", "UTC"],
            key="tw_timezone_choice",
            label_visibility="collapsed",
            help="All displayed dates use this time zone. HST is local at Kīlauea.",
        )

    with tb_meta:
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
            _sample_meta = f"latest sample {_local_fmt(_last_data)} ({_age})"
        else:
            _sample_meta = "latest sample —"

        if _last_poll_ts is not None:
            _successful = sum(1 for r in reports if r.error is None)
            _total = len(reports)
            _indicator = "🟢" if _successful == _total else ("🟡" if _successful else "🔴")
            _poll_meta = (
                f"{_indicator} {_successful}/{_total} sources · "
                f"last poll {_local_fmt(pd.Timestamp(_last_poll_ts))}"
            )
        else:
            _poll_meta = "no polls yet"

        st.markdown(
            f'<div class="kt-topbar__meta">'
            f'<div>{_sample_meta}</div>'
            f'<div>{_poll_meta}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Transition detector — flip back to slow cadence + repaint shell
    # when the running flag clears, and when on-disk data changes.
    if last_running is True and snap.refresh.running is False:
        st.rerun(scope="app")
    if last_mtime is not None and snap.history_mtime > last_mtime:
        st.rerun(scope="app")


_topbar_fragment()


# ─────────────────────────────────────────────────────────────────────────────
# Routing — st.navigation registers each page; the active page's script
# runs on every interaction. position="top" keeps the visual close to the
# previous st.tabs strip without taking over the sidebar.
# ─────────────────────────────────────────────────────────────────────────────

_pages = [
    st.Page("pages/now.py", title="Now", url_path="now", default=True),
    st.Page("pages/chart.py", title="Chart", url_path="chart"),
    st.Page("pages/backtest.py", title="Backtest", url_path="backtest"),
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
