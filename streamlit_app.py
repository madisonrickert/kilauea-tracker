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

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from kilauea_tracker.cache import load_history
from kilauea_tracker.config import (
    HISTORY_CSV,
    INGEST_CACHE_TTL_SECONDS,
    PEAK_DEFAULTS,
)
from kilauea_tracker.ingest.pipeline import IngestReport, ingest_all
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


# ─────────────────────────────────────────────────────────────────────────────
# Cached ingest — at most once per 15 minutes regardless of widget activity
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=INGEST_CACHE_TTL_SECONDS, show_spinner="Fetching latest USGS tilt data…")
def cached_ingest() -> list[IngestReport]:
    """Run all four USGS sources through the ingest pipeline.

    Wrapped in `st.cache_data` so the same browser session reuses results
    until the TTL expires (15 minutes by default — USGS updates these PNGs
    on roughly that cadence). Clearing the cache via `cached_ingest.clear()`
    forces a fresh fetch the next time this function runs.
    """
    return ingest_all()


# Initialize session state — used to remember whether ingestion has already
# run at least once in this Streamlit session, so we can show "Last update"
# accurately even after the cache TTL expires.
if "last_ingest_at" not in st.session_state:
    st.session_state.last_ingest_at = None
if "last_ingest_reports" not in st.session_state:
    st.session_state.last_ingest_reports = []


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
        st.session_state.last_ingest_at = None  # force the block below to re-run

    st.divider()
    st.subheader("Peak detection")

    min_prominence = st.slider(
        "Minimum prominence (µrad)",
        min_value=1.0,
        max_value=15.0,
        value=PEAK_DEFAULTS.min_prominence,
        step=0.5,
        help=(
            "How much the peak must rise above its surrounding troughs. "
            "Higher = fewer, more confident peaks."
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
    min_height = st.slider(
        "Minimum height (µrad)",
        min_value=-5.0,
        max_value=20.0,
        value=PEAK_DEFAULTS.min_height,
        step=0.5,
        help="Absolute tilt threshold a sample must clear to count as a peak.",
    )

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
            "trendline fit. v1.0 used 6."
        ),
    )

    st.divider()
    st.caption("**Data source**")
    st.caption(
        "Electronic tilt at the **UWD** station (Uēkahuna, summit), **azimuth 300°**. "
        "Published as auto-updating PNGs by USGS HVO at "
        "[volcanoes.usgs.gov](https://volcanoes.usgs.gov/vsc/captures/kilauea/). "
        "v2.0 traces those images directly via OpenCV + Tesseract OCR — no "
        "manual digitization required."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run ingest — populates `data/tilt_history.csv`
# ─────────────────────────────────────────────────────────────────────────────

reports = cached_ingest()
if st.session_state.last_ingest_at is None:
    st.session_state.last_ingest_at = datetime.now(tz=timezone.utc)
    st.session_state.last_ingest_reports = reports

# Surface ingest errors and warnings
ingest_errors = [r for r in reports if r.error]
ingest_warnings = [w for r in reports for w in r.warnings]

if ingest_errors:
    for r in ingest_errors:
        st.error(f"❌ **{r.source.name}**: {r.error}")
if ingest_warnings:
    with st.expander(f"⚠️ {len(ingest_warnings)} ingest warning(s)"):
        for w in ingest_warnings:
            st.warning(w)


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


def _fmt_date(ts: pd.Timestamp | None) -> str:
    if ts is None:
        return "—"
    return ts.strftime("%Y-%m-%d %H:%M HST")


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
    return f"{lo.strftime('%m/%d')} → {hi.strftime('%m/%d')}"


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Next fountain event",
        _fmt_date(prediction.next_event_date),
        delta=_fmt_band(prediction.confidence_band),
        delta_color="off",
        help=(
            "Intersection of the all-peaks linear trendline with the exp fit. "
            "The delta line is the 10th–90th percentile Monte Carlo "
            "confidence band over the exp fit's covariance."
        ),
    )
with col2:
    st.metric(
        "Earliest likely",
        _fmt_date(prediction.earliest_event_date),
        help="Intersection using only the most recent 3 peaks (steeper trend).",
    )
with col3:
    last_data = tilt_df[DATE_COL].max()
    st.metric(
        "Latest tilt sample",
        _fmt_date(last_data),
    )
with col4:
    if st.session_state.last_ingest_at is not None:
        successful = sum(1 for r in reports if r.error is None)
        total = len(reports)
        if successful == total:
            indicator = "🟢"
        elif successful > 0:
            indicator = "🟡"
        else:
            indicator = "🔴"
        st.metric(
            "Last refresh",
            f"{indicator} {_ago(st.session_state.last_ingest_at)}",
            delta=f"{successful}/{total} sources",
            delta_color="off",
        )
    else:
        st.metric("Last refresh", "—")


# ─────────────────────────────────────────────────────────────────────────────
# Main chart
# ─────────────────────────────────────────────────────────────────────────────

fig = build_figure(tilt_df, recent_peaks, prediction, title="")
st.plotly_chart(fig, width="stretch")


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
        st.json(diag)
        if prediction.exp_params:
            A, k, C = prediction.exp_params
            st.write(
                f"**Exp fit:** A = {A:.3f}, k = {k:.4f}, C = {C:.3f}  "
                f"(asymptotes to {A + C:.2f} µrad)"
            )

with st.expander("📡 Ingest pipeline status"):
    if not reports:
        st.write("No ingest reports — pipeline hasn't run yet.")
    else:
        for r in reports:
            status_icon = "✅" if r.error is None else "❌"
            with st.container():
                gap_str = (
                    f", `{r.rows_dropped_as_filled}` dropped (gap-fill mode)"
                    if r.gap_fill_mode
                    else ""
                )
                st.markdown(
                    f"{status_icon} **{r.source.name}** — "
                    f"`{r.rows_traced}` rows traced, "
                    f"`{r.rows_added_to_cache}` added, "
                    f"`{r.rows_updated_in_cache}` updated"
                    f"{gap_str}"
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
                if r.applied_y_offset is not None:
                    st.caption(
                        f"Cross-source y-offset: {r.applied_y_offset:+.3f} µrad "
                        f"(median over {r.overlap_buckets} overlap buckets)"
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
        on the summit caldera rim), **azimuth 300°**, published as
        auto-updating PNGs by USGS Hawaiian Volcano Observatory. v2.0's
        ingest pipeline traces those images directly via OpenCV + Tesseract
        OCR — no manual digitization required. The "Refresh" button
        re-fetches all five time windows (2-day, week, month, 3-month, and
        the long-history Dec 2024 → now plot) and merges new samples into
        the local history cache.

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
