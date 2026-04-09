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

import math
from datetime import datetime, timezone
from pathlib import Path

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

# Surface ingest errors and warnings (per-source AND reconciliation-layer)
ingest_errors = [r for r in reports if r.error]
ingest_warnings = [w for r in reports for w in r.warnings]
if reconcile_report is not None:
    ingest_warnings.extend(reconcile_report.warnings)

if ingest_errors:
    for r in ingest_errors:
        st.error(f"❌ **{r.source_name}**: {r.error}")
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
    st.metric(
        "Next fountain event",
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
# Main chart
# ─────────────────────────────────────────────────────────────────────────────

fig = build_figure(
    tilt_df,
    recent_peaks,
    prediction,
    all_peaks_df=all_peaks,
    title="",
)
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
