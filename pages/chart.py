"""Chart — full prediction model.

Trendline + exponential-fit chart, peak-detection tuning sliders, the
interval-based independent-sanity-check baseline, the CSV / debug-zip
exporters, and the detected-peaks table. This is the page for visitors
who want to dig past the headline answer the Now page gives them.

The chart is the workflow centerpiece: drag a box on the plot and the
selection prefills the CSV-export date pickers, hover-and-press-⌘C copies
the value to clipboard, and the trendline/peak sliders adjust live.
"""

from __future__ import annotations

import io
import json as _json_dbg
import zipfile

import pandas as pd
import streamlit as st

from kilauea_tracker import app_state
from kilauea_tracker.config import ALL_SOURCES, TILT_SOURCE_NAME, source_csv_path
from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.plotting import build_figure

_timezone_choice = st.session_state.get("tw_timezone_choice", "HST (Pacific/Honolulu)")
DISPLAY_TZ = "Pacific/Honolulu" if _timezone_choice.startswith("HST") else "UTC"
TZ_LABEL = "HST" if DISPLAY_TZ == "Pacific/Honolulu" else "UTC"


def _to_display_tz(ts: pd.Timestamp | None) -> pd.Timestamp | None:
    if ts is None or pd.isna(ts):
        return None
    aware = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    return aware.tz_convert(DISPLAY_TZ)


def _fmt_short(ts: pd.Timestamp | None) -> str:
    converted = _to_display_tz(ts)
    if converted is None:
        return "—"
    return converted.strftime("%b %-d")


def _fmt_band(band: tuple[pd.Timestamp, pd.Timestamp] | None) -> str:
    if band is None:
        return "—"
    lo, hi = band
    return f"{_fmt_short(lo)} → {_fmt_short(hi)}"


# ─────────────────────────────────────────────────────────────────────────────
# Compute (cached + shared with other pages via app_state)
# ─────────────────────────────────────────────────────────────────────────────

tilt_df = app_state.load_tilt_df()
ingest_result = app_state.load_run_report()
reconcile_report = ingest_result.reconcile

all_peaks = app_state.get_peaks(
    tilt_df,
    min_prominence=st.session_state["adv_min_prominence"],
    min_distance_days=st.session_state["adv_min_distance_days"],
    min_height=st.session_state["adv_min_height"],
)
recent_peaks = app_state.get_recent_peaks(all_peaks, st.session_state["tw_n_peaks_for_fit"])
prediction = app_state.get_prediction(tilt_df, recent_peaks)
eruption_state, _eruption_state_info = app_state.get_eruption_state(tilt_df, prediction)


# ─────────────────────────────────────────────────────────────────────────────
# Chart options — controls that only make sense next to the chart.
# ─────────────────────────────────────────────────────────────────────────────

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

show_per_source = st.session_state["tw_show_per_source"]

# Advanced model tuning — sliders that tune peak detection sensitivity.
# Writes directly into ``adv_*`` session_state keys that the entrypoint
# seeds on every rerun.
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
        ``(y - b) / a`` per ``_apply_ab_corrections``

    Returns ``{source_name → DataFrame}`` ready to hand to build_figure.
    Best-effort: a missing CSV or malformed row report returns an
    empty dict so overlay mode silently falls back to no overlay.
    """
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
# Phase 4 Commit 5: enable box-select on the chart so the user can drag a
# rectangle over a region to populate the CSV export date range.
chart_selection = st.plotly_chart(
    fig,
    width="stretch",
    on_select="rerun",
    selection_mode="box",
    key="main_chart",
)
# Plotly's native double-click-to-deselect is fiddly to hit on the chart
# area; give the user an explicit button that clears the box-select state
# so the CSV export date pickers fall back to their defaults. Only render
# when a selection is actually live to keep the UI quiet otherwise.
_has_chart_selection = (
    bool(chart_selection)
    and isinstance(chart_selection, dict)
    and chart_selection.get("selection", {}).get("box")
)
if _has_chart_selection and st.button(
    "✕ Clear chart selection", key="clear_chart_selection"
):
    st.session_state.pop("main_chart", None)
    st.rerun()

# Hover-to-clipboard: press ⌘C / Ctrl+C while hovering a datapoint on the
# chart to copy "YYYY-MM-DD HH:MM | X.XX µrad" to the clipboard. Uses
# st.html(unsafe_allow_javascript=True) which renders the <script> inline
# in the main Streamlit document — not in a sandboxed iframe — so the JS
# reaches Plotly's ``.js-plotly-plot`` node directly via ``document``.
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
          current = t + ' | ' + y + ' µrad';
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
    # Resolve initial range from box-selection on the chart, falling back
    # to the last 7 days of data.
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
