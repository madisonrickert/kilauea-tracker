"""Pipeline — behind-the-scenes instrumentation.

Reconcile diagnostics, model diagnostics, the inspector overlay layer
toggles, the per-source PNG transcription quality inspector, the ingest
pipeline status, the live USGS source plots, and the recent refresh-run
JSON reports. None of this is for casual visitors — it's the maintainer's
view into what the pipeline actually did this run.

Page is large by design: the inspector closures (`_build_overlay_png`,
`_inspector_calibrate`, `_fetch_png_bytes`) all share the same
calibration + raw-bytes lifecycle, and splitting them into ui modules
would mean threading a lot of state through function arguments without
buying much.
"""

from __future__ import annotations

import io
import math
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from kilauea_tracker import app_state
from kilauea_tracker.config import (
    ALL_SOURCES,
    HISTORY_CSV,
    INGEST_CACHE_TTL_SECONDS,
    TILT_SOURCE_NAME,
    USGS_TILT_URLS,
)
from kilauea_tracker.ingest.calibrate import AxisCalibration
from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.ui.diagnostics import (
    episode_samples_tint,
    exp_amplitude_tint,
    exp_k_tint,
    render_chip_html,
    trendline_slope_tint,
)

if TYPE_CHECKING:
    from kilauea_tracker.ingest.pipeline import IngestReport

from kilauea_tracker.state import get_state

state = get_state()
DISPLAY_TZ = (
    "Pacific/Honolulu"
    if state.widgets.chart.timezone_choice.startswith("HST")
    else "UTC"
)


# ─────────────────────────────────────────────────────────────────────────────
# Inspector-overlay dock — tag the "Inspector overlay layers" expander with
# .kt-overlay-dock so styles.py can float it on desktop. Only this page
# renders that expander, so the observer lives here. Streamlit doesn't
# expose a Python-level class hook on st.expander, hence the JS shim.
# ─────────────────────────────────────────────────────────────────────────────

st.html(
    """
    <script>
    (function() {
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


# ─────────────────────────────────────────────────────────────────────────────
# Compute (cached + shared with other pages via app_state)
# ─────────────────────────────────────────────────────────────────────────────

tilt_df = app_state.load_tilt_df()
ingest_result = app_state.load_run_report()
reports = ingest_result.per_source
reconcile_report = ingest_result.reconcile

all_peaks = app_state.get_peaks(
    tilt_df,
    min_prominence=state.widgets.peaks.min_prominence,
    min_distance_days=state.widgets.peaks.min_distance_days,
    min_height=state.widgets.peaks.min_height,
)
recent_peaks = app_state.get_recent_peaks(all_peaks, state.widgets.chart.n_peaks_for_fit)
prediction = app_state.get_prediction(tilt_df, recent_peaks)


# Overlay-layer toggles — typed read off the snapshot. Streamlit's
# widget→key auto-binding (in the inspector expander further down) still
# does the writes; we just read here.
_ovl = state.widgets.overlays
layer_dots = _ovl.dots
layer_bbox = _ovl.bbox
layer_yticks = _ovl.yticks
layer_ygrid = _ovl.ygrid
layer_corners = _ovl.corners
layer_blue = _ovl.blue
layer_legend = _ovl.legend
layer_dropcols = _ovl.dropcols
layer_outliers = _ovl.outliers
layer_now = _ovl.now
layer_green = _ovl.green
layer_csv = _ovl.csv


# ─────────────────────────────────────────────────────────────────────────────
# Reconcile diagnostics
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
        st.markdown("**Per-source alignments** — the scalar offset `b` applied to each source")
        align_rows = []
        for s in reconcile_report.sources:
            align_rows.append({
                "source": s.name,
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
            "**Pairwise fits** — each source pair's median scalar offset `β = median(y_i - y_j)` over overlapping buckets"
        )
        if reconcile_report.pairs:
            pair_rows = [
                {
                    "i": p.source_i,
                    "j": p.source_j,
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
# Detail expanders — peaks + model diagnostics
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
            col_tau.markdown(
                render_chip_html(
                    label="τ (time const)",
                    value=f"{tau_days:.1f}" if tau_days != float("inf") else "—",
                    unit="days",
                    tint=exp_k_tint(k),
                ),
                unsafe_allow_html=True,
            )
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


# ─────────────────────────────────────────────────────────────────────────────
# Inspector overlay layer toggles + transcription quality inspector
# ─────────────────────────────────────────────────────────────────────────────

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
        calibration: AxisCalibration | None,
        report: IngestReport | None,
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

            from kilauea_tracker.config import CURVE_MAX_COLUMN_WIDTH_PIXELS
            from kilauea_tracker.ingest.trace import (
                BLUE_HUE_MAX,
                BLUE_HUE_MIN,
                LEGEND_EXCLUSION_PLOT_RELATIVE,
                SATURATION_FLOOR,
                VALUE_FLOOR,
                trace_curve,
            )
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

        def date_to_px(ts: pd.Timestamp) -> float:
            return x0 + (ts - calibration.x_start).total_seconds() * px_span / span_seconds

        def val_to_py(val: float) -> float:
            return (val - calibration.y_intercept) / calibration.y_slope

        def fmt_dt_tz(ts: pd.Timestamp) -> str:
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
        _W, _H = img.size
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
            for py, px in zip(ys, xs, strict=False):
                overlay.putpixel((int(px + x0), int(py + y0)), (0, 200, 255, 140))

        # ── Green-mask highlight (magenta for high contrast vs green) ─
        if layers.get("green") and green_mask is not None:
            ys, xs = np.where(green_mask)
            stats["green_px"] = int(ys.size)
            for py, px in zip(ys, xs, strict=False):
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
            start = np.ceil(lo / step) * step
            end = np.floor(hi / step) * step
            n_lines = round((end - start) / step) + 1
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
                    if abs(step - round(step)) < 1e-6:
                        label = f"{round(val):+d}"
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
        if layers.get("now"):
            from datetime import datetime as _dt
            now = pd.Timestamp(_dt.now(tz=UTC)).tz_localize(None)
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
        # plot. Without correction the dots all fall off-plot.
        #
        # We subtract the median offset between CSV values and re-traced
        # values at overlapping timestamps. Any REMAINING offset between
        # dots and the blue line = a real per-sample alignment defect.
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

    # Map source_name → URL; replicated here because the original mapping
    # in the USGS-source-plots expander is built further down.
    _inspector_url_map = {TILT_SOURCE_NAME[s]: USGS_TILT_URLS[s] for s in ALL_SOURCES}

    def _inspector_calibrate(
        raw_bytes: bytes, source_name: str
    ) -> tuple[AxisCalibration | None, str | None]:
        """Decode raw PNG bytes and run the calibration pipeline.

        The pipeline's ``ingest()`` skips calibration on 304 Not Modified
        (the per-source CSV is already fresh), so on a re-visit to a
        warmed-cache app the IngestReports come back with
        ``calibration=None``. The inspector fetches fresh PNG bytes
        unconditionally, so we can always re-calibrate from those bytes
        here and show the overlays against the current PNG's axes.

        NOT cached via st.cache_data: @st.cache_data's hash-based cache
        invalidation only sees changes to THIS function's body, so
        pushing a fix to calibrate_axes (which this wrapper calls into)
        wouldn't invalidate the cache if raw_bytes were unchanged.
        Calibration is ~50ms; not worth the staleness risk.
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
        # Prefer the pipeline's calibration when present; otherwise
        # calibrate the PNG ourselves so we can still render overlays on
        # 304-Not-Modified runs.
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

        cal = calibration
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


# ─────────────────────────────────────────────────────────────────────────────
# Ingest pipeline status
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# USGS source plots
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Recent refresh runs
# ─────────────────────────────────────────────────────────────────────────────

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
