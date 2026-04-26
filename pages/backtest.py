"""Backtest — model accuracy across the inflation lifecycle.

Runs every registered prediction model against the last N complete
inflation phases at four quartiles (25/50/75/100% of inflation-phase
data visible). Shows which model is most reliable at which stage so
the user can pick deliberately rather than by gut feel.

Computation is cached by the merged-history file's mtime — flips to
fresh as soon as the daily cron commits new data.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from kilauea_tracker import app_state
from kilauea_tracker.backtest import (
    DEFAULT_N_SEGMENTS,
    DEFAULT_QUARTILES,
    BacktestResult,
    run_backtest,
)
from kilauea_tracker.config import HISTORY_CSV
from kilauea_tracker.model import DATE_COL
from kilauea_tracker.models import registry as model_registry
from kilauea_tracker.phase import LATE_PHASE_THRESHOLD, estimate_phase
from kilauea_tracker.state import get_state

state = get_state()

st.title("Model backtest")
st.caption(
    "How accurate is each prediction model, at each stage of the "
    "inflation phase? Tested against the recent complete episodes."
)


@st.cache_data(show_spinner="Running backtest across recent episodes…")
def _cached_backtest(
    history_mtime_ns: int,
    n_segments: int,
    fractions: tuple[float, ...],
) -> BacktestResult:
    """Cache key includes the history file's mtime so a fresh ingest
    invalidates automatically. Returns a frozen ``BacktestResult``."""
    tilt = pd.read_csv(HISTORY_CSV, parse_dates=[DATE_COL])
    return run_backtest(tilt, n_segments=n_segments, fractions=fractions)


# Re-fetch knobs.
opts_col1, opts_col2 = st.columns([1, 1])
with opts_col1:
    n_segments = st.slider(
        "Segments to backtest against",
        min_value=4,
        max_value=12,
        value=DEFAULT_N_SEGMENTS,
        help=(
            "How many of the most-recent complete inflation phases to "
            "evaluate against. Older episodes lived in a different "
            "regime (peaks were +12 µrad in mid-2025 vs −8 today), so "
            "going much further back may not reflect current behaviour."
        ),
    )
with opts_col2:
    if st.button("↻ Refresh backtest", help="Recompute against the latest data"):
        _cached_backtest.clear()
        st.rerun()

mtime_ns = HISTORY_CSV.stat().st_mtime_ns
result = _cached_backtest(mtime_ns, n_segments, DEFAULT_QUARTILES)


# ── Current-episode phase indicator ──────────────────────────────────
tilt_df = app_state.load_tilt_df()
all_peaks = app_state.get_peaks(
    tilt_df,
    min_prominence=state.widgets.peaks.min_prominence,
    min_distance_days=state.widgets.peaks.min_distance_days,
    min_height=state.widgets.peaks.min_height,
)
phase = estimate_phase(tilt_df, all_peaks)

st.subheader("Where the current episode is in its inflation phase")
if phase.fraction is None:
    st.info(
        "Not enough data to estimate the current phase. Need at least one "
        "post-trough sample plus several complete past episodes for the "
        "median-duration baseline."
    )
else:
    phase_pct = phase.fraction * 100.0
    elapsed_d = (phase.elapsed_hours or 0) / 24.0
    median_d = (phase.median_duration_hours or 1) / 24.0
    label = "**EARLY phase** (linear regime)" if not phase.is_late else "**LATE phase** (linear_naive regime)"
    progress_value = min(phase.fraction, 1.0)
    st.progress(progress_value, text=f"{phase_pct:.0f}% through expected inflation phase")
    st.markdown(
        f"{label} — elapsed since trough: **{elapsed_d:.1f} d**, median past "
        f"inflation duration: **{median_d:.1f} d** "
        f"(over {phase.n_historical_episodes} episodes).  \n"
        f"Halfway threshold for the `auto` model is "
        f"**{LATE_PHASE_THRESHOLD:.0%}**."
    )
    # Surface the recommended base model based on the backtest's per-quartile
    # winner closest to the current phase.
    nearest_q = min(result.fractions, key=lambda f: abs(f - phase.fraction))
    best = result.best_per_quartile().get(nearest_q)
    if best is not None:
        st.markdown(
            f"📌 At this phase the backtest's most reliable model is "
            f"**`{best.model_id}`** "
            f"(median |error| {best.median_abs_error_h:.0f}h at quartile "
            f"{nearest_q:.0%}, coverage {best.n_predictions}/{best.n_segments})."
        )

st.divider()


# ── Per-quartile rankings ────────────────────────────────────────────
st.subheader("Per-quartile rankings")
st.caption(
    "For each quartile of the inflation phase, models are ranked by median "
    "absolute error. Models with coverage <50% are not eligible for the "
    "ranking but still appear in the table for visibility."
)

_label_for = {m.id: m.label for m in model_registry.list_models()}


def _rank_dataframe(result: BacktestResult, fraction: float) -> pd.DataFrame:
    rows = []
    for mid in result.model_ids:
        s = result.stats(mid, fraction)
        rows.append({
            "Model": _label_for.get(mid, mid),
            "id": mid,
            "Coverage": f"{s.n_predictions}/{s.n_segments}",
            "Median |error| (h)": s.median_abs_error_h,
            "Median signed error (h)": s.median_signed_error_h,
            "Mean signed error (h)": s.mean_signed_error_h,
        })
    df = pd.DataFrame(rows)
    df["_sort"] = df["Median |error| (h)"].fillna(float("inf"))
    df = df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return df


_q_tabs = st.tabs([f"@ {f:.0%}" for f in result.fractions])
for tab, f in zip(_q_tabs, result.fractions, strict=True):
    with tab:
        df = _rank_dataframe(result, f)
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "id": None,  # hide
                "Median |error| (h)": st.column_config.NumberColumn(format="%.1f"),
                "Median signed error (h)": st.column_config.NumberColumn(format="%+.1f"),
                "Mean signed error (h)": st.column_config.NumberColumn(format="%+.1f"),
            },
        )

st.divider()


# ── Per-model trajectory across quartiles ─────────────────────────────
st.subheader("Per-model trajectory across quartiles")
st.caption(
    "How each model's accuracy evolves as more inflation-phase data "
    "becomes available. Lower is better; a flat line means the model "
    "doesn't use current-episode data."
)

trajectory_rows = []
for mid in result.model_ids:
    row = {"Model": _label_for.get(mid, mid), "id": mid}
    for f in result.fractions:
        s = result.stats(mid, f)
        row[f"@ {f:.0%}"] = s.median_abs_error_h
    trajectory_rows.append(row)
trajectory_df = pd.DataFrame(trajectory_rows)
st.dataframe(
    trajectory_df,
    width="stretch",
    hide_index=True,
    column_config={
        "id": None,
        **{
            f"@ {f:.0%}": st.column_config.NumberColumn(format="%.0f h")
            for f in result.fractions
        },
    },
)

st.divider()


# ── Recommended model per quartile ────────────────────────────────────
st.subheader("Recommended model by quartile")
st.caption(
    "Lowest median |error| with coverage ≥ 50%. The `auto` model picks "
    "between `linear` and `linear_naive` automatically based on the phase "
    "estimator above."
)
best_rows = []
for f, best in result.best_per_quartile().items():
    if best is None:
        best_rows.append({
            "Quartile": f"{f:.0%}",
            "Best model": "—",
            "Median |error| (h)": None,
            "Coverage": "—",
        })
        continue
    best_rows.append({
        "Quartile": f"{f:.0%}",
        "Best model": _label_for.get(best.model_id, best.model_id),
        "Median |error| (h)": best.median_abs_error_h,
        "Coverage": f"{best.n_predictions}/{best.n_segments}",
    })
st.dataframe(
    pd.DataFrame(best_rows),
    width="stretch",
    hide_index=True,
    column_config={
        "Median |error| (h)": st.column_config.NumberColumn(format="%.1f"),
    },
)

st.divider()


# ── Segment list (so the user can see what was evaluated) ─────────────
with st.expander(
    f"📋 Segments evaluated ({len(result.segments)} complete inflation phases)"
):
    seg_rows = [
        {
            "Trough (UTC)": s.trough_date,
            "Peak (UTC)": s.peak_date,
            "Duration (h)": s.duration_hours,
            "Duration (d)": s.duration_hours / 24.0,
        }
        for s in result.segments
    ]
    st.dataframe(
        pd.DataFrame(seg_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Duration (h)": st.column_config.NumberColumn(format="%.1f"),
            "Duration (d)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

with st.expander("ℹ️ How to read this"):
    st.markdown(
        """
**Sign convention.** Signed error > 0 means the model predicted the
next pulse LATER than it actually arrived (model was conservative).
Signed error < 0 means it predicted EARLIER.

**Why errors grow with more data for some models.** As the inflation
curve flattens approaching the peak, the projected slope shrinks and
the predicted intersection date drifts later. The eruption itself is
a fast deflation event — no curve-fit model can see it coming from
the inflation curve alone. Cross-cycle models (`linear_hist`,
`linear_stitched`, `power_law_hist_p`, `interval_median`) have flat
trajectories because they don't use current-episode data.

**Why pick `auto`.** It picks `linear` below the halfway mark and
`linear_naive` at and after, which matches the per-quartile winners
in the backtest. It's a strict improvement over either single model
unless the phase indicator is wildly miscalibrated for the current
regime.

**Caveats.** The data regime shifts — peaks have collapsed from
+12 µrad in mid-2025 to −8 µrad in 2026. Past performance is not
guaranteed; re-run after each new episode to see whether the same
model still wins.
        """
    )
