"""Pure backtest logic shared between the CLI script and the UI page.

For each complete inflation segment (trough → peak) in the recent past,
truncate the world to the data that would have been visible at each
quartile of the segment's progression and ask every registered model
"when's the next pulse?". The error vs the actual peak time is the
backtest's primary signal; per-quartile aggregations let us see which
model is most reliable at which stage of the inflation phase.

Pure: no I/O, no clock reads, no module-level mutable state. The
``run_backtest`` entry point takes already-loaded ``tilt_df`` and
``peaks_df`` and returns a frozen result the caller can render.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .model import DATE_COL, TILT_COL
from .models import registry as model_registry
from .peaks import detect_peaks

if TYPE_CHECKING:
    import pandas as pd

# Episodes have to clear this prominence to count as a real peak. The
# 2026-01-14 false peak in the live data has prom ≈ 5; bumping the
# threshold to 10 drops it cleanly without losing real peaks.
REAL_PEAK_PROMINENCE_FLOOR = 10.0

# Default quartiles. The "inflation phase" is trough → peak; quartiles
# are cumulative data fractions visible to the model at prediction time.
DEFAULT_QUARTILES: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

# Default number of recent complete segments to backtest. 7 lines up with
# the recent regime (peaks 38-45) and is the sweet spot between "enough
# data to rank" and "not so much that an old regime drowns the recent
# behaviour".
DEFAULT_N_SEGMENTS = 7

# Minimum coverage (fraction of segments where the model produced any
# prediction) for a model to be eligible for "best at this quartile"
# ranking. Below this we'd be ranking models on too few data points.
_RANK_MIN_COVERAGE = 0.5


@dataclass(frozen=True)
class SegmentSpec:
    """A single (trough, peak) pair the backtest evaluates against."""

    trough_date: pd.Timestamp
    peak_date: pd.Timestamp

    @property
    def duration_hours(self) -> float:
        return float(
            (self.peak_date - self.trough_date).total_seconds() / 3600.0
        )


@dataclass(frozen=True)
class ModelStageStats:
    """Per-(model, quartile) summary statistics."""

    model_id: str
    fraction: float
    n_predictions: int
    n_segments: int
    median_abs_error_h: float | None
    mean_abs_error_h: float | None
    median_signed_error_h: float | None
    mean_signed_error_h: float | None

    @property
    def coverage(self) -> float:
        return self.n_predictions / max(self.n_segments, 1)


@dataclass(frozen=True)
class BacktestResult:
    """Backtest outcome — every per-segment, per-model, per-quartile error.

    ``errors[segment_idx][model_id][fraction]`` returns the signed error
    in hours (predicted - actual), or ``None`` when the model produced no
    prediction at that quartile.
    """

    segments: list[SegmentSpec]
    fractions: tuple[float, ...]
    model_ids: list[str]
    errors: list[dict[str, dict[float, float | None]]] = field(default_factory=list)

    def stats(self, model_id: str, fraction: float) -> ModelStageStats:
        """Aggregate stats for one model at one quartile."""
        raw = [seg[model_id][fraction] for seg in self.errors]
        valid = [e for e in raw if e is not None]
        n_total = len(raw)
        if not valid:
            return ModelStageStats(
                model_id=model_id,
                fraction=fraction,
                n_predictions=0,
                n_segments=n_total,
                median_abs_error_h=None,
                mean_abs_error_h=None,
                median_signed_error_h=None,
                mean_signed_error_h=None,
            )
        arr = np.asarray(valid, dtype=float)
        return ModelStageStats(
            model_id=model_id,
            fraction=fraction,
            n_predictions=len(valid),
            n_segments=n_total,
            median_abs_error_h=float(np.median(np.abs(arr))),
            mean_abs_error_h=float(np.mean(np.abs(arr))),
            median_signed_error_h=float(np.median(arr)),
            mean_signed_error_h=float(np.mean(arr)),
        )

    def best_per_quartile(self) -> dict[float, ModelStageStats | None]:
        """The lowest-median-|error| model per quartile, with a coverage gate.

        Models below ``_RANK_MIN_COVERAGE`` of segments are excluded so
        partial-coverage flukes don't beat full-coverage workhorses.
        """
        out: dict[float, ModelStageStats | None] = {}
        for f in self.fractions:
            best: ModelStageStats | None = None
            for mid in self.model_ids:
                s = self.stats(mid, f)
                if s.median_abs_error_h is None:
                    continue
                if s.coverage < _RANK_MIN_COVERAGE:
                    continue
                if best is None or s.median_abs_error_h < (
                    best.median_abs_error_h or float("inf")
                ):
                    best = s
            out[f] = best
        return out


def find_recent_segments(
    tilt_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    n_segments: int = DEFAULT_N_SEGMENTS,
    prominence_floor: float = REAL_PEAK_PROMINENCE_FLOOR,
) -> list[SegmentSpec]:
    """Identify the last ``n_segments`` complete trough → peak inflation
    phases in the data, after filtering out low-prominence false peaks.
    """
    real = (
        peaks_df[peaks_df["prominence"] > prominence_floor]
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    sorted_tilt = tilt_df.sort_values(DATE_COL).reset_index(drop=True)
    segments: list[SegmentSpec] = []
    for i in range(1, len(real)):
        prev_peak = real[DATE_COL].iloc[i - 1]
        cur_peak = real[DATE_COL].iloc[i]
        between = sorted_tilt[
            (sorted_tilt[DATE_COL] > prev_peak)
            & (sorted_tilt[DATE_COL] <= cur_peak)
        ]
        if len(between) < 50:
            continue
        trough_idx = between[TILT_COL].idxmin()
        trough_ts = between.loc[trough_idx, DATE_COL]
        segments.append(
            SegmentSpec(trough_date=trough_ts, peak_date=cur_peak)
        )
    return segments[-n_segments:]


def _truncate(
    full_tilt: pd.DataFrame,
    full_peaks: pd.DataFrame,
    segment: SegmentSpec,
    fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (tilt, peaks) restricted to what would have been visible at
    ``fraction`` of the way through ``segment``'s inflation phase. The
    target peak itself is hidden from the peaks_df — that's what the
    model is being asked to predict."""
    cutoff = segment.trough_date + (
        segment.peak_date - segment.trough_date
    ) * fraction
    tilt = full_tilt[full_tilt[DATE_COL] <= cutoff].copy()
    peaks = full_peaks[full_peaks[DATE_COL] < segment.peak_date].copy()
    return tilt, peaks


def run_backtest(
    tilt_df: pd.DataFrame,
    peaks_df: pd.DataFrame | None = None,
    n_segments: int = DEFAULT_N_SEGMENTS,
    fractions: tuple[float, ...] = DEFAULT_QUARTILES,
) -> BacktestResult:
    """Run the backtest. Pure: no I/O, no clock reads.

    ``peaks_df`` is optional — if not provided we run ``detect_peaks``
    on ``tilt_df``. The caller can pre-detect (e.g., to apply different
    detection sensitivity than the default) and pass it in.
    """
    if peaks_df is None:
        peaks_df = detect_peaks(tilt_df)
    real_peaks = (
        peaks_df[peaks_df["prominence"] > REAL_PEAK_PROMINENCE_FLOOR]
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    segments = find_recent_segments(tilt_df, peaks_df, n_segments)
    model_ids = [m.id for m in model_registry.list_models()]

    errors: list[dict[str, dict[float, float | None]]] = []
    for seg in segments:
        per_segment: dict[str, dict[float, float | None]] = {
            mid: dict.fromkeys(fractions) for mid in model_ids
        }
        for f in fractions:
            tilt_t, _ = _truncate(tilt_df, real_peaks, seg, f)
            peaks_t = real_peaks[real_peaks[DATE_COL] < seg.peak_date]
            for model in model_registry.list_models():
                try:
                    out = model.predict(tilt_t, peaks_t)
                except Exception:
                    continue
                if out.next_event_date is None:
                    continue
                err_h = (
                    out.next_event_date - seg.peak_date
                ).total_seconds() / 3600.0
                per_segment[model.id][f] = err_h
        errors.append(per_segment)

    return BacktestResult(
        segments=segments,
        fractions=fractions,
        model_ids=model_ids,
        errors=errors,
    )
