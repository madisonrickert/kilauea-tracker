"""Stitched-inflation linear-trend model ("strip the drops").

Concatenates the post-trough inflation samples from the last N complete
episodes (dropping the deflation drops between them) and fits a single
Huber-robust linear regression jointly across all samples. The result is
a long-term inflation-rate estimate supported by ~12,000+ samples
instead of the ~30 peaks the standard trendline uses.

Two diagnostic outputs:

- ``stitched_slope_per_hour``: the average inflation rate across the
  recent regime, used as the curve slope.
- ``stitched_residual_std_microrad``: how well the concatenation
  actually looks linear. Small → steady-inflation regime; large →
  the volcano isn't in steady inflation and the model is unreliable.

The slope is anchored at the current episode's trough timestamp + tilt,
projected forward, intersected closed-form with the peak trendline.

Pure: no I/O, no clock reads, no module-level mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..config import HIST_FIT_MIN_EPISODES, HIST_FIT_N_EPISODES

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    _CurveEval = Callable[[np.ndarray], np.ndarray]
from ..model import DATE_COL, TILT_COL, to_days
from ._bootstrap import (
    DEFAULT_N_SAMPLES,
    DEFAULT_QUANTILES,
    bootstrap_peak_trend_sampler,
    joint_mc_bands,
)
from ._episodes import (
    find_current_episode_trough,
    iter_complete_inflation_segments,
)
from ._huber import fit_huber_linear
from ._intersection import find_linear_intersection
from .output import ModelOutput, NamedCurve

_CURVE_GRID_POINTS = 200
_PROJECTION_WINDOW_DAYS = 90.0


@dataclass(frozen=True)
class LinearStitchedModel:
    """Long-term inflation rate from stitched-together inflation segments.

    Drops the deflation drops between episodes and fits a single line
    through all the climbs. The resulting slope describes the volcano's
    average inflation rate; ``stitched_residual_std_microrad`` says how
    well a single line explains that concatenation.
    """

    id: str = "linear_stitched"
    label: str = "Stitched-inflation linear trend"
    description: str = (
        "Concatenates the post-trough inflation samples from the last "
        "N complete episodes (dropping the deflation drops between) "
        "and fits a single Huber line. The slope is the volcano's "
        "long-term inflation rate; the residual std (in diagnostics) "
        "indicates how steady that rate has been. Anchored at the "
        "current trough."
    )

    def predict(
        self, tilt_df: pd.DataFrame, peaks_df: pd.DataFrame
    ) -> ModelOutput:
        return _predict_stitched(tilt_df, peaks_df)


def _predict_stitched(
    tilt_df: pd.DataFrame, peaks_df: pd.DataFrame
) -> ModelOutput:
    diagnostics: dict = {}
    if len(peaks_df) < 2:
        diagnostics["error"] = "need at least 2 peaks for a linear trendline"
        return ModelOutput(
            next_event_date=None,
            confidence_band=None,
            headline_text=None,
            curves=[],
            diagnostics=diagnostics,
        )

    sorted_peaks = peaks_df.sort_values(DATE_COL).copy()
    peaks_day = to_days(sorted_peaks[DATE_COL]).astype(float)
    peaks_tilt = sorted_peaks[TILT_COL].to_numpy(dtype=float)

    trend_slope, trend_intercept, trend_quality = fit_huber_linear(
        peaks_day, peaks_tilt
    )
    diagnostics.update({
        "trendline_slope_per_day": trend_slope,
        "n_peaks_in_trendline": int(trend_quality.n_samples),
    })

    segments = iter_complete_inflation_segments(
        tilt_df, peaks_df, n=HIST_FIT_N_EPISODES
    )
    diagnostics["n_historical_episodes_available"] = len(segments)
    if len(segments) < HIST_FIT_MIN_EPISODES:
        diagnostics["warning"] = (
            f"have {len(segments)} complete episodes; "
            f"need ≥ {HIST_FIT_MIN_EPISODES}"
        )
        return _trendline_only_output(
            trend_slope, trend_intercept, peaks_day, len(sorted_peaks), diagnostics
        )

    # Concatenate all (t_hours_since_trough, tilt - trough_tilt) pairs.
    # We center each segment on its own trough so they're stitched
    # additively from a common zero-tilt-zero-time origin. The slope
    # estimate is in µrad per hour-since-trough.
    stitched_t: list[np.ndarray] = []
    stitched_y: list[np.ndarray] = []
    for seg in segments:
        stitched_t.append(seg.t_hours_since_trough)
        stitched_y.append(seg.tilt - seg.trough_tilt_microrad)
    t_h = np.concatenate(stitched_t)
    y = np.concatenate(stitched_y)
    diagnostics["n_stitched_samples"] = len(t_h)

    stitched_slope_per_hour, stitched_intercept_origin, stitched_quality = (
        fit_huber_linear(t_h, y)
    )
    stitched_slope_per_day = stitched_slope_per_hour * 24.0
    residuals = y - (stitched_slope_per_hour * t_h + stitched_intercept_origin)
    diagnostics.update({
        "stitched_slope_per_hour": stitched_slope_per_hour,
        "stitched_intercept_microrad": stitched_intercept_origin,
        "stitched_residual_std_microrad": float(np.std(residuals)),
        "stitched_huber_f_scale": stitched_quality.f_scale,
        "stitched_fraction_downweighted": stitched_quality.fraction_downweighted,
    })

    # Anchor the projection at the current episode's trough.
    trough = find_current_episode_trough(tilt_df, peaks_df)
    if trough is None:
        diagnostics["warning"] = "no current-episode trough found"
        return _trendline_only_output(
            trend_slope, trend_intercept, peaks_day, len(sorted_peaks), diagnostics
        )
    trough_date, trough_tilt = trough
    trough_day_val = float(to_days(trough_date))
    diagnostics.update({
        "anchor_trough_utc": str(trough_date),
        "anchor_trough_tilt_microrad": trough_tilt,
    })

    curve_intercept = float(trough_tilt) - stitched_slope_per_day * trough_day_val
    curve_slope = stitched_slope_per_day

    last_peak_day = float(peaks_day[-1])
    earliest_root = max(trough_day_val, last_peak_day)
    latest_root = earliest_root + _PROJECTION_WINDOW_DAYS

    next_event_date, next_event_tilt = find_linear_intersection(
        m_curve=curve_slope,
        b_curve=curve_intercept,
        m_trend=trend_slope,
        b_trend=trend_intercept,
        earliest_day=earliest_root,
        latest_day=latest_root,
    )
    if next_event_date is None:
        diagnostics["warning"] = (
            "stitched-slope projection does not intersect trendline within window"
        )

    # Joint MC: bootstrap stitched samples for the curve slope; bootstrap
    # peaks for the trendline.
    n_stitched = len(t_h)
    trend_grid = np.linspace(
        float(peaks_day.min()),
        trough_day_val + _PROJECTION_WINDOW_DAYS,
        _CURVE_GRID_POINTS,
    )
    curve_grid = np.linspace(
        trough_day_val, trough_day_val + _PROJECTION_WINDOW_DAYS, _CURVE_GRID_POINTS
    )

    def draw_curve(rng: np.random.Generator) -> _CurveEval | None:
        idx = rng.integers(0, n_stitched, size=n_stitched)
        try:
            m_b, _, _ = fit_huber_linear(t_h[idx], y[idx])
        except (ValueError, np.linalg.LinAlgError):
            return None
        m_per_day = m_b * 24.0
        b_b = float(trough_tilt) - m_per_day * trough_day_val
        return lambda x, _m=m_per_day, _b=b_b: _m * np.asarray(x, dtype=float) + _b

    draw_trend = bootstrap_peak_trend_sampler(peaks_day, peaks_tilt)

    def intersect(curve_eval: _CurveEval, trend_eval: _CurveEval) -> pd.Timestamp | None:
        x0 = float(earliest_root)
        x1 = float(earliest_root + 1.0)
        c0 = float(curve_eval(np.array([x0]))[0])
        c1 = float(curve_eval(np.array([x1]))[0])
        t0 = float(trend_eval(np.array([x0]))[0])
        t1 = float(trend_eval(np.array([x1]))[0])
        m_c = c1 - c0
        b_c = c0 - m_c * x0
        m_t = t1 - t0
        b_t = t0 - m_t * x0
        date, _ = find_linear_intersection(
            m_curve=m_c,
            b_curve=b_c,
            m_trend=m_t,
            b_trend=b_t,
            earliest_day=earliest_root,
            latest_day=latest_root,
        )
        return date

    confidence_band, trend_band, curve_band = joint_mc_bands(
        draw_curve=draw_curve,
        draw_trend=draw_trend,
        intersect=intersect,
        curve_grid=curve_grid,
        trend_grid=trend_grid,
        n_samples=DEFAULT_N_SAMPLES,
        quantiles=DEFAULT_QUANTILES,
    )

    curves: list[NamedCurve] = []
    if trend_band is not None:
        curves.append(
            NamedCurve(
                label="Trendline 80% CI",
                days=trend_band.days,
                values=(trend_band.lo + trend_band.hi) / 2.0,
                color_role="ribbon",
                band_lo=trend_band.lo,
                band_hi=trend_band.hi,
            )
        )
    if curve_band is not None:
        curves.append(
            NamedCurve(
                label="Stitched fit 80% CI",
                days=curve_band.days,
                values=(curve_band.lo + curve_band.hi) / 2.0,
                color_role="ribbon",
                band_lo=curve_band.lo,
                band_hi=curve_band.hi,
            )
        )
    curves.append(
        NamedCurve(
            label=f"Trendline (last {len(sorted_peaks)} peaks)",
            days=trend_grid,
            values=trend_slope * trend_grid + trend_intercept,
            style="dashed",
            color_role="primary",
        )
    )
    curves.append(
        NamedCurve(
            label=f"Stitched slope ({len(segments)} eps)",
            days=curve_grid,
            values=curve_slope * curve_grid + curve_intercept,
            style="solid",
            color_role="primary",
        )
    )

    headline = (
        f"stitched slope ({stitched_slope_per_hour:.3f} µrad/h)"
        if next_event_date is not None
        else None
    )
    return ModelOutput(
        next_event_date=next_event_date,
        confidence_band=confidence_band,
        headline_text=headline,
        curves=curves,
        diagnostics=diagnostics,
        next_event_tilt=next_event_tilt,
    )


def _trendline_only_output(
    trend_slope: float,
    trend_intercept: float,
    peaks_day: np.ndarray,
    n_peaks: int,
    diagnostics: dict,
) -> ModelOutput:
    grid = np.linspace(
        float(peaks_day.min()),
        float(peaks_day.max()) + _PROJECTION_WINDOW_DAYS,
        _CURVE_GRID_POINTS,
    )
    return ModelOutput(
        next_event_date=None,
        confidence_band=None,
        headline_text=None,
        curves=[
            NamedCurve(
                label=f"Trendline (last {n_peaks} peaks)",
                days=grid,
                values=trend_slope * grid + trend_intercept,
                style="dashed",
                color_role="primary",
            )
        ],
        diagnostics=diagnostics,
    )
