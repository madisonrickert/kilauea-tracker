"""Historical-median linear-slope projection model.

Uses the cross-cycle median of recent inflation slopes (Huber-fit per
episode) as the projection rate for the current in-progress episode,
anchored at the current trough. Independent of the current episode's
within-cycle data — falls back to "the typical episode shape" rather
than overfitting to whatever steep early-slope is visible right now.

Empirical: per-episode linear slopes have CV ≈ 0.28 across the last 8
episodes — the most stable parameter in the family. See
``.claude/plans/look-at-the-shape-spicy-island.md``.

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
class LinearHistModel:
    """Project the current episode's inflation rate from the median of
    the last N complete episodes' linear-fit slopes.

    Best-suited for early-inflation prediction when the current episode
    has too little data to fit reliably. Falls back to plain ``linear``
    behaviour when fewer than ``HIST_FIT_MIN_EPISODES`` complete
    episodes are available.
    """

    id: str = "linear_hist"
    label: str = "Historical-median linear slope"
    description: str = (
        "Predicts the next pulse using the median linear inflation "
        "slope from the last N complete episodes — independent of the "
        "current episode's data. Most accurate early in inflation when "
        "the within-cycle slope is still steep and biased; least "
        "accurate during regime changes when the median lags the new "
        "rate."
    )

    def predict(
        self, tilt_df: pd.DataFrame, peaks_df: pd.DataFrame
    ) -> ModelOutput:
        return _predict_linear_hist(tilt_df, peaks_df)


def _predict_linear_hist(
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

    # Per-episode linear fits across the last HIST_FIT_N_EPISODES.
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

    historical_slopes_per_hour: list[float] = []
    historical_intercepts: list[float] = []
    for seg in segments:
        m, b, _ = fit_huber_linear(seg.t_hours_since_trough, seg.tilt)
        historical_slopes_per_hour.append(m)
        historical_intercepts.append(b)

    median_slope_per_hour = float(np.median(historical_slopes_per_hour))
    median_slope_per_day = median_slope_per_hour * 24.0
    iqr_slope = (
        float(np.quantile(historical_slopes_per_hour, 0.25)),
        float(np.quantile(historical_slopes_per_hour, 0.75)),
    )
    diagnostics.update({
        "historical_slopes_per_hour": list(historical_slopes_per_hour),
        "median_slope_per_hour": median_slope_per_hour,
        "iqr_slope_per_hour": list(iqr_slope),
        "historical_episode_dates": [str(s.peak_date) for s in segments],
    })

    # Anchor at the current episode's trough.
    trough = find_current_episode_trough(tilt_df, peaks_df)
    if trough is None:
        diagnostics["warning"] = "no current-episode trough found; cannot anchor"
        return _trendline_only_output(
            trend_slope, trend_intercept, peaks_day, len(sorted_peaks), diagnostics
        )
    trough_date, trough_tilt = trough
    trough_day_val = float(to_days(trough_date))
    diagnostics.update({
        "anchor_trough_utc": str(trough_date),
        "anchor_trough_tilt_microrad": trough_tilt,
    })

    curve_intercept = float(trough_tilt) - median_slope_per_day * trough_day_val
    curve_slope = median_slope_per_day

    # Closed-form intersection.
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
            "historical-slope projection does not intersect trendline within window"
        )

    # Joint MC: bootstrap over (a) historical slope set, (b) peaks.
    trend_grid = np.linspace(
        float(peaks_day.min()),
        trough_day_val + _PROJECTION_WINDOW_DAYS,
        _CURVE_GRID_POINTS,
    )
    curve_grid = np.linspace(
        trough_day_val, trough_day_val + _PROJECTION_WINDOW_DAYS, _CURVE_GRID_POINTS
    )
    slope_array = np.array(historical_slopes_per_hour, dtype=float)

    def draw_curve(rng: np.random.Generator) -> _CurveEval | None:
        idx = rng.integers(0, len(slope_array), size=len(slope_array))
        m_b = float(np.median(slope_array[idx])) * 24.0
        b_b = float(trough_tilt) - m_b * trough_day_val
        return lambda x, _m=m_b, _b=b_b: _m * np.asarray(x, dtype=float) + _b

    draw_trend = bootstrap_peak_trend_sampler(peaks_day, peaks_tilt)

    def intersect(curve_eval: _CurveEval, trend_eval: _CurveEval) -> pd.Timestamp | None:
        # Recover slopes by sampling (lines are fully determined by 2 points).
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
                label="Historical-slope 80% CI",
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
            label=f"Hist median slope ({len(segments)} episodes)",
            days=curve_grid,
            values=curve_slope * curve_grid + curve_intercept,
            style="solid",
            color_role="primary",
        )
    )

    headline = (
        f"hist median slope ({median_slope_per_hour:.3f} µrad/h)"
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
