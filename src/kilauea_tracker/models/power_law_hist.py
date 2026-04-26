"""Power-law with historical-median exponent.

Variant B of the cross-cycle parameter projection technique: the *shape*
parameter ``p`` is fixed to the median of historical-fit ``p`` values
from the last N complete inflation phases (CV ≈ 0.31, regime-stable),
while the *amplitude* parameters ``a`` and ``c`` are fit fresh to the
current episode's data. Combines historical pattern stability with
current-episode scaling.

Falls back to plain ``power_law`` when fewer than
``HIST_FIT_MIN_EPISODES`` complete episodes are available — without
enough history we can't estimate a stable median ``p``.

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
    _FixedPCurve = Callable[[np.ndarray, float, float], np.ndarray]
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
from ._huber import fit_huber_curve, fit_huber_linear
from ._intersection import (
    PROJECTION_WINDOW_DAYS as _PROJECTION_WINDOW_DAYS,
)
from ._intersection import find_intersection
from .output import ModelOutput, NamedCurve

# Same exponent bounds as the within-cycle ``power_law`` model — the
# constraint applies during the historical-fit step too.
_POWER_P_MIN = 0.3
_POWER_P_MAX = 1.5

_MIN_FIT_SAMPLES = 12
_CURVE_GRID_POINTS = 200
_CURVE_FIT_MAXFEV = 10000
_T_FLOOR_HOURS = 1e-3


def _power_curve(t_hours: np.ndarray, a: float, p: float, c: float) -> np.ndarray:
    t = np.maximum(np.asarray(t_hours, dtype=float), _T_FLOOR_HOURS)
    return a * np.power(t, p) + c


def _power_curve_fixed_p(p: float) -> _FixedPCurve:
    """Closure: 2-param fit (a, c) at fixed p."""

    def _f(t_hours: np.ndarray, a: float, c: float) -> np.ndarray:
        return _power_curve(t_hours, a, p, c)

    return _f


@dataclass(frozen=True)
class PowerLawHistModel:
    """Power-law with the exponent fixed to the historical median.

    Constrains shape from history while still letting amplitude fit to
    the current episode. Best when current data is sparse and the
    regime is stable.
    """

    id: str = "power_law_hist_p"
    label: str = "Power-law w/ historical-median exponent"
    description: str = (
        "Power-law fit y = a·t^p + c, with p fixed to the median of "
        "historical fits across the last N complete episodes. The "
        "shape parameter is regime-stable (CV ≈ 0.31); amplitude "
        "varies cycle-to-cycle and is fit to the current episode. "
        "Combines cross-cycle pattern stability with current-cycle "
        "scaling. Falls back to plain power-law when historical data "
        "is insufficient."
    )

    def predict(
        self, tilt_df: pd.DataFrame, peaks_df: pd.DataFrame
    ) -> ModelOutput:
        return _predict_power_law_hist(tilt_df, peaks_df)


def _predict_power_law_hist(
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
            f"need ≥ {HIST_FIT_MIN_EPISODES}; "
            f"fall back to plain power_law"
        )
        return _trendline_only_output(
            trend_slope, trend_intercept, peaks_day, len(sorted_peaks), diagnostics
        )

    # Per-episode 3-param fits to recover historical p.
    historical_p: list[float] = []
    for seg in segments:
        a_guess = max(seg.amplitude_microrad, 1.0)
        try:
            params, _, _ = fit_huber_curve(
                _power_curve,
                seg.t_hours_since_trough,
                seg.tilt,
                p0=[a_guess, 0.6, seg.trough_tilt_microrad],
                bounds=(
                    [0.0, _POWER_P_MIN, -np.inf],
                    [np.inf, _POWER_P_MAX, np.inf],
                ),
                maxfev=_CURVE_FIT_MAXFEV,
            )
        except (RuntimeError, ValueError):
            continue
        historical_p.append(float(params[1]))

    if len(historical_p) < HIST_FIT_MIN_EPISODES:
        diagnostics["warning"] = (
            f"only {len(historical_p)} historical fits converged; "
            f"need ≥ {HIST_FIT_MIN_EPISODES}"
        )
        return _trendline_only_output(
            trend_slope, trend_intercept, peaks_day, len(sorted_peaks), diagnostics
        )

    median_p = float(np.median(historical_p))
    iqr_p = (
        float(np.quantile(historical_p, 0.25)),
        float(np.quantile(historical_p, 0.75)),
    )
    diagnostics.update({
        "historical_p": historical_p,
        "median_p": median_p,
        "iqr_p": list(iqr_p),
    })

    # Current-episode trough.
    trough = find_current_episode_trough(tilt_df, peaks_df)
    if trough is None:
        diagnostics["warning"] = "no current-episode trough found"
        return _trendline_only_output(
            trend_slope, trend_intercept, peaks_day, len(sorted_peaks), diagnostics
        )
    trough_date, trough_tilt = trough
    diagnostics["current_episode_start"] = str(trough_date)

    post_trough = tilt_df[tilt_df[DATE_COL] >= trough_date].sort_values(DATE_COL)
    if len(post_trough) < _MIN_FIT_SAMPLES:
        diagnostics["warning"] = (
            f"only {len(post_trough)} post-trough samples; "
            f"need ≥ {_MIN_FIT_SAMPLES}"
        )
        return _trendline_only_output(
            trend_slope, trend_intercept, peaks_day, len(sorted_peaks), diagnostics
        )

    post_day = to_days(post_trough[DATE_COL]).astype(float)
    post_tilt = post_trough[TILT_COL].to_numpy(dtype=float)
    trough_day_val = float(post_day[0])
    t_hours = (post_day - trough_day_val) * 24.0

    # 2-param fit (a, c) at fixed p = median_p.
    fixed_p_curve = _power_curve_fixed_p(median_p)
    a_guess = max(float(post_tilt.max() - post_tilt.min()), 1.0)
    try:
        params, _, fit_quality = fit_huber_curve(
            fixed_p_curve,
            t_hours,
            post_tilt,
            p0=[a_guess, float(trough_tilt)],
            bounds=([0.0, -np.inf], [np.inf, np.inf]),
            maxfev=_CURVE_FIT_MAXFEV,
        )
    except (RuntimeError, ValueError) as e:
        diagnostics["fit_error"] = str(e)
        return _trendline_only_output(
            trend_slope, trend_intercept, peaks_day, len(sorted_peaks), diagnostics
        )

    a = float(params[0])
    c = float(params[1])
    diagnostics.update({
        "power_a": a,
        "power_p_fixed": median_p,
        "power_c": c,
        "n_fit_samples": int(fit_quality.n_samples),
        "fit_rmse_microrad": fit_quality.rmse_microrad,
        "fit_huber_scale_microrad": fit_quality.f_scale,
        "fit_window_start_utc": str(post_trough[DATE_COL].iloc[0]),
        "fit_window_end_utc": str(post_trough[DATE_COL].iloc[-1]),
    })

    def _curve_eval_days(x_days: np.ndarray) -> np.ndarray:
        t_h = (np.asarray(x_days, dtype=float) - trough_day_val) * 24.0
        return _power_curve(t_h, a, median_p, c)

    def _trend_eval(x_days: np.ndarray) -> np.ndarray:
        return trend_slope * np.asarray(x_days, dtype=float) + trend_intercept

    last_current_day = float(post_day[-1])
    last_peak_day = float(peaks_day[-1])
    next_event_date, next_event_tilt = find_intersection(
        f_curve=_curve_eval_days,
        f_lin=_trend_eval,
        last_current_day=last_current_day,
        last_peak_day=last_peak_day,
    )
    if next_event_date is None:
        diagnostics["warning"] = "no intersection with trendline in projection window"

    # Joint MC: resample historical p; bootstrap-resample peaks for trend.
    fit_yhat = _power_curve(t_hours, a, median_p, c)
    fit_residuals = post_tilt - fit_yhat
    p_array = np.array(historical_p, dtype=float)
    trend_grid = np.linspace(
        float(peaks_day.min()),
        last_current_day + _PROJECTION_WINDOW_DAYS,
        _CURVE_GRID_POINTS,
    )
    curve_grid = np.linspace(
        trough_day_val,
        last_current_day + _PROJECTION_WINDOW_DAYS,
        _CURVE_GRID_POINTS,
    )

    def draw_curve(rng: np.random.Generator) -> _CurveEval | None:
        # Resample p (cross-episode), residuals (within-episode), refit (a, c).
        p_b = float(rng.choice(p_array))
        boot_residuals = rng.choice(
            fit_residuals, size=len(fit_residuals), replace=True
        )
        y_boot = fit_yhat + boot_residuals
        try:
            params_b, _, _ = fit_huber_curve(
                _power_curve_fixed_p(p_b),
                t_hours,
                y_boot,
                p0=[a, c],
                bounds=([0.0, -np.inf], [np.inf, np.inf]),
                maxfev=_CURVE_FIT_MAXFEV,
            )
        except (RuntimeError, ValueError):
            return None
        ab, cb = float(params_b[0]), float(params_b[1])

        def _eval(
            x_days: np.ndarray,
            _a: float = ab,
            _p: float = p_b,
            _c: float = cb,
        ) -> np.ndarray:
            t_h = (np.asarray(x_days, dtype=float) - trough_day_val) * 24.0
            return _power_curve(t_h, _a, _p, _c)

        return _eval

    draw_trend = bootstrap_peak_trend_sampler(peaks_day, peaks_tilt)

    def intersect(curve_eval: _CurveEval, trend_eval: _CurveEval) -> pd.Timestamp | None:
        date, _ = find_intersection(
            f_curve=curve_eval,
            f_lin=trend_eval,
            last_current_day=last_current_day,
            last_peak_day=last_peak_day,
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
                label="Power (hist p) 80% CI",
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
            label=f"Current episode (power, hist p={median_p:.2f})",
            days=curve_grid,
            values=_curve_eval_days(curve_grid),
            style="solid",
            color_role="primary",
        )
    )

    headline = (
        f"power × trendline (hist p={median_p:.2f})"
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
