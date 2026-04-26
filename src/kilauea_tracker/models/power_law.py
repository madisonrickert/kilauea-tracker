"""Trendline × constrained-power-law intersection model.

Fits ``y = a · (t − t₀)^p + c`` to the post-trough recovery, with the
exponent constrained to ``p ∈ [0.3, 1.5]``. The constraint matters:
unconstrained partial-fit power-law was the best-performing model in
the small-window backtest (15 h median |error| at 33% data) but also
the most fragile (one episode produced a +595 h overshoot when the
fitter let p drift > 1.5). The bound captures the natural slowdown
(p < 1) and degenerates gracefully to linear (p = 1).

Pure: no I/O, no clock reads, no module-level mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..model import DATE_COL, TILT_COL, to_days

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    _CurveEval = Callable[[np.ndarray], np.ndarray]
from ._bootstrap import (
    DEFAULT_N_SAMPLES,
    DEFAULT_QUANTILES,
    bootstrap_peak_trend_sampler,
    joint_mc_bands,
)
from ._episodes import find_current_episode_trough
from ._huber import fit_huber_curve, fit_huber_linear
from ._intersection import (
    PROJECTION_WINDOW_DAYS as _PROJECTION_WINDOW_DAYS,
)
from ._intersection import (
    find_intersection,
)
from .output import ModelOutput, NamedCurve

# Bound the exponent. p < 0.3 produces near-step-function fits that are
# numerically pathological; p > 1.5 produces runaway projections (the
# +595 h overshoot incident from the unconstrained backtest).
_POWER_P_MIN = 0.3
_POWER_P_MAX = 1.5

_MIN_FIT_SAMPLES = 12
_CURVE_GRID_POINTS = 200
_CURVE_FIT_MAXFEV = 10000

# Numerical floor on the time-since-trough argument so np.power never
# sees a 0 base raised to a fractional exponent (NaN).
_T_FLOOR_HOURS = 1e-3


def _power_curve(t_hours: np.ndarray, a: float, p: float, c: float) -> np.ndarray:
    """``y = a · t^p + c`` with t clamped above zero."""
    t = np.maximum(np.asarray(t_hours, dtype=float), _T_FLOOR_HOURS)
    return a * np.power(t, p) + c


@dataclass(frozen=True)
class PowerLawModel:
    """Constrained power-law fit through the post-trough recovery,
    intersected with the linear trendline through recent peaks. Best
    small-window prediction in the partial-fit backtest; the exponent
    constraint prevents unconstrained-fit overshoots.
    """

    id: str = "power_law"
    label: str = "Trendline × constrained power-law"
    description: str = (
        "Power-law fit y = a·t^p + c through the current episode's "
        "post-trough recovery, with p constrained to [0.3, 1.5] so the "
        "fitter can't escape into runaway-overshoot solutions. p < 1 "
        "captures the natural inflation slowdown observed in 6 of 7 "
        "recent episodes. Best small-window convergence in the backtest "
        "(15 h median error at 33% of episode data)."
    )

    def predict(
        self, tilt_df: pd.DataFrame, peaks_df: pd.DataFrame
    ) -> ModelOutput:
        return _predict_power_law(tilt_df, peaks_df)


def _predict_power_law(
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
        "trendline_huber_f_scale": trend_quality.f_scale,
        "n_peaks_in_trendline": int(trend_quality.n_samples),
    })

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

    a_guess = max(float(post_tilt.max() - post_tilt.min()), 1.0)
    p_guess = 0.6
    c_guess = float(trough_tilt)
    try:
        params, _cov, fit_quality = fit_huber_curve(
            _power_curve,
            t_hours,
            post_tilt,
            p0=[a_guess, p_guess, c_guess],
            bounds=([0.0, _POWER_P_MIN, -np.inf], [np.inf, _POWER_P_MAX, np.inf]),
            maxfev=_CURVE_FIT_MAXFEV,
        )
    except (RuntimeError, ValueError) as e:
        diagnostics["fit_error"] = str(e)
        return _trendline_only_output(
            trend_slope, trend_intercept, peaks_day, len(sorted_peaks), diagnostics
        )

    a, p, c = float(params[0]), float(params[1]), float(params[2])
    diagnostics.update({
        "power_a": a,
        "power_p": p,
        "power_c": c,
        "n_fit_samples": int(fit_quality.n_samples),
        "fit_rmse_microrad": fit_quality.rmse_microrad,
        "fit_huber_scale_microrad": fit_quality.f_scale,
        "fit_fraction_downweighted": fit_quality.fraction_downweighted,
        "fit_window_start_utc": str(post_trough[DATE_COL].iloc[0]),
        "fit_window_end_utc": str(post_trough[DATE_COL].iloc[-1]),
    })

    # Curve evaluator in float-days-since-epoch (the convention the
    # intersection helper uses).
    def _curve_eval_days(x_days: np.ndarray) -> np.ndarray:
        t_h = (np.asarray(x_days, dtype=float) - trough_day_val) * 24.0
        return _power_curve(t_h, a, p, c)

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
        diagnostics["warning"] = "no power-law × trendline crossing in projection window"

    # Joint MC: residual bootstrap on the curve, peak bootstrap on the trend.
    fit_yhat = _power_curve(t_hours, a, p, c)
    fit_residuals = post_tilt - fit_yhat
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
        boot_residuals = rng.choice(
            fit_residuals, size=len(fit_residuals), replace=True
        )
        y_boot = fit_yhat + boot_residuals
        try:
            params_b, _, _ = fit_huber_curve(
                _power_curve,
                t_hours,
                y_boot,
                p0=[a, p, c],
                bounds=([0.0, _POWER_P_MIN, -np.inf], [np.inf, _POWER_P_MAX, np.inf]),
                maxfev=_CURVE_FIT_MAXFEV,
            )
        except (RuntimeError, ValueError):
            return None
        ab, pb, cb = float(params_b[0]), float(params_b[1]), float(params_b[2])

        def _eval(
            x_days: np.ndarray,
            _a: float = ab,
            _p: float = pb,
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
                label="Power-law fit 80% CI",
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
            label=f"Current episode (power, p={p:.2f})",
            days=curve_grid,
            values=_curve_eval_days(curve_grid),
            style="solid",
            color_role="primary",
        )
    )

    headline = (
        f"trendline × power (p={p:.2f})" if next_event_date is not None else None
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
