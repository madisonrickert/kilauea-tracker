"""Linear inflation-rate intersection models.

``linear`` and ``linear_naive`` share fit + intersection machinery and
differ only in the post-trough window they fit on:

- ``linear`` (recommended) trims the steep-early portion of the recovery
  before fitting. Empirically across the last 7 episodes the first
  quartile's slope is 1.4–2.6× the last quartile's; fitting on the full
  window therefore biases predictions earlier than reality. Trim window
  is ``min(last INFLATION_LATE_FIT_FRACTION of samples, last
  INFLATION_LATE_FIT_MAX_DAYS days)`` from ``config.py``.

- ``linear_naive`` (reference baseline only) fits the entire post-trough
  window. Included so the chart's selector can show the size of the
  early-slope bias side-by-side with the corrected ``linear`` model.
  Description warns the user it's not the recommendation.

The intersection with the linear trendline is closed-form, so neither
model needs brentq. The only failure mode is parallel slopes (handled by
returning ``None`` with a diagnostic) — by construction this model
always emits a prediction otherwise, fixing the user's observed
"never intersects" failure on the exponential default.

Reasoning, slope-decay table, and HVO's "constant inflation rate to
threshold" precedent live in
``.claude/plans/look-at-the-shape-spicy-island.md``.

Pure: no I/O, no clock reads, no module-level mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..config import INFLATION_LATE_FIT_FRACTION, INFLATION_LATE_FIT_MAX_DAYS
from ..model import DATE_COL, TILT_COL, to_days
from ._bootstrap import (
    DEFAULT_N_SAMPLES,
    DEFAULT_QUANTILES,
    bootstrap_peak_trend_sampler,
    joint_mc_bands,
)
from ._episodes import find_current_episode_trough
from ._huber import fit_huber_linear
from ._intersection import find_linear_intersection
from .output import ModelOutput, NamedCurve

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    _CurveEval = Callable[[np.ndarray], np.ndarray]

# Below this point count we don't trust the fit — return no prediction
# with a diagnostic. 12 samples at 15-min cadence ≈ 3 h of post-trough
# data, which is also roughly the window where Huber needs enough
# residuals to estimate scale.
_MIN_FIT_SAMPLES = 12

# x-grid resolution for the rendered NamedCurve. Matches trendline_exp.
_CURVE_GRID_POINTS = 200

# Forward projection horizon — same 90-day default as the other curve
# models so the chart's overlay extents are consistent.
_PROJECTION_WINDOW_DAYS = 90.0


def _select_late_window(
    t_hours: np.ndarray,
    fraction: float,
    max_days: float,
) -> np.ndarray:
    """Indices of the late portion of a post-trough segment.

    Uses ``min(last fraction · n samples, last max_days · 24 hours)`` so
    short episodes degrade gracefully to the fraction cutoff and long
    ones cap at the wallclock window.
    """
    n = len(t_hours)
    n_keep_by_fraction = max(int(np.ceil(n * fraction)), 1)
    cutoff_h = t_hours[-1] - max_days * 24.0
    n_keep_by_days = int(np.sum(t_hours >= cutoff_h))
    n_keep = min(n_keep_by_fraction, n_keep_by_days) if n_keep_by_days > 0 else n_keep_by_fraction
    return np.arange(n - n_keep, n)


def _predict_linear(
    *,
    tilt_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    late_window_only: bool,
    label: str,
    headline_template: str,
) -> ModelOutput:
    """Shared core for both ``linear`` and ``linear_naive``.

    Pure. Never raises. On any unrecoverable failure returns a
    ``ModelOutput`` with ``None`` fields and a diagnostic.
    """
    diagnostics: dict = {}
    n_peaks = len(peaks_df)
    if n_peaks < 2:
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

    # Trendline through the recent peaks (Huber-robust).
    trend_slope, trend_intercept, trend_quality = fit_huber_linear(
        peaks_day, peaks_tilt
    )
    diagnostics.update({
        "trendline_slope_per_day": trend_slope,
        "trendline_huber_f_scale": trend_quality.f_scale,
        "trendline_fraction_downweighted": trend_quality.fraction_downweighted,
        "n_peaks_in_trendline": int(trend_quality.n_samples),
    })

    # Current-episode trough.
    trough = find_current_episode_trough(tilt_df, peaks_df)
    if trough is None:
        diagnostics["warning"] = "no current-episode trough found"
        return ModelOutput(
            next_event_date=None,
            confidence_band=None,
            headline_text=None,
            curves=_trendline_only_curves(
                trend_slope, trend_intercept, peaks_day, n_peaks
            ),
            diagnostics=diagnostics,
        )
    trough_date, _trough_tilt = trough
    diagnostics["current_episode_start"] = str(trough_date)

    # Slice post-trough samples.
    post_trough = tilt_df[tilt_df[DATE_COL] >= trough_date].sort_values(DATE_COL)
    if len(post_trough) < _MIN_FIT_SAMPLES:
        diagnostics["warning"] = (
            f"only {len(post_trough)} post-trough samples; "
            f"need ≥ {_MIN_FIT_SAMPLES}"
        )
        return ModelOutput(
            next_event_date=None,
            confidence_band=None,
            headline_text=None,
            curves=_trendline_only_curves(
                trend_slope, trend_intercept, peaks_day, n_peaks
            ),
            diagnostics=diagnostics,
        )

    post_day = to_days(post_trough[DATE_COL]).astype(float)
    post_tilt = post_trough[TILT_COL].to_numpy(dtype=float)
    t_hours = (post_day - post_day[0]) * 24.0

    if late_window_only:
        idx = _select_late_window(
            t_hours,
            INFLATION_LATE_FIT_FRACTION,
            INFLATION_LATE_FIT_MAX_DAYS,
        )
        if len(idx) < _MIN_FIT_SAMPLES:
            diagnostics["warning"] = (
                f"late window has {len(idx)} samples; need ≥ {_MIN_FIT_SAMPLES}"
            )
            return ModelOutput(
                next_event_date=None,
                confidence_band=None,
                headline_text=None,
                curves=_trendline_only_curves(
                    trend_slope, trend_intercept, peaks_day, n_peaks
                ),
                diagnostics=diagnostics,
            )
    else:
        idx = np.arange(len(post_day))

    fit_day = post_day[idx]
    fit_tilt = post_tilt[idx]

    curve_slope, curve_intercept, curve_quality = fit_huber_linear(
        fit_day, fit_tilt
    )
    diagnostics.update({
        "linear_slope_per_day": curve_slope,
        "linear_huber_f_scale": curve_quality.f_scale,
        "linear_fraction_downweighted": curve_quality.fraction_downweighted,
        "n_fit_samples": int(curve_quality.n_samples),
        "fit_window_start_utc": str(post_trough[DATE_COL].iloc[idx[0]]),
        "fit_window_end_utc": str(post_trough[DATE_COL].iloc[idx[-1]]),
        "fit_rmse_microrad": curve_quality.rmse_microrad,
    })

    # Closed-form intersection.
    last_current_day = float(post_day[-1])
    last_peak_day = float(peaks_day[-1])
    earliest_root = max(last_current_day, last_peak_day)
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
            "linear curve does not intersect trendline within projection window"
        )

    # Joint MC bands. Trendline draws are bootstrap resamples of the
    # peaks; curve draws are bootstrap resamples of the residuals
    # around the fit (parametric on the fit-noise scale via Huber's
    # f_scale, but we use a simple residual bootstrap here for
    # robustness on small n).
    trend_grid = np.linspace(
        float(peaks_day.min()),
        last_current_day + _PROJECTION_WINDOW_DAYS,
        _CURVE_GRID_POINTS,
    )
    curve_grid = np.linspace(
        last_current_day - max(t_hours[idx[-1]] - t_hours[idx[0]], 1.0) / 24.0,
        last_current_day + _PROJECTION_WINDOW_DAYS,
        _CURVE_GRID_POINTS,
    )

    fit_yhat = curve_slope * fit_day + curve_intercept
    fit_residuals = fit_tilt - fit_yhat

    def draw_curve(rng: np.random.Generator) -> _CurveEval | None:
        # Residual bootstrap: resample residuals with replacement, add
        # to fitted values, refit Huber. Captures within-cycle scatter
        # propagated through the fit.
        boot_residuals = rng.choice(
            fit_residuals, size=len(fit_residuals), replace=True
        )
        y_boot = fit_yhat + boot_residuals
        try:
            m_b, b_b, _ = fit_huber_linear(fit_day, y_boot)
        except (ValueError, np.linalg.LinAlgError):
            return None
        return lambda x, _m=m_b, _b=b_b: _m * np.asarray(x, dtype=float) + _b

    draw_trend = bootstrap_peak_trend_sampler(peaks_day, peaks_tilt)

    def intersect(curve_eval: _CurveEval, trend_eval: _CurveEval) -> pd.Timestamp | None:
        # Solve the same closed-form intersection per draw. Each draw's
        # curve_eval and trend_eval are linear lambdas; recover their
        # slope/intercept by sampling at two points to avoid touching
        # the lambda's closure.
        x0 = float(earliest_root)
        x1 = float(earliest_root + 1.0)
        c0, c1 = float(curve_eval(np.array([x0]))[0]), float(curve_eval(np.array([x1]))[0])
        t0, t1 = float(trend_eval(np.array([x0]))[0]), float(trend_eval(np.array([x1]))[0])
        m_c, b_c = c1 - c0, c0 - (c1 - c0) * x0
        m_t, b_t = t1 - t0, t0 - (t1 - t0) * x0
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

    # Build NamedCurves. Order matters — ribbons first so they render
    # underneath the lines.
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
                label="Linear fit 80% CI",
                days=curve_band.days,
                values=(curve_band.lo + curve_band.hi) / 2.0,
                color_role="ribbon",
                band_lo=curve_band.lo,
                band_hi=curve_band.hi,
            )
        )
    curves.append(
        NamedCurve(
            label=f"Trendline (last {n_peaks} peaks)",
            days=trend_grid,
            values=trend_slope * trend_grid + trend_intercept,
            style="dashed",
            color_role="primary",
        )
    )
    curves.append(
        NamedCurve(
            label=label,
            days=curve_grid,
            values=curve_slope * curve_grid + curve_intercept,
            style="solid",
            color_role="primary",
        )
    )

    headline_text = (
        headline_template if next_event_date is not None else None
    )
    return ModelOutput(
        next_event_date=next_event_date,
        confidence_band=confidence_band,
        headline_text=headline_text,
        curves=curves,
        diagnostics=diagnostics,
        next_event_tilt=next_event_tilt,
    )


def _trendline_only_curves(
    trend_slope: float,
    trend_intercept: float,
    peaks_day: np.ndarray,
    n_peaks: int,
) -> list[NamedCurve]:
    """When the curve fit fails but the trendline succeeded, still surface
    the trendline so the chart isn't bare."""
    grid = np.linspace(
        float(peaks_day.min()),
        float(peaks_day.max()) + _PROJECTION_WINDOW_DAYS,
        _CURVE_GRID_POINTS,
    )
    return [
        NamedCurve(
            label=f"Trendline (last {n_peaks} peaks)",
            days=grid,
            values=trend_slope * grid + trend_intercept,
            style="dashed",
            color_role="primary",
        )
    ]


@dataclass(frozen=True)
class LinearModel:
    """Trendline × late-window linear inflation-rate intersection.

    Recommended linear model. Fits a Huber-robust line to the late
    portion of the current episode's recovery (last
    ``INFLATION_LATE_FIT_FRACTION`` of samples or the trailing
    ``INFLATION_LATE_FIT_MAX_DAYS`` days, whichever yields fewer
    samples). Closed-form intersection with the peak trendline.
    Matches HVO's "constant inflation rate to threshold" working
    assumption applied to the asymptotic-rate portion of the recovery.
    """

    id: str = "linear"
    label: str = "Trendline × late-window linear"
    description: str = (
        "Linear inflation-rate fit on the trailing portion of the "
        "current episode's post-trough recovery, intersected with the "
        "trendline through recent peaks. Trims the steep early portion "
        "of the recovery (1.4–2.6× the asymptotic slope across recent "
        "episodes) so the projection isn't biased early. Matches HVO's "
        "constant-inflation-rate forecasting practice."
    )

    def predict(
        self, tilt_df: pd.DataFrame, peaks_df: pd.DataFrame
    ) -> ModelOutput:
        return _predict_linear(
            tilt_df=tilt_df,
            peaks_df=peaks_df,
            late_window_only=True,
            label="Current episode (linear, late window)",
            headline_template="trendline × linear (late window)",
        )


@dataclass(frozen=True)
class LinearNaiveModel:
    """Trendline × full-window linear (reference baseline).

    Same fit as ``LinearModel`` but on the entire post-trough recovery
    rather than just the late portion. Included so the chart can show
    the early-steep-slope bias next to the corrected ``linear`` model.
    """

    id: str = "linear_naive"
    label: str = "Trendline × full-window linear (baseline)"
    description: str = (
        "Reference baseline only — fits a linear inflation rate through "
        "the FULL post-trough recovery, including the steep early hours "
        "that bias predictions toward earlier eruption times. Use this "
        "to gauge how much the late-window correction in the recommended "
        "`linear` model is doing. Not a recommendation."
    )

    def predict(
        self, tilt_df: pd.DataFrame, peaks_df: pd.DataFrame
    ) -> ModelOutput:
        return _predict_linear(
            tilt_df=tilt_df,
            peaks_df=peaks_df,
            late_window_only=False,
            label="Current episode (linear, full window)",
            headline_template="trendline × linear (full window, biased)",
        )
