"""Curve-fit + intersection prediction model.

Mathematical content is a refactor of `legacy/eruption_projection.py:94-299`.
The math is preserved verbatim; only the structure changes:

- The hardcoded "current episode start" date (v1.0:112) is replaced with
  "the first tilt sample after the most recent peak" — derived dynamically
  from `peaks_df`.
- The matplotlib `mdates.date2num` conversion (v1.0:74) is replaced with a
  simple "float days since the Unix epoch" helper, which lets us drop the
  matplotlib runtime dependency entirely.
- All plotting/axis logic (v1.0:78-336) is removed; that's `plotting.py`'s job.
- `predict()` is a pure function: same inputs → same outputs, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit

DATE_COL = "Date"
TILT_COL = "Tilt (microradians)"

# Sanity bounds on the predicted intersection's tilt value. v1.0 used
# (-20, 15) at `eruption_projection.py:177` because that matched its November
# 2025 plot range; the live data eventually exceeded those bounds (peaks
# above 16 µrad in early 2026), which silently dropped valid predictions.
# We widen to (-50, 50) so any plausible intersection inside the USGS plot
# window passes — brentq + the projection-time bracket already guarantee the
# root is in a sensible date range, so this check is just a "no-NaN" guard.
_INTERSECTION_TILT_MIN = -50.0
_INTERSECTION_TILT_MAX = 50.0

# How far into the future the intersection search will look. v1.0 used a
# 3-month linear projection window (line 100); we keep that.
_PROJECTION_WINDOW_DAYS = 90.0

# `curve_fit` retry budget — same as v1.0:142.
_CURVE_FIT_MAXFEV = 5000


# ─────────────────────────────────────────────────────────────────────────────
# Public dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Curve:
    """A fitted curve, callable in float-days-since-epoch."""

    name: str
    f: Callable[[float], float]
    domain: tuple[float, float]  # (min_day, max_day) in float days since epoch


@dataclass(frozen=True)
class CurveBand:
    """A confidence ribbon around a fitted curve, evaluated on a fixed x-grid.

    Each entry in `lo` and `hi` is the 10th / 90th percentile of the curve's
    value at the corresponding `days` x-coordinate, computed from a Monte
    Carlo sample of the curve's fit parameters. Renders as a filled region
    between the two y-arrays.
    """

    days: np.ndarray   # float days since epoch, shape (n_grid,)
    lo: np.ndarray     # shape (n_grid,)
    hi: np.ndarray     # shape (n_grid,)


@dataclass(frozen=True)
class Prediction:
    """Result of `predict()`. Any field can be `None` if the underlying fit failed."""

    next_event_date: Optional[pd.Timestamp]
    next_event_tilt: Optional[float]
    trendline: Optional[Curve]          # linear fit through the fit-window peaks
    exp_curve: Optional[Curve]          # current-episode exponential saturation fit
    exp_params: Optional[tuple[float, float, float]]  # (A, k, C)
    exp_x0: Optional[float]             # the date offset baked into the exp fit
    exp_covariance: Optional[np.ndarray]  # for confidence band
    confidence_band: Optional[tuple[pd.Timestamp, pd.Timestamp]]
    trendline_band: Optional[CurveBand]   # 80% CI ribbon around the trendline
    exp_band: Optional[CurveBand]         # 80% CI ribbon around the exp curve
    n_peaks_in_fit: int                 # count of peaks that fed the trendline
    fit_diagnostics: dict


# ─────────────────────────────────────────────────────────────────────────────
# Time helpers — float days since the Unix epoch
# ─────────────────────────────────────────────────────────────────────────────


_EPOCH = pd.Timestamp("1970-01-01")
_SECONDS_PER_DAY = 86400.0


def to_days(t):
    """Convert Timestamp/Series/DatetimeIndex to float days since the Unix epoch.

    Implementation note: modern pandas (>=2.x) chose `datetime64[us]` as the
    default storage unit instead of `[ns]`, so we cannot divide an int64 view by
    a hardcoded constant. Going through `.dt.total_seconds()` (or its scalar
    equivalent) is unit-agnostic and works for any underlying resolution.

    Naive datetimes are interpreted at face value (treated as if UTC); tz-aware
    datetimes are converted properly.
    """
    if isinstance(t, pd.Series):
        return ((t - _EPOCH).dt.total_seconds() / _SECONDS_PER_DAY).to_numpy()
    if isinstance(t, pd.DatetimeIndex):
        return ((t - _EPOCH).total_seconds() / _SECONDS_PER_DAY).to_numpy()
    if isinstance(t, pd.Timestamp):
        return (t - _EPOCH).total_seconds() / _SECONDS_PER_DAY
    raise TypeError(f"to_days: unsupported type {type(t).__name__}")


def from_days(d: float) -> pd.Timestamp:
    """Inverse of `to_days` — float days since epoch back to Timestamp (naive)."""
    return _EPOCH + pd.Timedelta(seconds=float(d) * _SECONDS_PER_DAY)


# ─────────────────────────────────────────────────────────────────────────────
# Curve definitions
# ─────────────────────────────────────────────────────────────────────────────


def exp_saturation(x, A: float, k: float, C: float, x0: float):
    """Exponential saturation curve.

    Same as v1.0:128 — the x0 shift makes the k parameter easier to fit by
    anchoring the curve to the start of the current episode.
    """
    return A * (1.0 - np.exp(-k * (x - x0))) + C


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def predict(tilt_df: pd.DataFrame, peaks_df: pd.DataFrame) -> Prediction:
    """Run the curve-fit + intersection prediction.

    Args:
        tilt_df:  DataFrame with columns ['Date', 'Tilt (microradians)'].
        peaks_df: DataFrame with columns ['Date', 'Tilt (microradians)'] —
                  one row per detected episodic peak.

    Returns:
        A `Prediction`. Fields are `None` when the underlying fit could not be
        computed (e.g. too few data points). The function never raises.
    """
    diagnostics: dict = {}
    n_peaks_in_fit = len(peaks_df)

    if n_peaks_in_fit < 2:
        diagnostics["error"] = "need at least 2 peaks for a linear trendline"
        return _empty(diagnostics, n_peaks_in_fit=n_peaks_in_fit)

    peaks_df = peaks_df.sort_values(DATE_COL).copy()
    peaks_df["_day"] = to_days(peaks_df[DATE_COL])

    # ── linear trendline through the fit-window peaks ────────────────────────
    slope, intercept = np.polyfit(peaks_df["_day"], peaks_df[TILT_COL], 1)
    trendline = _make_linear_curve(
        "trendline",
        slope,
        intercept,
        domain=(float(peaks_df["_day"].min()), float(peaks_df["_day"].max())),
    )
    diagnostics["trendline_slope_per_day"] = float(slope)

    # ── current episode = tilt samples strictly after the most recent peak ──
    last_peak_date = peaks_df[DATE_COL].max()
    current = (
        tilt_df[tilt_df[DATE_COL] > last_peak_date]
        .sort_values(DATE_COL)
        .copy()
    )
    current["_day"] = to_days(current[DATE_COL])
    diagnostics["current_episode_n"] = int(len(current))

    if len(current) < 4:
        diagnostics["warning"] = "not enough current-episode points for exp fit"
        return Prediction(
            next_event_date=None,
            next_event_tilt=None,
            trendline=trendline,
            exp_curve=None,
            exp_params=None,
            exp_x0=None,
            exp_covariance=None,
            confidence_band=None,
            trendline_band=None,
            exp_band=None,
            n_peaks_in_fit=n_peaks_in_fit,
            fit_diagnostics=diagnostics,
        )

    # ── exponential saturation fit (v1.0:122-145) ────────────────────────────
    x_data = current["_day"].to_numpy()
    y_data = current[TILT_COL].to_numpy()
    x0_fit = float(x_data.min())

    A_guess = max(float(y_data.max() - y_data.min()), 1.0)  # v1.0:134
    k_guess = 0.01                                            # v1.0:135
    C_guess = float(y_data.min())                             # v1.0:136
    p0 = [A_guess, k_guess, C_guess]

    try:
        params, covariance = curve_fit(
            lambda x, A, k, C: exp_saturation(x, A, k, C, x0_fit),
            x_data,
            y_data,
            p0=p0,
            maxfev=_CURVE_FIT_MAXFEV,
            bounds=([0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
        )
    except (RuntimeError, ValueError) as e:
        diagnostics["exp_fit_error"] = str(e)
        return Prediction(
            next_event_date=None,
            next_event_tilt=None,
            trendline=trendline,
            exp_curve=None,
            exp_params=None,
            exp_x0=None,
            exp_covariance=None,
            confidence_band=None,
            trendline_band=None,
            exp_band=None,
            n_peaks_in_fit=n_peaks_in_fit,
            fit_diagnostics=diagnostics,
        )

    A_exp, k_exp, C_exp = float(params[0]), float(params[1]), float(params[2])
    diagnostics.update({"exp_A": A_exp, "exp_k": k_exp, "exp_C": C_exp})

    def _exp_eval(x, _A=A_exp, _k=k_exp, _C=C_exp, _x0=x0_fit):
        # Vectorized — accepts scalar or array, returns same shape.
        return exp_saturation(np.asarray(x, dtype=float), _A, _k, _C, _x0)

    exp_curve = Curve(
        name="current_episode_exp",
        f=_exp_eval,
        domain=(x0_fit, float(x_data.max()) + _PROJECTION_WINDOW_DAYS),
    )

    last_current_day = float(x_data.max())

    # ── intersection of trendline × exponential ──────────────────────────────
    next_event_date, next_event_tilt = _find_intersection(
        f_exp=_exp_eval,
        f_lin=trendline.f,
        last_current_day=last_current_day,
        last_peak_day=float(peaks_df["_day"].max()),
    )

    # ── Joint Monte Carlo: date band + trendline ribbon + exp ribbon ────────
    # Each draw samples a fresh (A, k, C) from the exp covariance AND a fresh
    # (slope, intercept) from a bootstrap resample of the peak set. We compute
    # three things from those draws:
    #   1. confidence_band: 10/90 percentile of the predicted intersection DATE
    #   2. trendline_band:  10/90 percentile RIBBON around the linear trendline
    #   3. exp_band:        10/90 percentile RIBBON around the exp saturation
    confidence_band, trendline_band, exp_band = _monte_carlo_bands(
        next_event_date=next_event_date,
        exp_params=params,
        exp_covariance=covariance,
        x0_fit=x0_fit,
        last_current_day=last_current_day,
        last_peak_day=float(peaks_df["_day"].max()),
        peaks_day=peaks_df["_day"].to_numpy(),
        peaks_tilt=peaks_df[TILT_COL].to_numpy(),
        trendline_domain=trendline.domain,
        exp_domain=exp_curve.domain,
    )

    return Prediction(
        next_event_date=next_event_date,
        next_event_tilt=next_event_tilt,
        trendline=trendline,
        exp_curve=exp_curve,
        exp_params=(A_exp, k_exp, C_exp),
        exp_x0=x0_fit,
        exp_covariance=covariance,
        confidence_band=confidence_band,
        trendline_band=trendline_band,
        exp_band=exp_band,
        n_peaks_in_fit=n_peaks_in_fit,
        fit_diagnostics=diagnostics,
    )


def _monte_carlo_bands(
    next_event_date: Optional[pd.Timestamp],
    exp_params: np.ndarray,
    exp_covariance: np.ndarray,
    x0_fit: float,
    last_current_day: float,
    last_peak_day: float,
    peaks_day: np.ndarray,
    peaks_tilt: np.ndarray,
    trendline_domain: tuple[float, float],
    exp_domain: tuple[float, float],
    n_samples: int = 200,
    quantiles: tuple[float, float] = (0.10, 0.90),
    seed: int = 0,
    n_grid: int = 200,
) -> tuple[
    Optional[tuple[pd.Timestamp, pd.Timestamp]],
    Optional[CurveBand],
    Optional[CurveBand],
]:
    """Joint Monte Carlo: returns (date_band, trendline_band, exp_band).

    See `predict()` for the high-level explanation. This function does the
    actual sampling loop and computes three percentile-based confidence
    products from the same draws.
    """
    if exp_covariance is None or not np.all(np.isfinite(exp_covariance)):
        return None, None, None
    n_peaks = len(peaks_day)
    if n_peaks < 2:
        return None, None, None

    rng = np.random.default_rng(seed)
    try:
        exp_draws = rng.multivariate_normal(
            mean=exp_params, cov=exp_covariance, size=n_samples
        )
    except (ValueError, np.linalg.LinAlgError):
        return None, None, None

    # x-grids for the curve ribbons. We extend the trendline grid forward to
    # the same end as the exp curve's domain so the trendline ribbon visually
    # reaches the predicted intersection.
    trend_x = np.linspace(trendline_domain[0], exp_domain[1], n_grid)
    exp_x = np.linspace(exp_domain[0], exp_domain[1], n_grid)

    trend_samples: list[np.ndarray] = []  # each entry shape (n_grid,)
    exp_samples: list[np.ndarray] = []
    valid_dates: list[pd.Timestamp] = []

    for A, k, C in exp_draws:
        # A and k must stay positive for the saturation curve to make sense.
        if A <= 0 or k <= 0:
            continue

        # Bootstrap resample of the peaks → fresh trendline parameters.
        idx = rng.integers(0, n_peaks, size=n_peaks)
        sampled_x = peaks_day[idx]
        sampled_y = peaks_tilt[idx]
        if len(np.unique(sampled_x)) < 2:
            continue
        try:
            slope_b, intercept_b = np.polyfit(sampled_x, sampled_y, 1)
        except (np.linalg.LinAlgError, ValueError):
            continue

        # Evaluate both curves on their grids for the ribbon bands
        trend_samples.append(slope_b * trend_x + intercept_b)
        exp_samples.append(exp_saturation(exp_x, A, k, C, x0_fit))

        # Solve the intersection for the date band
        def f_lin_b(x, _s=slope_b, _i=intercept_b):
            return _s * np.asarray(x, dtype=float) + _i

        def f_exp_b(x, _A=A, _k=k, _C=C, _x0=x0_fit):
            return exp_saturation(np.asarray(x, dtype=float), _A, _k, _C, _x0)

        if next_event_date is not None:
            date, _ = _find_intersection(
                f_exp=f_exp_b,
                f_lin=f_lin_b,
                last_current_day=last_current_day,
                last_peak_day=last_peak_day,
            )
            if date is not None:
                valid_dates.append(date)

    if not trend_samples or not exp_samples:
        return None, None, None

    # ── Date confidence band ─────────────────────────────────────────────────
    if len(valid_dates) < 10 or next_event_date is None:
        date_band = None
    else:
        sorted_dates = sorted(valid_dates)
        lo_idx = int(quantiles[0] * len(sorted_dates))
        hi_idx = min(int(quantiles[1] * len(sorted_dates)), len(sorted_dates) - 1)
        date_band = (sorted_dates[lo_idx], sorted_dates[hi_idx])

    # ── Curve ribbons ────────────────────────────────────────────────────────
    trend_arr = np.asarray(trend_samples)  # (n_valid, n_grid)
    exp_arr = np.asarray(exp_samples)      # (n_valid, n_grid)
    trend_lo = np.quantile(trend_arr, quantiles[0], axis=0)
    trend_hi = np.quantile(trend_arr, quantiles[1], axis=0)
    exp_lo = np.quantile(exp_arr, quantiles[0], axis=0)
    exp_hi = np.quantile(exp_arr, quantiles[1], axis=0)

    return (
        date_band,
        CurveBand(days=trend_x, lo=trend_lo, hi=trend_hi),
        CurveBand(days=exp_x, lo=exp_lo, hi=exp_hi),
    )


def _confidence_band(
    next_event_date: Optional[pd.Timestamp],
    params: np.ndarray,
    covariance: np.ndarray,
    x0_fit: float,
    last_current_day: float,
    last_peak_day: float,
    peaks_day: np.ndarray,
    peaks_tilt: np.ndarray,
    n_samples: int = 200,
    quantiles: tuple[float, float] = (0.10, 0.90),
    seed: int = 0,
) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
    """10th-90th percentile dates from a joint Monte Carlo over BOTH fits.

    Each draw independently:
      1. Samples (A, k, C) from the multivariate normal defined by curve_fit's
         3×3 covariance matrix on the exponential parameters.
      2. Bootstrap-resamples the peak set (with replacement) and refits a
         fresh linear trendline through the resampled peaks. Captures
         trendline uncertainty without assuming Gaussian residuals.
      3. Solves the intersection of the resampled exp × resampled trendline.

    Reports the 10th/90th percentile of the resulting predicted dates as an
    80% confidence interval. The combined band correctly widens when EITHER
    the exponential or the trendline is shaky — the previous version only
    sampled the exp fit, which understated the uncertainty when the
    trendline was based on few peaks.

    Returns `None` if the point estimate is None, the exp covariance is
    degenerate, fewer than 2 peaks are available to bootstrap from, or
    fewer than 10 valid Monte Carlo intersections were found.
    """
    if next_event_date is None:
        return None
    if covariance is None or not np.all(np.isfinite(covariance)):
        return None
    n_peaks = len(peaks_day)
    if n_peaks < 2:
        return None

    rng = np.random.default_rng(seed)
    try:
        exp_draws = rng.multivariate_normal(
            mean=params, cov=covariance, size=n_samples
        )
    except (ValueError, np.linalg.LinAlgError):
        return None

    valid_dates: list[pd.Timestamp] = []
    for A, k, C in exp_draws:
        # A and k must stay positive for the saturation curve to make sense.
        if A <= 0 or k <= 0:
            continue

        # Bootstrap resample of the peaks (with replacement) → fresh trendline.
        idx = rng.integers(0, n_peaks, size=n_peaks)
        sampled_x = peaks_day[idx]
        sampled_y = peaks_tilt[idx]
        # Reject degenerate samples (all peaks ended up at the same date) —
        # polyfit can't fit a line through a single x value.
        if len(np.unique(sampled_x)) < 2:
            continue
        try:
            slope_b, intercept_b = np.polyfit(sampled_x, sampled_y, 1)
        except (np.linalg.LinAlgError, ValueError):
            continue

        def f_lin_b(x, _s=slope_b, _i=intercept_b):
            return _s * np.asarray(x, dtype=float) + _i

        def f_exp_b(x, _A=A, _k=k, _C=C, _x0=x0_fit):
            return exp_saturation(np.asarray(x, dtype=float), _A, _k, _C, _x0)

        date, _ = _find_intersection(
            f_exp=f_exp_b,
            f_lin=f_lin_b,
            last_current_day=last_current_day,
            last_peak_day=last_peak_day,
        )
        if date is not None:
            valid_dates.append(date)

    if len(valid_dates) < 10:
        return None

    sorted_dates = sorted(valid_dates)
    lo_idx = int(quantiles[0] * len(sorted_dates))
    hi_idx = int(quantiles[1] * len(sorted_dates))
    hi_idx = min(hi_idx, len(sorted_dates) - 1)
    return sorted_dates[lo_idx], sorted_dates[hi_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────


def _make_linear_curve(
    name: str,
    slope: float,
    intercept: float,
    domain: tuple[float, float],
) -> Curve:
    poly = np.poly1d([float(slope), float(intercept)])

    def _eval(x, _p=poly):
        # Vectorized via np.poly1d — accepts scalar or array.
        return _p(np.asarray(x, dtype=float))

    return Curve(name=name, f=_eval, domain=domain)


def _find_intersection(
    f_exp: Callable[[float], float],
    f_lin: Callable[[float], float],
    last_current_day: float,
    last_peak_day: float,
) -> tuple[Optional[pd.Timestamp], Optional[float]]:
    """Solve `f_exp(x) == f_lin(x)` in the future projection window.

    Strategy:
      1. Scan the diff function over the window in 1-day steps.
      2. Find the first interval where it changes sign.
      3. Use `brentq` (bracketed bisection) to pinpoint the root in that
         interval. Brentq is rock-solid when given a valid bracket — it can
         never wander, unlike fsolve.
    Replaces the v1.0:158-183 fsolve-with-magic-initial-guess approach, which
    was prone to converging to spurious far-future roots.
    """
    projection_start = max(last_peak_day, last_current_day)
    projection_end = projection_start + _PROJECTION_WINDOW_DAYS

    def diff(x: float) -> float:
        return float(f_exp(x) - f_lin(x))

    # 1-day scan resolution. Window is 90 days → 91 samples. Cheap.
    scan = np.linspace(projection_start, projection_end, 91)
    try:
        diffs = np.array([diff(x) for x in scan])
    except Exception:
        return None, None

    # Find the first sign change.
    sign_change_idx = None
    for i in range(len(diffs) - 1):
        if np.isnan(diffs[i]) or np.isnan(diffs[i + 1]):
            continue
        if diffs[i] == 0:
            sign_change_idx = i
            break
        if diffs[i] * diffs[i + 1] < 0:
            sign_change_idx = i
            break

    if sign_change_idx is None:
        return None, None  # no intersection in the projection window

    # Bracket the root and refine with brentq.
    a, b = scan[sign_change_idx], scan[sign_change_idx + 1]
    if diffs[sign_change_idx] == 0:
        root = a
    else:
        try:
            root = brentq(diff, a, b, xtol=1e-4, maxiter=100)
        except Exception:
            return None, None

    # v1.0:177 — sanity bounds on the predicted tilt
    tilt_at_root = float(f_lin(root))
    if not (_INTERSECTION_TILT_MIN < tilt_at_root < _INTERSECTION_TILT_MAX):
        return None, None

    return from_days(root), tilt_at_root


def _empty(diagnostics: dict, n_peaks_in_fit: int = 0) -> Prediction:
    return Prediction(
        next_event_date=None,
        next_event_tilt=None,
        trendline=None,
        exp_curve=None,
        exp_params=None,
        exp_x0=None,
        exp_covariance=None,
        confidence_band=None,
        trendline_band=None,
        exp_band=None,
        n_peaks_in_fit=n_peaks_in_fit,
        fit_diagnostics=diagnostics,
    )
