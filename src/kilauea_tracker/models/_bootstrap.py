"""Joint Monte Carlo confidence bands.

Generalization of the trendline×exp ``_monte_carlo_bands`` routine so
linear, power-law, and historical-projection models can produce the
same shape of (date_band, trendline_ribbon, curve_ribbon) output. Each
draw samples a fresh trendline AND a fresh curve from their joint
parameter uncertainty, recomputes the intersection, and aggregates
quantiles across draws.

Pure: deterministic given a seed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..model import CurveBand

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

# 200 MC draws. Enough for a stable 80% quantile (10th/90th percentile)
# without bloating the per-render time. Matches the old default in
# trendline_exp.
DEFAULT_N_SAMPLES = 200

# Default 80% interval (10th–90th percentile). Matches the existing
# ribbon convention.
DEFAULT_QUANTILES = (0.10, 0.90)

# Minimum valid intersection dates required to emit a date band — below
# this the band would be a noisy IQR over a handful of samples.
_MIN_VALID_DATES_FOR_BAND = 10


def joint_mc_bands(
    *,
    draw_curve: Callable[[np.random.Generator], Callable[[np.ndarray], np.ndarray] | None],
    draw_trend: Callable[[np.random.Generator], tuple[float, float] | None],
    intersect: Callable[
        [Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]],
        pd.Timestamp | None,
    ],
    curve_grid: np.ndarray,
    trend_grid: np.ndarray,
    n_samples: int = DEFAULT_N_SAMPLES,
    quantiles: tuple[float, float] = DEFAULT_QUANTILES,
    seed: int = 0,
) -> tuple[
    tuple[pd.Timestamp, pd.Timestamp] | None,
    CurveBand | None,
    CurveBand | None,
]:
    """Run joint MC over a (trendline, curve) pair and return three bands.

    ``draw_curve`` and ``draw_trend`` each take an RNG and return a fresh
    sample — a curve evaluator (``f(x) -> y``) and a ``(slope, intercept)``
    pair respectively. They return ``None`` to skip a sample (e.g.,
    parameter draw was out-of-bounds).

    ``intersect(curve_eval, trend_eval) -> pd.Timestamp | None`` runs the
    domain-specific intersection logic on a single sampled pair.

    Returns ``(date_band, trendline_band, curve_band)`` — any can be
    ``None`` if the underlying samples were unusable.
    """
    rng = np.random.default_rng(seed)

    trend_samples: list[np.ndarray] = []
    curve_samples: list[np.ndarray] = []
    valid_dates: list[pd.Timestamp] = []

    for _ in range(n_samples):
        curve_eval = draw_curve(rng)
        if curve_eval is None:
            continue
        trend_params = draw_trend(rng)
        if trend_params is None:
            continue
        slope_b, intercept_b = trend_params

        try:
            curve_sample = np.asarray(curve_eval(curve_grid), dtype=float)
        except Exception:
            continue

        trend_sample = slope_b * trend_grid + intercept_b
        trend_samples.append(trend_sample)
        curve_samples.append(curve_sample)

        def trend_eval(
            x: np.ndarray,
            _s: float = slope_b,
            _i: float = intercept_b,
        ) -> np.ndarray:
            return _s * np.asarray(x, dtype=float) + _i

        date = intersect(curve_eval, trend_eval)
        if date is not None:
            valid_dates.append(date)

    if not trend_samples or not curve_samples:
        return None, None, None

    if len(valid_dates) < _MIN_VALID_DATES_FOR_BAND:
        date_band = None
    else:
        sorted_dates = sorted(valid_dates)
        lo_idx = int(quantiles[0] * len(sorted_dates))
        hi_idx = min(
            int(quantiles[1] * len(sorted_dates)), len(sorted_dates) - 1
        )
        date_band = (sorted_dates[lo_idx], sorted_dates[hi_idx])

    trend_arr = np.asarray(trend_samples)
    curve_arr = np.asarray(curve_samples)
    trend_lo = np.quantile(trend_arr, quantiles[0], axis=0)
    trend_hi = np.quantile(trend_arr, quantiles[1], axis=0)
    curve_lo = np.quantile(curve_arr, quantiles[0], axis=0)
    curve_hi = np.quantile(curve_arr, quantiles[1], axis=0)

    return (
        date_band,
        CurveBand(days=trend_grid, lo=trend_lo, hi=trend_hi),
        CurveBand(days=curve_grid, lo=curve_lo, hi=curve_hi),
    )


def bootstrap_peak_trend_sampler(
    peaks_day: np.ndarray, peaks_tilt: np.ndarray
) -> Callable[[np.random.Generator], tuple[float, float] | None]:
    """Build a ``draw_trend`` callable that bootstrap-resamples peaks.

    Each call resamples the peaks with replacement, refits OLS, returns
    ``(slope, intercept)``. Returns ``None`` when the resample
    happens to land on collinear x's (e.g., all-same-day).
    """
    n = len(peaks_day)

    def _draw(rng: np.random.Generator) -> tuple[float, float] | None:
        if n < 2:
            return None
        idx = rng.integers(0, n, size=n)
        sx = peaks_day[idx]
        sy = peaks_tilt[idx]
        if len(np.unique(sx)) < 2:
            return None
        try:
            slope, intercept = np.polyfit(sx, sy, 1)
        except (np.linalg.LinAlgError, ValueError):
            return None
        return float(slope), float(intercept)

    return _draw
