"""Huber-robust fit wrappers around the existing scipy stack.

Every model in this package fits curves through possibly-noisy tilt data
(or possibly-anomalous historical peak heights). A single bad sample on
either side can drag an OLS fit far enough to break the predicted-pulse
intersection — which is exactly what we want a robust loss to prevent.

Choosing scipy over scikit-learn was deliberate: scipy is already a
runtime dep, sklearn would add ~20 MB of wheel weight to the Streamlit
Cloud cold-start, and the linear-Huber wrapper is a 15-line scipy
expression. See `.claude/plans/look-at-the-shape-spicy-island.md` for the
full deps tradeoff.

The two public entry points share the same f_scale convention: scale
auto-derived from the MAD of an initial OLS residual so the Huber
kick-in threshold tracks the data's natural scatter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit, least_squares

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# 1.345 is the standard Huber tuning constant — at this z value the
# influence function transitions from quadratic to linear, giving the
# estimator ~95% efficiency on Gaussian errors while bounding the
# influence of outliers. Multiplied by an MAD-based σ estimate to
# convert from "data scatter" to "Huber inflection distance".
_HUBER_TUNING_CONSTANT = 1.345

# Floor on the auto-derived f_scale. When residuals are ~zero (e.g. a
# perfect fit on synthetic data) the MAD is zero too, which would make
# every point an outlier. Floor keeps the fit well-defined.
_MIN_F_SCALE = 1e-6

# MAD-to-sigma conversion factor (assuming Gaussian residuals). Standard.
_MAD_TO_SIGMA = 1.0 / 0.6745


@dataclass(frozen=True)
class HuberFitQuality:
    """Diagnostics emitted alongside every Huber fit.

    Surfaced into ``ModelOutput.diagnostics`` so per-run reports can show
    drift in the noise-scale estimate or the down-weighted-fraction.
    """

    rmse_microrad: float
    f_scale: float
    n_samples: int
    min_weight: float
    mean_weight: float
    fraction_downweighted: float


def _mad(residuals: np.ndarray) -> float:
    return float(np.median(np.abs(residuals - np.median(residuals))))


def _auto_f_scale(residuals: np.ndarray) -> float:
    """Pick f_scale = 1.345 * σ_MAD where σ_MAD = MAD / 0.6745."""
    sigma = _mad(residuals) * _MAD_TO_SIGMA
    return max(_HUBER_TUNING_CONSTANT * sigma, _MIN_F_SCALE)


def _huber_weights(residuals: np.ndarray, f_scale: float) -> np.ndarray:
    """Per-sample weights from the Huber influence function.

    For |z| ≤ 1: weight is 1 (treated as inlier). For |z| > 1: weight is
    1/|z| (linearly down-weighted). z = residual / f_scale.
    """
    z = np.abs(residuals) / max(f_scale, _MIN_F_SCALE)
    return np.where(z <= 1.0, 1.0, 1.0 / np.maximum(z, _MIN_F_SCALE))


def fit_huber_linear(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, HuberFitQuality]:
    """Robust linear regression ``y ≈ m·x + b`` via scipy.least_squares.

    Two-stage: an initial OLS fit estimates the residual scale, then the
    robust fit runs with ``f_scale`` derived from that scale's MAD. The
    standard k=1.345 tuning constant gives ~95% efficiency on Gaussian
    errors while clamping the influence of large residuals.

    Returns the fitted slope, intercept, and a quality struct.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = len(x_arr)
    if n < 2:
        raise ValueError(f"fit_huber_linear: need ≥ 2 samples, got {n}")

    # Stage 1 — OLS fit for scale estimation. np.polyfit is well-tested
    # and sidesteps any iterative-solver convergence concern at this step.
    m0, b0 = np.polyfit(x_arr, y_arr, 1)
    initial_residuals = y_arr - (m0 * x_arr + b0)
    f_scale = _auto_f_scale(initial_residuals)

    # Stage 2 — robust refinement.
    def _residuals(params: np.ndarray) -> np.ndarray:
        m, b = params
        return y_arr - (m * x_arr + b)

    result = least_squares(
        _residuals,
        x0=np.array([m0, b0], dtype=float),
        loss="huber",
        f_scale=f_scale,
    )
    m, b = float(result.x[0]), float(result.x[1])

    final_residuals = y_arr - (m * x_arr + b)
    weights = _huber_weights(final_residuals, f_scale)
    quality = HuberFitQuality(
        rmse_microrad=float(np.sqrt(np.mean(final_residuals**2))),
        f_scale=float(f_scale),
        n_samples=int(n),
        min_weight=float(weights.min()),
        mean_weight=float(weights.mean()),
        fraction_downweighted=float(np.mean(weights < 1.0)),
    )
    return m, b, quality


def fit_huber_curve(
    f: Callable[..., np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    p0: Sequence[float],
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    maxfev: int = 5000,
) -> tuple[np.ndarray, np.ndarray, HuberFitQuality]:
    """Robust nonlinear curve fit via ``scipy.optimize.curve_fit(loss='huber')``.

    Same two-stage f_scale derivation as :func:`fit_huber_linear`. Returns
    ``(params, covariance, quality)``. Raises ``RuntimeError`` /
    ``ValueError`` (curve_fit's native errors) when the fit can't
    converge — callers handle the exception narrowly.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = len(x_arr)
    if n < len(p0):
        raise ValueError(
            f"fit_huber_curve: need ≥ {len(p0)} samples, got {n}"
        )

    # Stage 1 — OLS curve_fit (no robust loss) for scale estimation.
    # If the initial fit fails we still try the robust pass with
    # f_scale=1 — better than dying on the diagnostic step.
    try:
        if bounds is None:
            p_initial, _ = curve_fit(f, x_arr, y_arr, p0=list(p0), maxfev=maxfev)
        else:
            p_initial, _ = curve_fit(
                f, x_arr, y_arr, p0=list(p0), bounds=bounds, maxfev=maxfev
            )
        initial_residuals = y_arr - f(x_arr, *p_initial)
        f_scale = _auto_f_scale(initial_residuals)
    except (RuntimeError, ValueError):
        f_scale = _MIN_F_SCALE

    # Stage 2 — robust refinement. curve_fit's TRF method supports loss.
    if bounds is None:
        params, covariance = curve_fit(
            f,
            x_arr,
            y_arr,
            p0=list(p0),
            method="trf",
            loss="huber",
            f_scale=f_scale,
            maxfev=maxfev,
        )
    else:
        params, covariance = curve_fit(
            f,
            x_arr,
            y_arr,
            p0=list(p0),
            bounds=bounds,
            method="trf",
            loss="huber",
            f_scale=f_scale,
            maxfev=maxfev,
        )

    final_residuals = y_arr - f(x_arr, *params)
    weights = _huber_weights(final_residuals, f_scale)
    quality = HuberFitQuality(
        rmse_microrad=float(np.sqrt(np.mean(final_residuals**2))),
        f_scale=float(f_scale),
        n_samples=int(n),
        min_weight=float(weights.min()),
        mean_weight=float(weights.mean()),
        fraction_downweighted=float(np.mean(weights < 1.0)),
    )
    return params, covariance, quality
