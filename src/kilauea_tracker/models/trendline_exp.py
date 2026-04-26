"""Trendline × exponential intersection model.

Predicts the next pulse as the date where the linear trendline through
recent peaks meets the exponential-saturation fit through the current
post-trough recovery. Uncertainty comes from a joint Monte Carlo over
both fits (bootstrap-resampled peaks for the trendline; multivariate-
normal samples of the exp covariance for the saturation curve).

The math is the same the app has always used — historically inlined in
``model.predict()``. Lifting it here makes the trendline+exp pair just
one of N registered models. ``model.predict()`` is now a thin facade
that delegates here and reassembles the legacy ``Prediction`` shape for
back-compat with existing consumers.

Pure: no I/O, no clock reads, no module-level mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..model import (
    DATE_COL,
    TILT_COL,
    Curve,
    CurveBand,
    exp_saturation,
    to_days,
)
from ._intersection import (
    PROJECTION_WINDOW_DAYS as _PROJECTION_WINDOW_DAYS,
)
from ._intersection import (
    find_intersection as _find_intersection,
)
from .output import ModelOutput, NamedCurve

_CURVE_FIT_MAXFEV = 5000
# Cap exp-saturation amplitude so the fitter can't escape into degenerate
# (A→∞, k→0) solutions when the recovery hasn't developed enough curvature.
# Tilt cycles peak at 5-15 µrad and the asymptote is no more than ~50 µrad
# above the trough.
_MAX_EXP_AMPLITUDE_MICRORAD = 50.0
# Cap the trough search at 14 days post-peak. Deflations bottom within ~24h;
# 14d safely brackets the worst case without grabbing an old trough.
_TROUGH_SEARCH_WINDOW_DAYS = 14
# Minimum samples in the post-trough current episode for the exp fit to
# be stable. Below this we still return the trendline but skip the exp
# fit and the intersection.
_MIN_CURRENT_EPISODE_SAMPLES = 4
# x-grid resolution for surfaces (curves the chart will draw).
_CURVE_GRID_POINTS = 200


# ─────────────────────────────────────────────────────────────────────────────
# Internal raw-result struct (used both by this model and by the legacy
# ``model.predict()`` facade)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrendlineExpRaw:
    """The full set of computed primitives — what both ``ModelOutput``
    and the legacy ``Prediction`` are projected from."""

    next_event_date: pd.Timestamp | None
    next_event_tilt: float | None
    trendline: Curve | None
    exp_curve: Curve | None
    exp_params: tuple[float, float, float] | None
    exp_x0: float | None
    exp_covariance: np.ndarray | None
    confidence_band: tuple[pd.Timestamp, pd.Timestamp] | None
    trendline_band: CurveBand | None
    exp_band: CurveBand | None
    n_peaks_in_fit: int
    fit_diagnostics: dict = field(default_factory=dict)


def compute_trendline_exp(
    tilt_df: pd.DataFrame, peaks_df: pd.DataFrame
) -> TrendlineExpRaw:
    """Run the trendline+exp+intersection math. Returns a raw struct.

    Pure. Same inputs → same outputs. Never raises.
    """
    diagnostics: dict = {}
    n_peaks_in_fit = len(peaks_df)

    if n_peaks_in_fit < 2:
        diagnostics["error"] = "need at least 2 peaks for a linear trendline"
        return TrendlineExpRaw(
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

    # ── current episode = post-trough recovery ───────────────────────────────
    # The fountain event is the sharp deflation at the most recent peak.
    # The bottom of the deflation is where the saturating-rebuild begins.
    # Fitting the exp from the peak itself is wrong — the deflation drop
    # is not a saturation curve and ruins the fit.
    last_peak_date = peaks_df[DATE_COL].max()
    post_peak = (
        tilt_df[tilt_df[DATE_COL] > last_peak_date]
        .sort_values(DATE_COL)
        .copy()
    )
    if len(post_peak) > 0:
        search_end = last_peak_date + pd.Timedelta(days=_TROUGH_SEARCH_WINDOW_DAYS)
        trough_search = post_peak[post_peak[DATE_COL] <= search_end]
        if len(trough_search) > 0:
            trough_idx = trough_search[TILT_COL].idxmin()
            trough_date = trough_search.loc[trough_idx, DATE_COL]
            current = post_peak[post_peak[DATE_COL] >= trough_date].copy()
        else:
            current = post_peak
    else:
        current = post_peak

    current["_day"] = to_days(current[DATE_COL])
    diagnostics["current_episode_n"] = len(current)
    if len(current) > 0:
        diagnostics["current_episode_start"] = str(current[DATE_COL].iloc[0])

    if len(current) < _MIN_CURRENT_EPISODE_SAMPLES:
        diagnostics["warning"] = "not enough current-episode points for exp fit"
        return TrendlineExpRaw(
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

    # ── exponential saturation fit ───────────────────────────────────────────
    x_data = current["_day"].to_numpy()
    y_data = current[TILT_COL].to_numpy()
    x0_fit = float(x_data.min())

    A_guess = max(float(y_data.max() - y_data.min()), 1.0)
    k_guess = 0.01
    C_guess = float(y_data.min())
    p0 = [A_guess, k_guess, C_guess]

    try:
        params, covariance = curve_fit(
            lambda x, A, k, C: exp_saturation(x, A, k, C, x0_fit),
            x_data,
            y_data,
            p0=p0,
            maxfev=_CURVE_FIT_MAXFEV,
            bounds=(
                [0.0, 0.0, -np.inf],
                [_MAX_EXP_AMPLITUDE_MICRORAD, np.inf, np.inf],
            ),
        )
    except (RuntimeError, ValueError) as e:
        diagnostics["exp_fit_error"] = str(e)
        return TrendlineExpRaw(
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

    def _exp_eval(
        x: float,
        _A: float = A_exp,
        _k: float = k_exp,
        _C: float = C_exp,
        _x0: float = x0_fit,
    ) -> float:
        return exp_saturation(np.asarray(x, dtype=float), _A, _k, _C, _x0)

    exp_curve = Curve(
        name="current_episode_exp",
        f=_exp_eval,
        domain=(x0_fit, float(x_data.max()) + _PROJECTION_WINDOW_DAYS),
    )

    last_current_day = float(x_data.max())

    # ── intersection of trendline × exp ──────────────────────────────────────
    next_event_date, next_event_tilt = _find_intersection(
        f_curve=_exp_eval,
        f_lin=trendline.f,
        last_current_day=last_current_day,
        last_peak_day=float(peaks_df["_day"].max()),
    )

    # ── joint Monte Carlo: date band + trendline ribbon + exp ribbon ─────────
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

    return TrendlineExpRaw(
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


def _make_linear_curve(
    name: str,
    slope: float,
    intercept: float,
    domain: tuple[float, float],
) -> Curve:
    poly = np.poly1d([float(slope), float(intercept)])

    def _eval(x: float, _p: np.poly1d = poly) -> float:
        return _p(np.asarray(x, dtype=float))

    return Curve(name=name, f=_eval, domain=domain)


def _monte_carlo_bands(
    next_event_date: pd.Timestamp | None,
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
    n_grid: int = _CURVE_GRID_POINTS,
) -> tuple[
    tuple[pd.Timestamp, pd.Timestamp] | None,
    CurveBand | None,
    CurveBand | None,
]:
    """Joint Monte Carlo: returns (date_band, trendline_band, exp_band).

    Each draw samples (A, k, C) from the exp covariance AND a fresh
    (slope, intercept) from a bootstrap resample of the peak set.
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

    trend_x = np.linspace(trendline_domain[0], exp_domain[1], n_grid)
    exp_x = np.linspace(exp_domain[0], exp_domain[1], n_grid)

    trend_samples: list[np.ndarray] = []
    exp_samples: list[np.ndarray] = []
    valid_dates: list[pd.Timestamp] = []

    for A, k, C in exp_draws:
        if A <= 0 or k <= 0:
            continue

        idx = rng.integers(0, n_peaks, size=n_peaks)
        sampled_x = peaks_day[idx]
        sampled_y = peaks_tilt[idx]
        if len(np.unique(sampled_x)) < 2:
            continue
        try:
            slope_b, intercept_b = np.polyfit(sampled_x, sampled_y, 1)
        except (np.linalg.LinAlgError, ValueError):
            continue

        trend_samples.append(slope_b * trend_x + intercept_b)
        exp_samples.append(exp_saturation(exp_x, A, k, C, x0_fit))

        def f_lin_b(x: float, _s: float = slope_b, _i: float = intercept_b) -> float:
            return _s * np.asarray(x, dtype=float) + _i

        def f_exp_b(
            x: float,
            _A: float = A,
            _k: float = k,
            _C: float = C,
            _x0: float = x0_fit,
        ) -> float:
            return exp_saturation(np.asarray(x, dtype=float), _A, _k, _C, _x0)

        if next_event_date is not None:
            date, _ = _find_intersection(
                f_curve=f_exp_b,
                f_lin=f_lin_b,
                last_current_day=last_current_day,
                last_peak_day=last_peak_day,
            )
            if date is not None:
                valid_dates.append(date)

    if not trend_samples or not exp_samples:
        return None, None, None

    if len(valid_dates) < 10 or next_event_date is None:
        date_band = None
    else:
        sorted_dates = sorted(valid_dates)
        lo_idx = int(quantiles[0] * len(sorted_dates))
        hi_idx = min(int(quantiles[1] * len(sorted_dates)), len(sorted_dates) - 1)
        date_band = (sorted_dates[lo_idx], sorted_dates[hi_idx])

    trend_arr = np.asarray(trend_samples)
    exp_arr = np.asarray(exp_samples)
    trend_lo = np.quantile(trend_arr, quantiles[0], axis=0)
    trend_hi = np.quantile(trend_arr, quantiles[1], axis=0)
    exp_lo = np.quantile(exp_arr, quantiles[0], axis=0)
    exp_hi = np.quantile(exp_arr, quantiles[1], axis=0)

    return (
        date_band,
        CurveBand(days=trend_x, lo=trend_lo, hi=trend_hi),
        CurveBand(days=exp_x, lo=exp_lo, hi=exp_hi),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model — wraps the raw computation in the registry's protocol
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrendlineExpModel:
    """Linear trendline through recent peaks × exponential-saturation
    fit through the current post-trough recovery. Predicts the next
    pulse as the date where these curves intersect, with an 80% Monte
    Carlo confidence band over both fits' uncertainty.
    """

    id: str = "trendline_exp"
    label: str = "Trendline × exponential intersection"
    description: str = (
        "The default model. Linear trendline through the recent N peaks "
        "meets an exponential-saturation fit through the current post-"
        "trough recovery; the intersection date is the predicted pulse. "
        "80% confidence band from a joint bootstrap + multivariate-normal "
        "sample over both fits."
    )

    def predict(
        self, tilt_df: pd.DataFrame, peaks_df: pd.DataFrame
    ) -> ModelOutput:
        raw = compute_trendline_exp(tilt_df, peaks_df)

        # Ribbons emitted FIRST so the chart renderer draws them under
        # the line curves — Plotly's trace order is its z-order, and a
        # ribbon on top would obscure the line through it.
        # Trace labels match the legacy ``build_figure`` names so existing
        # palette/structure assertions in test_plotting.py keep working.
        curves: list[NamedCurve] = []
        if raw.trendline_band is not None:
            curves.append(
                NamedCurve(
                    label="Trendline 80% CI",
                    days=raw.trendline_band.days,
                    values=(raw.trendline_band.lo + raw.trendline_band.hi) / 2.0,
                    color_role="ribbon",
                    band_lo=raw.trendline_band.lo,
                    band_hi=raw.trendline_band.hi,
                )
            )
        if raw.exp_band is not None:
            curves.append(
                NamedCurve(
                    label="Exp fit 80% CI",
                    days=raw.exp_band.days,
                    values=(raw.exp_band.lo + raw.exp_band.hi) / 2.0,
                    color_role="ribbon",
                    band_lo=raw.exp_band.lo,
                    band_hi=raw.exp_band.hi,
                )
            )
        if raw.trendline is not None:
            # Sample on a grid spanning the trendline's domain extended
            # forward to the exp curve's projection end (so the chart can
            # see where the trendline is going).
            t_start = raw.trendline.domain[0]
            t_end = (
                raw.exp_curve.domain[1]
                if raw.exp_curve is not None
                else raw.trendline.domain[1]
            )
            xs = np.linspace(t_start, t_end, _CURVE_GRID_POINTS)
            curves.append(
                NamedCurve(
                    label=f"Trendline (last {raw.n_peaks_in_fit} peaks)",
                    days=xs,
                    values=np.asarray(raw.trendline.f(xs), dtype=float),
                    style="dashed",
                    color_role="primary",
                )
            )
        if raw.exp_curve is not None:
            xs = np.linspace(
                raw.exp_curve.domain[0],
                raw.exp_curve.domain[1],
                _CURVE_GRID_POINTS,
            )
            curves.append(
                NamedCurve(
                    label="Current episode (exp fit)",
                    days=xs,
                    values=np.asarray(raw.exp_curve.f(xs), dtype=float),
                    style="solid",
                    color_role="primary",
                )
            )

        return ModelOutput(
            next_event_date=raw.next_event_date,
            confidence_band=raw.confidence_band,
            headline_text="trendline × exp intersection"
            if raw.next_event_date is not None
            else None,
            curves=curves,
            diagnostics=raw.fit_diagnostics,
            next_event_tilt=raw.next_event_tilt,
        )
