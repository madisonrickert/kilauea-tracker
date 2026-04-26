"""Material-Failure Forecast Method (Voight 1988) — experimental.

Voight 1988 (*Nature* 332:125) introduced the failure-forecast method:
the rate of an observable Ω accelerates as a power of itself,
``d²Ω/dt² = A · (dΩ/dt)^α``. For α=2 the inverse-rate ``1/(dΩ/dt)``
plot is linear in t, with x-intercept at the predicted failure time.

**Important caveat for this dataset.** Voight's method targets
*accelerating*-rate-to-failure (Mt St Helens 1985–86, Redoubt 1989–90).
Kīlauea's inflation rate *decelerates* — see the slope-decay table in
``.claude/plans/look-at-the-shape-spicy-island.md``: 6 of 7 recent
episodes show first-quartile slopes 1.4–2.6× the last-quartile slopes.
And the deflation/eruption is threshold-triggered, not failure-
triggered — there's no diverging precursor.

So this model will frequently fail to find a positive-x-intercept fit
and emit ``next_event_date=None`` with a "no convergence" diagnostic.
That's the *expected* behaviour for this regime, surfaced honestly.
The model is included for academic completeness so the user can
compare against the canonical published method.

Pure: no I/O, no clock reads, no module-level mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..model import DATE_COL, TILT_COL, from_days, to_days
from ._episodes import find_current_episode_trough
from ._huber import fit_huber_linear
from .output import ModelOutput, NamedCurve

_TRENDLINE_GRID_POINTS = 200
_TRENDLINE_PROJECTION_DAYS = 90.0


def _trendline_curve(peaks_df: pd.DataFrame) -> NamedCurve | None:
    """Compute a NamedCurve for the peak trendline so the chart isn't bare
    when FFM no-ops. Returns None if the peak set is too small to fit."""
    if len(peaks_df) < 2:
        return None
    sorted_peaks = peaks_df.sort_values(DATE_COL)
    days = to_days(sorted_peaks[DATE_COL]).astype(float)
    tilt = sorted_peaks[TILT_COL].to_numpy(dtype=float)
    slope, intercept, _ = fit_huber_linear(days, tilt)
    grid = np.linspace(
        float(days.min()),
        float(days.max()) + _TRENDLINE_PROJECTION_DAYS,
        _TRENDLINE_GRID_POINTS,
    )
    return NamedCurve(
        label=f"Trendline (last {len(sorted_peaks)} peaks)",
        days=grid,
        values=slope * grid + intercept,
        style="dashed",
        color_role="primary",
    )

# Centered-difference smoothing window. 6 samples × 15 min ≈ 1.5 h —
# enough to suppress sample noise without losing the slow rate trend.
_RATE_SMOOTH_WINDOW = 6

# Below this many post-trough samples the rate fit is too noisy to
# extract a meaningful inverse-rate trend.
_MIN_SAMPLES = 24

_CURVE_GRID_POINTS = 200
_PROJECTION_WINDOW_DAYS = 90.0

# Minimum positive rate (µrad/h) to include in the inverse-rate fit. A
# sample with near-zero or negative rate either is sample noise or
# represents inflation having stalled — in either case the inverse
# rate's reciprocal is nonsense at that point.
_MIN_POSITIVE_RATE_MICRORAD_PER_HOUR = 0.005


@dataclass(frozen=True)
class FFMVoightModel:
    """Voight 1988 inverse-rate failure-forecast (experimental).

    Caveat: targets accelerating-rate-to-failure regimes; Kīlauea
    inflation decelerates. This model will frequently no-op with a
    "no convergence" diagnostic, which is the correct behaviour for
    the regime. Included for academic completeness.
    """

    id: str = "ffm_voight"
    label: str = "FFM / Voight inverse-rate (experimental)"
    description: str = (
        "Voight 1988 inverse-rate method — fits 1/(dy/dt) vs t and "
        "reads off the x-intercept as failure time. **Experimental** "
        "for Kīlauea: the method assumes accelerating-rate-to-failure "
        "(Mt St Helens 1985, Redoubt 1989), but Kīlauea's inflation "
        "decelerates and eruptions are threshold-triggered, not "
        "failure-triggered. Frequently no-ops with a diagnostic — that "
        "no-op is itself the regime signal."
    )

    def predict(
        self, tilt_df: pd.DataFrame, peaks_df: pd.DataFrame
    ) -> ModelOutput:
        return _predict_ffm(tilt_df, peaks_df)


def _predict_ffm(tilt_df: pd.DataFrame, peaks_df: pd.DataFrame) -> ModelOutput:
    diagnostics: dict = {"method": "voight_1988_alpha=2"}
    # Always-present trendline so the chart shows context even when the
    # FFM fit no-ops (which is the regime-mismatch expectation).
    base_curves: list[NamedCurve] = []
    tl = _trendline_curve(peaks_df)
    if tl is not None:
        base_curves.append(tl)

    trough = find_current_episode_trough(tilt_df, peaks_df)
    if trough is None:
        diagnostics["warning"] = "no current-episode trough found"
        return _empty_output(diagnostics, base_curves=base_curves)
    trough_date, _trough_tilt = trough

    post_trough = tilt_df[tilt_df[DATE_COL] >= trough_date].sort_values(DATE_COL)
    if len(post_trough) < _MIN_SAMPLES:
        diagnostics["warning"] = (
            f"only {len(post_trough)} post-trough samples; need ≥ {_MIN_SAMPLES}"
        )
        return _empty_output(diagnostics, base_curves=base_curves)

    post_day = to_days(post_trough[DATE_COL]).astype(float)
    post_tilt = post_trough[TILT_COL].to_numpy(dtype=float)
    trough_day_val = float(post_day[0])
    t_hours = (post_day - trough_day_val) * 24.0

    # Centered-difference rate, then rolling-median smooth.
    rate_microrad_per_hour = _smoothed_rate(t_hours, post_tilt)

    # Keep only points where rate is positively bounded — we need to
    # invert and fit a line to 1/rate.
    mask = rate_microrad_per_hour > _MIN_POSITIVE_RATE_MICRORAD_PER_HOUR
    if mask.sum() < _MIN_SAMPLES:
        diagnostics["warning"] = (
            f"only {int(mask.sum())} positive-rate samples after smoothing; "
            f"need ≥ {_MIN_SAMPLES}"
        )
        diagnostics["regime_signal"] = (
            "inflation rate not reliably positive — inconsistent with FFM regime"
        )
        return _empty_output(diagnostics)

    t_h_fit = t_hours[mask]
    rate_fit = rate_microrad_per_hour[mask]
    inverse_rate = 1.0 / rate_fit

    inv_slope_per_hour, inv_intercept, inv_quality = fit_huber_linear(
        t_h_fit, inverse_rate
    )
    diagnostics.update({
        "n_fit_samples": int(inv_quality.n_samples),
        "fit_rmse_inverse_rate": inv_quality.rmse_microrad,
        "inverse_rate_slope": inv_slope_per_hour,
        "inverse_rate_intercept": inv_intercept,
        "fit_window_start_utc": str(post_trough[DATE_COL].iloc[0]),
        "fit_window_end_utc": str(post_trough[DATE_COL].iloc[-1]),
    })

    # Voight α=2: failure time t_f satisfies 1/rate(t_f) = 0.
    # Decelerating rate ⇒ slope > 0 (inverse rate increasing) ⇒ no
    # positive-x intercept. That's the regime mismatch.
    if inv_slope_per_hour >= 0:
        diagnostics["warning"] = (
            f"inverse-rate slope is non-negative ({inv_slope_per_hour:.4g}); "
            f"inflation rate is decelerating, not accelerating. No FFM "
            f"failure-time intercept exists."
        )
        diagnostics["regime_signal"] = (
            "decelerating inflation — Voight α=2 not applicable; expected "
            "for Kīlauea's threshold-bounded regime"
        )
        return _empty_output(diagnostics, base_curves=base_curves)

    failure_t_hours = -inv_intercept / inv_slope_per_hour
    if failure_t_hours <= float(t_hours[-1]):
        diagnostics["warning"] = (
            "inferred failure time is in the past or already reached"
        )
        return _empty_output(diagnostics, base_curves=base_curves)

    failure_day = trough_day_val + failure_t_hours / 24.0
    if failure_day - trough_day_val > _PROJECTION_WINDOW_DAYS:
        diagnostics["warning"] = (
            f"inferred failure time is > {_PROJECTION_WINDOW_DAYS} days out"
        )
        return _empty_output(diagnostics, base_curves=base_curves)

    next_event_date = from_days(failure_day)
    diagnostics["failure_t_hours_since_trough"] = failure_t_hours

    # Surface the predicted-event marker plus the trendline so the chart
    # has visible context. The inverse-rate fit itself isn't drawn — it
    # lives in 1/rate units that don't share the µrad y-axis.
    return ModelOutput(
        next_event_date=next_event_date,
        confidence_band=None,
        headline_text=f"FFM α=2 failure t = {failure_t_hours / 24.0:.1f} d",
        curves=base_curves,
        diagnostics=diagnostics,
        next_event_tilt=None,
    )


def _smoothed_rate(t_hours: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Centered finite-difference rate dy/dt with rolling-median smooth.

    Uses ``np.gradient`` for the centered diff (handles uniform and
    non-uniform spacing) followed by a centered rolling median of width
    ``_RATE_SMOOTH_WINDOW`` to suppress sample-noise spikes. End samples
    fall through to single-sided differences via np.gradient's natural
    boundary handling.

    Defensively drops strictly-non-increasing duplicates in ``t_hours``
    so np.gradient never sees a zero ``dx`` (which would emit
    divide-by-zero RuntimeWarnings and produce NaN/inf in the result).
    Real merged history has no duplicate timestamps today; this
    survives any future ingest-anomaly that might introduce them.
    """
    if len(t_hours) > 1:
        keep = np.concatenate([[True], np.diff(t_hours) > 0])
        if not keep.all():
            t_hours = t_hours[keep]
            y = y[keep]
    if len(t_hours) < 2:
        return np.zeros_like(y)
    raw = np.gradient(y, t_hours)
    if len(raw) < _RATE_SMOOTH_WINDOW:
        return raw
    s = pd.Series(raw)
    smooth = s.rolling(
        window=_RATE_SMOOTH_WINDOW, center=True, min_periods=1
    ).median()
    return smooth.to_numpy(dtype=float)


def _empty_output(
    diagnostics: dict,
    base_curves: list[NamedCurve] | None = None,
) -> ModelOutput:
    return ModelOutput(
        next_event_date=None,
        confidence_band=None,
        headline_text=None,
        curves=list(base_curves) if base_curves else [],
        diagnostics=diagnostics,
    )
