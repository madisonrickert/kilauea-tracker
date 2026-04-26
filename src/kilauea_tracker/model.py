"""Legacy prediction-model facade + shared primitives.

Historically this module owned all the curve-fit + intersection math.
That logic has moved into the ``models/`` package — see
``models/trendline_exp.py`` (the trendline × exp intersection math) and
``models/interval_median.py`` (the interval-baseline forecast).

What stays here:

- The shared primitives (``DATE_COL``, ``TILT_COL``, ``Curve``,
  ``CurveBand``, ``Prediction``, ``to_days``, ``from_days``,
  ``exp_saturation``) — used by the new model modules and by every
  consumer that already imports from ``kilauea_tracker.model``.
- A back-compat ``predict()`` facade that delegates to both new models
  and reassembles the legacy ``Prediction`` dataclass so existing
  callers (``hero.py``, ``plotting.build_figure``, the test suite)
  keep working unchanged.

New code should use ``app_state.get_prediction(model_id=...)``, which
returns a ``ModelOutput`` directly and allows picking any registered
model. The ``Prediction`` shape and this facade are deferred-deletion
boundary code — once consumers migrate, both can go.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

DATE_COL = "Date"
TILT_COL = "Tilt (microradians)"


# ─────────────────────────────────────────────────────────────────────────────
# Public dataclasses (used by both new model modules and back-compat callers)
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

    Each entry in ``lo`` and ``hi`` is the 10th / 90th percentile of the
    curve's value at the corresponding ``days`` x-coordinate, computed
    from a Monte Carlo sample of the curve's fit parameters. Renders as
    a filled region between the two y-arrays.
    """

    days: np.ndarray   # float days since epoch, shape (n_grid,)
    lo: np.ndarray     # shape (n_grid,)
    hi: np.ndarray     # shape (n_grid,)


@dataclass(frozen=True)
class Prediction:
    """Legacy back-compat dataclass — populated by the facade ``predict()``.

    Most fields originate from the trendline×exp model
    (``models.trendline_exp``); the ``interval_based_*`` fields and
    ``median_peak_interval_days`` originate from the interval-median
    baseline (``models.interval_median``). New code should consume
    ``ModelOutput`` directly via ``app_state.get_prediction()``.
    """

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
    fit_diagnostics: dict

    # Independent baseline — interval-median model's outputs surfaced
    # back into the legacy shape so the Chart-page caption rendering
    # continues to work without changes.
    interval_based_next_event_date: pd.Timestamp | None = None
    interval_based_band: tuple[pd.Timestamp, pd.Timestamp] | None = None
    median_peak_interval_days: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Time helpers — float days since the Unix epoch
# ─────────────────────────────────────────────────────────────────────────────


_EPOCH = pd.Timestamp("1970-01-01")
_SECONDS_PER_DAY = 86400.0


def to_days(
    t: pd.Timestamp | pd.Series | pd.DatetimeIndex,
) -> float | np.ndarray:
    """Convert Timestamp/Series/DatetimeIndex to float days since the Unix epoch.

    Modern pandas (>=2.x) chose ``datetime64[us]`` as the default storage
    unit instead of ``[ns]`` — going through ``.dt.total_seconds()`` (or
    its scalar equivalent) is unit-agnostic and works for any underlying
    resolution. Naive datetimes are interpreted at face value (treated
    as UTC); tz-aware datetimes are converted properly.
    """
    if isinstance(t, pd.Series):
        return ((t - _EPOCH).dt.total_seconds() / _SECONDS_PER_DAY).to_numpy()
    if isinstance(t, pd.DatetimeIndex):
        return ((t - _EPOCH).total_seconds() / _SECONDS_PER_DAY).to_numpy()
    if isinstance(t, pd.Timestamp):
        return (t - _EPOCH).total_seconds() / _SECONDS_PER_DAY
    raise TypeError(f"to_days: unsupported type {type(t).__name__}")


def from_days(d: float) -> pd.Timestamp:
    """Inverse of ``to_days`` — float days since epoch back to Timestamp (naive)."""
    return _EPOCH + pd.Timedelta(seconds=float(d) * _SECONDS_PER_DAY)


# ─────────────────────────────────────────────────────────────────────────────
# Curve definitions
# ─────────────────────────────────────────────────────────────────────────────


def exp_saturation(
    x: float | np.ndarray, A: float, k: float, C: float, x0: float
) -> float | np.ndarray:
    """Exponential saturation curve.

    The x0 shift makes the k parameter easier to fit by anchoring the
    curve to the start of the current episode.
    """
    return A * (1.0 - np.exp(-k * (x - x0))) + C


# ─────────────────────────────────────────────────────────────────────────────
# Back-compat facade
# ─────────────────────────────────────────────────────────────────────────────


def predict(tilt_df: pd.DataFrame, peaks_df: pd.DataFrame) -> Prediction:
    """Legacy facade — delegates to the new ``models/`` package.

    Calls ``models.trendline_exp.compute_trendline_exp`` and
    ``models.interval_median.IntervalMedianModel().predict`` and
    reassembles the legacy ``Prediction`` shape so existing consumers
    (``plotting.build_figure``, ``hero.compose``, the test suite) stay
    green. The function never raises; on failure individual fields are
    ``None`` and ``fit_diagnostics`` carries the cause.

    New code should use ``app_state.get_prediction(model_id=...)`` for
    direct access to a chosen model's ``ModelOutput``.
    """
    # Lazy imports break the circular dependency: the new model modules
    # import primitives (``Curve``, ``CurveBand``, ``to_days``, etc.)
    # from this module, so we can't import them at module top-level.
    from .models.interval_median import IntervalMedianModel
    from .models.trendline_exp import compute_trendline_exp

    raw = compute_trendline_exp(tilt_df, peaks_df)
    interval = IntervalMedianModel().predict(tilt_df, peaks_df)

    # Merge diagnostics — keep trendline+exp's keys, then layer the
    # interval baseline's median/mean on top so the legacy diagnostics
    # dict carries the same keys it always has.
    diagnostics: dict = dict(raw.fit_diagnostics)
    if "median_peak_interval_days" in interval.diagnostics:
        diagnostics["median_peak_interval_days"] = interval.diagnostics[
            "median_peak_interval_days"
        ]
    if "mean_peak_interval_days" in interval.diagnostics:
        diagnostics["mean_peak_interval_days"] = interval.diagnostics[
            "mean_peak_interval_days"
        ]

    return Prediction(
        next_event_date=raw.next_event_date,
        next_event_tilt=raw.next_event_tilt,
        trendline=raw.trendline,
        exp_curve=raw.exp_curve,
        exp_params=raw.exp_params,
        exp_x0=raw.exp_x0,
        exp_covariance=raw.exp_covariance,
        confidence_band=raw.confidence_band,
        trendline_band=raw.trendline_band,
        exp_band=raw.exp_band,
        n_peaks_in_fit=raw.n_peaks_in_fit,
        fit_diagnostics=diagnostics,
        interval_based_next_event_date=interval.next_event_date,
        interval_based_band=interval.confidence_band,
        median_peak_interval_days=interval.diagnostics.get(
            "median_peak_interval_days"
        ),
    )
