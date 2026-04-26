"""Interval-median baseline model.

Predicts the next event as ``last_peak + median(consecutive_peak_intervals)``.
Uses the median (not mean) so a single anomalously short or long interval
doesn't dominate. The 80% confidence band uses the 25th/75th percentile
intervals — a roughly interquartile range around the point estimate.

This is the simple statistical baseline historically inlined in
``model.predict()`` (formerly lines 195-223). Lifting it into its own
``Model`` validates the registry's protocol with the smallest possible
real implementation.

Pure: no I/O, no clock reads, no module-level mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..model import DATE_COL, from_days, to_days
from .output import ModelOutput

if TYPE_CHECKING:
    import pandas as pd

# Minimum number of intervals required to compute the IQR-based band.
# With 2 peaks (1 interval) the median equals the only sample, but the
# 25/75 quantile is degenerate — fall back to point-estimate-only.
_MIN_INTERVALS_FOR_BAND = 3


@dataclass(frozen=True)
class IntervalMedianModel:
    """Median-cycle-length forecast — independent of any tilt fit.

    Always returns ``curves=[]``; this model has no overlay visualization
    on the chart, only a marker + band at the predicted date. The
    headline text reports the median cycle length so the visitor sees
    the basis for the forecast.
    """

    id: str = "interval_median"
    label: str = "Median peak interval"
    description: str = (
        "Forecasts the next pulse as the most recent peak plus the median "
        "of all observed peak-to-peak intervals. Independent sanity check "
        "against the curve-fit model — no tilt math involved."
    )

    def predict(
        self, tilt_df: pd.DataFrame, peaks_df: pd.DataFrame
    ) -> ModelOutput:
        if len(peaks_df) < 2:
            return ModelOutput(
                next_event_date=None,
                confidence_band=None,
                headline_text=None,
                curves=[],
                diagnostics={"error": "need at least 2 peaks for an interval"},
            )

        sorted_peaks = peaks_df.sort_values(DATE_COL)
        days = to_days(sorted_peaks[DATE_COL])
        intervals_days = np.diff(np.asarray(days, dtype=float))

        if len(intervals_days) < 1:
            return ModelOutput(
                next_event_date=None,
                confidence_band=None,
                headline_text=None,
                curves=[],
                diagnostics={"error": "no valid intervals after sorting"},
            )

        last_peak_day = float(np.asarray(days, dtype=float).max())
        median_interval = float(np.median(intervals_days))
        next_date = from_days(last_peak_day + median_interval)

        confidence_band: tuple[pd.Timestamp, pd.Timestamp] | None = None
        if len(intervals_days) >= _MIN_INTERVALS_FOR_BAND:
            lo_interval = float(np.quantile(intervals_days, 0.25))
            hi_interval = float(np.quantile(intervals_days, 0.75))
            confidence_band = (
                from_days(last_peak_day + lo_interval),
                from_days(last_peak_day + hi_interval),
            )

        diagnostics = {
            "median_peak_interval_days": median_interval,
            "mean_peak_interval_days": float(np.mean(intervals_days)),
            "n_intervals": len(intervals_days),
        }

        # Use a compact integer day count in the headline if it rounds
        # cleanly; otherwise show one decimal so the visitor can tell
        # short cycles (12.5 d) apart from long ones (32.0 d).
        median_str = (
            f"{round(median_interval)}d"
            if abs(median_interval - round(median_interval)) < 0.05
            else f"{median_interval:.1f}d"
        )
        headline_text = f"median {median_str} cycle"

        return ModelOutput(
            next_event_date=next_date,
            confidence_band=confidence_band,
            headline_text=headline_text,
            curves=[],
            diagnostics=diagnostics,
        )
