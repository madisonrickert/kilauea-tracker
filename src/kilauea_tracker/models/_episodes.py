"""Historical inflation segment iterator.

Cross-cycle prediction models (``linear_hist``, ``linear_stitched``,
``power_law_hist``) all need the same primitive: given the full tilt
history and the detected peak set, hand back the trough → peak inflation
segments from the last N *complete* episodes (excluding the current
in-progress one). One canonical implementation here keeps episode
boundaries consistent across every model that uses them.

Mirrors the trough-finding logic embedded in
``models/trendline_exp.compute_trendline_exp`` (post-peak deflation
search, then take the minimum) so a "complete" episode = trough → peak.

Pure: no I/O, no clock reads, no module-level mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from ..model import DATE_COL, TILT_COL, to_days

if TYPE_CHECKING:
    import numpy as np

# Mirror of ``trendline_exp._TROUGH_SEARCH_WINDOW_DAYS`` so episode
# segmentation here matches the current-episode segmentation there.
# Deflations bottom within ~24 h; 14 d safely brackets the worst case
# without grabbing an unrelated old trough.
_TROUGH_SEARCH_WINDOW_DAYS = 14

# Below this point count a segment isn't useful for parametric fitting.
# 50 samples at 15-min cadence = 12.5 h, less than one inflation phase.
_MIN_SEGMENT_SAMPLES = 50


@dataclass(frozen=True)
class InflationSegment:
    """A single trough → peak inflation phase.

    ``t_hours_since_trough`` is float hours since the trough (so each
    segment can be fitted on a normalized clock); ``tilt`` matches it
    elementwise. ``trough_date`` and ``peak_date`` are the absolute
    timestamps for context.
    """

    trough_date: pd.Timestamp
    peak_date: pd.Timestamp
    trough_tilt_microrad: float
    peak_tilt_microrad: float
    t_hours_since_trough: np.ndarray
    tilt: np.ndarray

    @property
    def duration_hours(self) -> float:
        return float(
            (self.peak_date - self.trough_date).total_seconds() / 3600.0
        )

    @property
    def amplitude_microrad(self) -> float:
        return float(self.peak_tilt_microrad - self.trough_tilt_microrad)

    @property
    def n_samples(self) -> int:
        return len(self.t_hours_since_trough)


def iter_complete_inflation_segments(
    tilt_df: pd.DataFrame, peaks_df: pd.DataFrame, n: int
) -> list[InflationSegment]:
    """Return the last ``n`` *complete* trough → peak inflation segments.

    "Complete" means the segment ends at a detected peak — i.e., we
    excluded the current in-progress episode (whose trough may exist but
    whose peak hasn't been detected yet). The most recent peak in
    ``peaks_df`` is therefore the right-edge of the last complete
    segment, and the segment immediately preceding it provides the
    left-edge trough.

    Returns up to ``n`` segments in chronological order. Returns fewer
    when the data is too sparse (e.g., fewer than ``n+1`` peaks
    available) or when individual segments fail the
    ``_MIN_SEGMENT_SAMPLES`` floor.
    """
    if len(peaks_df) < 2:
        return []

    sorted_peaks = peaks_df.sort_values(DATE_COL).reset_index(drop=True)
    sorted_tilt = tilt_df.sort_values(DATE_COL).reset_index(drop=True)

    segments: list[InflationSegment] = []
    # Iterate consecutive peak pairs. The (i-1, i) pair defines a segment
    # that ENDS at peak i — the trough lives somewhere between them.
    for i in range(1, len(sorted_peaks)):
        prev_peak_date = sorted_peaks[DATE_COL].iloc[i - 1]
        curr_peak_date = sorted_peaks[DATE_COL].iloc[i]

        # Slice tilt rows strictly after the previous peak and up to and
        # including the current peak. The trough is the minimum tilt
        # within ``_TROUGH_SEARCH_WINDOW_DAYS`` of the previous peak.
        between = sorted_tilt[
            (sorted_tilt[DATE_COL] > prev_peak_date)
            & (sorted_tilt[DATE_COL] <= curr_peak_date)
        ]
        if len(between) < _MIN_SEGMENT_SAMPLES:
            continue

        search_end = prev_peak_date + pd.Timedelta(
            days=_TROUGH_SEARCH_WINDOW_DAYS
        )
        trough_search = between[between[DATE_COL] <= search_end]
        if len(trough_search) == 0:
            continue
        trough_idx = trough_search[TILT_COL].idxmin()
        trough_date = trough_search.loc[trough_idx, DATE_COL]
        trough_tilt = float(trough_search.loc[trough_idx, TILT_COL])

        inflation = between[between[DATE_COL] >= trough_date].copy()
        if len(inflation) < _MIN_SEGMENT_SAMPLES:
            continue

        t_hours = (
            (inflation[DATE_COL] - trough_date).dt.total_seconds() / 3600.0
        ).to_numpy()
        tilt = inflation[TILT_COL].to_numpy(dtype=float)

        segments.append(
            InflationSegment(
                trough_date=trough_date,
                peak_date=curr_peak_date,
                trough_tilt_microrad=trough_tilt,
                peak_tilt_microrad=float(sorted_peaks[TILT_COL].iloc[i]),
                t_hours_since_trough=t_hours,
                tilt=tilt,
            )
        )

    return segments[-n:]


def find_current_episode_trough(
    tilt_df: pd.DataFrame, peaks_df: pd.DataFrame
) -> tuple[pd.Timestamp, float] | None:
    """Locate the current in-progress episode's trough.

    Mirrors ``compute_trendline_exp``'s post-peak deflation search so the
    cross-cycle models anchor at the same trough as the within-cycle
    models. Returns ``(trough_timestamp, trough_tilt_microrad)`` or
    ``None`` if no usable trough is found.
    """
    if len(peaks_df) == 0:
        return None
    sorted_peaks = peaks_df.sort_values(DATE_COL).reset_index(drop=True)
    last_peak_date = sorted_peaks[DATE_COL].iloc[-1]

    post_peak = tilt_df[tilt_df[DATE_COL] > last_peak_date].sort_values(
        DATE_COL
    )
    if len(post_peak) == 0:
        return None

    search_end = last_peak_date + pd.Timedelta(
        days=_TROUGH_SEARCH_WINDOW_DAYS
    )
    trough_search = post_peak[post_peak[DATE_COL] <= search_end]
    if len(trough_search) == 0:
        return None

    trough_idx = trough_search[TILT_COL].idxmin()
    return (
        trough_search.loc[trough_idx, DATE_COL],
        float(trough_search.loc[trough_idx, TILT_COL]),
    )


def trough_day(trough_date: pd.Timestamp) -> float:
    """Helper: convert a trough timestamp to float days since epoch."""
    return float(to_days(trough_date))
