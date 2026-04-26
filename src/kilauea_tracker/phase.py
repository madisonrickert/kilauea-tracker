"""Inflation-phase progress estimator.

The "halfway through inflation" question is the canonical signal we use
to switch the active prediction model (linear → linear_naive). It can
only be known retroactively — by the time the next peak arrives we know
the duration and can compute the fraction. We approximate it forward
by anchoring at the *current* trough and dividing elapsed time by the
median duration of recent complete inflation phases.

Pure: no I/O, no clock reads. The current time enters via the optional
``now`` parameter — pass ``pd.Timestamp.now(tz='UTC').tz_localize(None)``
at the boundary if you need wall-clock semantics, or hand in the last
data sample's timestamp for "phase as of latest data."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import HIST_FIT_MIN_EPISODES, HIST_FIT_N_EPISODES
from .model import DATE_COL
from .models._episodes import (
    find_current_episode_trough,
    iter_complete_inflation_segments,
)

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class PhaseEstimate:
    """Where in the inflation phase the current episode is.

    ``fraction`` is elapsed-since-trough divided by median historical
    inflation duration. Values < 1 mean we're still rising; values > 1
    mean the current episode has already lasted longer than typical.
    Returns ``None`` for fields when there isn't enough data to estimate.
    """

    elapsed_hours: float | None
    median_duration_hours: float | None
    fraction: float | None
    n_historical_episodes: int
    historical_durations_hours: list[float]
    trough_date_utc: str | None

    @property
    def is_late(self) -> bool:
        """``True`` if the model selector should prefer the late-window
        model (``linear_naive``) over the early-window one (``linear``).

        Threshold is 0.5 — the empirical halfway point between when
        ``linear`` stops winning the backtest and ``linear_naive``
        starts winning.
        """
        return self.fraction is not None and self.fraction >= 0.5


# Halfway threshold above. Lifted to a module constant so callers can
# import it for display ("transitions at fraction X"). 0.5 is the
# empirical handoff between linear and linear_naive in the quartile
# backtest (linear wins at 25/50%, linear_naive wins at 75/100%).
LATE_PHASE_THRESHOLD = 0.5


def estimate_phase(
    tilt_df: pd.DataFrame,
    peaks_df: pd.DataFrame,
    now: pd.Timestamp | None = None,
    n_history: int = HIST_FIT_N_EPISODES,
    min_history: int = HIST_FIT_MIN_EPISODES,
) -> PhaseEstimate:
    """Estimate where the current episode is in its inflation phase.

    ``now`` defaults to the latest tilt sample's timestamp — passing it
    explicitly lets the caller decouple display time from data time.
    """
    trough = find_current_episode_trough(tilt_df, peaks_df)
    if trough is None:
        return PhaseEstimate(
            elapsed_hours=None,
            median_duration_hours=None,
            fraction=None,
            n_historical_episodes=0,
            historical_durations_hours=[],
            trough_date_utc=None,
        )
    trough_date, _ = trough

    if now is None and len(tilt_df) > 0:
        now = tilt_df[DATE_COL].max()

    elapsed_hours: float | None = None
    if now is not None:
        elapsed_hours = max(
            float((now - trough_date).total_seconds() / 3600.0), 0.0
        )

    segments = iter_complete_inflation_segments(tilt_df, peaks_df, n=n_history)
    durations = [seg.duration_hours for seg in segments]
    median_duration: float | None = None
    fraction: float | None = None
    if len(durations) >= min_history:
        median_duration = float(np.median(durations))
        if elapsed_hours is not None and median_duration > 0:
            fraction = elapsed_hours / median_duration

    return PhaseEstimate(
        elapsed_hours=elapsed_hours,
        median_duration_hours=median_duration,
        fraction=fraction,
        n_historical_episodes=len(durations),
        historical_durations_hours=list(durations),
        trough_date_utc=str(trough_date),
    )
