"""Tests for the inflation-phase progress estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.phase import LATE_PHASE_THRESHOLD, estimate_phase


def _multi_episode_synthetic(
    n_episodes: int, inflation_hours: int = 240, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build N complete sawtooth episodes plus a partial in-progress one."""
    rng = np.random.default_rng(seed)
    rows = []
    peaks_rows = []
    base = pd.Timestamp("2026-01-01 00:00:00")
    cur_t = base
    for _ep in range(n_episodes):
        for i in range(inflation_hours * 4):
            t_h = i / 4.0
            y = -10.0 + 0.04 * t_h + rng.normal(0, 0.05)
            rows.append((cur_t + pd.Timedelta(hours=t_h), y))
        peak_t = cur_t + pd.Timedelta(hours=inflation_hours - 0.25)
        peak_y = -10.0 + 0.04 * inflation_hours
        peaks_rows.append((peak_t, peak_y))
        for j in range(8):
            t = peak_t + pd.Timedelta(minutes=15 * (j + 1))
            y = peak_y - (j + 1) * (peak_y - (-10.0)) / 8.0
            rows.append((t, y))
        cur_t = peak_t + pd.Timedelta(hours=2)
    return (
        pd.DataFrame(rows, columns=[DATE_COL, TILT_COL]).sort_values(DATE_COL),
        pd.DataFrame(peaks_rows, columns=[DATE_COL, TILT_COL]),
    )


def test_phase_recovers_halfway_in_synthetic_data():
    """6 complete episodes of 240h each, then a partial episode 120h in.
    Phase should be close to 0.5."""
    tilt, peaks = _multi_episode_synthetic(n_episodes=6, inflation_hours=240)
    # Build a partial 7th episode 120h past last peak.
    last_peak = peaks[DATE_COL].iloc[-1]
    rng = np.random.default_rng(99)
    partial_rows = []
    for i in range(120 * 4):
        t_h = i / 4.0 + 2.0  # offset by deflation
        y = -10.0 + 0.04 * t_h + rng.normal(0, 0.05)
        partial_rows.append((last_peak + pd.Timedelta(hours=t_h), y))
    extra = pd.DataFrame(partial_rows, columns=[DATE_COL, TILT_COL])
    tilt_with_partial = pd.concat([tilt, extra]).sort_values(DATE_COL).reset_index(drop=True)

    p = estimate_phase(tilt_with_partial, peaks)
    assert p.fraction is not None
    # Allow ±10% slack for trough-finding noise.
    assert 0.40 < p.fraction < 0.60


def test_phase_returns_none_with_too_few_episodes():
    tilt, peaks = _multi_episode_synthetic(n_episodes=2, inflation_hours=240)
    p = estimate_phase(tilt, peaks)
    # Below HIST_FIT_MIN_EPISODES (4) the median has no fraction.
    assert p.fraction is None
    assert p.n_historical_episodes < 4


def test_is_late_threshold():
    """Check is_late uses 0.5 threshold consistently."""
    from kilauea_tracker.phase import PhaseEstimate

    early = PhaseEstimate(
        elapsed_hours=10.0,
        median_duration_hours=100.0,
        fraction=0.10,
        n_historical_episodes=6,
        historical_durations_hours=[],
        trough_date_utc="2026-01-01",
    )
    late = PhaseEstimate(
        elapsed_hours=80.0,
        median_duration_hours=100.0,
        fraction=0.80,
        n_historical_episodes=6,
        historical_durations_hours=[],
        trough_date_utc="2026-01-01",
    )
    threshold = PhaseEstimate(
        elapsed_hours=50.0,
        median_duration_hours=100.0,
        fraction=LATE_PHASE_THRESHOLD,
        n_historical_episodes=6,
        historical_durations_hours=[],
        trough_date_utc="2026-01-01",
    )
    assert not early.is_late
    assert late.is_late
    assert threshold.is_late  # >= 0.5 is "late"
