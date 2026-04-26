"""Tests for the historical inflation segment iterator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.models._episodes import (
    InflationSegment,
    find_current_episode_trough,
    iter_complete_inflation_segments,
)


def _synthetic_sawtooth(n_episodes: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a synthetic sawtooth tilt series + matching peaks_df.

    Each episode: 12-day inflation linearly from 0 → 10 µrad, then a
    1-h drop back to 0. 15-min cadence.
    """
    rng = np.random.default_rng(seed)
    sample_h = 0.25  # 15 min
    inflation_days = 12
    dt_inflation = pd.Timedelta(hours=sample_h)
    base = pd.Timestamp("2026-01-01")
    rows: list[tuple[pd.Timestamp, float]] = []
    peak_rows: list[tuple[pd.Timestamp, float]] = []
    cur = base
    for _ in range(n_episodes):
        n_inf = int(inflation_days * 24 / sample_h)
        for i in range(n_inf):
            t = cur + i * dt_inflation
            y = (i / n_inf) * 10.0 + rng.normal(0, 0.05)
            rows.append((t, y))
        peak_t = cur + (n_inf - 1) * dt_inflation
        peak_y = 10.0
        peak_rows.append((peak_t, peak_y))
        # Deflation drop: 4 samples back to 0
        for j in range(1, 5):
            t = peak_t + j * dt_inflation
            y = 10.0 - (j / 4.0) * 10.0
            rows.append((t, y))
        cur = peak_t + 4 * dt_inflation
    tilt_df = pd.DataFrame(rows, columns=[DATE_COL, TILT_COL])
    peaks_df = pd.DataFrame(peak_rows, columns=[DATE_COL, TILT_COL])
    return tilt_df, peaks_df


def test_iter_returns_at_most_n_complete_segments():
    tilt, peaks = _synthetic_sawtooth(n_episodes=5)
    segs = iter_complete_inflation_segments(tilt, peaks, n=3)
    assert len(segs) == 3
    for s in segs:
        assert isinstance(s, InflationSegment)
        assert s.n_samples > 0


def test_segments_are_chronological():
    tilt, peaks = _synthetic_sawtooth(n_episodes=5)
    segs = iter_complete_inflation_segments(tilt, peaks, n=4)
    dates = [s.peak_date for s in segs]
    assert dates == sorted(dates)


def test_segment_amplitudes_match_synthetic_truth():
    tilt, peaks = _synthetic_sawtooth(n_episodes=4)
    segs = iter_complete_inflation_segments(tilt, peaks, n=3)
    # Each segment should rise ~10 µrad.
    for s in segs:
        assert 8.0 < s.amplitude_microrad < 12.0


def test_too_few_peaks_returns_empty():
    base = pd.Timestamp("2026-01-01")
    tilt = pd.DataFrame({DATE_COL: [base], TILT_COL: [0.0]})
    peaks = pd.DataFrame({DATE_COL: [base], TILT_COL: [10.0]})
    assert iter_complete_inflation_segments(tilt, peaks, n=3) == []


def test_find_current_episode_trough_locates_minimum():
    tilt, peaks = _synthetic_sawtooth(n_episodes=3)
    # After the synthetic peak rows end, there's nothing — but we kept
    # the deflation drop too. So the "current" trough is the last
    # synthetic deflation low.
    trough = find_current_episode_trough(tilt, peaks)
    assert trough is not None
    _, trough_tilt = trough
    assert trough_tilt < 1.0  # synthetic deflation ends near zero
