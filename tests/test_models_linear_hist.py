"""Tests for the historical-median linear-slope model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.models.linear_hist import LinearHistModel
from kilauea_tracker.models.output import ModelOutput


def _multi_episode_synthetic(
    n_episodes: int, target_slope_per_hour: float, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a tilt history with planted episodes — each rises at the
    given slope, drops, repeats."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[pd.Timestamp, float]] = []
    peaks_rows: list[tuple[pd.Timestamp, float]] = []
    base = pd.Timestamp("2026-01-01 00:00:00")
    cur_t = base
    inflation_hours = 240  # 10 days
    for _ep in range(n_episodes):
        # Inflation
        for i in range(inflation_hours * 4):
            t_h = i / 4.0
            y = -20.0 + target_slope_per_hour * t_h + rng.normal(0, 0.1)
            rows.append((cur_t + pd.Timedelta(hours=t_h), y))
        peak_t = cur_t + pd.Timedelta(hours=inflation_hours - 0.25)
        peak_y = -20.0 + target_slope_per_hour * inflation_hours
        peaks_rows.append((peak_t, peak_y))
        # Deflation drop
        for j in range(8):
            t = peak_t + pd.Timedelta(minutes=15 * (j + 1))
            y = peak_y - (j + 1) * (peak_y - (-20.0)) / 8.0
            rows.append((t, y))
        cur_t = peak_t + pd.Timedelta(hours=2)
    tilt_df = pd.DataFrame(rows, columns=[DATE_COL, TILT_COL]).sort_values(DATE_COL)
    peaks_df = pd.DataFrame(peaks_rows, columns=[DATE_COL, TILT_COL])
    return tilt_df, peaks_df


def test_recovers_planted_slope():
    """If every historical episode has the same planted slope, the
    median should recover it."""
    tilt, peaks = _multi_episode_synthetic(n_episodes=6, target_slope_per_hour=0.05)
    out = LinearHistModel().predict(tilt, peaks)
    assert isinstance(out, ModelOutput)
    median_slope = out.diagnostics.get("median_slope_per_hour")
    assert median_slope is not None
    assert abs(median_slope - 0.05) < 0.005


def test_falls_back_when_too_few_episodes():
    tilt, peaks = _multi_episode_synthetic(n_episodes=3, target_slope_per_hour=0.05)
    out = LinearHistModel().predict(tilt, peaks)
    assert out.next_event_date is None
    assert "warning" in out.diagnostics


def test_metadata():
    assert LinearHistModel().id == "linear_hist"
    assert "historical" in LinearHistModel().description.lower() or "median" in LinearHistModel().description.lower()
