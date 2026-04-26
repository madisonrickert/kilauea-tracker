"""Tests for the stitched-inflation linear-trend model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.models.linear_stitched import LinearStitchedModel
from kilauea_tracker.models.output import ModelOutput


def _multi_episode_synthetic(
    n_episodes: int, slope_per_hour: float, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[tuple[pd.Timestamp, float]] = []
    peaks_rows: list[tuple[pd.Timestamp, float]] = []
    base = pd.Timestamp("2026-01-01 00:00:00")
    cur_t = base
    inflation_hours = 240
    for _ in range(n_episodes):
        for i in range(inflation_hours * 4):
            t_h = i / 4.0
            y = -20.0 + slope_per_hour * t_h + rng.normal(0, 0.1)
            rows.append((cur_t + pd.Timedelta(hours=t_h), y))
        peak_t = cur_t + pd.Timedelta(hours=inflation_hours - 0.25)
        peak_y = -20.0 + slope_per_hour * inflation_hours
        peaks_rows.append((peak_t, peak_y))
        for j in range(8):
            t = peak_t + pd.Timedelta(minutes=15 * (j + 1))
            y = peak_y - (j + 1) * (peak_y - (-20.0)) / 8.0
            rows.append((t, y))
        cur_t = peak_t + pd.Timedelta(hours=2)
    return (
        pd.DataFrame(rows, columns=[DATE_COL, TILT_COL]).sort_values(DATE_COL),
        pd.DataFrame(peaks_rows, columns=[DATE_COL, TILT_COL]),
    )


def test_recovers_stitched_slope():
    tilt, peaks = _multi_episode_synthetic(n_episodes=6, slope_per_hour=0.05)
    out = LinearStitchedModel().predict(tilt, peaks)
    assert isinstance(out, ModelOutput)
    slope = out.diagnostics.get("stitched_slope_per_hour")
    assert slope is not None
    assert abs(slope - 0.05) < 0.005


def test_residual_std_reported():
    tilt, peaks = _multi_episode_synthetic(n_episodes=6, slope_per_hour=0.05)
    out = LinearStitchedModel().predict(tilt, peaks)
    assert "stitched_residual_std_microrad" in out.diagnostics
    # Synthetic clean data should have residual ~ 0.1 (the planted noise).
    rs = out.diagnostics["stitched_residual_std_microrad"]
    assert 0.0 <= rs < 1.0


def test_metadata():
    assert LinearStitchedModel().id == "linear_stitched"
