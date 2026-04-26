"""Tests for the historical-median-p power-law model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.models.output import ModelOutput
from kilauea_tracker.models.power_law_hist import PowerLawHistModel


def _multi_episode_power_synthetic(
    n_episodes: int, p_target: float, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[tuple[pd.Timestamp, float]] = []
    peaks_rows: list[tuple[pd.Timestamp, float]] = []
    base = pd.Timestamp("2026-01-01 00:00:00")
    cur_t = base
    inflation_hours = 240
    for _ep in range(n_episodes):
        # Vary amplitude per episode so the model has to find shape, not amplitude.
        amplitude = 20.0 + rng.uniform(-3, 3)
        for i in range(inflation_hours * 4):
            t_h = i / 4.0
            y = -20.0 + amplitude * np.power(max(t_h / inflation_hours, 1e-3), p_target) + rng.normal(0, 0.1)
            rows.append((cur_t + pd.Timedelta(hours=t_h), y))
        peak_t = cur_t + pd.Timedelta(hours=inflation_hours - 0.25)
        peak_y = -20.0 + amplitude
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


def test_recovers_planted_p():
    tilt, peaks = _multi_episode_power_synthetic(n_episodes=6, p_target=0.6)
    out = PowerLawHistModel().predict(tilt, peaks)
    assert isinstance(out, ModelOutput)
    median_p = out.diagnostics.get("median_p")
    assert median_p is not None
    assert abs(median_p - 0.6) < 0.1


def test_falls_back_when_too_few_episodes():
    tilt, peaks = _multi_episode_power_synthetic(n_episodes=3, p_target=0.6)
    out = PowerLawHistModel().predict(tilt, peaks)
    assert out.next_event_date is None
    assert "warning" in out.diagnostics


def test_metadata():
    assert PowerLawHistModel().id == "power_law_hist_p"
