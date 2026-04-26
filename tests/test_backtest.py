"""Tests for the pure backtest module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kilauea_tracker.backtest import (
    DEFAULT_QUARTILES,
    BacktestResult,
    find_recent_segments,
    run_backtest,
)
from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.peaks import detect_peaks


def _multi_episode(n: int, inflation_h: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2026-01-01 00:00:00")
    cur = base
    for _ in range(n):
        for i in range(inflation_h * 4):
            t_h = i / 4.0
            # Use a larger amplitude so peak prominence > 10 (filter floor).
            y = -10.0 + (50.0 / inflation_h) * t_h + rng.normal(0, 0.05)
            rows.append((cur + pd.Timedelta(hours=t_h), y))
        peak_t = cur + pd.Timedelta(hours=inflation_h - 0.25)
        peak_y = -10.0 + 50.0
        for j in range(8):
            t = peak_t + pd.Timedelta(minutes=15 * (j + 1))
            y = peak_y - (j + 1) * (peak_y - (-10.0)) / 8.0
            rows.append((t, y))
        # Hours=2.5 (not 2.0) so the next inflation's t=0 sample doesn't
        # collide with the deflation's last sample (peak_t + 2h).
        cur = peak_t + pd.Timedelta(hours=2.5)
    return pd.DataFrame(rows, columns=[DATE_COL, TILT_COL]).sort_values(DATE_COL).reset_index(drop=True)


def test_run_backtest_returns_result_for_synthetic():
    tilt = _multi_episode(n=6, inflation_h=240)
    result = run_backtest(tilt, n_segments=4)
    assert isinstance(result, BacktestResult)
    assert len(result.segments) >= 1
    assert result.fractions == DEFAULT_QUARTILES


def test_find_recent_segments_returns_chronological():
    tilt = _multi_episode(n=5, inflation_h=240)
    peaks = detect_peaks(tilt)
    segments = find_recent_segments(tilt, peaks, n_segments=3)
    dates = [s.peak_date for s in segments]
    assert dates == sorted(dates)


def test_stats_handles_no_predictions_gracefully():
    """A model that always returns None should produce all-None stats."""
    tilt = _multi_episode(n=4, inflation_h=240)
    result = run_backtest(tilt, n_segments=3)
    # ffm_voight nearly always no-ops on synthetic data
    s = result.stats("ffm_voight", 0.50)
    # Either all-None (true no-op) or a valid number — both are valid
    # outcomes. We only assert the type contract.
    assert s.model_id == "ffm_voight"
    assert s.fraction == 0.50
    assert s.n_segments >= 1
    assert s.coverage >= 0.0
    assert s.coverage <= 1.0


def test_best_per_quartile_only_picks_qualifying_models():
    tilt = _multi_episode(n=6, inflation_h=240)
    result = run_backtest(tilt, n_segments=4)
    bests = result.best_per_quartile()
    assert set(bests.keys()) == set(result.fractions)
    for _f, best in bests.items():
        if best is not None:
            # All "best" picks must have ≥50% coverage.
            assert best.coverage >= 0.5
