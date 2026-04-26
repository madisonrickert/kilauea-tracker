"""Tests for the phase-aware ``auto`` ensemble model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.models.auto import AutoModel
from kilauea_tracker.models.linear import LinearModel, LinearNaiveModel
from kilauea_tracker.models.output import ModelOutput


def _multi_episode(n: int, inflation_h: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    peaks_rows = []
    base = pd.Timestamp("2026-01-01 00:00:00")
    cur = base
    for _ in range(n):
        for i in range(inflation_h * 4):
            t_h = i / 4.0
            y = -10.0 + 0.04 * t_h + rng.normal(0, 0.05)
            rows.append((cur + pd.Timedelta(hours=t_h), y))
        peak_t = cur + pd.Timedelta(hours=inflation_h - 0.25)
        peak_y = -10.0 + 0.04 * inflation_h
        peaks_rows.append((peak_t, peak_y))
        for j in range(8):
            t = peak_t + pd.Timedelta(minutes=15 * (j + 1))
            y = peak_y - (j + 1) * (peak_y - (-10.0)) / 8.0
            rows.append((t, y))
        # Hours=2.5 so the next inflation's t=0 sample doesn't collide
        # with the deflation's last sample at peak_t + 2h.
        cur = peak_t + pd.Timedelta(hours=2.5)
    return (
        pd.DataFrame(rows, columns=[DATE_COL, TILT_COL]).sort_values(DATE_COL).reset_index(drop=True),
        pd.DataFrame(peaks_rows, columns=[DATE_COL, TILT_COL]),
    )


def _append_partial(
    tilt: pd.DataFrame, last_peak: pd.Timestamp, hours: int, seed: int = 1
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(hours * 4):
        t_h = i / 4.0 + 2.0
        y = -10.0 + 0.04 * t_h + rng.normal(0, 0.05)
        rows.append((last_peak + pd.Timedelta(hours=t_h), y))
    return pd.concat([tilt, pd.DataFrame(rows, columns=tilt.columns)]).sort_values(DATE_COL).reset_index(drop=True)


def test_auto_picks_linear_in_early_phase():
    """Phase ~10% → auto should pick `linear`."""
    tilt, peaks = _multi_episode(n=6, inflation_h=240)
    last_peak = peaks[DATE_COL].iloc[-1]
    tilt = _append_partial(tilt, last_peak, hours=24)  # 24h of inflation = ~10%

    out = AutoModel().predict(tilt, peaks)
    assert isinstance(out, ModelOutput)
    assert out.diagnostics["auto_chosen_base"] == "linear"
    # Compare with the standalone linear model — same prediction.
    direct = LinearModel().predict(tilt, peaks)
    assert out.next_event_date == direct.next_event_date


def test_auto_picks_linear_naive_in_late_phase():
    """Phase ~80% → auto should pick `linear_naive`."""
    tilt, peaks = _multi_episode(n=6, inflation_h=240)
    last_peak = peaks[DATE_COL].iloc[-1]
    tilt = _append_partial(tilt, last_peak, hours=192)  # 192h ≈ 80% of 240h

    out = AutoModel().predict(tilt, peaks)
    assert out.diagnostics["auto_chosen_base"] == "linear_naive"
    direct = LinearNaiveModel().predict(tilt, peaks)
    assert out.next_event_date == direct.next_event_date


def test_auto_diagnostics_carry_phase_context():
    tilt, peaks = _multi_episode(n=6, inflation_h=240)
    last_peak = peaks[DATE_COL].iloc[-1]
    tilt = _append_partial(tilt, last_peak, hours=60)
    out = AutoModel().predict(tilt, peaks)
    for k in (
        "auto_chosen_base",
        "auto_phase_fraction",
        "auto_elapsed_hours",
        "auto_median_duration_hours",
        "auto_threshold",
    ):
        assert k in out.diagnostics


def test_metadata():
    m = AutoModel()
    assert m.id == "auto"
    assert "phase" in m.description.lower()
