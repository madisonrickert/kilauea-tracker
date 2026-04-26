"""Tests for the constrained power-law model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.models.output import ModelOutput
from kilauea_tracker.models.power_law import PowerLawModel


def _synthetic_post_trough(
    base: pd.Timestamp, hours: int, p: float
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(hours * 4):
        t_h = i / 4.0
        y = -20.0 + 1.0 * np.power(max(t_h, 1e-3), p) + rng.normal(0, 0.05)
        rows.append((base + pd.Timedelta(hours=t_h), y))
    return pd.DataFrame(rows, columns=[DATE_COL, TILT_COL])


def _baseline_history(end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for i in range(30 * 24 * 4):
        t = end - pd.Timedelta(hours=30 * 24) + pd.Timedelta(minutes=15 * i)
        rows.append((t, -25.0))
    return pd.DataFrame(rows, columns=[DATE_COL, TILT_COL])


def _peaks_df(end_peak: pd.Timestamp, n: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n):
        t = end_peak - pd.Timedelta(days=14 * (n - i - 1))
        rows.append((t, 10.0))
    return pd.DataFrame(rows, columns=[DATE_COL, TILT_COL])


def test_predict_returns_model_output():
    end_peak = pd.Timestamp("2026-04-23 10:00:00")
    history = _baseline_history(end_peak)
    inflation = _synthetic_post_trough(end_peak + pd.Timedelta(hours=1), 200, p=0.6)
    tilt = pd.concat([history, inflation], ignore_index=True).sort_values(DATE_COL)
    peaks = _peaks_df(end_peak)
    out = PowerLawModel().predict(tilt, peaks)
    assert isinstance(out, ModelOutput)


def test_p_is_constrained_to_band():
    """Synthetic data with extreme exponent (e.g. p=0.2 below floor)
    should still fit within the [0.3, 1.5] bounds."""
    end_peak = pd.Timestamp("2026-04-23 10:00:00")
    history = _baseline_history(end_peak)
    inflation = _synthetic_post_trough(end_peak + pd.Timedelta(hours=1), 200, p=0.2)
    tilt = pd.concat([history, inflation], ignore_index=True).sort_values(DATE_COL)
    peaks = _peaks_df(end_peak)
    out = PowerLawModel().predict(tilt, peaks)
    p = out.diagnostics.get("power_p")
    if p is not None:
        assert 0.3 <= p <= 1.5


def test_no_peaks_returns_empty_output():
    base = pd.Timestamp("2026-04-01")
    tilt = pd.DataFrame({DATE_COL: [base], TILT_COL: [0.0]})
    peaks = pd.DataFrame({DATE_COL: pd.Series(dtype="datetime64[ns]"), TILT_COL: pd.Series(dtype="float")})
    out = PowerLawModel().predict(tilt, peaks)
    assert out.next_event_date is None


def test_metadata():
    m = PowerLawModel()
    assert m.id == "power_law"
    assert "power-law" in m.description.lower() or "p ∈" in m.description or "exponent" in m.description.lower()
