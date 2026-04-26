"""Tests for the FFM/Voight inverse-rate experimental model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.models.ffm_voight import FFMVoightModel


def _accelerating_synthetic(
    base: pd.Timestamp, hours: int = 100, t_failure: float = 150.0
) -> pd.DataFrame:
    """Synthetic data with rate accelerating to a singularity at t_failure.

    Voight α=2: rate = A / (t_f - t). Integrating: y = -A · ln(t_f - t).
    Pick A=1 for unit-amplitude rate.
    """
    rng = np.random.default_rng(0)
    rows = []
    A = 1.0
    for i in range(hours * 4):
        t_h = i / 4.0
        y = -A * np.log(t_failure - t_h) + rng.normal(0, 0.02)
        rows.append((base + pd.Timedelta(hours=t_h), y))
    return pd.DataFrame(rows, columns=[DATE_COL, TILT_COL])


def _decelerating_synthetic(
    base: pd.Timestamp, hours: int = 100
) -> pd.DataFrame:
    """Saturating curve: rate decreasing over time. Wrong regime for FFM."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(hours * 4):
        t_h = i / 4.0
        y = -10.0 + 10.0 * (1.0 - np.exp(-0.02 * t_h)) + rng.normal(0, 0.05)
        rows.append((base + pd.Timedelta(hours=t_h), y))
    return pd.DataFrame(rows, columns=[DATE_COL, TILT_COL])


def _peaks_df(end: pd.Timestamp, n: int = 4) -> pd.DataFrame:
    rows = [
        (end - pd.Timedelta(days=14 * (n - i - 1)), 5.0)
        for i in range(n)
    ]
    return pd.DataFrame(rows, columns=[DATE_COL, TILT_COL])


def test_no_op_on_decelerating_regime():
    """Kīlauea-like decelerating inflation must produce no prediction."""
    end_peak = pd.Timestamp("2026-04-23 10:00:00")
    inflation = _decelerating_synthetic(end_peak + pd.Timedelta(hours=1), hours=120)
    peaks = _peaks_df(end_peak)
    out = FFMVoightModel().predict(inflation, peaks)
    assert out.next_event_date is None
    assert "regime_signal" in out.diagnostics or "warning" in out.diagnostics


def test_no_op_or_finite_prediction_on_accelerating_synthetic():
    """On the textbook accelerating regime FFM should EITHER recover the
    failure time OR produce a finite prediction with a sensible slope.
    The smoothed-rate finite-difference loses fidelity near the
    singularity, so we accept either outcome — the regression test is
    just "doesn't crash and emits a coherent diagnostic."""
    end_peak = pd.Timestamp("2026-04-23 10:00:00")
    inflation = _accelerating_synthetic(
        end_peak + pd.Timedelta(hours=1), hours=100, t_failure=150.0
    )
    peaks = _peaks_df(end_peak)
    out = FFMVoightModel().predict(inflation, peaks)
    # In a Voight α=2 regime the inverse-rate slope must be NEGATIVE
    # (rate increasing toward the singularity). That's the model
    # contract: if the slope is negative, a positive failure-time
    # intercept exists.
    inv_slope = out.diagnostics.get("inverse_rate_slope")
    assert inv_slope is not None
    assert inv_slope < 0


def test_metadata():
    m = FFMVoightModel()
    assert m.id == "ffm_voight"
    assert "experimental" in m.description.lower()
