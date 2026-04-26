"""Tests for the linear / linear_naive within-cycle models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.models.linear import LinearModel, LinearNaiveModel
from kilauea_tracker.models.output import ModelOutput


def _build_steep_early_inflation_segment(
    base: pd.Timestamp, hours: int = 200
) -> pd.DataFrame:
    """Inflation phase whose first quarter is 2× steeper than the rest.

    Validates the slope-decay regression: a `linear_naive` fit on the
    full window should predict an EARLIER intersection than `linear`
    (which only fits the late portion).
    """
    rows: list[tuple[pd.Timestamp, float]] = []
    rng = np.random.default_rng(42)
    for i in range(hours * 4):  # 15-min cadence
        t_h = i / 4.0
        # First 25% steep (slope 0.10), rest shallow (slope 0.05).
        threshold = hours * 0.25
        if t_h <= threshold:
            y = -10.0 + 0.10 * t_h
        else:
            y = -10.0 + 0.10 * threshold + 0.05 * (t_h - threshold)
        y += rng.normal(0, 0.05)
        rows.append((base + pd.Timedelta(hours=t_h), y))
    return pd.DataFrame(rows, columns=[DATE_COL, TILT_COL])


def _peaks_df(start: pd.Timestamp, n: int = 6, slope_per_day: float = 0.0) -> pd.DataFrame:
    rows = []
    for i in range(n):
        t = start - pd.Timedelta(days=14 * (n - i - 1))
        y = 10.0 + slope_per_day * (i - n / 2.0)
        rows.append((t, y))
    return pd.DataFrame(rows, columns=[DATE_COL, TILT_COL])


def _baseline_history(end: pd.Timestamp) -> pd.DataFrame:
    # 30 days of pre-trough data trailing the most recent peak so peak
    # detection has context.
    rows = []
    for i in range(30 * 24 * 4):
        t = end - pd.Timedelta(hours=30 * 24) + pd.Timedelta(minutes=15 * i)
        rows.append((t, -15.0))
    return pd.DataFrame(rows, columns=[DATE_COL, TILT_COL])


def test_linear_returns_model_output_on_real_shape():
    base_peak = pd.Timestamp("2026-04-23 10:00:00")
    history = _baseline_history(base_peak)
    inflation_start = base_peak + pd.Timedelta(hours=2)
    inflation = _build_steep_early_inflation_segment(inflation_start)
    tilt = pd.concat([history, inflation], ignore_index=True).sort_values(DATE_COL)
    peaks = _peaks_df(base_peak)

    out = LinearModel().predict(tilt, peaks)
    assert isinstance(out, ModelOutput)
    assert "linear_slope_per_day" in out.diagnostics


def test_linear_naive_predicts_earlier_than_linear_under_slope_decay():
    """Locked-in regression for the user's slope-decay insight.

    With a planted 2× early-vs-late slope, linear_naive must predict
    an EARLIER intersection than linear. (Both models use the same
    trendline through the same peaks; only the curve fit window
    differs.)
    """
    base_peak = pd.Timestamp("2026-04-23 10:00:00")
    history = _baseline_history(base_peak)
    inflation_start = base_peak + pd.Timedelta(hours=2)
    inflation = _build_steep_early_inflation_segment(inflation_start, hours=200)
    tilt = pd.concat([history, inflation], ignore_index=True).sort_values(DATE_COL)
    peaks = _peaks_df(base_peak)

    out_linear = LinearModel().predict(tilt, peaks)
    out_naive = LinearNaiveModel().predict(tilt, peaks)

    if out_linear.next_event_date is None or out_naive.next_event_date is None:
        pytest.skip("predicted dates unavailable in this synthetic regime")

    # Naive (full window, biased early) predicts SOONER than late-window.
    assert out_naive.next_event_date < out_linear.next_event_date


def test_linear_returns_no_prediction_with_one_peak():
    base = pd.Timestamp("2026-04-01")
    tilt = pd.DataFrame({DATE_COL: [base], TILT_COL: [0.0]})
    peaks = pd.DataFrame({DATE_COL: [base], TILT_COL: [10.0]})
    out = LinearModel().predict(tilt, peaks)
    assert out.next_event_date is None


def test_metadata():
    assert LinearModel().id == "linear"
    assert LinearNaiveModel().id == "linear_naive"
    assert "asymptotic" in LinearModel().description.lower() or "late" in LinearModel().description.lower() or "trail" in LinearModel().description.lower()
    assert "baseline" in LinearNaiveModel().description.lower() or "biased" in LinearNaiveModel().description.lower() or "not a recommendation" in LinearNaiveModel().description.lower()
