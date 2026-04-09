"""Tests for `kilauea_tracker.peaks.detect_peaks`.

The big one: detect_peaks() must recover the 6 hardcoded v1.0 peaks from
`legacy/eruption_projection.py:50-67` when run against the bootstrap CSV.
That's the proof we can throw away v1.0's hardcoded peak literal entirely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.peaks import detect_peaks

# Same fixture as test_model.py — single source of truth for the v1.0 peaks.
V1_HARDCODED_PEAKS = pd.DataFrame(
    {
        DATE_COL: pd.to_datetime(
            [
                "2025-08-22 07:42:37",
                "2025-09-02 00:47:22",
                "2025-09-18 17:18:56",
                "2025-09-30 17:22:22",
                "2025-10-17 11:02:47",
                "2025-11-09 00:19:32",
            ]
        ),
        TILT_COL: [
            9.471433662,
            11.79226069,
            9.867617108,
            11.29327902,
            9.511201629,
            8.238434164,
        ],
    }
)

BOOTSTRAP_CSV = (
    Path(__file__).resolve().parents[1] / "legacy" / "Tiltmeter Data - Sheet1.csv"
)


@pytest.fixture
def bootstrap_tilt() -> pd.DataFrame:
    df = pd.read_csv(BOOTSTRAP_CSV)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed", dayfirst=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# v1.0 reproduction
# ─────────────────────────────────────────────────────────────────────────────


def test_detects_all_v1_hardcoded_peaks(bootstrap_tilt):
    """For every v1.0 hardcoded peak, detect_peaks must return one within ±12h.

    The detector is allowed to report MORE peaks than v1.0's 6 (e.g. earlier
    episodes from the start of the bootstrap CSV in late 2024). What matters
    is that no v1.0 peak is missed.
    """
    detected = detect_peaks(bootstrap_tilt)
    assert len(detected) > 0, "no peaks detected at all — defaults are too tight"

    tolerance = pd.Timedelta(hours=12)
    detected_dates = detected[DATE_COL].to_numpy()

    for _, expected_peak in V1_HARDCODED_PEAKS.iterrows():
        expected_dt = expected_peak[DATE_COL]
        # Find the nearest detected peak in time.
        deltas = np.abs(detected[DATE_COL] - expected_dt)
        nearest_delta = deltas.min()
        assert nearest_delta <= tolerance, (
            f"v1.0 peak at {expected_dt} not detected within ±12h. "
            f"Closest detected peak is {nearest_delta} away. "
            f"All detected dates: {list(detected_dates)}"
        )


def test_detect_peaks_returns_canonical_schema(bootstrap_tilt):
    """The returned DataFrame must have the schema downstream code expects."""
    detected = detect_peaks(bootstrap_tilt)
    assert list(detected.columns) == [DATE_COL, TILT_COL, "prominence"]
    assert detected[DATE_COL].dtype.kind == "M"  # datetime64
    assert (detected["prominence"] >= 0).all()


def test_detect_peaks_sorted(bootstrap_tilt):
    detected = detect_peaks(bootstrap_tilt)
    dates = detected[DATE_COL].to_numpy()
    assert (dates[1:] >= dates[:-1]).all()


# ─────────────────────────────────────────────────────────────────────────────
# Threshold sensitivity (the Streamlit sidebar wires straight into these args)
# ─────────────────────────────────────────────────────────────────────────────


def test_increasing_prominence_reduces_peak_count(bootstrap_tilt):
    loose = detect_peaks(bootstrap_tilt, min_prominence=2.0)
    tight = detect_peaks(bootstrap_tilt, min_prominence=10.0)
    assert len(tight) <= len(loose)


def test_increasing_distance_reduces_peak_count(bootstrap_tilt):
    short = detect_peaks(bootstrap_tilt, min_distance_days=1.0)
    long = detect_peaks(bootstrap_tilt, min_distance_days=30.0)
    assert len(long) <= len(short)


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_input_returns_empty_peaks():
    empty = pd.DataFrame({DATE_COL: pd.Series(dtype="datetime64[ns]"), TILT_COL: []})
    result = detect_peaks(empty)
    assert len(result) == 0
    assert list(result.columns) == [DATE_COL, TILT_COL, "prominence"]


def test_synthetic_dense_series_finds_planted_peaks():
    """Plant peaks at known positions in a 5-minute-spacing series.

    This guards against the resample-step regression: if the 1-hour resample
    interacts badly with the `distance_samples = days * 24` conversion, the
    detection breaks. Densely-spaced data should still give back the planted
    peaks.
    """
    base = pd.Timestamp("2026-01-01")
    n = 24 * 12 * 60  # 60 days at 12 samples/hour (5-minute spacing)
    times = pd.date_range(base, periods=n, freq="5min")
    # baseline at 2 µrad with 3 prominent humps spaced 20 days apart
    y = 2.0 + np.zeros(n)
    for hump_day in [10, 30, 50]:
        center = hump_day * 24 * 12  # samples
        width = 24 * 12  # ~1 day wide
        idx = np.arange(max(0, center - width), min(n, center + width))
        y[idx] += 8.0 * np.exp(-((idx - center) ** 2) / (2 * (width / 4) ** 2))

    df = pd.DataFrame({DATE_COL: times, TILT_COL: y})
    detected = detect_peaks(df, min_prominence=3.0, min_distance_days=5.0, min_height=5.0)
    assert len(detected) == 3
