"""Tests for the interval-median baseline model.

The math is lifted from the inlined block in ``model.predict()`` (lines
195-223 in the pre-refactor file). These tests pin its behavior so the
extraction is provably faithful and the model is directly comparable to
``Prediction.interval_based_*`` fields on the same input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.models.interval_median import IntervalMedianModel
from kilauea_tracker.models.output import ModelOutput


def _peaks_at_days(days: list[float], base: pd.Timestamp) -> pd.DataFrame:
    """Build a peaks DataFrame from days-from-base offsets (tilt fixed at 10)."""
    return pd.DataFrame(
        {
            DATE_COL: [base + pd.Timedelta(days=d) for d in days],
            TILT_COL: [10.0] * len(days),
        }
    )


def test_predict_returns_model_output():
    model = IntervalMedianModel()
    base = pd.Timestamp("2026-01-01")
    peaks = _peaks_at_days([0, 22, 44, 66], base=base)
    tilt = pd.DataFrame({DATE_COL: [base], TILT_COL: [0.0]})

    out = model.predict(tilt, peaks)
    assert isinstance(out, ModelOutput)


def test_evenly_spaced_peaks_predict_one_more_interval():
    """Four peaks 22 days apart → next prediction = last peak + 22 days."""
    model = IntervalMedianModel()
    base = pd.Timestamp("2026-01-01")
    peaks = _peaks_at_days([0, 22, 44, 66], base=base)
    tilt = pd.DataFrame({DATE_COL: [base], TILT_COL: [0.0]})

    out = model.predict(tilt, peaks)
    assert out.next_event_date == base + pd.Timedelta(days=66 + 22)


def test_diagnostics_include_median_and_mean():
    model = IntervalMedianModel()
    base = pd.Timestamp("2026-01-01")
    peaks = _peaks_at_days([0, 20, 40, 60], base=base)
    tilt = pd.DataFrame({DATE_COL: [base], TILT_COL: [0.0]})

    out = model.predict(tilt, peaks)
    assert out.diagnostics["median_peak_interval_days"] == 20.0
    assert out.diagnostics["mean_peak_interval_days"] == 20.0


def test_band_uses_iqr_when_at_least_three_intervals():
    """Bands at 25/75 percentile of intervals."""
    model = IntervalMedianModel()
    base = pd.Timestamp("2026-01-01")
    # intervals: 10, 20, 30, 40 → median 25, q25=17.5, q75=32.5
    peaks = _peaks_at_days([0, 10, 30, 60, 100], base=base)
    tilt = pd.DataFrame({DATE_COL: [base], TILT_COL: [0.0]})

    out = model.predict(tilt, peaks)
    assert out.confidence_band is not None
    lo, hi = out.confidence_band
    last_peak = base + pd.Timedelta(days=100)
    assert lo == last_peak + pd.Timedelta(days=17.5)
    assert hi == last_peak + pd.Timedelta(days=32.5)


def test_two_peaks_yield_point_estimate_no_band():
    """One interval is enough for a point estimate; need 3+ for the band."""
    model = IntervalMedianModel()
    base = pd.Timestamp("2026-01-01")
    peaks = _peaks_at_days([0, 22], base=base)
    tilt = pd.DataFrame({DATE_COL: [base], TILT_COL: [0.0]})

    out = model.predict(tilt, peaks)
    assert out.next_event_date == base + pd.Timedelta(days=44)
    assert out.confidence_band is None


def test_too_few_peaks_returns_empty_output():
    """One peak → cannot compute any interval. Empty output, no exception."""
    model = IntervalMedianModel()
    base = pd.Timestamp("2026-01-01")
    peaks = _peaks_at_days([0], base=base)
    tilt = pd.DataFrame({DATE_COL: [base], TILT_COL: [0.0]})

    out = model.predict(tilt, peaks)
    assert out.next_event_date is None
    assert out.confidence_band is None
    assert out.curves == []


def test_zero_peaks_returns_empty_output():
    model = IntervalMedianModel()
    base = pd.Timestamp("2026-01-01")
    peaks = pd.DataFrame({DATE_COL: pd.to_datetime([]), TILT_COL: []})
    tilt = pd.DataFrame({DATE_COL: [base], TILT_COL: [0.0]})

    out = model.predict(tilt, peaks)
    assert out.next_event_date is None
    assert out.confidence_band is None


def test_no_curves_returned():
    """Interval-median is a point+band model — no overlay curves."""
    model = IntervalMedianModel()
    base = pd.Timestamp("2026-01-01")
    peaks = _peaks_at_days([0, 22, 44, 66, 88], base=base)
    tilt = pd.DataFrame({DATE_COL: [base], TILT_COL: [0.0]})

    out = model.predict(tilt, peaks)
    assert out.curves == []


def test_headline_text_mentions_median_cycle_days():
    model = IntervalMedianModel()
    base = pd.Timestamp("2026-01-01")
    peaks = _peaks_at_days([0, 20, 40, 60], base=base)
    tilt = pd.DataFrame({DATE_COL: [base], TILT_COL: [0.0]})

    out = model.predict(tilt, peaks)
    assert out.headline_text is not None
    assert "20" in out.headline_text  # median is 20 days


def test_model_metadata():
    model = IntervalMedianModel()
    assert model.id == "interval_median"
    assert model.label
    assert model.description


def test_parity_with_legacy_inlined_baseline():
    """The new model returns the same numbers as the inlined baseline in
    ``model.predict()`` for the same peak set. This is the migration
    safety net — if these diverge, the extraction was lossy.
    """
    from kilauea_tracker.model import predict as legacy_predict

    base = pd.Timestamp("2026-01-01")
    days = [0.0, 11.5, 22.7, 33.4, 47.0, 60.2, 71.5]
    peaks = pd.DataFrame(
        {
            DATE_COL: [base + pd.Timedelta(days=d) for d in days],
            TILT_COL: [10.0] * len(days),
        }
    )
    tilt = pd.DataFrame(
        {
            DATE_COL: [base + pd.Timedelta(days=d) for d in np.linspace(0, 80, 200)],
            TILT_COL: np.linspace(-2.0, 9.0, 200),
        }
    )

    legacy = legacy_predict(tilt, peaks)
    new = IntervalMedianModel().predict(tilt, peaks)

    assert new.next_event_date == legacy.interval_based_next_event_date
    assert new.confidence_band == legacy.interval_based_band
    assert (
        new.diagnostics["median_peak_interval_days"]
        == legacy.median_peak_interval_days
    )
