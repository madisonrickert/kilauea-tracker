"""Tests for the trendline×exp intersection model.

The migration safety-net is the parity test: this model returns the same
``next_event_date``, ``confidence_band``, and per-curve evaluations as
the legacy ``model.predict()`` did before the refactor. If those diverge,
the extraction was lossy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.model import predict as legacy_predict
from kilauea_tracker.models.output import ModelOutput
from kilauea_tracker.models.trendline_exp import TrendlineExpModel

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


def _bootstrap_tilt() -> pd.DataFrame:
    df = pd.read_csv(BOOTSTRAP_CSV)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed", dayfirst=False)
    return df


def test_predict_returns_model_output():
    out = TrendlineExpModel().predict(_bootstrap_tilt(), V1_HARDCODED_PEAKS)
    assert isinstance(out, ModelOutput)


def test_metadata():
    model = TrendlineExpModel()
    assert model.id == "trendline_exp"
    assert model.label
    assert model.description


def test_parity_with_legacy_next_event_date():
    """Same predicted date as the pre-refactor inline math."""
    tilt = _bootstrap_tilt()
    legacy = legacy_predict(tilt, V1_HARDCODED_PEAKS)
    new = TrendlineExpModel().predict(tilt, V1_HARDCODED_PEAKS)
    assert new.next_event_date == legacy.next_event_date


def test_parity_with_legacy_confidence_band():
    tilt = _bootstrap_tilt()
    legacy = legacy_predict(tilt, V1_HARDCODED_PEAKS)
    new = TrendlineExpModel().predict(tilt, V1_HARDCODED_PEAKS)
    assert new.confidence_band == legacy.confidence_band


def test_curves_include_trendline_and_exp():
    """The trendline + exp curves are surfaced as NamedCurves the chart
    can render. Both must be present and labelled distinctly."""
    out = TrendlineExpModel().predict(_bootstrap_tilt(), V1_HARDCODED_PEAKS)
    labels = [c.label for c in out.curves]
    # Must include both a trendline-like curve and an exp/episode-like curve.
    assert any("trendline" in label.lower() for label in labels)
    assert any(
        "episode" in label.lower() or "exp" in label.lower() for label in labels
    )


def test_curves_include_ribbons_when_band_available():
    """When the Monte Carlo bands fit, both ribbons should appear with
    band_lo / band_hi populated."""
    out = TrendlineExpModel().predict(_bootstrap_tilt(), V1_HARDCODED_PEAKS)
    ribbons = [c for c in out.curves if c.color_role == "ribbon"]
    # The legacy model produces both trendline and exp ribbons under the
    # bootstrap fixture; assert both are present and well-formed.
    assert len(ribbons) == 2
    for r in ribbons:
        assert r.band_lo is not None
        assert r.band_hi is not None
        assert r.band_lo.shape == r.band_hi.shape
        assert r.band_lo.shape == r.values.shape


def test_curve_evaluations_match_legacy_curves():
    """Sample each curve at its own x-grid and compare to the legacy
    ``Curve.f``-evaluated values. They must be numerically identical."""
    tilt = _bootstrap_tilt()
    legacy = legacy_predict(tilt, V1_HARDCODED_PEAKS)
    new = TrendlineExpModel().predict(tilt, V1_HARDCODED_PEAKS)

    # Find the line curves (not ribbons) by color role and label.
    new_trend = next(
        c for c in new.curves if "trendline" in c.label.lower() and c.color_role == "primary"
    )
    new_exp = next(
        c for c in new.curves if "episode" in c.label.lower() and c.color_role == "primary"
    )

    # Legacy trendline.f / exp_curve.f sampled at the same x-grid.
    legacy_trend_y = np.asarray(legacy.trendline.f(new_trend.days), dtype=float)
    legacy_exp_y = np.asarray(legacy.exp_curve.f(new_exp.days), dtype=float)
    np.testing.assert_allclose(new_trend.values, legacy_trend_y, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(new_exp.values, legacy_exp_y, rtol=1e-9, atol=1e-9)


def test_too_few_peaks_returns_empty_output():
    """One peak → no trendline possible. Empty output, no exception."""
    base = pd.Timestamp("2026-01-01")
    peaks = pd.DataFrame({DATE_COL: [base], TILT_COL: [10.0]})
    tilt = pd.DataFrame({DATE_COL: [base], TILT_COL: [0.0]})

    out = TrendlineExpModel().predict(tilt, peaks)
    assert out.next_event_date is None
    assert out.confidence_band is None
    assert out.curves == []
