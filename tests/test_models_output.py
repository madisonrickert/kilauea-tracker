"""Tests for the common-contract dataclasses in `models/output.py`."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from kilauea_tracker.models.output import ModelOutput, NamedCurve


def test_named_curve_is_frozen():
    curve = NamedCurve(
        label="trend",
        days=np.array([1.0, 2.0, 3.0]),
        values=np.array([0.1, 0.2, 0.3]),
    )
    assert dataclasses.is_dataclass(curve)
    with pytest.raises(dataclasses.FrozenInstanceError):
        curve.label = "renamed"


def test_named_curve_defaults():
    curve = NamedCurve(
        label="trend",
        days=np.array([1.0, 2.0]),
        values=np.array([0.0, 1.0]),
    )
    assert curve.style == "solid"
    assert curve.color_role == "primary"
    assert curve.band_lo is None
    assert curve.band_hi is None


def test_named_curve_ribbon_carries_bounds():
    days = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 0.5, 1.0])
    hi = np.array([1.0, 1.5, 2.0])
    curve = NamedCurve(
        label="trend ribbon",
        days=days,
        values=(lo + hi) / 2.0,
        color_role="ribbon",
        band_lo=lo,
        band_hi=hi,
    )
    assert curve.color_role == "ribbon"
    np.testing.assert_array_equal(curve.band_lo, lo)
    np.testing.assert_array_equal(curve.band_hi, hi)


def test_model_output_is_frozen():
    out = ModelOutput(
        next_event_date=pd.Timestamp("2026-05-01"),
        confidence_band=(pd.Timestamp("2026-04-29"), pd.Timestamp("2026-05-03")),
        headline_text="median 22d cycle",
        curves=[],
        diagnostics={},
    )
    assert dataclasses.is_dataclass(out)
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.headline_text = "changed"


def test_model_output_with_curves_round_trips():
    days = np.array([1.0, 2.0, 3.0])
    curve = NamedCurve(label="trend", days=days, values=days * 0.5)
    out = ModelOutput(
        next_event_date=None,
        confidence_band=None,
        headline_text=None,
        curves=[curve],
        diagnostics={"slope": 0.5},
    )
    assert out.next_event_date is None
    assert out.confidence_band is None
    assert out.headline_text is None
    assert len(out.curves) == 1
    assert out.curves[0].label == "trend"
    assert out.diagnostics == {"slope": 0.5}
