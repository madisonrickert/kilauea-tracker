"""Tests for the Huber-robust fit wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from kilauea_tracker.models._huber import (
    HuberFitQuality,
    fit_huber_curve,
    fit_huber_linear,
)


def test_linear_recovers_truth_on_clean_data():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 100.0, 60)
    y_true = 2.0 * x + 5.0
    y = y_true + rng.normal(0, 0.1, len(x))
    m, b, q = fit_huber_linear(x, y)
    assert m == pytest.approx(2.0, abs=0.01)
    assert b == pytest.approx(5.0, abs=0.5)
    assert isinstance(q, HuberFitQuality)
    assert q.n_samples == 60
    # On clean Gaussian data, Huber's auto-tuned f_scale puts the cutoff
    # near 1.6σ — so ~20% of samples land in the linear-influence
    # region. That's expected, not a bug. Mean weight should still be
    # close to 1 because the soft down-weighting is gentle.
    assert q.mean_weight > 0.9


def test_linear_downweights_planted_outliers():
    """Single large outlier must not corrupt the slope."""
    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 100.0, 60)
    y = 2.0 * x + 5.0 + rng.normal(0, 0.1, 60)
    y[10] = 200.0  # planted outlier
    m, _b, q = fit_huber_linear(x, y)
    assert m == pytest.approx(2.0, abs=0.05)
    assert q.fraction_downweighted > 0


def test_linear_raises_on_too_few_samples():
    with pytest.raises(ValueError):
        fit_huber_linear(np.array([1.0]), np.array([1.0]))


def test_curve_recovers_power_law_truth():
    rng = np.random.default_rng(2)
    t = np.linspace(0.0, 200.0, 100)
    a, p, c = 1.5, 0.6, -10.0
    y_true = a * np.power(np.maximum(t, 1e-3), p) + c
    y = y_true + rng.normal(0, 0.3, len(t))

    def power(t, a, p, c):
        return a * np.power(np.maximum(t, 1e-3), p) + c

    params, _, q = fit_huber_curve(power, t, y, p0=[1.0, 0.5, 0.0])
    assert params[0] == pytest.approx(a, rel=0.2)
    assert params[1] == pytest.approx(p, abs=0.1)
    assert params[2] == pytest.approx(c, abs=2.0)
    assert q.n_samples == 100


def test_curve_with_bounds_respects_them():
    """Bounds should clamp the parameter space."""
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 200.0, 100)

    def power(t, a, p, c):
        return a * np.power(np.maximum(t, 1e-3), p) + c

    # Generate data with p=0.4, fit with bounds p ∈ [0.5, 1.0] — must
    # clamp to 0.5.
    y = power(t, 1.0, 0.4, 0.0) + rng.normal(0, 0.2, len(t))
    params, _, _ = fit_huber_curve(
        power,
        t,
        y,
        p0=[1.0, 0.7, 0.0],
        bounds=([0.0, 0.5, -np.inf], [np.inf, 1.0, np.inf]),
    )
    assert 0.5 <= params[1] <= 1.0
