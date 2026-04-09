"""Tests for `kilauea_tracker.ingest.trace`.

Like test_calibrate.py, these run against the committed fixture
`tests/fixtures/UWD-TILT-3month_2026-04-08.png`. They guard against:
  - the HSV blue range drifting (USGS changing plot colors),
  - the trace producing implausibly few or implausibly many samples,
  - the y/date ranges escaping the calibration's bounds.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from kilauea_tracker.ingest import trace as trace_mod
from kilauea_tracker.ingest.calibrate import calibrate_axes
from kilauea_tracker.ingest.exceptions import TraceError
from kilauea_tracker.ingest.trace import trace_curve
from kilauea_tracker.model import DATE_COL, TILT_COL

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "UWD-TILT-3month_2026-04-08.png"
)


@pytest.fixture(scope="module")
def fixture_img():
    img = cv2.imread(str(FIXTURE))
    assert img is not None
    return img


@pytest.fixture(scope="module")
def fixture_calib(fixture_img):
    return calibrate_axes(fixture_img)


def test_trace_returns_canonical_schema(fixture_img, fixture_calib):
    df = trace_curve(fixture_img, fixture_calib)
    assert list(df.columns) == [DATE_COL, TILT_COL]
    assert df[DATE_COL].dtype.kind == "M"
    assert df[TILT_COL].dtype.kind == "f"


def test_trace_produces_dense_samples(fixture_img, fixture_calib):
    df = trace_curve(fixture_img, fixture_calib)
    x0, _, x1, _ = fixture_calib.plot_bbox
    plot_width = x1 - x0
    # Should hit ≥80% of plot columns on a healthy capture
    assert len(df) >= 0.8 * plot_width


def test_trace_dates_within_calibration_range(fixture_img, fixture_calib):
    df = trace_curve(fixture_img, fixture_calib)
    cal_start, cal_end = fixture_calib.x_range
    assert df[DATE_COL].min() >= cal_start - pd.Timedelta(seconds=1)
    assert df[DATE_COL].max() <= cal_end + pd.Timedelta(seconds=1)


def test_trace_tilts_within_calibration_y_range(fixture_img, fixture_calib):
    df = trace_curve(fixture_img, fixture_calib)
    x0, y0, x1, y1 = fixture_calib.plot_bbox
    top = fixture_calib.pixel_to_microradians(y0)
    bot = fixture_calib.pixel_to_microradians(y1)
    assert df[TILT_COL].min() >= bot - 1.0  # 1 µrad tolerance for line thickness
    assert df[TILT_COL].max() <= top + 1.0


def test_trace_sorted_chronologically(fixture_img, fixture_calib):
    df = trace_curve(fixture_img, fixture_calib)
    dates = df[DATE_COL].to_numpy()
    assert (dates[1:] >= dates[:-1]).all()


def test_trace_reasonable_tilt_dynamics(fixture_img, fixture_calib):
    """The fixture clearly shows multiple eruption cycles between roughly
    -22 and +14 µrad. A working trace must produce a tilt span >20 µrad."""
    df = trace_curve(fixture_img, fixture_calib)
    span = df[TILT_COL].max() - df[TILT_COL].min()
    assert span > 20, f"trace span {span} too small — mask may be off"


def test_trace_raises_on_blank_image(fixture_calib):
    """A solid white image has no curve — must raise TraceError, not silent NaN."""
    blank = np.full((300, 900, 3), 255, dtype=np.uint8)
    with pytest.raises(TraceError):
        trace_curve(blank, fixture_calib)


# ─────────────────────────────────────────────────────────────────────────────
# Calibrate + trace + downstream pipeline integration
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Legend exclusion — guards against the phantom-swatch bug
# ─────────────────────────────────────────────────────────────────────────────


def test_legend_swatch_does_not_produce_phantom_samples(fixture_img, fixture_calib):
    """USGS overlays a 'UWD Raw Data 300.0' legend whose blue color swatch
    sits at a fixed pixel position on every capture. Without exclusion the
    HSV mask picks it up as ~26 phantom samples all at the same y (and
    therefore the same tilt) clustered in the leftmost columns. Lock that
    bug out by checking the leftmost samples don't show the swatch's
    fingerprint: many identical tilt values in a row.
    """
    df = trace_curve(fixture_img, fixture_calib)
    leftmost = df.head(30)[TILT_COL].round(4)
    # If the legend swatch was leaking through, ~26 of the first 30 samples
    # would share the exact same tilt value (the y of the swatch line).
    # Real curve data has at least a few µrad of column-to-column variation
    # in this region of the 3-month plot — no single value should dominate.
    most_common_count = leftmost.value_counts().iloc[0]
    assert most_common_count < 10, (
        f"{most_common_count} of the first 30 samples share an identical tilt — "
        "looks like the legend swatch is leaking through the HSV mask"
    )


def test_legend_exclusion_actually_changes_the_trace(fixture_img, fixture_calib, monkeypatch):
    """Sanity-check the exclusion is doing real work: with it disabled, the
    leftmost samples should look qualitatively different. If this test fails
    while the previous one passes, the exclusion has been removed but
    something else is masking the swatch — investigate before deleting.
    """
    monkeypatch.setattr(trace_mod, "LEGEND_EXCLUSION_PLOT_RELATIVE", (0, 0, 0, 0))
    unmasked = trace_curve(fixture_img, fixture_calib)
    monkeypatch.undo()
    masked = trace_curve(fixture_img, fixture_calib)

    # The unmasked trace should have at least 5 more identical-value samples
    # in its leftmost 30 rows than the masked one.
    unmasked_top = unmasked.head(30)[TILT_COL].round(4).value_counts().iloc[0]
    masked_top = masked.head(30)[TILT_COL].round(4).value_counts().iloc[0]
    assert unmasked_top > masked_top + 4, (
        f"unmasked top-value count ({unmasked_top}) should exceed "
        f"masked ({masked_top}) by at least 5 — exclusion appears inert"
    )


def test_traced_data_is_consumable_by_peak_detection(fixture_img, fixture_calib):
    """Smoke test: a fresh trace must flow through detect_peaks and predict
    without exceptions, even though the fixture is from a different time
    window than v1.0's bootstrap CSV.
    """
    from kilauea_tracker.model import predict
    from kilauea_tracker.peaks import detect_peaks

    df = trace_curve(fixture_img, fixture_calib)
    peaks = detect_peaks(df)
    # Don't assert peak count — the fixture's tilt regime may be different
    # from the bootstrap CSV. Just verify no crash.
    assert isinstance(peaks, pd.DataFrame)
    pred = predict(df, peaks)
    assert pred is not None
    assert pred.fit_diagnostics is not None
