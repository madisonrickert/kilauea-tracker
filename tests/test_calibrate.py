"""Tests for `kilauea_tracker.ingest.calibrate`.

These tests run against the committed PNG fixture
`tests/fixtures/UWD-TILT-3month_2026-04-08.png` — a real USGS capture frozen
at a known date. The fixture filename is dated so we can tell when the
upstream layout last drifted.

If USGS changes their plot style (different colors, fonts, canvas size,
legend position), these tests will fail loudly — which is exactly the
"layout regression alarm" we want.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import pytest

from kilauea_tracker.ingest.calibrate import (
    AxisCalibration,
    calibrate_axes,
    detect_plot_bbox,
    ocr_title_timestamps,
    ocr_y_axis_labels,
)
from kilauea_tracker.ingest.exceptions import CalibrationError

FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "UWD-TILT-3month_2026-04-08.png"
)


@pytest.fixture(scope="module")
def fixture_img():
    assert FIXTURE.exists(), f"missing fixture: {FIXTURE}"
    img = cv2.imread(str(FIXTURE))
    assert img is not None
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Plot bbox detection
# ─────────────────────────────────────────────────────────────────────────────


def test_detect_plot_bbox_finds_main_axes(fixture_img):
    h, w = fixture_img.shape[:2]
    bbox = detect_plot_bbox(fixture_img)
    x0, y0, x1, y1 = bbox
    assert 0 < x0 < w / 4, f"plot left edge unexpectedly far right: {x0}"
    assert 0 < y0 < h / 4
    assert 3 * w / 4 < x1 < w
    assert h / 2 < y1 < h
    assert x1 - x0 > w * 0.7
    assert y1 - y0 > h * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Y-axis OCR
# ─────────────────────────────────────────────────────────────────────────────


def test_ocr_y_axis_labels_recovers_at_least_three_ticks(fixture_img):
    bbox = detect_plot_bbox(fixture_img)
    labels = ocr_y_axis_labels(fixture_img, bbox)
    assert len(labels) >= 3, (
        f"expected ≥3 confident y-axis labels, got {len(labels)}: {labels}"
    )

    # Each label is (pixel_y, value)
    pixels = [p for p, _ in labels]
    values = [v for _, v in labels]

    # Pixel positions should be monotonic with the OCR'd values:
    # higher pixel_y == lower numeric value (matplotlib y-axis is inverted).
    sorted_by_pixel = sorted(zip(pixels, values))
    sorted_pixels = [p for p, _ in sorted_by_pixel]
    sorted_values = [v for _, v in sorted_by_pixel]
    # Values must be strictly decreasing as pixel_y increases.
    for i in range(len(sorted_values) - 1):
        assert sorted_values[i] > sorted_values[i + 1], (
            f"y-axis ordering violated at index {i}: "
            f"{sorted_pixels[i]}→{sorted_values[i]} then "
            f"{sorted_pixels[i+1]}→{sorted_values[i+1]}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Title timestamp OCR
# ─────────────────────────────────────────────────────────────────────────────


def test_ocr_title_timestamps_extracts_iso_range(fixture_img):
    bbox = detect_plot_bbox(fixture_img)
    start, end = ocr_title_timestamps(fixture_img, bbox)
    assert isinstance(start, pd.Timestamp)
    assert isinstance(end, pd.Timestamp)
    assert start < end
    span = end - start
    # 3-month plot → ~90 days. Allow 60-120 day window for safety.
    assert pd.Timedelta(days=60) < span < pd.Timedelta(days=120), (
        f"title span outside expected 60-120 days: {span}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────


def test_calibrate_axes_returns_usable_transforms(fixture_img):
    calib = calibrate_axes(fixture_img)
    assert isinstance(calib, AxisCalibration)

    # Sanity 1: invoking the y transform at the top-of-bbox pixel gives a value
    # higher than at the bottom-of-bbox pixel (matplotlib y is inverted).
    x0, y0, x1, y1 = calib.plot_bbox
    top_val = calib.pixel_to_microradians(y0)
    bot_val = calib.pixel_to_microradians(y1)
    assert top_val > bot_val, f"y axis is inverted wrong: top={top_val}, bot={bot_val}"

    # Sanity 2: the implied y-axis range is plausible for tilt data — should
    # span at least 10 µrad and at most a few hundred.
    span = top_val - bot_val
    assert 10 < span < 200, f"implausible y-span: {span}"

    # Sanity 3: invoking the x transform at the bbox left and right gives the
    # title's start/end timestamps respectively (within 1 second).
    x_start_actual = calib.pixel_to_datetime(x0)
    x_end_actual = calib.pixel_to_datetime(x1)
    assert calib.x_range is not None
    assert abs((x_start_actual - calib.x_range[0]).total_seconds()) < 1
    assert abs((x_end_actual - calib.x_range[1]).total_seconds()) < 1

    # Sanity 4: y-fit residuals are small (sub-microradian for these clean PNGs)
    assert calib.fit_residual_per_axis["y_max_residual_microrad"] < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Error paths
# ─────────────────────────────────────────────────────────────────────────────


def test_calibrate_axes_raises_on_garbage_image():
    """A solid white image has no axes — calibration must raise, not crash."""
    import numpy as np
    img = np.full((300, 900, 3), 255, dtype=np.uint8)
    with pytest.raises(CalibrationError):
        calibrate_axes(img)
