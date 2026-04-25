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
    _recover_ocr_year_misread,
    _try_parse_title_at_psm,
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
    sorted_by_pixel = sorted(zip(pixels, values, strict=False))
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
    start, end, psm_used, raw_text = ocr_title_timestamps(fixture_img, bbox)
    assert isinstance(start, pd.Timestamp)
    assert isinstance(end, pd.Timestamp)
    assert start < end
    span = end - start
    # 3-month plot → ~90 days. Allow 60-120 day window for safety.
    assert pd.Timedelta(days=60) < span < pd.Timedelta(days=120), (
        f"title span outside expected 60-120 days: {span}"
    )
    # PSM diagnostics round-trip for post-hoc debugging.
    assert psm_used in ("psm7", "psm6")
    assert raw_text  # non-empty


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


# ─────────────────────────────────────────────────────────────────────────────
# OCR year recovery — guards against single-digit year misreads in the
# title timestamp OCR. The dominant failure mode in 2026-04 was a 2 → 0
# flip in one of the year digits, producing a non-chronological range.
# ─────────────────────────────────────────────────────────────────────────────


def test_recover_ocr_year_misread_fixes_real_2026_to_2006_failure():
    """The exact failure observed on prod on 2026-04-09: the week PNG's
    title OCR returned end_year=2006 instead of 2026. Recovery should
    substitute end.year = start.year and produce a valid 6-day range.
    """
    start = pd.Timestamp("2026-04-03 02:51:26")
    end = pd.Timestamp("2006-04-09 02:51:26")
    out_start, out_end = _recover_ocr_year_misread(start, end)
    assert out_start == start
    assert out_end == pd.Timestamp("2026-04-09 02:51:26")
    assert out_end - out_start == pd.Timedelta(days=6)


def test_recover_ocr_year_misread_fixes_start_year_misread():
    """Less common but possible: start's year is the OCR victim."""
    start = pd.Timestamp("2006-04-03 02:51:26")
    end = pd.Timestamp("2026-04-09 02:51:26")
    out_start, out_end = _recover_ocr_year_misread(start, end)
    # When start.year < end.year - 1 the recovery substitutes start.year = end.year.
    assert out_start == pd.Timestamp("2026-04-03 02:51:26")
    assert out_end == end


def test_recover_ocr_year_misread_passes_through_chronological_input():
    """No recovery needed when the range is already valid — return verbatim."""
    start = pd.Timestamp("2026-04-03 00:00:00")
    end = pd.Timestamp("2026-04-09 00:00:00")
    out_start, out_end = _recover_ocr_year_misread(start, end)
    assert out_start == start
    assert out_end == end


def test_recover_ocr_year_misread_passes_through_new_year_boundary():
    """A start/end spanning a new-year boundary differs by exactly 1 year and
    is chronologically valid — must NOT be flagged as an OCR misread.
    """
    start = pd.Timestamp("2025-12-28 00:00:00")
    end = pd.Timestamp("2026-01-04 00:00:00")
    out_start, out_end = _recover_ocr_year_misread(start, end)
    assert (out_start, out_end) == (start, end)


def test_recover_ocr_year_misread_returns_unchanged_when_recovery_doesnt_help():
    """If substituting the year doesn't yield a chronological range (e.g.
    the wrong digit was somewhere else entirely), return the original
    inputs unchanged so the caller raises a proper CalibrationError.
    """
    # Both within 2026 but end is BEFORE start (impossible OCR state where
    # the day was misread, not the year). Recovery shouldn't touch it.
    start = pd.Timestamp("2026-04-09 00:00:00")
    end = pd.Timestamp("2026-04-03 00:00:00")
    out_start, out_end = _recover_ocr_year_misread(start, end)
    assert (out_start, out_end) == (start, end)


def test_try_parse_title_at_psm_returns_error_on_invalid_day():
    """The 2026-04-09 prod failure was a different OCR misread: PSM 7
    produced a date with day=30 in February (or similar), which makes
    `pd.Timestamp(year=y, month=mo, day=d)` raise "day is out of range
    for month". The new helper must report this as a clean error string
    instead of letting the exception propagate up — that's what enables
    the PSM 7 → PSM 6 fallback in `ocr_title_timestamps` to kick in.
    """
    import numpy as np
    # We can't easily make Tesseract emit a specific bad string, but we
    # can build a fake OCR-output image of pure white and verify the
    # helper returns (None, raw_text, error_str) instead of raising.
    fake_strip = np.full((50, 400, 3), 255, dtype=np.uint8)
    result, _text, err = _try_parse_title_at_psm(fake_strip, "--psm 7")
    assert result is None
    assert err  # non-empty error message
    # Pure white → either no regex match or unparseable; both are
    # legitimate "PSM 7 failed, try PSM 6" outcomes.
    assert (
        "regex did not match" in err
        or "unparseable timestamp" in err
        or "non-chronological" in err
    )


def test_recover_ocr_year_misread_handles_feb_29_safely():
    """`pd.Timestamp.replace(year=...)` raises on Feb 29 in non-leap years.
    Recovery must catch that and return inputs unchanged rather than
    crashing — degraded output is fine; a hard crash inside calibrate is not.
    """
    # 2024 is a leap year; 2025 is not. Feb 29 2024 cannot be replaced
    # with year=2025 because Feb 29 2025 doesn't exist.
    start = pd.Timestamp("2025-02-15 00:00:00")
    end = pd.Timestamp("2024-02-29 00:00:00")
    # Should not raise; should return unchanged so the caller errors cleanly.
    out_start, out_end = _recover_ocr_year_misread(start, end)
    assert (out_start, out_end) == (start, end)


# ─────────────────────────────────────────────────────────────────────────────
# Lenient timestamp parser — guards against USGS publishing field overflows
# ─────────────────────────────────────────────────────────────────────────────


def test_lenient_timestamp_parser_handles_seconds_60():
    """USGS occasionally publishes titles with seconds=60. Real bug seen on
    2026-04-09: title contained '2026-04-09 23:60:09', which `pd.Timestamp`
    rejects strictly. The lenient parser should roll over to the next minute.

    Also: USGS plot titles are labeled HST so the parser converts to UTC.
    23:60:09 HST → 24:00:09 HST → 00:00:09 next day HST → 10:00:09 next day UTC.
    """
    from kilauea_tracker.ingest.calibrate import _parse_lenient_timestamp

    assert _parse_lenient_timestamp("2026-04-09 23:60:09") == pd.Timestamp(
        "2026-04-10 10:00:09"
    )


def test_lenient_timestamp_parser_converts_hst_to_utc():
    """USGS plots embed Pacific/Honolulu Time (HST = UTC-10). The parser
    converts to UTC so the rest of the pipeline (which treats naive
    timestamps as UTC) lands at the correct absolute moment.
    """
    from kilauea_tracker.ingest.calibrate import _parse_lenient_timestamp

    # 12:34:56 HST = 22:34:56 UTC same day
    assert _parse_lenient_timestamp("2026-04-09 12:34:56") == pd.Timestamp(
        "2026-04-09 22:34:56"
    )
    # 11:00:23 HST (typical 2-day right-edge) = 21:00:23 UTC same day
    assert _parse_lenient_timestamp("2026-04-09 11:00:23") == pd.Timestamp(
        "2026-04-09 21:00:23"
    )


def test_lenient_timestamp_parser_rejects_unparseable():
    from kilauea_tracker.ingest.calibrate import _parse_lenient_timestamp

    with pytest.raises(ValueError):
        _parse_lenient_timestamp("not a timestamp")
    with pytest.raises(ValueError):
        _parse_lenient_timestamp("2026-04-09")  # missing time
