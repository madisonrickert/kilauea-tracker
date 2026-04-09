"""Extract the tilt curve from the plot interior pixels.

USGS plots two series on every chart: blue (azimuth 300°) is the one v1.0
uses; green (azimuth 30°) is auxiliary. We isolate the blue curve via HSV
color masking, then collapse the lit pixels in each column to a single y
value and convert through the calibration to (datetime, microradians).

Why median per column instead of mean: during eruption transitions the curve
plunges nearly vertically, leaving a tall vertical stripe of blue pixels in
one or two columns. The mean of that stripe is meaningless, but the median is
a reasonable midpoint and the downstream curve fit is robust to a few
mid-transition samples (the legacy v1.0 CSV at `legacy/Tiltmeter Data - Sheet1.csv`
shows the same kind of mid-drop noise).
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from ..model import DATE_COL, TILT_COL
from .calibrate import AxisCalibration
from .exceptions import TraceError

# Hue ranges in OpenCV HSV (H ∈ [0, 180]).
# Tuned against `tests/fixtures/UWD-TILT-3month_2026-04-08.png` — pure blue is
# H≈120, pure green is H≈60. Saturation floor of 100 rejects gridlines / text.
BLUE_HUE_MIN = 100
BLUE_HUE_MAX = 130
SATURATION_FLOOR = 100
VALUE_FLOOR = 50

# We need at least this fraction of plot columns to contain the curve before
# we accept the trace; below this threshold something is wrong with the mask.
MIN_COLUMN_COVERAGE = 0.5


def trace_curve(img: np.ndarray, calib: AxisCalibration) -> pd.DataFrame:
    """Extract the blue (Az 300°) tilt curve from a calibrated image.

    Args:
        img:   The full BGR image (as returned by `cv2.imdecode`).
        calib: An `AxisCalibration` produced by `calibrate.calibrate_axes`.

    Returns:
        A DataFrame `[Date, Tilt (microradians)]` sorted by Date with one row
        per plot column where the curve was detected.

    Raises:
        TraceError: if the curve mask is empty or covers too few columns to
                    be plausibly the real time series.
    """
    x0, y0, x1, y1 = calib.plot_bbox
    plot = img[y0:y1, x0:x1]
    if plot.size == 0:
        raise TraceError("plot crop is empty — bbox is degenerate")

    hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)
    mask = (
        (hsv[:, :, 0] >= BLUE_HUE_MIN)
        & (hsv[:, :, 0] <= BLUE_HUE_MAX)
        & (hsv[:, :, 1] >= SATURATION_FLOOR)
        & (hsv[:, :, 2] >= VALUE_FLOOR)
    )

    n_columns = plot.shape[1]
    columns_with_curve = int(np.count_nonzero(mask.any(axis=0)))
    coverage = columns_with_curve / max(1, n_columns)
    if coverage < MIN_COLUMN_COVERAGE:
        raise TraceError(
            f"blue curve detected in only {columns_with_curve}/{n_columns} "
            f"plot columns ({coverage:.0%}); needed ≥{MIN_COLUMN_COVERAGE:.0%}. "
            "USGS may have changed plot colors or the calibration is off."
        )

    rows: list[tuple[pd.Timestamp, float]] = []
    for col_offset in range(n_columns):
        column_mask = mask[:, col_offset]
        if not column_mask.any():
            continue
        # Median row index of lit pixels — robust to mid-transition stripes.
        row_indices = np.where(column_mask)[0]
        median_row_in_crop = float(np.median(row_indices))

        # Convert back to full-image pixel coordinates.
        original_pixel_x = float(col_offset + x0)
        original_pixel_y = median_row_in_crop + y0

        timestamp = calib.pixel_to_datetime(original_pixel_x)
        microrad = calib.pixel_to_microradians(original_pixel_y)

        rows.append((timestamp, microrad))

    if not rows:
        raise TraceError("no curve samples produced — empty after column scan")

    df = pd.DataFrame(rows, columns=[DATE_COL, TILT_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df
