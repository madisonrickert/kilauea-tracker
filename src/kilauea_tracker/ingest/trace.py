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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd

from ..config import (
    CURVE_MAX_COLUMN_WIDTH_PIXELS,
    MAX_PHYSICAL_RATE_MICRORAD_PER_HOUR,
    TRACE_OUTLIER_MIN_SAMPLES,
    TRACE_OUTLIER_THRESHOLD_MICRORAD,
    WIDE_COLUMN_THRESHOLD_PIXELS,
)
from ..model import DATE_COL, TILT_COL
from .exceptions import TraceError

if TYPE_CHECKING:
    from .calibrate import AxisCalibration

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

# Legend exclusion: USGS overlays a "UWD Raw Data 300.0 / 30.0" legend in the
# upper-left corner of every plot. The "300.0" entry's color swatch is the
# SAME blue (H≈120) as the Az 300° curve, so without exclusion the HSV mask
# picks up the swatch as 26 phantom samples at fixed pixel positions on every
# capture — they then map to ~26 timestamps near the start of the plot's
# window, all at the same fake tilt value (~13 µrad), polluting the leftmost
# region of every source.
#
# Coordinates are plot-relative (i.e. inside plot_bbox after cropping), in
# (x0, y0, x1, y1) order. The exclusion zone is generously padded to cover
# the entire legend rectangle and not just the swatch — on THREE_MONTH the
# real curve also passes through this region (the user has confirmed this is
# the only source where the curve and legend overlap on the canvas), and
# losing those samples is fine because reconciliation fills them in from
# higher-resolution sources (TWO_DAY/WEEK/MONTH) for any recent dates.
#
# Measured directly against all 5 fixtures in tests/fixtures/ by scanning
# for long horizontal dark pixel runs (legend top/bottom borders) and
# vertical dark runs (left/right borders). Every source has the legend
# in the exact same absolute position (81, 26, 239, 64), which is
# (6, 6, 164, 44) plot-relative given bbox (75, 20, ...).
#
# Earlier constant was (0, 0, 145, 45): over-extended 6px past the
# legend into the top-left plot interior (clipping real curve pixels
# on three_month) AND under-extended 19px on the right (missing the
# legend swatch's right edge on some fetches where OCR rendered the
# swatch a few pixels wider). New zone is tight to the legend rectangle.
LEGEND_EXCLUSION_PLOT_RELATIVE = (6, 6, 164, 44)


@dataclass
class TraceReport:
    """Diagnostics from a `trace_curve` call.

    Attached to the traced DataFrame via `DataFrame.attrs["trace_report"]`
    so callers can surface the numbers without the pure-data columns
    needing a source-tag column.
    """

    rows_raw: int = 0
    rows_after_outlier_filter: int = 0
    outliers_dropped: int = 0
    dropped_rows: list[tuple[pd.Timestamp, float, float]] = field(default_factory=list)
    """Each entry is `(timestamp, raw_tilt, local_median)` for one dropped row."""
    # Phase 1e: columns rejected because they were wider than the expected
    # curve thickness (gridline / axis-label crossings). Counted BEFORE
    # rows are emitted, so they're invisible in `rows_raw`.
    columns_dropped_width: int = 0
    # Phase 1d: per-column samples rejected because their tilt slope vs a
    # time-sorted neighbour exceeded the maximum physical rate. Catches
    # single-column gridline spikes the rolling-median filter can't isolate
    # when the artifact spans enough neighbours to drag the window.
    rows_dropped_rate: int = 0
    # Phase 0a: fraction of plot columns that ended up contributing a
    # sample (before filters). Persisted in the capture-quality CSV to let
    # the post-cutover analysis spot slow drift in trace coverage.
    column_coverage: float = 0.0


def trace_curve(img: np.ndarray, calib: AxisCalibration) -> pd.DataFrame:
    """Extract the blue (Az 300°) tilt curve from a calibrated image.

    Args:
        img:   The full BGR image (as returned by `cv2.imdecode`).
        calib: An `AxisCalibration` produced by `calibrate.calibrate_axes`.

    Returns:
        A DataFrame `[Date, Tilt (microradians)]` sorted by Date with one row
        per plot column where the curve was detected. Diagnostics about the
        outlier filter are attached on `df.attrs["trace_report"]`.

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

    # Zero out the legend region so the blue swatch doesn't get traced as
    # phantom samples. Coordinates are plot-relative because `plot` is the
    # already-cropped interior. Clamp to the actual plot dimensions in case
    # a future calibration produces an unusually small plot bbox.
    lx0, ly0, lx1, ly1 = LEGEND_EXCLUSION_PLOT_RELATIVE
    lx1 = min(lx1, plot.shape[1])
    ly1 = min(ly1, plot.shape[0])
    if lx1 > lx0 and ly1 > ly0:
        mask[ly0:ly1, lx0:lx1] = False

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
    columns_dropped_width = 0
    # Track the last emitted row so we can pick trend-consistent endpoints
    # on wide (near-vertical) columns instead of collapsing to the median.
    # The median-per-column behavior was observably flattening sharp
    # eruption drops on three_month/dec2024_to_now (user reported that
    # the re-traced dots "aren't capturing the sharp downward drop").
    prev_row_in_crop: float | None = None
    for col_offset in range(n_columns):
        column_mask = mask[:, col_offset]
        lit_count = int(np.count_nonzero(column_mask))
        if lit_count == 0:
            continue
        # Phase 1e: reject columns wider than a plausible curve thickness.
        # The real Az-300 blue curve is 1-3 pixels thick; anything ≥ 8
        # pixels is crossing a gridline, axis-tick label, or a JPEG artifact
        # cluster. Dropping the column surrenders the sample at that
        # timestamp; the rolling-median pass downstream can interpolate
        # across the gap, and re-tracing the same PNG on a later run (the
        # sliding window usually still contains it) gets another shot.
        if lit_count > CURVE_MAX_COLUMN_WIDTH_PIXELS:
            columns_dropped_width += 1
            continue
        row_indices = np.where(column_mask)[0]
        if lit_count <= WIDE_COLUMN_THRESHOLD_PIXELS or prev_row_in_crop is None:
            # Narrow column (static or near-static curve) — median row is
            # robust to anti-aliasing and picks up the exact line position.
            row_in_crop = float(np.median(row_indices))
        else:
            # Wide column — the curve crosses vertically in this
            # timestep, lighting a tall stripe. Median of that stripe
            # lands mid-transition (which flattens peaks/troughs),
            # so instead pick the endpoint FARTHER from the previous
            # emitted row. That continues the direction of motion and
            # preserves the real extremum at the end of the transition.
            top = float(row_indices[0])
            bot = float(row_indices[-1])
            row_in_crop = (
                top if abs(top - prev_row_in_crop) > abs(bot - prev_row_in_crop)
                else bot
            )
        prev_row_in_crop = row_in_crop

        # Convert back to full-image pixel coordinates.
        original_pixel_x = float(col_offset + x0)
        original_pixel_y = row_in_crop + y0

        timestamp = calib.pixel_to_datetime(original_pixel_x)
        microrad = calib.pixel_to_microradians(original_pixel_y)

        rows.append((timestamp, microrad))

    if not rows:
        raise TraceError("no curve samples produced — empty after column scan")

    df = pd.DataFrame(rows, columns=[DATE_COL, TILT_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Phase 1d: enforce the physical tilt-rate ceiling. Drops columns whose
    # tilt differs from a neighbour by more than the instrument can plausibly
    # change in the intervening time (see `MAX_PHYSICAL_RATE_MICRORAD_PER_HOUR`).
    # Runs BEFORE the rolling-median filter because a rate-outlier cluster
    # can drag the rolling window enough to hide an individual offender.
    df, rate_dropped_rows = _filter_by_max_physical_rate(df)

    # Drop per-row outliers produced by non-curve blue pixels (gridlines,
    # tick-label bleed, JPEG hue drift near the Az-30° green curve). The
    # 2026-04 archive contamination was caused by rows like these slipping
    # through into the archive on days the higher-priority sources hadn't
    # yet produced overlapping buckets. Real eruption transitions drop
    # gradually across many columns, so they stay close to their rolling
    # median; phantom spikes sit ~10 µrad off it.
    df, report = _filter_rolling_median_outliers(df)
    report.columns_dropped_width = columns_dropped_width
    report.rows_dropped_rate = rate_dropped_rows
    report.column_coverage = coverage
    df.attrs["trace_report"] = report
    return df


def _filter_by_max_physical_rate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows whose tilt change vs the previous sample exceeds the
    configured physical rate ceiling.

    Kīlauea's fastest real DI transitions sit around 5 µrad/hour; the
    threshold (`MAX_PHYSICAL_RATE_MICRORAD_PER_HOUR`) is set at 3× that to
    avoid false positives during legitimate eruption transitions. Anything
    above it is a gridline hit, an axis-label artifact, or a JPEG blue-hue
    spike — never real.

    Returns `(df_filtered, n_dropped)`.
    """
    if len(df) < 2:
        return df, 0
    dates = df[DATE_COL].to_numpy()
    tilts = df[TILT_COL].to_numpy()
    dt_hours = np.diff(dates).astype("timedelta64[s]").astype(float) / 3600.0
    # Avoid div-by-zero: adjacent columns in a wide plot can land in the
    # same second after calibration rounding. Treat 0-hour gaps as 1 min.
    dt_hours = np.where(dt_hours <= 0, 1.0 / 60.0, dt_hours)
    rate = np.abs(np.diff(tilts)) / dt_hours
    # Flag BOTH rows of any pair whose rate is too high; we can't yet tell
    # which of the two is the outlier, but downstream the rolling-median
    # pass will judge each surviving row against its wider context.
    pair_bad = rate > MAX_PHYSICAL_RATE_MICRORAD_PER_HOUR
    if not pair_bad.any():
        return df, 0

    # Row i is "bad" if either adjacent pair involving it violates the rate.
    bad_rows = np.zeros(len(df), dtype=bool)
    bad_rows[:-1] |= pair_bad
    bad_rows[1:] |= pair_bad
    # Don't kill every row — if EVERY row were bad (e.g. a corrupt PNG with
    # all gridlines), the rolling-median filter would take over downstream.
    # Limit to removing at most 20% of rows here; above that, the rate
    # threshold is miscalibrated for the current plot and we defer to
    # rolling-median instead.
    if bad_rows.mean() > 0.20:
        return df, 0

    keep_mask = ~bad_rows
    n_dropped = int(bad_rows.sum())
    return df.loc[keep_mask].reset_index(drop=True), n_dropped


def _filter_rolling_median_outliers(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, TraceReport]:
    """Drop rows whose tilt deviates from a centred rolling median by more
    than `TRACE_OUTLIER_THRESHOLD_MICRORAD`.

    The filter is centred (not trailing) so a spike doesn't shift the
    reference on the rows immediately after it. A 5-sample window balances
    sensitivity (wide enough to resist one-off noise) against locality
    (narrow enough that real eruption transitions don't look like outliers
    relative to their own neighbours).

    Skips the filter entirely for very short inputs where the rolling
    median is dominated by its own padding.
    """
    report = TraceReport(rows_raw=len(df), rows_after_outlier_filter=len(df))
    if len(df) < TRACE_OUTLIER_MIN_SAMPLES:
        return df, report

    tilt = df[TILT_COL].to_numpy()
    # Centered rolling median with a 5-sample window. At the edges pandas'
    # min_periods=1 keeps the window active with whatever samples exist,
    # which is the right behaviour for a USGS plot where the first/last
    # few columns often carry real curve data.
    rolling = (
        pd.Series(tilt)
        .rolling(window=5, center=True, min_periods=3)
        .median()
        .to_numpy()
    )
    delta = np.abs(tilt - rolling)
    keep_mask = delta <= TRACE_OUTLIER_THRESHOLD_MICRORAD
    # Preserve rows where the rolling median couldn't be computed (only
    # happens at the extreme edges when min_periods isn't met).
    keep_mask = keep_mask | np.isnan(rolling)

    n_dropped = int((~keep_mask).sum())
    if n_dropped == 0:
        return df, report

    df.loc[~keep_mask]
    report.dropped_rows = [
        (row[DATE_COL], float(row[TILT_COL]), float(rolling[idx]))
        for idx, (_, row) in enumerate(df.iterrows())
        if not keep_mask[idx]
    ]
    report.outliers_dropped = n_dropped
    report.rows_after_outlier_filter = len(df) - n_dropped

    filtered = df.loc[keep_mask].reset_index(drop=True)
    return filtered, report
