"""Re-calibrate the USGS tilt plot's axes from the raw PNG.

Why this exists: USGS publishes auto-updating PNG plots whose y-axis range
shifts between updates (the user has confirmed this with HVO directly). We
cannot persist a single calibration — every fetch must re-derive the pixel ↔
data transform from the labels visible in the current image.

Strategy:
  1. Detect the plot bounding box geometrically (cv2.findContours over a
     thresholded image — the rectangular axis frame is the largest dark
     rectangle in the figure).
  2. Crop the strip just left of the bounding box and OCR it for y-axis
     numeric labels (`pytesseract.image_to_data` with PSM 11 + a digits-only
     whitelist returns confident token bboxes).
  3. Crop the strip just below the bounding box and OCR it for the title
     line, which contains the full ISO timestamp range
     (`Pacific/Honolulu Time (YYYY-MM-DD HH:MM:SS to YYYY-MM-DD HH:MM:SS)`).
     Regex extracts the start/end timestamps — far more reliable than OCRing
     the abbreviated MM/DD x-axis tick labels.
  4. Fit linear pixel ↔ data transforms for both axes.

The OCR'd y-axis labels are intentionally tolerated to be incomplete: if 3 of
6 visible labels are recognized with high confidence, we can still fit a
clean linear y-axis transform. Tesseract on these tiny PNGs typically gets the
lower labels (-10, -20, -30) more reliably than the upper ones, but it varies
across captures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
import pytesseract

from .exceptions import CalibrationError

# OCR upscale factor — these PNGs are tiny (~900×300) so Tesseract benefits
# from cubic resampling before recognition.
OCR_UPSCALE = 4

# Minimum number of recognized y-axis numeric labels to accept a calibration.
MIN_Y_LABELS_REQUIRED = 3

# Tesseract confidence floor; the values come back as ints in [0, 100].
MIN_OCR_CONFIDENCE = 70

# Title strip search regex. Tolerates the common Tesseract slip "to" → "t0".
TITLE_TIMESTAMP_RE = re.compile(
    r"\(\s*"
    r"(?P<start>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
    r"\s+t[o0]\s+"
    r"(?P<end>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
    r"\s*\)"
)


@dataclass
class AxisCalibration:
    """Calibrated pixel ↔ data transforms for a single PNG.

    Stored as plain numeric coefficients (not closures) so the dataclass is
    picklable — required for `st.cache_data` to memoize ingest results.
    The transform methods reconstruct the linear math from those coefficients
    on every call.
    """

    plot_bbox: tuple[int, int, int, int]   # (x0, y0, x1, y1)
    y_slope: float                          # microradians per pixel (typically negative)
    y_intercept: float                      # microradians at pixel y = 0
    x_start: pd.Timestamp                   # datetime at plot_bbox left edge
    x_end: pd.Timestamp                     # datetime at plot_bbox right edge
    y_labels_found: list[tuple[int, float]] = field(default_factory=list)
    fit_residual_per_axis: dict[str, float] = field(default_factory=dict)

    def pixel_to_microradians(self, py: float) -> float:
        return float(self.y_slope * float(py) + self.y_intercept)

    def pixel_to_datetime(self, px: float) -> pd.Timestamp:
        x0, _, x1, _ = self.plot_bbox
        span_seconds = (self.x_end - self.x_start).total_seconds()
        px_span = max(1.0, float(x1 - x0))
        offset_seconds = (float(px) - x0) * span_seconds / px_span
        return self.x_start + pd.Timedelta(seconds=offset_seconds)

    @property
    def x_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (self.x_start, self.x_end)

    def microradians_per_pixel(self) -> float:
        """How many µrad each vertical pixel represents (positive number)."""
        x0, y0, x1, y1 = self.plot_bbox
        top_value = self.pixel_to_microradians(y0)
        bot_value = self.pixel_to_microradians(y1)
        return abs(top_value - bot_value) / max(1, (y1 - y0))


# ─────────────────────────────────────────────────────────────────────────────
# Plot bbox detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_plot_bbox(img: np.ndarray) -> tuple[int, int, int, int]:
    """Locate the rectangular plot region inside the figure.

    The strategy: threshold to dark pixels, find contours, return the largest
    contour whose bounding box is near-axis-aligned and covers a substantial
    fraction of the image.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape[:2]
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int, int]] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 0.5 * w or ch < 0.4 * h:
            continue
        # Reject contours that touch the very edges (they're frame artifacts).
        if x == 0 or y == 0 or x + cw == w or y + ch == h:
            continue
        candidates.append((x, y, cw, ch, cw * ch))

    if not candidates:
        raise CalibrationError(
            f"could not locate plot region in {w}×{h} image — no contour "
            "covered ≥50% width and ≥40% height"
        )

    candidates.sort(key=lambda r: -r[4])
    x, y, cw, ch, _ = candidates[0]
    return (x, y, x + cw, y + ch)


# ─────────────────────────────────────────────────────────────────────────────
# Y-axis OCR
# ─────────────────────────────────────────────────────────────────────────────


def ocr_y_axis_labels(
    img: np.ndarray,
    plot_bbox: tuple[int, int, int, int],
) -> list[tuple[int, float]]:
    """Return `[(pixel_y_in_original_image, numeric_value), ...]` for each
    confident y-axis label found by Tesseract.
    """
    x0, y0, x1, y1 = plot_bbox

    # Strip just left of the plot, with a small vertical bleed so we don't
    # clip labels whose centers are near the bbox edges.
    strip = img[max(0, y0 - 5) : y1 + 5, 0:x0]
    if strip.size == 0:
        return []
    strip_top_in_original = max(0, y0 - 5)

    upscaled = cv2.resize(
        strip,
        None,
        fx=OCR_UPSCALE,
        fy=OCR_UPSCALE,
        interpolation=cv2.INTER_CUBIC,
    )

    # Whitelist must include `.` because shorter-window plots (week, 2-day) use
    # fractional y-axis labels (1.5, 0.5, -0.5, -1.5, …) — without the period
    # Tesseract drops the decimal and we read "1.5" as "15".
    data = pytesseract.image_to_data(
        upscaled,
        config="--psm 11 -c tessedit_char_whitelist=-.0123456789",
        output_type=pytesseract.Output.DICT,
    )

    found: list[tuple[int, float]] = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if conf < MIN_OCR_CONFIDENCE:
            continue

        # Convert the upscaled bbox center back to original-image coordinates.
        upscaled_center_y = data["top"][i] + data["height"][i] / 2.0
        original_y = (upscaled_center_y / OCR_UPSCALE) + strip_top_in_original
        found.append((int(round(original_y)), value))

    # Deduplicate by value (occasionally Tesseract finds the same label twice).
    seen: dict[float, int] = {}
    for py, val in found:
        if val not in seen:
            seen[val] = py
    return sorted([(py, val) for val, py in seen.items()], key=lambda t: t[0])


# ─────────────────────────────────────────────────────────────────────────────
# Title timestamp OCR
# ─────────────────────────────────────────────────────────────────────────────


def ocr_title_timestamps(
    img: np.ndarray,
    plot_bbox: tuple[int, int, int, int],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Extract the (start, end) ISO timestamps from the title line below the plot.

    The USGS plots embed the full date range as text:
        "Pacific/Honolulu Time (2026-01-08 15:42:31 to 2026-04-08 15:42:31)"

    OCRing this with PSM 7 produces a near-perfect string that a regex
    cleanly recovers. The only common error is "to" → "t0", which the
    regex tolerates.
    """
    h, w = img.shape[:2]
    _, _, _, y1 = plot_bbox

    title_strip = img[y1 + 5 : h, 0:w]
    if title_strip.size == 0:
        raise CalibrationError("title strip is empty — plot bbox extends to image bottom")

    upscaled = cv2.resize(
        title_strip,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC,
    )

    text = pytesseract.image_to_string(upscaled, config="--psm 7")
    match = TITLE_TIMESTAMP_RE.search(text)
    if match is None:
        # Try PSM 6 as a fallback — sometimes the date region is on a separate
        # line from the tick labels and PSM 7 (single line) can't see it.
        text = pytesseract.image_to_string(upscaled, config="--psm 6")
        match = TITLE_TIMESTAMP_RE.search(text)

    if match is None:
        raise CalibrationError(
            "could not parse timestamp range from title strip; "
            f"OCR returned: {text!r}"
        )

    try:
        start = _parse_lenient_timestamp(match.group("start"))
        end = _parse_lenient_timestamp(match.group("end"))
    except (ValueError, TypeError) as e:
        raise CalibrationError(f"unparseable timestamp in title: {e}") from e

    if end <= start:
        raise CalibrationError(
            f"title timestamps are not chronological: {start} → {end}"
        )

    return start, end


# Lenient timestamp parser for the title regex output. USGS occasionally
# publishes titles with field overflows like `23:60:09` (which `pd.Timestamp`
# strictly rejects with "second must be in 0..59") because their plotting
# code rounds without rolling over. We handle the overflow by parsing the
# fields manually and adding via `pd.Timedelta`, which carries naturally.
_LENIENT_TS_RE = re.compile(
    r"^(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})$"
)

# USGS plot titles say "Pacific/Honolulu Time (...)" — the embedded
# timestamps are HST, not UTC. We localize to HST then convert to UTC so
# every traced sample lands at the correct absolute moment in time. The
# rest of the codebase treats naive timestamps as UTC, so we strip the
# tz info on the way out.
_USGS_PLOT_TIMEZONE = "Pacific/Honolulu"


def _parse_lenient_timestamp(s: str) -> pd.Timestamp:
    """Parse `YYYY-MM-DD HH:MM:SS` from a USGS plot title and convert
    HST→UTC, returning a naive UTC timestamp.

    Tolerates field overflows like seconds=60 because USGS's plotting code
    rounds without rolling over (e.g. ``23:60:09``). Handled by parsing the
    fields manually and adding via ``pd.Timedelta``.

    Examples:
      "2026-04-09 11:00:23" (HST) → 2026-04-09 21:00:23 (UTC, returned naive)
      "2026-04-09 23:60:09" (HST) → 2026-04-10 09:00:09 (UTC, returned naive)
    """
    m = _LENIENT_TS_RE.match(s.strip())
    if not m:
        raise ValueError(f"unparseable timestamp format: {s!r}")
    y, mo, d, h, mi, sec = (int(x) for x in m.groups())
    base = pd.Timestamp(year=y, month=mo, day=d)
    naive_local = base + pd.Timedelta(hours=h, minutes=mi, seconds=sec)
    # Localize to HST, convert to UTC, drop the tz so callers can keep
    # treating naive timestamps as UTC throughout the pipeline.
    aware_hst = naive_local.tz_localize(_USGS_PLOT_TIMEZONE)
    return aware_hst.tz_convert("UTC").tz_localize(None)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestration
# ─────────────────────────────────────────────────────────────────────────────


def calibrate_axes(img: np.ndarray) -> AxisCalibration:
    """Run the full calibration pipeline on a freshly-decoded image.

    Raises `CalibrationError` if any of the steps fail their sanity checks.
    """
    plot_bbox = detect_plot_bbox(img)

    # ── y-axis ──────────────────────────────────────────────────────────────
    y_labels = ocr_y_axis_labels(img, plot_bbox)
    if len(y_labels) < MIN_Y_LABELS_REQUIRED:
        raise CalibrationError(
            f"y-axis OCR found only {len(y_labels)} label(s) "
            f"(need ≥{MIN_Y_LABELS_REQUIRED}): {y_labels}"
        )

    pixel_ys = np.array([p for p, _ in y_labels], dtype=float)
    values = np.array([v for _, v in y_labels], dtype=float)
    # Linear fit: value = a * pixel_y + b. (Note pixel y is inverted vs data.)
    a_y, b_y = np.polyfit(pixel_ys, values, 1)
    y_residuals = values - (a_y * pixel_ys + b_y)
    y_resid_max = float(np.abs(y_residuals).max())

    # ── x-axis ──────────────────────────────────────────────────────────────
    x_start, x_end = ocr_title_timestamps(img, plot_bbox)

    return AxisCalibration(
        plot_bbox=plot_bbox,
        y_slope=float(a_y),
        y_intercept=float(b_y),
        x_start=x_start,
        x_end=x_end,
        y_labels_found=y_labels,
        fit_residual_per_axis={"y_max_residual_microrad": y_resid_max},
    )
