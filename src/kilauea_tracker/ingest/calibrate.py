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

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import pytesseract

from ..config import (
    ANCHOR_FIT_A_WARNING_FRACTION,
    ANCHOR_FIT_B_WARNING_MICRORAD,
    ANCHOR_FIT_MIN_OVERLAP_BUCKETS,
    ANCHOR_FIT_TRIM_HOURS,
    DATA_DIR,
    X_END_MAX_AGE_HOURS,
    X_WINDOW_EXPECTED_HOURS,
    X_WINDOW_TOLERANCE_HOURS,
    Y_CALIBRATION_MAX_RESIDUAL_MICRORAD,
    Y_SLOPE_HISTORY_LENGTH,
    Y_SLOPE_REGRESSION_TOLERANCE_PERCENT,
)
from ..model import DATE_COL, TILT_COL
from .exceptions import CalibrationError

# Where the per-source rolling history of fitted y-slopes lives. Each entry
# is appended after a successful calibration and consulted by the next one
# to detect sudden slope drift (symptom of a mis-OCR'd label).
Y_SLOPE_HISTORY_FILE = DATA_DIR / "y_slope_history.json"

# OCR upscale factor — these PNGs are tiny (~900×300) so Tesseract benefits
# from cubic resampling before recognition.
OCR_UPSCALE = 4

# Minimum number of recognized y-axis numeric labels to accept a calibration.
MIN_Y_LABELS_REQUIRED = 3

# Tesseract confidence floor; the values come back as ints in [0, 100].
MIN_OCR_CONFIDENCE = 70

logger = logging.getLogger(__name__)

# Title strip search regex. Tolerates the common Tesseract slip "to" → "t0".
TITLE_TIMESTAMP_RE = re.compile(
    r"\(\s*"
    r"(?P<start>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
    r"\s+t[o0]\s+"
    r"(?P<end>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})"
    r"\s*\)"
)

# X-axis tick label regexes per source. USGS uses a different format on
# each plot — rolling sources with shorter windows show `MM/DD` or
# `HH:MM`, and the long dec2024_to_now plot shows `MMM-YYYY`.
# Validating each detected tick against these tells us whether the
# time-range OCR's linear interpolation between x_start and x_end
# lands on the same pixel where USGS drew the tick's label.
_MMDD_RE = re.compile(r"^(?P<m>\d{2})[/\-.](?P<d>\d{2})$")
_HHMM_RE = re.compile(r"^(?P<h>\d{2})[:.](?P<mi>\d{2})$")
_MONTH_ABBREV_RE = re.compile(
    r"^(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[-\s](?P<y>\d{4})$",
    re.IGNORECASE,
)
_MONTH_ABBREV_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
X_TICK_FORMAT: dict[str, str] = {
    "two_day": "HH:MM",
    "week": "MM/DD",
    "month": "MM/DD",
    "three_month": "MM/DD",
    "dec2024_to_now": "MMM-YYYY",
}


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
    # Title OCR diagnostics — which PSM mode succeeded ("psm7" or "psm6"),
    # and the raw text Tesseract returned for that mode. Empty strings
    # when the caller didn't capture the title (legacy paths).
    title_psm_used: str = ""
    title_raw_text: str = ""
    # Phase 1a diagnostics: per-label Tesseract confidences used in the
    # weighted polyfit, and whether the slope was taken from history
    # fallback instead of this run's labels.
    y_label_confidences: list[float] = field(default_factory=list)
    y_slope_fallback_used: bool = False
    y_slope_history_median: Optional[float] = None

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

    Back-compat entry point: strips the confidence column returned by the
    richer `ocr_y_axis_labels_with_conf` so existing callers (and tests)
    continue to work unchanged.
    """
    return [(py, val) for py, val, _ in ocr_y_axis_labels_with_conf(img, plot_bbox)]


def ocr_y_axis_labels_with_conf(
    img: np.ndarray,
    plot_bbox: tuple[int, int, int, int],
) -> list[tuple[int, float, float]]:
    """Return `[(pixel_y_in_original_image, numeric_value, confidence), ...]`
    for each confident y-axis label.

    The confidence is Tesseract's own `conf` in [0, 100]. Callers that want
    to weight the linear fit (`calibrate_axes`) use the confidence squared
    so a 95%-confident label dominates a 72%-confident one by ~1.7×.
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
    #
    # We run two passes with different page-segmentation modes and union the
    # results: PSM 11 ("sparse text") was the original single-pass mode but
    # misses single-digit labels like "0" and values at the extremes of
    # wide-range axes (dec2024_to_now's +40 / +60 / +80 labels on current
    # PNGs). PSM 6 ("uniform block of text") catches those but sometimes
    # produces false positives like misreading a tick spacer as "8.0" or
    # a plot-interior gridline as "2.0". RANSAC below filters the false
    # positives by selecting the largest set of points that lies on a single
    # line within 2px of OCR error, which is easy to satisfy for real ticks
    # (they're colinear by construction) and hard for OCR noise.
    def _run_psm(psm: int) -> list[tuple[int, float, float]]:
        data = pytesseract.image_to_data(
            upscaled,
            config=f"--psm {psm} -c tessedit_char_whitelist=-.0123456789",
            output_type=pytesseract.Output.DICT,
        )
        out: list[tuple[int, float, float]] = []
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
            upscaled_center_y = data["top"][i] + data["height"][i] / 2.0
            original_y = (upscaled_center_y / OCR_UPSCALE) + strip_top_in_original
            out.append((int(round(original_y)), value, conf))
        return out

    # Union PSM 6 + PSM 11 hits, dedup by value keeping highest confidence.
    by_value: dict[float, tuple[int, float]] = {}
    for py, val, conf in _run_psm(6) + _run_psm(11):
        prior = by_value.get(val)
        if prior is None or conf > prior[1]:
            by_value[val] = (py, conf)
    points = sorted(
        [(py, val, conf) for val, (py, conf) in by_value.items()],
        key=lambda t: t[0],
    )

    # RANSAC: find the subset that maximally agrees on a single linear fit.
    # Real y-axis ticks are perfectly colinear; OCR false positives
    # generally sit off the line. Threshold of 2px is tight enough to
    # catch misreads but loose enough to absorb antialiasing noise on
    # the ±1px glyph-center estimate.
    if len(points) >= 3:
        best_inliers: list[tuple[int, float, float]] = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                y_i, v_i, _ = points[i]
                y_j, v_j, _ = points[j]
                if y_i == y_j:
                    continue
                a = (v_j - v_i) / (y_j - y_i)
                if abs(a) < 1e-9:
                    continue
                b = v_i - a * y_i
                inliers = [
                    p for p in points
                    if abs(p[1] - (a * p[0] + b)) / abs(a) <= 2.0
                ]
                # Prefer more inliers; tie-break by higher confidence sum
                # so pairs that agree with high-conf labels win over pairs
                # that agree with low-conf noise.
                if len(inliers) > len(best_inliers) or (
                    len(inliers) == len(best_inliers)
                    and sum(p[2] for p in inliers)
                    > sum(p[2] for p in best_inliers)
                ):
                    best_inliers = inliers
        if len(best_inliers) >= 3:
            points = sorted(best_inliers, key=lambda t: t[0])

    return points


# ─────────────────────────────────────────────────────────────────────────────
# Title timestamp OCR
# ─────────────────────────────────────────────────────────────────────────────


def _preprocess_for_ocr(crop: np.ndarray, upscale: int = 6) -> np.ndarray:
    """Upscale + binarize a text crop for Tesseract.

    Tesseract on small anti-aliased USGS glyphs (often 10-12 px tall)
    benefits from 6× bicubic upscale plus a per-crop Otsu binarization
    that removes anti-alias gradient greys. Skips binarization if the
    crop is already near-binary (most glyphs < 30 px tall after upscale
    would over-binarize and lose thin strokes).
    """
    if crop is None or crop.size == 0:
        return crop
    big = cv2.resize(
        crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY) if big.ndim == 3 else big
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def _detect_text_rows_below(
    img: np.ndarray,
    plot_bbox: tuple[int, int, int, int],
    dark_threshold: int = 120,
    min_dark_pixels: int = 10,
    min_row_gap: int = 3,
) -> list[tuple[int, int]]:
    """Find contiguous horizontal bands of dark pixels in the strip
    below the plot bbox. Returns `[(y_top, y_bot), ...]` in absolute
    image coords. USGS plots consistently have two rows: x-axis tick
    labels (first/upper) and the `Pacific/Honolulu Time (...)` string
    (second/lower).
    """
    h = img.shape[0]
    _, _, _, y1 = plot_bbox
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    strip = gray[y1 + 1 : h, :]
    dark_per_row = (strip < dark_threshold).sum(axis=1)
    rows = np.where(dark_per_row > min_dark_pixels)[0]
    if len(rows) == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(rows[0])
    for y in rows[1:]:
        y = int(y)
        if y - prev > min_row_gap:
            groups.append((start + y1 + 1, prev + y1 + 1))
            start = y
        prev = y
    groups.append((start + y1 + 1, prev + y1 + 1))
    return groups


def ocr_title_timestamps(
    img: np.ndarray,
    plot_bbox: tuple[int, int, int, int],
    *,
    source_name: Optional[str] = None,
) -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    """Extract the (start, end) ISO timestamps from the title line below the plot.

    The USGS plots embed the full date range as text:
        "Pacific/Honolulu Time (2026-01-08 15:42:31 to 2026-04-08 15:42:31)"

    For rolling sources (two_day/week/month/three_month) the window is a
    fixed duration, so USGS always prints the same HH:MM:SS at both
    endpoints. We use that invariant to detect and reject OCR misreads
    (e.g. month on 2026-04-22 read start=12:03:33 end=12:08:33 — a
    3↔8 digit confusion); when a candidate result fails the time-of-day
    consistency check we fall through to the other PSM mode and score
    the best remaining result.

    Returns `(start, end, psm_used, raw_text)` — the last two fields are
    diagnostic-only so the ingest pipeline can log which PSM mode won and
    exactly what Tesseract spat out.
    """
    h, w = img.shape[:2]

    # Prefer a tight crop on just the time-range text row (second text
    # band below the plot). Falls back to the full strip when row
    # detection can't isolate two bands (e.g. damaged PNG).
    text_rows = _detect_text_rows_below(img, plot_bbox)
    if len(text_rows) >= 2:
        # Time-range line is always the LOWER band.
        top, bot = text_rows[-1]
        pad = 3
        trange_crop = img[max(0, top - pad) : min(h, bot + pad + 1), 0:w]
    else:
        trange_crop = img[plot_bbox[3] + 5 : h, 0:w]
    if trange_crop.size == 0:
        raise CalibrationError("title strip is empty — plot bbox extends to image bottom")

    preprocessed = _preprocess_for_ocr(trange_crop, upscale=6)

    # Rolling sources have a fixed-duration window → start.time() must
    # equal end.time(). dec2024_to_now doesn't (start is typically
    # midnight on some fixed date; end is "now").
    requires_time_match = source_name in {
        "two_day", "week", "month", "three_month",
    }

    def _time_of_day_consistent(start: pd.Timestamp, end: pd.Timestamp) -> bool:
        return start.time() == end.time()

    # Try FOUR PSM modes on the time-range crop. Each has a different
    # tradeoff:
    #   PSM 7  — single text line, fastest, best for one-liner input
    #   PSM 6  — uniform text block, best when the crop spans 2 rows
    #   PSM 4  — single column of variable-size text
    #   PSM 13 — raw line, bypasses Tesseract's preprocessing heuristics
    # Collect every parseable candidate and prefer the one that's
    # internally consistent (start.time() == end.time()) on rolling
    # sources. Ties broken by PSM priority order.
    psms = ("7", "6", "4", "13")
    all_candidates: list[tuple[str, tuple[pd.Timestamp, pd.Timestamp], str, str]] = []
    errors: dict[str, str] = {}
    for psm in psms:
        result, text, err = _try_parse_title_at_psm(preprocessed, f"--psm {psm}")
        if result is not None:
            all_candidates.append((f"psm{psm}", result, text, err))
        else:
            errors[f"psm{psm}"] = f"{err} (OCR={text!r})"

    if not all_candidates:
        err_summary = "; ".join(f"PSM{psm}: {msg}" for psm, msg in errors.items())
        raise CalibrationError(f"could not extract title timestamps; {err_summary}")

    scored = []
    for psm, (start, end), raw, _ in all_candidates:
        score = 0
        if requires_time_match and not _time_of_day_consistent(start, end):
            score = 1
        scored.append((score, psm, start, end, raw))

    psm_priority = {"psm7": 0, "psm6": 1, "psm4": 2, "psm13": 3}
    scored.sort(key=lambda t: (t[0], psm_priority.get(t[1], 99)))
    best_score, best_psm, best_start, best_end, best_raw = scored[0]

    # Verify the OCR'd text actually mentions the expected timezone —
    # protects against a silent regression if USGS ever rewrites the
    # label in a different timezone (the `tz_convert` call downstream
    # would otherwise shift every timestamp by ±10h without complaint).
    if "honolulu" not in best_raw.lower() and "hawaii" not in best_raw.lower():
        logger.warning(
            "calibrate %s: time-range OCR did not contain a recognizable "
            "Honolulu/Hawaii timezone marker — if USGS changed the label's "
            "timezone, all downstream timestamps may be off. Raw: %r",
            source_name, best_raw,
        )

    if best_score > 0:
        logger.warning(
            "calibrate %s: time-range OCR produced inconsistent "
            "start.time()=%s != end.time()=%s on best candidate "
            "(%s); accepting anyway. Raw text: %r",
            source_name, best_start.time(), best_end.time(),
            best_psm, best_raw,
        )

    return best_start, best_end, best_psm, best_raw


def ocr_x_axis_ticks(
    img: np.ndarray,
    plot_bbox: tuple[int, int, int, int],
    source_name: Optional[str],
    x_start_utc: Optional[pd.Timestamp] = None,
    x_end_utc: Optional[pd.Timestamp] = None,
) -> list[tuple[int, pd.Timestamp]]:
    """Cross-check: OCR the x-axis tick row and parse each hit against
    the source's expected label format. Returns `[(pixel_x_center_utc,
    parsed_datetime_utc), ...]` — one entry per recognized label.

    Pass the time-range's `x_start_utc` and `x_end_utc` so we can map
    each tick's pixel position to a predicted UTC and use that as the
    disambiguation reference for `HH:MM` / `MM/DD` labels (whose
    calendar date isn't on the label itself). This makes each OCR'd
    tick INDEPENDENTLY parseable against the linear interpolation the
    main calibration uses — a genuine cross-check.

    Returns an empty list if tick row can't be isolated or no labels
    match the expected format. Never raises — cross-checks are
    advisory, not gating.
    """
    if source_name not in X_TICK_FORMAT:
        return []
    fmt = X_TICK_FORMAT[source_name]

    text_rows = _detect_text_rows_below(img, plot_bbox)
    if not text_rows:
        return []
    top, bot = text_rows[0]  # upper band = x-axis ticks
    pad = 3
    h, w = img.shape[:2]
    tick_crop = img[max(0, top - pad) : min(h, bot + pad + 1), 0:w]
    if tick_crop.size == 0:
        return []

    preprocessed = _preprocess_for_ocr(tick_crop, upscale=6)

    # PSM 11 (sparse text) finds labels positioned anywhere on the
    # strip; PSM 4 (single column of variable-size text) handles the
    # horizontally-separated tick layout equally well. Combine.
    hits: list[tuple[int, str]] = []  # (pixel_x_center_in_original, text)
    for psm in ("11", "4"):
        whitelist = (
            "0123456789:/-"
            if fmt != "MMM-YYYY"
            else "0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                 "abcdefghijklmnopqrstuvwxyz"
        )
        try:
            data = pytesseract.image_to_data(
                preprocessed,
                config=f"--psm {psm} -c tessedit_char_whitelist={whitelist}",
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            continue
        upscale = 6
        for i in range(len(data["text"])):
            t = data["text"][i].strip()
            if not t:
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError):
                conf = -1.0
            if conf < MIN_OCR_CONFIDENCE:
                continue
            cx_upscaled = data["left"][i] + data["width"][i] / 2.0
            cx_original = int(round(cx_upscaled / upscale))
            hits.append((cx_original, t))

    # Deduplicate by pixel_x within ±3 px (PSM 11 + PSM 4 often double-
    # count the same label). Keep first occurrence.
    hits.sort()
    deduped: list[tuple[int, str]] = []
    for cx, t in hits:
        if deduped and abs(cx - deduped[-1][0]) < 4:
            continue
        deduped.append((cx, t))

    x0, _, x1, _ = plot_bbox
    px_span = max(1.0, float(x1 - x0))
    span_seconds = (
        (x_end_utc - x_start_utc).total_seconds()
        if (x_start_utc is not None and x_end_utc is not None)
        else 0
    )
    parsed: list[tuple[int, pd.Timestamp]] = []
    for cx, t in deduped:
        # Per-tick predicted UTC = linear interpolation between
        # x_start_utc and x_end_utc at this pixel. Used as the
        # disambiguation reference so HH:MM ticks resolve to the right
        # calendar date.
        if x_start_utc is not None and span_seconds > 0:
            frac = (cx - x0) / px_span
            predicted_utc = x_start_utc + pd.Timedelta(
                seconds=frac * span_seconds
            )
        else:
            predicted_utc = pd.Timestamp.now("UTC").tz_localize(None)
        dt = _parse_x_tick_label(t, fmt, predicted_utc)
        if dt is not None:
            parsed.append((cx, dt))
    return parsed


def _parse_x_tick_label(
    text: str, fmt: str, reference_utc: pd.Timestamp
) -> Optional[pd.Timestamp]:
    """Parse an OCR'd tick label (`04/13`, `12:00`, `Jan-2025`, ...) into
    a UTC Timestamp, using `reference_utc` (UTC-naive) to fill in
    missing fields. USGS tick labels are in Pacific/Honolulu (HST, a
    fixed UTC-10) — this function returns UTC after applying that
    offset so results compare apples-to-apples against the time-range
    OCR's UTC output. `MMM-YYYY` labels are calendar markers with no
    time-of-day component; we leave them at midnight (HST = UTC either
    way for the integer-month comparison we use them for).
    """
    if fmt == "MM/DD":
        m = _MMDD_RE.match(text)
        if not m:
            return None
        mon, day = int(m.group("m")), int(m.group("d"))
        if not (1 <= mon <= 12 and 1 <= day <= 31):
            return None
        year = reference_utc.year
        try:
            hst_midnight = pd.Timestamp(year=year, month=mon, day=day)
        except (ValueError, OverflowError):
            return None
        # Convert HST midnight (UTC-10) to UTC.
        utc = hst_midnight + pd.Timedelta(hours=10)
        # If the tick is implausibly far in the future relative to
        # reference, it belongs to the prior year (USGS rolling windows
        # can straddle new year).
        if utc > reference_utc + pd.Timedelta(days=30):
            try:
                hst_midnight = pd.Timestamp(year=year - 1, month=mon, day=day)
                utc = hst_midnight + pd.Timedelta(hours=10)
            except (ValueError, OverflowError):
                return None
        return utc
    if fmt == "HH:MM":
        m = _HHMM_RE.match(text)
        if not m:
            return None
        hh, mm = int(m.group("h")), int(m.group("mi"))
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return None
        # Tick is in HST. Pick the date whose HST-to-UTC conversion
        # lands closest to the pixel's predicted UTC. Search ±3 days
        # so we cover two_day's 48h window (ticks near the far edge
        # can be 2 days away from any single anchor date). The
        # cross-check then measures the remaining delta.
        hst_ref = reference_utc - pd.Timedelta(hours=10)
        hst_candidate = hst_ref.normalize() + pd.Timedelta(hours=hh, minutes=mm)
        best = None
        best_delta = None
        for offset in range(-3, 4):
            c_hst = hst_candidate + pd.Timedelta(days=offset)
            c_utc = c_hst + pd.Timedelta(hours=10)
            delta = abs((c_utc - reference_utc).total_seconds())
            if best_delta is None or delta < best_delta:
                best, best_delta = c_utc, delta
        return best
    if fmt == "MMM-YYYY":
        m = _MONTH_ABBREV_RE.match(text)
        if not m:
            return None
        mon = _MONTH_ABBREV_NUM.get(m.group("mon").lower())
        y = int(m.group("y"))
        if mon is None or not (2020 <= y <= 2099):
            return None
        try:
            # Month-start in HST ≈ month-start in UTC (10h offset is
            # negligible for month-level cross-check).
            return pd.Timestamp(year=y, month=mon, day=1)
        except (ValueError, OverflowError):
            return None
    return None


def _try_parse_title_at_psm(
    upscaled_strip: np.ndarray, config: str
) -> tuple[Optional[tuple[pd.Timestamp, pd.Timestamp]], str, str]:
    """Run OCR + regex + parse + recover at one PSM config.

    Returns `(result, raw_text, error_message)` where:
      - `result` is `(start, end)` on success, `None` on any failure
      - `raw_text` is whatever Tesseract spat out (for diagnostics)
      - `error_message` is a human-readable description of why it failed
        (empty string on success)

    Never raises — all failures are reported via the return value so the
    caller can try a fallback PSM mode without unwinding the stack.
    """
    text = pytesseract.image_to_string(upscaled_strip, config=config)
    match = TITLE_TIMESTAMP_RE.search(text)
    if match is None:
        return None, text, "regex did not match any timestamp range"

    try:
        start = _parse_lenient_timestamp(match.group("start"))
        end = _parse_lenient_timestamp(match.group("end"))
    except (ValueError, TypeError) as e:
        return None, text, f"unparseable timestamp: {e}"

    start, end = _recover_ocr_year_misread(start, end)

    if end <= start:
        return None, text, f"non-chronological after year recovery: {start} → {end}"

    return (start, end), text, ""


def _recover_ocr_year_misread(
    start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Heuristically repair single-digit year OCR misreads in title timestamps.

    The dominant title-OCR failure mode is a single digit in the YEAR field
    being misread (most often a `2` ↔ `0` flip — they look similar at low
    resolution). USGS PNG titles always show ranges where start/end years
    differ by at most 1 (across new-year boundaries), so a year diff > 1 is
    a near-certain OCR error.

    Two failure modes are handled:

    A) Non-chronological after misread (the common case):
       OCR returned end_year smaller than start_year, e.g.
           start = 2026-04-03 02:51:26
           end   = 2006-04-09 02:51:26      ← misread
       The strict `end <= start` check fires. We try substituting
       `end.year = start.year` and accept the result if it's chronological.

    B) Chronological-but-implausibly-wide (rarer, silently dangerous):
       OCR returned a smaller year for `start` than for `end`, e.g.
           start = 2006-04-03 02:51:26      ← misread
           end   = 2026-04-09 02:51:26
       The strict check passes (2006 < 2026) but the implied 20-year
       range would produce a wildly wrong x-axis calibration. Caught by
       `abs(year diff) > 1` regardless of chronological order, then
       recovered by substituting `start.year = end.year` (or vice versa
       for the analogous reverse case).

    Returns the (possibly-recovered) `(start, end)` tuple. Always pure;
    never raises — ill-behaved inputs round-trip unchanged so the caller
    can raise a proper CalibrationError.
    """
    if abs(end.year - start.year) <= 1:
        # Year diff plausible (same year or new-year crossing) — trust it.
        return start, end

    if end.year < start.year:
        # end's year is the OCR victim (covers failure mode A and the
        # less-common "end was misread to be older than the real start").
        try:
            recovered_end = end.replace(year=start.year)
        except ValueError:
            # `replace(year=...)` fails on Feb 29 in non-leap years.
            return start, end
        if recovered_end > start:
            return start, recovered_end
    else:
        # end.year > start.year by more than 1 → start's year is the OCR
        # victim (failure mode B). Substitute start.year = end.year.
        try:
            recovered_start = start.replace(year=end.year)
        except ValueError:
            return start, end
        if recovered_start < end:
            return recovered_start, end

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


def calibrate_axes(
    img: np.ndarray, *, source_name: Optional[str] = None
) -> AxisCalibration:
    """Run the full calibration pipeline on a freshly-decoded image.

    Args:
        img:          freshly-decoded BGR image.
        source_name:  if supplied (canonical reconcile name, e.g. `"two_day"`),
                      enables Phase 1 guards: x-window sanity against the
                      advertised window length, and y-slope regression
                      against the rolling history.

    Raises `CalibrationError` when any guard fails. Callers above us (the
    per-source ingest loop) catch `CalibrationError` and skip the source
    for this run, leaving its previous CSV untouched.
    """
    plot_bbox = detect_plot_bbox(img)

    # ── y-axis ──────────────────────────────────────────────────────────────
    labels_with_conf = ocr_y_axis_labels_with_conf(img, plot_bbox)
    if len(labels_with_conf) < MIN_Y_LABELS_REQUIRED:
        raise CalibrationError(
            f"y-axis OCR found only {len(labels_with_conf)} label(s) "
            f"(need ≥{MIN_Y_LABELS_REQUIRED}): {labels_with_conf}"
        )

    pixel_ys = np.array([p for p, _, _ in labels_with_conf], dtype=float)
    values = np.array([v for _, v, _ in labels_with_conf], dtype=float)
    confs = np.array([c for _, _, c in labels_with_conf], dtype=float)

    # Phase 1a: weighted polyfit with w = (conf/100)^2. A 95%-confident
    # label contributes ~1.7× the leverage of a 72% one, which is the
    # regime where OCR occasionally misreads a stray pixel-cluster as a
    # digit. Falls back to equal-weight polyfit if all confidences are
    # identical (numerically equivalent anyway).
    weights = (confs / 100.0) ** 2
    if weights.sum() <= 0:
        weights = np.ones_like(confs)
    a_y, b_y = np.polyfit(pixel_ys, values, 1, w=weights)
    y_residuals = values - (a_y * pixel_ys + b_y)
    y_resid_max = float(np.abs(y_residuals).max())
    y_resid_std = float(np.sqrt(np.average(y_residuals**2, weights=weights)))

    # Phase 1a: enforce a µrad-space residual ceiling on the label fit.
    # See `Y_CALIBRATION_MAX_RESIDUAL_MICRORAD` in config.py for the
    # threshold rationale and fixture baseline numbers. A pixel-space
    # threshold doesn't work because sources vary ~20× in µrad/px.
    if abs(a_y) > 1e-9:
        y_resid_max_pixels = y_resid_max / abs(a_y)
    else:
        y_resid_max_pixels = float("inf")
    if y_resid_max > Y_CALIBRATION_MAX_RESIDUAL_MICRORAD:
        raise CalibrationError(
            f"y-axis linear fit residual too large: "
            f"max={y_resid_max:.3f} µrad (limit "
            f"{Y_CALIBRATION_MAX_RESIDUAL_MICRORAD} µrad, = "
            f"{y_resid_max_pixels:.1f} px at this plot's scale). "
            f"Labels: {[(float(v), float(c)) for _, v, c in labels_with_conf]}"
        )

    # Phase 1a: compare this run's slope to the rolling history of the same
    # source. Two scenarios can produce slope drift:
    #   (a) USGS rescaled the PNG's y-axis (legitimate — they do this
    #       periodically when the data range shifts). The current fit is
    #       clean (sub-pixel residuals across all OCR labels) and is the
    #       correct answer — history is stale.
    #   (b) OCR misread a label value or pixel position, pulling the fit
    #       off the real ticks. Residuals will be noticeable (multi-pixel)
    #       because the bogus point doesn't sit on the true line.
    # Distinguish by the residual quality we already computed: clean fit
    # = trust the new slope; noisy fit = trust history and refit the
    # intercept so the pair stays mathematically consistent.
    slope_history_median: Optional[float] = None
    slope_fallback_used = False
    if source_name is not None:
        history = _load_y_slope_history(source_name)
        if history:
            slope_history_median = float(np.median(history))
            denom = abs(slope_history_median) if slope_history_median else 1.0
            drift_pct = 100.0 * abs(a_y - slope_history_median) / denom
            if drift_pct > Y_SLOPE_REGRESSION_TOLERANCE_PERCENT:
                if y_resid_max_pixels <= 1.0:
                    # Clean fit with drift → USGS rescaled. Trust the fit.
                    # History will catch up as more fetches append.
                    logger.info(
                        "calibrate %s: y-slope drift %.1f%% but residuals "
                        "are sub-pixel (%.2f px); accepting new fit "
                        "(likely USGS axis rescale)",
                        source_name, drift_pct, y_resid_max_pixels,
                    )
                else:
                    # Noisy fit AND drift → OCR probably mis-read a label.
                    # Use the historical slope and refit the intercept as
                    # a weighted-mean of `y_i - a_hist·px_i` so the line
                    # passes through the OCR labels as closely as
                    # possible under the fixed slope constraint. Without
                    # the refit, the intercept was computed assuming a
                    # different slope and the calibration ends up with
                    # mismatched (a,b) that diverges from truth at the
                    # pixel extremes.
                    a_y = slope_history_median
                    b_y = float(
                        np.average(values - a_y * pixel_ys, weights=weights)
                    )
                    slope_fallback_used = True
                    logger.warning(
                        "calibrate %s: y-slope drift %.1f%% AND noisy fit "
                        "(%.2f px max resid) — falling back to history "
                        "median %.4f and refitting intercept",
                        source_name, drift_pct, y_resid_max_pixels,
                        slope_history_median,
                    )

    # ── x-axis ──────────────────────────────────────────────────────────────
    x_start, x_end, psm_used, title_raw = ocr_title_timestamps(
        img, plot_bbox, source_name=source_name,
    )

    # Phase 1b: sanity-check the x-window span against the advertised window
    # length. USGS plots each have a fixed duration (two_day=48 h, week=7 d,
    # etc.) that a minute/hour OCR misread would violate even if the regex
    # still matches. `dec2024_to_now` has a monotonically-growing window
    # and gets a loose 30-day-to-5-year bracket instead.
    if source_name is not None:
        span_hours = (x_end - x_start).total_seconds() / 3600.0
        expected = X_WINDOW_EXPECTED_HOURS.get(source_name)
        if expected is not None:
            if abs(span_hours - expected) > X_WINDOW_TOLERANCE_HOURS:
                raise CalibrationError(
                    f"x-window span implausible for {source_name}: "
                    f"{span_hours:.2f} h (expected {expected} ± "
                    f"{X_WINDOW_TOLERANCE_HOURS} h). "
                    f"Title OCR returned [{x_start} → {x_end}] via {psm_used}"
                )
        elif source_name == "dec2024_to_now":
            if span_hours < 30 * 24 or span_hours > 5 * 365 * 24:
                raise CalibrationError(
                    f"dec2024_to_now x-window span out of plausible range: "
                    f"{span_hours:.1f} h"
                )

        # Phase 1b extension: x_end must be close to "now". A multi-digit
        # year OCR misread (e.g. 2026 → 2008 observed 2026-04-22) produces
        # a VALID-span range that nonetheless places the capture in the
        # wrong era; the existing per-source CSV has no temporal overlap
        # and the downstream frame-alignment then appends in raw frame,
        # poisoning the CSV. Reject such captures here.
        now_utc = pd.Timestamp.now("UTC").tz_localize(None)
        age_hours = (now_utc - x_end).total_seconds() / 3600.0
        if abs(age_hours) > X_END_MAX_AGE_HOURS:
            raise CalibrationError(
                f"x_end implausibly far from now for {source_name}: "
                f"x_end={x_end} is {age_hours:.1f} h from now "
                f"(limit ±{X_END_MAX_AGE_HOURS} h). "
                f"Likely a multi-digit year OCR misread. PSM={psm_used}, "
                f"raw={title_raw!r}"
            )

    # Record this run's slope in the rolling history so future runs can
    # check against it. Done only after all guards have passed — a rejected
    # calibration must not pollute the reference.
    if source_name is not None and not slope_fallback_used:
        _append_y_slope_history(source_name, float(a_y))

    return AxisCalibration(
        plot_bbox=plot_bbox,
        y_slope=float(a_y),
        y_intercept=float(b_y),
        x_start=x_start,
        x_end=x_end,
        y_labels_found=[(py, val) for py, val, _ in labels_with_conf],
        fit_residual_per_axis={
            "y_max_residual_microrad": y_resid_max,
            "y_std_residual_microrad": y_resid_std,
            "y_max_residual_pixels": y_resid_max_pixels,
        },
        title_psm_used=psm_used,
        title_raw_text=title_raw,
        y_label_confidences=[float(c) for c in confs],
        y_slope_fallback_used=slope_fallback_used,
        y_slope_history_median=slope_history_median,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Y-slope rolling history — Phase 1a drift detection
# ─────────────────────────────────────────────────────────────────────────────


def _load_y_slope_history(source_name: str) -> list[float]:
    """Return the rolling history of successful y-slope fits for `source_name`.

    Empty list on a fresh checkout or if the file is unreadable — both
    cases are fine: the guard just no-ops on the first run and starts
    accumulating history.
    """
    if not Y_SLOPE_HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(Y_SLOPE_HISTORY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    values = data.get(source_name) or []
    return [float(v) for v in values if isinstance(v, (int, float))]


def _append_y_slope_history(
    source_name: str, slope: float, *, max_length: int = Y_SLOPE_HISTORY_LENGTH
) -> None:
    """Append `slope` to the rolling history for `source_name`, truncating
    from the front once the list exceeds `max_length`.
    """
    Y_SLOPE_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = (
            json.loads(Y_SLOPE_HISTORY_FILE.read_text())
            if Y_SLOPE_HISTORY_FILE.exists()
            else {}
        )
    except (json.JSONDecodeError, OSError):
        existing = {}
    history = list(existing.get(source_name) or [])
    history.append(float(slope))
    if len(history) > max_length:
        history = history[-max_length:]
    existing[source_name] = history
    Y_SLOPE_HISTORY_FILE.write_text(json.dumps(existing, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1c: anchor-referenced calibration cross-check (digital → source)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AnchorFitResult:
    """Outcome of `recalibrate_by_anchor_fit()` for one source."""

    source_name: str
    ran: bool = False                      # True iff regression was attempted
    overlap_buckets: int = 0               # after bucketing + trimming
    a: float = 1.0                         # slope correction (identity = 1.0)
    b: float = 0.0                         # intercept correction (identity = 0.0)
    residual_std_microrad: float = 0.0     # Huber-loss-weighted residual spread
    warning: Optional[str] = None          # human-readable reason to warn
    note: Optional[str] = None             # reason regression was skipped


def recalibrate_by_anchor_fit(
    source_name: str,
    source_df: pd.DataFrame,
    digital_df: pd.DataFrame,
    *,
    bucket_freq: str = "1h",
) -> AnchorFitResult:
    """Fit `digital_tilt = a * source_tilt + b` via Huber-robust regression
    over the temporal overlap and return the (a, b) correction.

    The model assumes each PNG source is a linear image of the same
    underlying tilt signal that digital captures authoritatively. Fitting
    against digital recovers a source-specific `(a, b)` that collapses the
    systematic y-slope / y-intercept errors the legacy alignment offset
    couldn't absorb (because it was a single scalar).

    Implementation notes:
      * Buckets both sides at `bucket_freq` (default 1 h) and keeps only
        buckets present in both frames. Below `ANCHOR_FIT_MIN_OVERLAP_BUCKETS`
        the result is not meaningful — caller should leave the source
        untouched.
      * Trims the first and last `ANCHOR_FIT_TRIM_HOURS` of digital's range
        to avoid endpoint pathologies where the research-release file
        starts/ends mid-sample.
      * Uses `scipy.optimize.least_squares` with Huber loss. `np.polyfit`
        would be OLS and pay dearly for the tail outliers this data has.
    """
    result = AnchorFitResult(source_name=source_name)

    if source_df is None or len(source_df) == 0:
        result.note = "source_df empty"
        return result
    if digital_df is None or len(digital_df) == 0:
        result.note = "digital_df empty"
        return result

    src = source_df[[DATE_COL, TILT_COL]].copy()
    dig = digital_df[[DATE_COL, TILT_COL]].copy()
    src[DATE_COL] = pd.to_datetime(src[DATE_COL]).astype("datetime64[ns]")
    dig[DATE_COL] = pd.to_datetime(dig[DATE_COL]).astype("datetime64[ns]")
    src = src.dropna().sort_values(DATE_COL)
    dig = dig.dropna().sort_values(DATE_COL)
    if src.empty or dig.empty:
        result.note = "no rows after cleaning"
        return result

    # Trim digital's endpoints — the research-release file begins/ends
    # mid-sample and the first/last few hours inflate the Huber residual.
    dig_start = dig[DATE_COL].min() + pd.Timedelta(hours=ANCHOR_FIT_TRIM_HOURS)
    dig_end = dig[DATE_COL].max() - pd.Timedelta(hours=ANCHOR_FIT_TRIM_HOURS)
    dig_trimmed = dig[(dig[DATE_COL] >= dig_start) & (dig[DATE_COL] <= dig_end)]
    if len(dig_trimmed) == 0:
        result.note = "digital overlap window empty after trim"
        return result

    # Bucket-align both sides at `bucket_freq`, take the mean in each bucket.
    src_b = src.assign(_b=lambda d: d[DATE_COL].dt.floor(bucket_freq)).groupby("_b")[TILT_COL].mean()
    dig_b = (
        dig_trimmed.assign(_b=lambda d: d[DATE_COL].dt.floor(bucket_freq))
        .groupby("_b")[TILT_COL]
        .mean()
    )
    overlap = src_b.index.intersection(dig_b.index)
    result.overlap_buckets = len(overlap)
    if len(overlap) < ANCHOR_FIT_MIN_OVERLAP_BUCKETS:
        result.note = (
            f"overlap too small ({len(overlap)} buckets, need "
            f"≥{ANCHOR_FIT_MIN_OVERLAP_BUCKETS})"
        )
        return result

    x = src_b.loc[overlap].to_numpy(dtype=float)
    y = dig_b.loc[overlap].to_numpy(dtype=float)

    # Huber-robust regression via scipy.optimize.least_squares. Initial
    # guess (a=1, b=0) is the identity transform — correct when calibration
    # is already clean. `f_scale` sets the Huber elbow: residuals larger
    # than f_scale µrad are treated as outliers (L1); smaller are L2. 1.0
    # is conservative — most real inter-source disagreement should be
    # under it.
    try:
        from scipy.optimize import least_squares  # type: ignore
    except ImportError:  # pragma: no cover — scipy is a pinned dep
        result.note = "scipy not available"
        return result

    def residuals(params: np.ndarray) -> np.ndarray:
        a, b = params
        return y - (a * x + b)

    fit = least_squares(
        residuals,
        x0=np.array([1.0, 0.0]),
        loss="huber",
        f_scale=1.0,
        max_nfev=200,
    )
    a_fit, b_fit = float(fit.x[0]), float(fit.x[1])
    resid = residuals(fit.x)
    resid_std = float(np.sqrt(np.mean(resid**2)))

    result.ran = True
    result.a = a_fit
    result.b = b_fit
    result.residual_std_microrad = resid_std

    drift_a = abs(a_fit - 1.0)
    drift_b = abs(b_fit)
    if drift_a > ANCHOR_FIT_A_WARNING_FRACTION or drift_b > ANCHOR_FIT_B_WARNING_MICRORAD:
        result.warning = (
            f"anchor cross-check flags {source_name}: digital = "
            f"{a_fit:.4f} · png + {b_fit:+.3f}  "
            f"(|a-1|={drift_a:.1%}, |b|={drift_b:.2f} µrad, "
            f"residual_std={resid_std:.2f} µrad over {len(overlap)} buckets)"
        )
    return result


def apply_anchor_fit(df: pd.DataFrame, fit: AnchorFitResult) -> pd.DataFrame:
    """Apply a (a, b) correction from `recalibrate_by_anchor_fit` to a
    source's DataFrame in-place-copy fashion: returns a new DataFrame with
    tilt transformed as `tilt ← a * tilt + b` so it matches digital's frame.

    Identity when `fit.ran` is False or the fit is within tolerance.
    """
    if not fit.ran:
        return df
    if fit.warning is None:
        # Within tolerance — don't nudge the data, but record that the
        # fit ran cleanly.
        return df
    out = df.copy()
    out[TILT_COL] = fit.a * out[TILT_COL] + fit.b
    return out
