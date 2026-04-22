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
    data = pytesseract.image_to_data(
        upscaled,
        config="--psm 11 -c tessedit_char_whitelist=-.0123456789",
        output_type=pytesseract.Output.DICT,
    )

    found: list[tuple[int, float, float]] = []
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
        found.append((int(round(original_y)), value, conf))

    # Deduplicate by value — keep the highest-confidence occurrence.
    seen: dict[float, tuple[int, float]] = {}
    for py, val, conf in found:
        prior = seen.get(val)
        if prior is None or conf > prior[1]:
            seen[val] = (py, conf)
    return sorted(
        [(py, val, conf) for val, (py, conf) in seen.items()],
        key=lambda t: t[0],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Title timestamp OCR
# ─────────────────────────────────────────────────────────────────────────────


def ocr_title_timestamps(
    img: np.ndarray,
    plot_bbox: tuple[int, int, int, int],
) -> tuple[pd.Timestamp, pd.Timestamp, str, str]:
    """Extract the (start, end) ISO timestamps from the title line below the plot.

    The USGS plots embed the full date range as text:
        "Pacific/Honolulu Time (2026-01-08 15:42:31 to 2026-04-08 15:42:31)"

    OCRing this with PSM 7 produces a near-perfect string that a regex
    cleanly recovers. The only common error is "to" → "t0", which the
    regex tolerates.

    Returns `(start, end, psm_used, raw_text)` — the last two fields are
    diagnostic-only so the ingest pipeline can log which PSM mode won and
    exactly what Tesseract spat out.
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

    # Try PSM 7 first (single-line, fastest, usually best for the title
    # strip). Fall through to PSM 6 if PSM 7 fails for *any* reason —
    # regex mismatch, invalid day-of-month, non-chronological even after
    # year recovery, etc. PSM 6 is multi-block and has historically read
    # the title cleanly even when PSM 7 misreads individual digits.
    psm7_result, psm7_text, psm7_error = _try_parse_title_at_psm(upscaled, "--psm 7")
    if psm7_result is not None:
        return psm7_result[0], psm7_result[1], "psm7", psm7_text

    psm6_result, psm6_text, psm6_error = _try_parse_title_at_psm(upscaled, "--psm 6")
    if psm6_result is not None:
        return psm6_result[0], psm6_result[1], "psm6", psm6_text

    # Both PSM modes failed. Surface whichever error is more informative —
    # parsing errors beat regex-mismatch errors because they prove the OCR
    # at least produced something date-shaped.
    raise CalibrationError(
        f"could not extract title timestamps; "
        f"PSM 7: {psm7_error} (OCR={psm7_text!r}); "
        f"PSM 6: {psm6_error} (OCR={psm6_text!r})"
    )


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
    # source. If it drifts too far, fall back to the last-known-good slope
    # and warn — USGS shifts the y-range across updates but shouldn't
    # change the per-pixel scale between consecutive captures.
    slope_history_median: Optional[float] = None
    slope_fallback_used = False
    if source_name is not None:
        history = _load_y_slope_history(source_name)
        if history:
            slope_history_median = float(np.median(history))
            denom = abs(slope_history_median) if slope_history_median else 1.0
            drift_pct = 100.0 * abs(a_y - slope_history_median) / denom
            if drift_pct > Y_SLOPE_REGRESSION_TOLERANCE_PERCENT:
                # Fallback: keep this run's intercept (USGS legitimately
                # shifts it) but restore the historical slope so the pixel
                # scale stays stable across runs.
                a_y = slope_history_median
                slope_fallback_used = True

    # ── x-axis ──────────────────────────────────────────────────────────────
    x_start, x_end, psm_used, title_raw = ocr_title_timestamps(img, plot_bbox)

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
