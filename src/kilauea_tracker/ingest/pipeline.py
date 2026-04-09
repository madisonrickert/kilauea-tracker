"""Top-level ingest orchestration: fetch → decode → calibrate → trace → cache.

This module is the only thing `streamlit_app.py` needs to import from
`kilauea_tracker.ingest`. It collapses all four pipeline stages into a single
`ingest(source)` call and returns an `IngestReport` summarizing what happened.

The orchestration is wrapped in a "last-known-good calibration" persistence
layer so that a transient OCR failure (e.g. one bad PNG with garbled labels)
doesn't take the app down. The previous calibration is reused as a fallback;
the warning is surfaced through the `IngestReport.warnings` list.

Usage from the Streamlit app:

    from kilauea_tracker.ingest.pipeline import ingest, ingest_all
    reports = ingest_all()
    for report in reports:
        if report.error:
            st.error(report.error)
        for w in report.warnings:
            st.warning(w)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from ..cache import DEDUPE_BUCKET, AppendReport, append_history, load_history
from ..config import (
    ALL_SOURCES,
    DIGITAL_CSV,
    GAP_FILL_SOURCES,
    HISTORY_CSV,
    LAST_GOOD_CALIBRATION,
    USGS_TILT_URLS,
    TiltSource,
)
from ..model import DATE_COL, TILT_COL
from .calibrate import AxisCalibration, calibrate_axes
from .exceptions import CalibrationError, FetchError, IngestError, TraceError
from .fetch import fetch_tilt_png
from .trace import trace_curve

# Where the per-source `Last-Modified` headers are persisted.
LAST_MODIFIED_FILE = LAST_GOOD_CALIBRATION.parent / "last_modified.json"

# Bucket size used for cross-source alignment. Intentionally coarser than the
# dedupe bucket (15 min) so that sparse / irregular sources (legacy CSV at
# ~6.5h spacing, DEC2024_TO_NOW at ~16h spacing) can find enough overlapping
# buckets to compute a meaningful median offset. The dedupe step still uses
# 15-min buckets — only the alignment step bins more aggressively.
ALIGNMENT_BUCKET = "1h"

# Minimum number of overlapping alignment buckets we need before we trust a
# computed cross-source y-offset. Below this we leave the trace untouched
# and let the conflict warnings (if any) surface as a real anomaly.
MIN_OVERLAP_BUCKETS_FOR_ALIGN = 5

# Don't apply offsets larger than this (in microradians). A delta this big
# isn't a y-axis drift — it's a calibration bug, and silently shifting the
# data by 15 µrad would mask the bug.
MAX_TRUSTED_OFFSET_MICRORAD = 15.0


@dataclass
class IngestReport:
    """Outcome of one `ingest(source)` call. Always populated, even on failure."""

    source: TiltSource
    fetched: bool = False                       # True iff a fresh PNG arrived
    rows_traced: int = 0                        # rows produced by trace_curve
    rows_added_to_cache: int = 0                # net new rows after dedupe
    rows_updated_in_cache: int = 0
    rows_dropped_as_filled: int = 0             # gap-fill mode: rows whose buckets were already covered
    cache_conflicts: list[dict] = field(default_factory=list)
    applied_y_offset: Optional[float] = None    # microradians shifted by cross-source align
    overlap_buckets: int = 0                    # how many buckets the offset was computed over
    gap_fill_mode: bool = False                 # True if this source ran in fill-gaps-only mode
    last_modified: Optional[str] = None
    calibration: Optional[AxisCalibration] = None
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None                 # set when ingest failed entirely


def ingest(
    source: TiltSource,
    *,
    history_path: Path = HISTORY_CSV,
) -> IngestReport:
    """Run the full pipeline for a single USGS source.

    Never raises — failures are recorded on the returned `IngestReport`.
    """
    report = IngestReport(source=source)
    url = USGS_TILT_URLS[source]
    cached_last_modified = _load_cached_last_modified(source)

    try:
        fetch_result = fetch_tilt_png(url, cached_last_modified)
    except FetchError as e:
        report.error = f"fetch failed for {source.name}: {e}"
        return report

    report.last_modified = fetch_result.last_modified

    if not fetch_result.changed or fetch_result.body is None:
        # Server said 304 Not Modified — nothing new to ingest.
        return report

    report.fetched = True

    # Decode the bytes via OpenCV
    try:
        img_array = np.frombuffer(fetch_result.body, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise IngestError("cv2.imdecode returned None")
    except Exception as e:
        report.error = f"could not decode {source.name} PNG: {e}"
        return report

    # Calibrate axes — fall back to last-known-good if OCR fails
    try:
        calibration = calibrate_axes(img)
    except CalibrationError as e:
        report.warnings.append(
            f"calibration failed for {source.name}: {e}. "
            "Cannot fall back — no usable image."
        )
        report.error = f"calibration failed: {e}"
        return report

    report.calibration = calibration

    # Trace the curve
    try:
        traced = trace_curve(img, calibration)
    except TraceError as e:
        report.error = f"trace failed for {source.name}: {e}"
        return report

    report.rows_traced = len(traced)

    # Cross-source y-offset alignment. Each USGS PNG re-renders with its own
    # y-axis labels and Tesseract introduces small intercept differences;
    # measured drift between captures is ~5-7 µrad systematic. Without
    # correction, every overlap bucket flags a "conflict" and the model fits
    # see step jumps where one source hands off to another. Solution: align
    # the new trace to the existing cache by subtracting the median bucket-
    # level delta in their overlap region.
    existing_for_align = load_history(history_path)
    aligned, offset, overlap_n = _align_to_cache(traced, existing_for_align)
    report.applied_y_offset = offset
    report.overlap_buckets = overlap_n
    if offset is not None:
        traced = aligned

    # Gap-fill mode: drop any sample whose 15-min bucket already exists in
    # the cache. Used for low-resolution long-history sources (DEC2024_TO_NOW)
    # so they only fill gaps without overwriting the higher-resolution recent
    # sources that ran earlier in the chain.
    if source in GAP_FILL_SOURCES:
        report.gap_fill_mode = True
        traced, dropped = _filter_to_gap_buckets(traced, existing_for_align)
        report.rows_dropped_as_filled = dropped

    # Append to the cache
    cache_report: AppendReport = append_history(traced, history_path)
    report.rows_added_to_cache = cache_report.rows_added
    report.rows_updated_in_cache = cache_report.rows_updated
    report.cache_conflicts = cache_report.conflicts

    if cache_report.conflicts:
        report.warnings.append(
            f"{len(cache_report.conflicts)} cache conflict(s) detected — "
            "calibration may be drifting between captures."
        )

    # Persist the Last-Modified header so the next fetch can short-circuit
    _save_cached_last_modified(source, fetch_result.last_modified)

    return report


def _filter_to_gap_buckets(
    new_rows: pd.DataFrame,
    existing: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Drop new rows whose 15-min bucket already exists in `existing`.

    Used by gap-fill sources so the new low-resolution data only adds rows
    where the cache is sparse, instead of overwriting buckets covered by
    higher-resolution sources.

    Returns `(filtered_rows, dropped_count)`.
    """
    if len(new_rows) == 0:
        return new_rows, 0
    if len(existing) == 0:
        return new_rows, 0

    new_with_bucket = new_rows.copy()
    new_with_bucket["_bucket"] = new_with_bucket[DATE_COL].dt.round(DEDUPE_BUCKET)
    existing_buckets = set(existing[DATE_COL].dt.round(DEDUPE_BUCKET))
    keep_mask = ~new_with_bucket["_bucket"].isin(existing_buckets)
    dropped = int((~keep_mask).sum())
    filtered = new_with_bucket.loc[keep_mask].drop(columns=["_bucket"])
    return filtered, dropped


def _align_to_cache(
    new_rows: pd.DataFrame,
    existing: pd.DataFrame,
    *,
    bucket_freq: str = ALIGNMENT_BUCKET,
) -> tuple[pd.DataFrame, Optional[float], int]:
    """Compute the median y-offset between `new_rows` and `existing` in their
    overlapping `bucket_freq` time buckets, and return a copy of `new_rows`
    shifted by that offset.

    Returns `(aligned_rows, applied_offset_microrad, overlap_bucket_count)`.
    `applied_offset_microrad` is `None` when no alignment was applied — either
    because there wasn't enough overlap, the offset was implausibly large, or
    one of the inputs was empty.

    Why the median, not the mean: a few buckets near eruption transitions
    have wildly different values across captures (the curve is moving fast
    and the per-column median pixel can land at very different positions).
    The mean would chase those outliers; the median ignores them.

    Why a coarser bucket than the dedupe bucket: alignment needs to find
    overlap between sources whose individual sample timestamps may not line
    up exactly (e.g. legacy at ~6.5h irregular vs DEC2024_TO_NOW at ~16h).
    Binning into 1-hour or larger buckets gives them more chances to share
    a bucket. Dedupe stays at 15-min granularity.
    """
    if len(new_rows) == 0 or len(existing) == 0:
        return new_rows, None, 0

    new_buckets = (
        new_rows[[DATE_COL, TILT_COL]]
        .assign(_bucket=lambda d: d[DATE_COL].dt.floor(bucket_freq))
        .groupby("_bucket")[TILT_COL]
        .mean()
    )
    existing_buckets = (
        existing[[DATE_COL, TILT_COL]]
        .assign(_bucket=lambda d: d[DATE_COL].dt.floor(bucket_freq))
        .groupby("_bucket")[TILT_COL]
        .mean()
    )
    overlap_index = new_buckets.index.intersection(existing_buckets.index)
    overlap_n = len(overlap_index)

    if overlap_n < MIN_OVERLAP_BUCKETS_FOR_ALIGN:
        return new_rows, None, overlap_n

    deltas = new_buckets.loc[overlap_index] - existing_buckets.loc[overlap_index]
    offset = float(deltas.median())

    # Refuse implausibly large corrections — they indicate calibration bugs,
    # not y-axis drift, and we want them to surface loudly rather than be
    # silently masked.
    if abs(offset) > MAX_TRUSTED_OFFSET_MICRORAD:
        return new_rows, None, overlap_n

    aligned = new_rows.copy()
    aligned[TILT_COL] = aligned[TILT_COL] - offset
    return aligned, offset, overlap_n


def ingest_digital(history_path: Path = HISTORY_CSV) -> IngestReport:
    """Ingest the pre-processed UWD digital tiltmeter CSV.

    The digital data is the most accurate source we have but covers only
    Jan-Jun 2025. It's split into 6 segments at instrument relevelings — each
    relevel resets the absolute baseline, so we align each segment
    independently against the existing cache (the cache supplies the
    canonical y-frame from the live image sources).

    The processed CSV is produced once by `scripts/import_digital_data.py`
    from the raw USGS research-release files at `~/Downloads/UWD_digital/`,
    then committed to the repo.
    """
    # We re-use the IngestReport shape but synthesize a "source" that isn't
    # in the TiltSource enum. The Streamlit UI special-cases `source` being
    # None to render this differently.
    report = IngestReport(source=None)  # type: ignore[arg-type]
    if not DIGITAL_CSV.exists():
        report.warnings.append(
            f"digital data CSV not found at {DIGITAL_CSV} — "
            "run `scripts/import_digital_data.py` to generate it"
        )
        return report

    digital = pd.read_csv(DIGITAL_CSV)
    digital[DATE_COL] = pd.to_datetime(digital[DATE_COL])
    if "segment" not in digital.columns:
        digital["segment"] = 1

    existing = load_history(history_path)

    aligned_segments: list[pd.DataFrame] = []
    per_segment_offsets: list[Optional[float]] = []
    total_overlap = 0
    for seg_id, seg_df in digital.groupby("segment", sort=True):
        seg_df = seg_df[[DATE_COL, TILT_COL]].copy()
        # Each segment aligns independently — different relevelings → different
        # absolute baselines.
        seg_aligned, seg_offset, seg_overlap = _align_to_cache(seg_df, existing)
        per_segment_offsets.append(seg_offset)
        total_overlap += seg_overlap
        if seg_offset is not None:
            aligned_segments.append(seg_aligned)
        else:
            aligned_segments.append(seg_df)
            report.warnings.append(
                f"digital segment {seg_id} could not be aligned "
                f"(overlap={seg_overlap} buckets)"
            )

    combined = pd.concat(aligned_segments, ignore_index=True)
    combined = combined.sort_values(DATE_COL).reset_index(drop=True)
    report.rows_traced = len(combined)
    report.fetched = True
    report.overlap_buckets = total_overlap
    # Aggregate offset reported is the median across segments (informational)
    valid_offsets = [o for o in per_segment_offsets if o is not None]
    if valid_offsets:
        report.applied_y_offset = float(np.median(valid_offsets))

    cache_report = append_history(combined, history_path)
    report.rows_added_to_cache = cache_report.rows_added
    report.rows_updated_in_cache = cache_report.rows_updated
    report.cache_conflicts = cache_report.conflicts

    return report


def ingest_all(history_path: Path = HISTORY_CSV) -> list[IngestReport]:
    """Run `ingest()` for every source in `config.ALL_SOURCES`, plus the
    one-shot digital data ingest at the end.

    Order matters because each subsequent source aligns to the cache built
    by the previous ones. The digital data runs LAST so it can anchor itself
    to the y-frame already established by the image sources — and because
    its samples are dense enough (~30 min) to dominate the cache for the
    Jan-Jun 2025 region after the dedupe.
    """
    reports = [ingest(s, history_path=history_path) for s in ALL_SOURCES]
    reports.append(ingest_digital(history_path=history_path))
    return reports


# ─────────────────────────────────────────────────────────────────────────────
# Per-source `Last-Modified` persistence
# ─────────────────────────────────────────────────────────────────────────────


def _load_cached_last_modified(source: TiltSource) -> Optional[str]:
    if not LAST_MODIFIED_FILE.exists():
        return None
    try:
        data = json.loads(LAST_MODIFIED_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data.get(source.value)


def _save_cached_last_modified(source: TiltSource, value: Optional[str]) -> None:
    if value is None:
        return
    LAST_MODIFIED_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = (
            json.loads(LAST_MODIFIED_FILE.read_text())
            if LAST_MODIFIED_FILE.exists()
            else {}
        )
    except (json.JSONDecodeError, OSError):
        existing = {}
    existing[source.value] = value
    LAST_MODIFIED_FILE.write_text(json.dumps(existing, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint — `python -m kilauea_tracker.ingest.pipeline`
# ─────────────────────────────────────────────────────────────────────────────


def _cli_main() -> int:
    """Run the full ingest and print a one-line summary per source.

    Used by the GitHub Actions cache-refresh workflow. Returns 0 if at least
    one source ingested successfully (workflow continues, may commit data),
    or 1 if every source failed (workflow fails loudly).
    """
    reports = ingest_all()
    failures = sum(1 for r in reports if r.error is not None)
    print(f"Ingested {len(reports)} sources ({len(reports) - failures} ok, {failures} failed):")
    for r in reports:
        flag = "OK  " if r.error is None else "FAIL"
        line = (
            f"  [{flag}] {r.source.name:12s} "
            f"fetched={int(r.fetched)}  "
            f"traced={r.rows_traced:5d}  "
            f"added={r.rows_added_to_cache:5d}  "
            f"updated={r.rows_updated_in_cache:5d}"
        )
        if r.error:
            line += f"  error={r.error}"
        print(line)
    if failures == len(reports):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
