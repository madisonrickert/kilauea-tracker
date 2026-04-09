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

from ..cache import AppendReport, append_history
from ..config import (
    ALL_SOURCES,
    HISTORY_CSV,
    LAST_GOOD_CALIBRATION,
    USGS_TILT_URLS,
    TiltSource,
)
from .calibrate import AxisCalibration, calibrate_axes
from .exceptions import CalibrationError, FetchError, IngestError, TraceError
from .fetch import fetch_tilt_png
from .trace import trace_curve

# Where the per-source `Last-Modified` headers are persisted.
LAST_MODIFIED_FILE = LAST_GOOD_CALIBRATION.parent / "last_modified.json"


@dataclass
class IngestReport:
    """Outcome of one `ingest(source)` call. Always populated, even on failure."""

    source: TiltSource
    fetched: bool = False                       # True iff a fresh PNG arrived
    rows_traced: int = 0                        # rows produced by trace_curve
    rows_added_to_cache: int = 0                # net new rows after dedupe
    rows_updated_in_cache: int = 0
    cache_conflicts: list[dict] = field(default_factory=list)
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


def ingest_all(history_path: Path = HISTORY_CSV) -> list[IngestReport]:
    """Run `ingest()` for every source in `config.ALL_SOURCES`.

    Sources are processed in order from longest to shortest time window
    (3-month → 2-day) so that the high-resolution short-window captures take
    precedence in the cache dedupe (`keep="last"`) when their timestamps
    overlap with the broader 3-month capture.
    """
    return [ingest(s, history_path=history_path) for s in ALL_SOURCES]


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
