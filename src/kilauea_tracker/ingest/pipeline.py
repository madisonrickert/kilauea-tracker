"""Top-level ingest orchestration: per-source raw storage + reconciliation.

The architecture is intentionally split into two layers:

  1. **Raw layer** (`data/sources/<source>.csv`): each USGS PNG source has
     its own CSV. `ingest(source)` fetches the PNG, traces the curve, and
     APPENDS the new rows to that source's CSV (with intra-source dedupe at
     15-min buckets). Sources never touch each other's files.

  2. **Reconciled view** (`data/tilt_history.csv`): rebuilt by
     `reconcile.reconcile_sources()` after every ingest run. Reads every
     per-source CSV plus the legacy and digital reference files, computes
     a single y-offset per source against the topologically-prior union of
     other sources, and emits one merged history with priority-based
     per-bucket selection.

This decoupling is what makes the cache deterministic and order-independent
— a problem we used to have when sources cascaded into the cache via
last-write-wins dedupe. See `reconcile.py` for the algorithm and rationale.

Usage from the Streamlit app:

    from kilauea_tracker.ingest.pipeline import ingest_all
    result = ingest_all()
    for r in result.per_source:
        if r.error:
            st.error(r.error)
    for w in result.reconcile.warnings:
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

from ..cache import append_history
from ..config import (
    ALL_SOURCES,
    DIGITAL_CSV,
    DIGITAL_SOURCE_NAME,
    HISTORY_CSV,
    LAST_GOOD_CALIBRATION,
    TILT_SOURCE_NAME,
    USGS_TILT_URLS,
    TiltSource,
    source_csv_path,
)
from ..model import DATE_COL, TILT_COL
from ..reconcile import ReconcileReport, reconcile_sources
from .calibrate import AxisCalibration, calibrate_axes
from .exceptions import CalibrationError, FetchError, IngestError, TraceError
from .fetch import fetch_tilt_png
from .trace import trace_curve

# Where the per-source `Last-Modified` headers are persisted.
LAST_MODIFIED_FILE = LAST_GOOD_CALIBRATION.parent / "last_modified.json"


@dataclass
class IngestReport:
    """Outcome of one `ingest(source)` call. Always populated, even on failure."""

    source: Optional[TiltSource]                # None for the digital source
    source_name: str                            # canonical name in SOURCE_PRIORITY
    fetched: bool = False                       # True iff a fresh PNG arrived
    rows_traced: int = 0                        # rows produced by trace_curve
    rows_appended: int = 0                      # net new rows in the per-source CSV
    last_modified: Optional[str] = None
    calibration: Optional[AxisCalibration] = None
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class IngestRunResult:
    """Outcome of an `ingest_all()` call: per-source reports + reconciliation."""

    per_source: list[IngestReport] = field(default_factory=list)
    reconcile: Optional[ReconcileReport] = None
    history_path: Optional[Path] = None         # where the merged CSV was written


def ingest(
    source: TiltSource,
    *,
    sources_dir: Optional[Path] = None,
) -> IngestReport:
    """Run the fetch → calibrate → trace pipeline for ONE USGS source and
    append the result to that source's per-source CSV.

    Never raises — failures are recorded on the returned `IngestReport`.
    Reconciliation into the merged history is done separately by
    `ingest_all()`; this function is purely about getting the raw rows from
    a single PNG into a single per-source file.
    """
    name = TILT_SOURCE_NAME[source]
    report = IngestReport(source=source, source_name=name)
    url = USGS_TILT_URLS[source]
    cached_last_modified = _load_cached_last_modified(source)

    try:
        fetch_result = fetch_tilt_png(url, cached_last_modified)
    except FetchError as e:
        report.error = f"fetch failed for {source.name}: {e}"
        return report

    report.last_modified = fetch_result.last_modified

    if not fetch_result.changed or fetch_result.body is None:
        # 304 Not Modified — nothing new to ingest. The per-source CSV is
        # already up to date as of the last successful fetch.
        return report

    report.fetched = True

    try:
        img_array = np.frombuffer(fetch_result.body, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise IngestError("cv2.imdecode returned None")
    except Exception as e:
        report.error = f"could not decode {source.name} PNG: {e}"
        return report

    try:
        calibration = calibrate_axes(img)
    except CalibrationError as e:
        report.warnings.append(f"calibration failed for {source.name}: {e}")
        report.error = f"calibration failed: {e}"
        return report

    report.calibration = calibration

    try:
        traced = trace_curve(img, calibration)
    except TraceError as e:
        report.error = f"trace failed for {source.name}: {e}"
        return report

    report.rows_traced = len(traced)

    # Append to the per-source CSV. `append_history` does intra-source dedupe
    # at 15-min buckets — re-tracing the same time period (because the PNG's
    # sliding window still includes it) refreshes those buckets via "keep
    # latest." No cross-source contamination because each source has its own
    # file.
    csv_path = source_csv_path(name)
    if sources_dir is not None:
        csv_path = sources_dir / f"{name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    append_result = append_history(traced, csv_path)
    report.rows_appended = append_result.rows_added + append_result.rows_updated

    _save_cached_last_modified(source, fetch_result.last_modified)
    return report


def ingest_all(
    history_path: Path = HISTORY_CSV,
    *,
    sources_dir: Optional[Path] = None,
) -> IngestRunResult:
    """Run the per-source ingest for every USGS source, then reconcile all
    raw sources into the merged tilt history.

    The merged history at `history_path` is overwritten on every call —
    it's a derived view, not a primary store. The primary stores are the
    per-source CSVs (under `data/sources/`) and the static legacy + digital
    reference files.
    """
    result = IngestRunResult(history_path=history_path)

    # 1. Per-source ingest. Order doesn't matter for correctness anymore —
    #    each ingest writes to its own file. Iterate ALL_SOURCES purely for
    #    a deterministic report ordering.
    for s in ALL_SOURCES:
        result.per_source.append(ingest(s, sources_dir=sources_dir))

    # 2. Read every per-source CSV that exists into the reconcile inputs.
    #    Sources whose ingest just failed will still have their previously-
    #    cached CSV on disk and contribute to reconciliation.
    sources_for_reconcile: dict[str, pd.DataFrame] = {}
    for s in ALL_SOURCES:
        name = TILT_SOURCE_NAME[s]
        csv_path = source_csv_path(name) if sources_dir is None else sources_dir / f"{name}.csv"
        if csv_path.exists():
            try:
                sources_for_reconcile[name] = _read_canonical_csv(csv_path)
            except Exception as e:
                result.per_source[-1].warnings.append(
                    f"could not load {csv_path.name} for reconciliation: {e}"
                )

    # 3. Read the digital reference file. This is NOT a per-source CSV in
    #    data/sources/ — it's the checked-in canonical file produced by
    #    scripts/import_digital_data.py. The legacy hand-traced CSV used to
    #    feed the reconciler too, but it was removed in 2026-04 because its
    #    samples didn't reliably match dec2024_to_now's auto-traced frame
    #    and were creating systemic ~6 µrad offsets. dec2024_to_now covers
    #    the same Jul-Nov 2025 range with one consistent y-frame.
    if DIGITAL_CSV.exists():
        try:
            digital = _read_canonical_csv(DIGITAL_CSV)
            if len(digital) > 0:
                sources_for_reconcile[DIGITAL_SOURCE_NAME] = digital
        except Exception as e:
            print(f"WARNING: could not load digital CSV: {e}")

    # 4. Reconcile and write the merged history.
    merged, reconcile_report = reconcile_sources(sources_for_reconcile)
    result.reconcile = reconcile_report

    history_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(history_path, index=False)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────


def _read_canonical_csv(path: Path) -> pd.DataFrame:
    """Read a tilt CSV with the canonical `[Date, Tilt (microradians)]`
    schema. Tolerant of mixed datetime formats so the same helper handles
    legacy CSVs (US-style dates) and freshly-traced PNG outputs.
    """
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed", dayfirst=False)
    df = df[[DATE_COL, TILT_COL]].dropna()
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


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
    """Run the full ingest+reconcile pipeline and print a one-line summary
    per source plus the reconciliation report.

    Used by the GitHub Actions cache-refresh workflow. Returns 0 if the
    reconcile produced any rows, 1 if everything failed.
    """
    result = ingest_all()
    print(f"Per-source ingest ({len(result.per_source)} sources):")
    for r in result.per_source:
        flag = "OK  " if r.error is None else "FAIL"
        line = (
            f"  [{flag}] {r.source_name:14s} "
            f"fetched={int(r.fetched)}  "
            f"traced={r.rows_traced:5d}  "
            f"appended={r.rows_appended:5d}"
        )
        if r.error:
            line += f"  error={r.error}"
        print(line)

    if result.reconcile is not None:
        print(f"\nReconciliation: {result.reconcile.rows_out} rows in merged history")
        for s in result.reconcile.sources:
            anchor = " (anchor)" if s.is_anchor else ""
            offset_str = (
                f"offset={s.offset_microrad:+.3f} µrad over {s.overlap_buckets} buckets"
                if s.offset_microrad is not None
                else f"UNALIGNED ({s.note})"
            )
            print(f"  {s.name:14s} rows={s.rows_in:6d}  {offset_str}{anchor}")
        if result.reconcile.conflicts:
            print(f"  {len(result.reconcile.conflicts)} bucket conflict(s) detected")
        if result.reconcile.warnings:
            for w in result.reconcile.warnings:
                print(f"  WARN: {w}")

    if result.reconcile is None or result.reconcile.rows_out == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
