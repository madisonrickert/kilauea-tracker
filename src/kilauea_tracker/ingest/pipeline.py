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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from ..archive import ArchivePromotionReport, load_archive, promote_to_archive
from ..cache import append_history
from ..config import (
    ALL_SOURCES,
    ARCHIVE_CSV,
    ARCHIVE_SOURCE_NAME,
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

# Where structured per-run diagnostic reports are written. One JSON file per
# ingest_all() invocation; the GitHub Actions workflow commits them alongside
# the CSV updates so prod diagnoses don't depend on stdout that vanishes when
# the workflow run is retired. See _write_run_report below.
RUN_REPORTS_DIR = LAST_GOOD_CALIBRATION.parent / "run_reports"

# How many run report files to keep on disk. Older ones are pruned by
# `_prune_old_run_reports` on every write. 90 is roughly one quarter of
# daily cron runs; enough to investigate a slow-burn issue but not so many
# that the repo bloats.
MAX_RUN_REPORTS = 90


@dataclass
class IngestReport:
    """Outcome of one `ingest(source)` call. Always populated, even on failure."""

    source: Optional[TiltSource]                # None for the digital source
    source_name: str                            # canonical name in SOURCE_PRIORITY
    fetched: bool = False                       # True iff a fresh PNG arrived
    rows_traced: int = 0                        # rows produced by trace_curve (after outlier filter)
    rows_raw: int = 0                           # raw samples from the HSV mask before filtering
    rows_outlier_dropped: int = 0               # rows the rolling-median filter removed
    rows_appended: int = 0                      # net new rows in the per-source CSV
    last_modified: Optional[str] = None
    calibration: Optional[AxisCalibration] = None
    # Intra-source frame alignment diagnostics — populated from the
    # AppendReport returned by cache.append_history. A non-zero offset
    # means Part 1 of the 2026-04 drift fix corrected for a y-axis shift
    # between this fetch and the existing per-source CSV.
    frame_offset_microrad: float = 0.0
    frame_overlap_buckets: int = 0
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None
    # PSM mode that succeeded on the title OCR ("psm7" or "psm6"), plus the
    # raw title text Tesseract returned. Useful for post-hoc diagnosis when
    # a calibration produces wrong timestamps.
    title_psm_used: Optional[str] = None
    title_raw_text: Optional[str] = None
    # Rows dropped by the per-row outlier filter, each as (timestamp, value,
    # local_median). Kept terse; full detail goes to the JSON run report.
    dropped_outlier_samples: list[tuple[pd.Timestamp, float, float]] = field(
        default_factory=list
    )


@dataclass
class IngestRunResult:
    """Outcome of an `ingest_all()` call: per-source reports + reconciliation."""

    per_source: list[IngestReport] = field(default_factory=list)
    reconcile: Optional[ReconcileReport] = None
    history_path: Optional[Path] = None         # where the merged CSV was written
    archive: Optional[ArchivePromotionReport] = None
    archive_path: Optional[Path] = None         # where the archive CSV was written
    run_report_path: Optional[Path] = None       # where the diagnostic JSON was written
    run_started_at_utc: Optional[datetime] = None
    run_finished_at_utc: Optional[datetime] = None


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
        # Set the error ONLY — don't also append to warnings, or the
        # Streamlit panel will show the same message both as a red ❌
        # banner (from the error path) AND as a yellow item in the
        # warnings expander, which is the duplication the user reported
        # in 2026-04.
        report.error = f"calibration failed: {e}"
        return report

    report.calibration = calibration
    report.title_psm_used = calibration.title_psm_used or None
    report.title_raw_text = calibration.title_raw_text or None

    try:
        traced = trace_curve(img, calibration)
    except TraceError as e:
        report.error = f"trace failed for {source.name}: {e}"
        return report

    report.rows_traced = len(traced)
    trace_report = traced.attrs.get("trace_report")
    if trace_report is not None:
        report.rows_raw = trace_report.rows_raw
        report.rows_outlier_dropped = trace_report.outliers_dropped
        report.dropped_outlier_samples = list(trace_report.dropped_rows)
        if trace_report.outliers_dropped > 0:
            report.warnings.append(
                f"dropped {trace_report.outliers_dropped} per-row outlier(s) "
                f"(|delta| > threshold vs rolling median)"
            )

    # Append to the per-source CSV. `append_history` does intra-source
    # frame alignment + dedupe at 15-min buckets — re-tracing the same
    # time period (because the PNG's sliding window still includes it)
    # has its frame anchored to the existing CSV's first-fetch frame
    # before the dedupe runs, so the per-source CSV stays in one
    # consistent y-frame across many fetches. No cross-source contamination
    # because each source has its own file.
    csv_path = source_csv_path(name)
    if sources_dir is not None:
        csv_path = sources_dir / f"{name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    append_result = append_history(traced, csv_path)
    report.rows_appended = append_result.rows_added + append_result.rows_updated
    # Surface frame alignment diagnostics + warnings so the CLI / UI can
    # show them. Without this propagation Part 1's diagnostics would be
    # invisible in production.
    report.frame_offset_microrad = append_result.frame_offset_microrad
    report.frame_overlap_buckets = append_result.frame_overlap_buckets
    report.warnings.extend(append_result.warnings)

    _save_cached_last_modified(source, fetch_result.last_modified)
    return report


def ingest_all(
    history_path: Path = HISTORY_CSV,
    *,
    sources_dir: Optional[Path] = None,
    archive_path: Path = ARCHIVE_CSV,
) -> IngestRunResult:
    """Run the per-source ingest for every USGS source, reconcile all raw
    sources into the merged tilt history, then promote any new rows into
    the append-only archive.

    The merged history at `history_path` is overwritten on every call —
    it's a derived view, not a primary store. The primary stores are
    the per-source CSVs (under `data/sources/`), the static digital
    reference file, and the append-only archive at `archive_path`.

    The archive is BOTH an input to reconciliation AND an output target:

      - As an input, it sits in `SOURCE_PRIORITY` just below `digital`,
        so reconciled timestamps that already live in the archive get
        sourced from the archive (immune to live-source drift).
      - As an output, any newly-merged timestamps not already in the
        archive get appended to it via `promote_to_archive()` (keep-
        first dedupe). Once a row lands in the archive, the next
        ingest_all() run will source it from there forever.
    """
    result = IngestRunResult(history_path=history_path, archive_path=archive_path)
    result.run_started_at_utc = datetime.now(tz=timezone.utc)

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
    #    scripts/import_digital_data.py.
    if DIGITAL_CSV.exists():
        try:
            digital = _read_canonical_csv(DIGITAL_CSV)
            if len(digital) > 0:
                sources_for_reconcile[DIGITAL_SOURCE_NAME] = digital
        except Exception as e:
            print(f"WARNING: could not load digital CSV: {e}")

    # 4. Read the append-only archive. On first run after a fresh checkout
    #    this is empty; subsequent runs see whatever rows have already been
    #    promoted from prior reconciles.
    archive_df = load_archive(archive_path)
    if len(archive_df) > 0:
        sources_for_reconcile[ARCHIVE_SOURCE_NAME] = archive_df

    # 5. Reconcile and write the merged history.
    merged, reconcile_report = reconcile_sources(sources_for_reconcile)
    result.reconcile = reconcile_report

    history_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(history_path, index=False)

    # 6. Promote any new merged rows into the archive (keep-first dedupe
    #    + quorum gate). We pass the reconcile inputs through so
    #    promote_to_archive can verify that ≥ARCHIVE_QUORUM_MIN_SOURCES
    #    contributed to each bucket before it becomes permanent — the fix
    #    for the 2026-04 week-PNG phantom-spike contamination.
    result.archive = promote_to_archive(
        merged, archive_path, sources=sources_for_reconcile
    )

    result.run_finished_at_utc = datetime.now(tz=timezone.utc)

    # 7. Write a structured JSON diagnostic per-run for persistent
    #    prod observability. Best-effort: a write failure here must never
    #    fail the pipeline — the CSVs are what the app actually reads.
    try:
        result.run_report_path = _write_run_report(result)
        _prune_old_run_reports(RUN_REPORTS_DIR)
    except Exception as e:  # pragma: no cover — defensive
        print(f"WARNING: could not write run report: {e}")

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
    per source plus the reconciliation and archive reports.

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
        # Surface intra-source frame-alignment diagnostics from cache.
        # append_history. These tell us whether Part 1 of the 2026-04 drift
        # fix is actually doing its job in production: a non-zero offset
        # means we corrected for a y-axis shift between fetches.
        if hasattr(r, "frame_offset_microrad") and r.frame_overlap_buckets:
            line += (
                f"  frame_offset={r.frame_offset_microrad:+.3f} µrad "
                f"over {r.frame_overlap_buckets} buckets"
            )
        if r.error:
            line += f"  error={r.error}"
        print(line)
        for w in r.warnings:
            print(f"      WARN: {w}")

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

    # Archive growth — Part 2 of the drift fix. The archive accumulates
    # one row per timestamp the first time we ever observe it; subsequent
    # runs that see the same timestamp report it as already_archived. A
    # healthy run should show monotonic growth equal to (or close to) the
    # number of genuinely new timestamps the latest fetch produced.
    if result.archive is not None:
        a = result.archive
        print(
            f"\nArchive: {a.rows_in_archive_before} → {a.rows_in_archive_after} rows  "
            f"(promoted={a.rows_promoted}, already_archived={a.rows_already_archived})"
        )
        for w in a.warnings:
            print(f"  WARN: {w}")

    if result.reconcile is None or result.reconcile.rows_out == 0:
        return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Run report — persistent prod diagnostic
# ─────────────────────────────────────────────────────────────────────────────


def _write_run_report(result: IngestRunResult) -> Path:
    """Serialize the full IngestRunResult to `data/run_reports/<ts>.json`.

    The committed report is the primary prod observability surface — stdout
    from the cron evaporates with the workflow run. The JSON is stable
    enough to grep, diff, and replay for post-hoc diagnosis of bad rows.

    Path format uses a colon-free UTC timestamp so the filename works on
    any filesystem: `YYYY-MM-DDTHH-MM-SSZ.json`.
    """
    RUN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = result.run_finished_at_utc or datetime.now(tz=timezone.utc)
    stem = ts.strftime("%Y-%m-%dT%H-%M-%SZ")
    out_path = RUN_REPORTS_DIR / f"{stem}.json"

    payload: dict = {
        "run_started_at_utc": _dt_str(result.run_started_at_utc),
        "run_finished_at_utc": _dt_str(result.run_finished_at_utc),
        "per_source": [_serialize_source_report(r) for r in result.per_source],
    }
    if result.reconcile is not None:
        payload["reconcile"] = _serialize_reconcile(result.reconcile)
    if result.archive is not None:
        a = result.archive
        payload["archive"] = {
            "rows_in_archive_before": a.rows_in_archive_before,
            "rows_in_archive_after": a.rows_in_archive_after,
            "rows_promoted": a.rows_promoted,
            "rows_already_archived": a.rows_already_archived,
            "rows_deferred_by_quorum": a.rows_deferred_by_quorum,
            "warnings": list(a.warnings),
        }

    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path


def _serialize_source_report(r: IngestReport) -> dict:
    payload = {
        "source_name": r.source_name,
        "fetched": r.fetched,
        "rows_raw": r.rows_raw,
        "rows_outlier_dropped": r.rows_outlier_dropped,
        "rows_traced": r.rows_traced,
        "rows_appended": r.rows_appended,
        "frame_offset_microrad": r.frame_offset_microrad,
        "frame_overlap_buckets": r.frame_overlap_buckets,
        "last_modified": r.last_modified,
        "title_psm_used": r.title_psm_used,
        "title_raw_text": r.title_raw_text,
        "warnings": list(r.warnings),
        "error": r.error,
    }
    if r.calibration is not None:
        c = r.calibration
        payload["calibration"] = {
            "plot_bbox": list(c.plot_bbox),
            "y_slope": c.y_slope,
            "y_intercept": c.y_intercept,
            "x_start_utc": _dt_str(c.x_start),
            "x_end_utc": _dt_str(c.x_end),
            "y_labels_found": [[int(py), float(v)] for py, v in c.y_labels_found],
            "y_max_residual_microrad": c.fit_residual_per_axis.get(
                "y_max_residual_microrad"
            ),
        }
    if r.dropped_outlier_samples:
        payload["dropped_outlier_samples"] = [
            {"date_utc": _dt_str(ts), "tilt": tilt, "local_median": med}
            for ts, tilt, med in r.dropped_outlier_samples
        ]
    return payload


def _serialize_reconcile(rep: ReconcileReport) -> dict:
    return {
        "rows_out": rep.rows_out,
        "sources": [
            {
                "name": s.name,
                "rows_in": s.rows_in,
                "offset_microrad": s.offset_microrad,
                "overlap_buckets": s.overlap_buckets,
                "is_anchor": s.is_anchor,
                "note": s.note,
                "rows_proximity_dropped": s.rows_proximity_dropped,
                "piecewise_residuals": dict(s.piecewise_residuals),
            }
            for s in rep.sources
        ],
        "conflicts_top": [
            {
                "bucket_utc": _dt_str(c.bucket),
                "winning_source": c.winning_source,
                "losing_source": c.losing_source,
                "winning_tilt": c.winning_tilt,
                "losing_tilt": c.losing_tilt,
                "delta": c.delta,
            }
            for c in sorted(
                rep.conflicts, key=lambda c: abs(c.delta), reverse=True
            )[:20]
        ],
        "conflicts_total": len(rep.conflicts),
        "warnings": list(rep.warnings),
    }


def _dt_str(value) -> Optional[str]:
    """ISO-8601 serialization for datetime-like values.

    Handles both pd.Timestamp (nanosecond-precision) and stdlib datetime
    without going through `.to_pydatetime()`, which drops nanoseconds and
    emits a UserWarning for any timestamp that has them. Nanoseconds don't
    survive round-tripping to ISO anyway (Python's datetime is microsecond
    resolution), but we can format a pandas Timestamp directly at
    microsecond precision without the warning.
    """
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        fmt = "%Y-%m-%dT%H:%M:%S.%f"
        if value.tzinfo is None:
            return value.strftime(fmt)
        return value.tz_convert("UTC").strftime(fmt) + "Z"
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.strftime("%Y-%m-%dT%H:%M:%S")
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(value)


def _prune_old_run_reports(
    reports_dir: Path, *, keep: int = MAX_RUN_REPORTS
) -> None:
    """Delete the oldest reports when more than `keep` files are present.

    Keeps the directory bounded so the repo doesn't bloat over time while
    still preserving enough history for quarterly investigations.
    """
    if not reports_dir.exists():
        return
    files = sorted(reports_dir.glob("*.json"))
    excess = len(files) - keep
    if excess <= 0:
        return
    for old in files[:excess]:
        try:
            old.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(_cli_main())
