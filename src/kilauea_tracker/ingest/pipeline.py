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

import contextlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING

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
from .calibrate import (
    AnchorFitResult,
    AxisCalibration,
    apply_anchor_fit,
    calibrate_axes,
    recalibrate_by_anchor_fit,
)
from .exceptions import CalibrationError, FetchError, IngestError, TraceError
from .fetch import fetch_tilt_png
from .trace import TraceReport, trace_curve

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger(__name__)

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

    source: TiltSource | None                # None for the digital source
    source_name: str                            # canonical name in SOURCE_PRIORITY
    fetched: bool = False                       # True iff a fresh PNG arrived
    rows_traced: int = 0                        # rows produced by trace_curve (after outlier filter)
    rows_raw: int = 0                           # raw samples from the HSV mask before filtering
    rows_outlier_dropped: int = 0               # rows the rolling-median filter removed
    rows_appended: int = 0                      # net new rows in the per-source CSV
    last_modified: str | None = None
    calibration: AxisCalibration | None = None
    # Intra-source frame alignment diagnostics — populated from the
    # AppendReport returned by cache.append_history. A non-zero offset
    # means Part 1 of the 2026-04 drift fix corrected for a y-axis shift
    # between this fetch and the existing per-source CSV.
    frame_offset_microrad: float = 0.0
    frame_overlap_buckets: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    # PSM mode that succeeded on the title OCR ("psm7" or "psm6"), plus the
    # raw title text Tesseract returned. Useful for post-hoc diagnosis when
    # a calibration produces wrong timestamps.
    title_psm_used: str | None = None
    title_raw_text: str | None = None
    # Rows dropped by the per-row outlier filter, each as (timestamp, value,
    # local_median). Kept terse; full detail goes to the JSON run report.
    dropped_outlier_samples: list[tuple[pd.Timestamp, float, float]] = field(
        default_factory=list
    )


@dataclass
class IngestRunResult:
    """Outcome of an `ingest_all()` call: per-source reports + reconciliation."""

    per_source: list[IngestReport] = field(default_factory=list)
    reconcile: ReconcileReport | None = None
    history_path: Path | None = None         # where the merged CSV was written
    archive: ArchivePromotionReport | None = None
    archive_path: Path | None = None         # where the archive CSV was written
    run_report_path: Path | None = None       # where the diagnostic JSON was written
    run_started_at_utc: datetime | None = None
    run_finished_at_utc: datetime | None = None
    # Phase 1c: per-source anchor cross-check results. Empty unless digital
    # is present AND at least one rolling source overlaps digital's Jan-Jun
    # 2025 window (today: dec2024_to_now only).
    anchor_fits: list[AnchorFitResult] = field(default_factory=list)


def ingest(
    source: TiltSource,
    *,
    sources_dir: Path | None = None,
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
        logger.error("fetch failed for %s: %s", source.name, e)
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
        logger.error("PNG decode failed for %s: %s", source.name, e)
        return report

    try:
        calibration = calibrate_axes(img, source_name=name)
    except CalibrationError as e:
        # Set the error ONLY — don't also append to warnings, or the
        # Streamlit panel will show the same message both as a red ❌
        # banner (from the error path) AND as a yellow item in the
        # warnings expander, which is the duplication the user reported
        # in 2026-04.
        report.error = f"calibration failed: {e}"
        logger.error("calibration failed for %s: %s", source.name, e)
        return report

    report.calibration = calibration
    report.title_psm_used = calibration.title_psm_used or None
    report.title_raw_text = calibration.title_raw_text or None

    try:
        traced = trace_curve(img, calibration)
    except TraceError as e:
        report.error = f"trace failed for {source.name}: {e}"
        logger.error("trace failed for %s: %s", source.name, e)
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
            logger.warning(
                "%s: dropped %d per-row outliers (rolling-median gate)",
                source.name, trace_report.outliers_dropped,
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

    # Phase 0a: append a capture-quality row to the per-source quality CSV.
    # Best-effort: a write failure here must never abort the ingest (the
    # per-source CSV is what the pipeline actually reads downstream).
    try:
        _append_quality_row(
            source_name=name,
            sources_dir=sources_dir,
            report=report,
            trace_report=trace_report,
        )
    except Exception as e:  # pragma: no cover — defensive
        report.warnings.append(f"could not write quality CSV: {e}")
        logger.warning("%s: could not write quality CSV: %s", source.name, e)

    return report


def ingest_all(
    history_path: Path = HISTORY_CSV,
    *,
    sources_dir: Path | None = None,
    archive_path: Path = ARCHIVE_CSV,
    on_stage: Callable[[str], None] | None = None,
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
    result.run_started_at_utc = datetime.now(tz=UTC)

    def _emit(msg: str) -> None:
        if on_stage is not None:
            on_stage(msg)

    # 1. Per-source ingest. Order doesn't matter for correctness anymore —
    #    each ingest writes to its own file. Iterate ALL_SOURCES purely for
    #    a deterministic report ordering.
    for s in ALL_SOURCES:
        _emit(f"Fetching {TILT_SOURCE_NAME[s]}…")
        result.per_source.append(ingest(s, sources_dir=sources_dir))

    _emit("Loading source files…")

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
            logger.warning("could not load digital CSV: %s", e)

    # 4. Read the append-only archive. On first run after a fresh checkout
    #    this is empty; subsequent runs see whatever rows have already been
    #    promoted from prior reconciles.
    archive_df = load_archive(archive_path)
    if len(archive_df) > 0:
        sources_for_reconcile[ARCHIVE_SOURCE_NAME] = archive_df

    # 4b. Phase 1c: anchor-referenced calibration cross-check. For every
    #     PNG source that temporally overlaps `digital` (today: only
    #     dec2024_to_now), fit `digital = a * png + b` via Huber-robust
    #     regression and apply the correction IN MEMORY for this reconcile
    #     pass when the fit deviates from identity. The per-source CSV on
    #     disk is NOT modified — this gives us a reversible, rerun-stable
    #     correction layer that can be tuned without touching the raw data.
    digital_df = sources_for_reconcile.get(DIGITAL_SOURCE_NAME)
    if digital_df is not None and len(digital_df) > 0:
        _emit("Calibrating against digital reference…")
        for source_name in list(sources_for_reconcile.keys()):
            if source_name in (DIGITAL_SOURCE_NAME, ARCHIVE_SOURCE_NAME):
                continue
            src_df = sources_for_reconcile[source_name]
            fit = recalibrate_by_anchor_fit(source_name, src_df, digital_df)
            result.anchor_fits.append(fit)
            if fit.warning is not None:
                # Apply the correction and surface the warning on the
                # originating source's report (or a free-floating reconcile
                # warning if we can't match it).
                sources_for_reconcile[source_name] = apply_anchor_fit(src_df, fit)
                matched = next(
                    (r for r in result.per_source if r.source_name == source_name),
                    None,
                )
                if matched is not None:
                    matched.warnings.append(fit.warning)

    _emit("Reconciling sources…")

    # 5. Reconcile and write the merged history.
    merged, reconcile_report = reconcile_sources(sources_for_reconcile)
    result.reconcile = reconcile_report

    history_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(history_path, index=False)

    _emit("Updating archive…")

    # 6. Promote any new merged rows into the archive (keep-first dedupe
    #    + quorum gate). We pass the reconcile inputs through so
    #    promote_to_archive can verify that ≥ARCHIVE_QUORUM_MIN_SOURCES
    #    contributed to each bucket before it becomes permanent — the fix
    #    for the 2026-04 week-PNG phantom-spike contamination.
    result.archive = promote_to_archive(
        merged, archive_path, sources=sources_for_reconcile
    )

    result.run_finished_at_utc = datetime.now(tz=UTC)

    # 7. Write a structured JSON diagnostic per-run for persistent
    #    prod observability. Best-effort: a write failure here must never
    #    fail the pipeline — the CSVs are what the app actually reads.
    try:
        result.run_report_path = _write_run_report(result)
        _prune_old_run_reports(RUN_REPORTS_DIR)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("could not write run report: %s", e)

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
# Phase 0a: Per-source capture-quality CSV
# ─────────────────────────────────────────────────────────────────────────────


# Column order is stable across runs so simple CSV tools can diff the files.
_QUALITY_CSV_COLUMNS = [
    "run_timestamp_utc",
    "y_slope",
    "y_intercept",
    "y_max_residual_microrad",
    "y_max_residual_pixels",
    "y_std_residual_microrad",
    "y_labels_used",
    "y_slope_fallback_used",
    "y_slope_history_median",
    "x_start_utc",
    "x_end_utc",
    "x_span_hours",
    "x_tick_cross_check_count",
    "x_tick_cross_check_median_err_px",
    "x_tick_cross_check_max_err_px",
    "title_psm_used",
    "plot_bbox",
    "column_coverage",
    "rows_raw",
    "rows_after_outlier_filter",
    "rows_dropped_outlier",
    "rows_dropped_rate",
    "columns_dropped_width",
    "rows_appended",
    "frame_offset_microrad",
    "frame_overlap_buckets",
]


def _append_quality_row(
    *,
    source_name: str,
    sources_dir: Path | None,
    report: IngestReport,
    trace_report: TraceReport | None,
) -> None:
    """Append a single row to `data/sources/<source>_quality.csv`.

    Builds the row from the freshly-computed IngestReport + its attached
    TraceReport. Best-effort: callers catch and log failures so the ingest
    pipeline never aborts on a quality-log error.
    """
    calib = report.calibration
    quality_path = (
        source_csv_path(source_name).with_name(f"{source_name}_quality.csv")
        if sources_dir is None
        else sources_dir / f"{source_name}_quality.csv"
    )
    quality_path.parent.mkdir(parents=True, exist_ok=True)

    row: dict[str, object] = {
        "run_timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "y_slope": getattr(calib, "y_slope", ""),
        "y_intercept": getattr(calib, "y_intercept", ""),
        "y_max_residual_microrad": (
            calib.fit_residual_per_axis.get("y_max_residual_microrad", "")
            if calib is not None
            else ""
        ),
        "y_max_residual_pixels": (
            calib.fit_residual_per_axis.get("y_max_residual_pixels", "")
            if calib is not None
            else ""
        ),
        "y_std_residual_microrad": (
            calib.fit_residual_per_axis.get("y_std_residual_microrad", "")
            if calib is not None
            else ""
        ),
        "y_labels_used": len(calib.y_labels_found) if calib is not None else 0,
        "y_slope_fallback_used": (
            bool(getattr(calib, "y_slope_fallback_used", False))
            if calib is not None
            else False
        ),
        "y_slope_history_median": (
            getattr(calib, "y_slope_history_median", None) or ""
            if calib is not None
            else ""
        ),
        "x_start_utc": _dt_str(getattr(calib, "x_start", None)) or "",
        "x_end_utc": _dt_str(getattr(calib, "x_end", None)) or "",
        "x_span_hours": (
            round((calib.x_end - calib.x_start).total_seconds() / 3600.0, 4)
            if calib is not None
            else ""
        ),
        "x_tick_cross_check_count": (
            getattr(calib, "x_tick_cross_check_count", 0) if calib is not None else ""
        ),
        "x_tick_cross_check_median_err_px": (
            (getattr(calib, "x_tick_cross_check_median_err_px", None) or "")
            if calib is not None
            else ""
        ),
        "x_tick_cross_check_max_err_px": (
            (getattr(calib, "x_tick_cross_check_max_err_px", None) or "")
            if calib is not None
            else ""
        ),
        "title_psm_used": (getattr(calib, "title_psm_used", "") or "") if calib is not None else "",
        "plot_bbox": (
            "|".join(str(v) for v in calib.plot_bbox) if calib is not None else ""
        ),
        "column_coverage": (
            trace_report.column_coverage if trace_report is not None else ""
        ),
        "rows_raw": trace_report.rows_raw if trace_report is not None else "",
        "rows_after_outlier_filter": (
            trace_report.rows_after_outlier_filter if trace_report is not None else ""
        ),
        "rows_dropped_outlier": (
            trace_report.outliers_dropped if trace_report is not None else ""
        ),
        "rows_dropped_rate": (
            trace_report.rows_dropped_rate if trace_report is not None else ""
        ),
        "columns_dropped_width": (
            trace_report.columns_dropped_width if trace_report is not None else ""
        ),
        "rows_appended": report.rows_appended,
        "frame_offset_microrad": report.frame_offset_microrad,
        "frame_overlap_buckets": report.frame_overlap_buckets,
    }

    # Write header + row on first write; append thereafter. Using pandas
    # to serialize handles quoting edge cases automatically.
    df_row = pd.DataFrame([row], columns=_QUALITY_CSV_COLUMNS)
    header_needed = not quality_path.exists()
    df_row.to_csv(quality_path, mode="a", index=False, header=header_needed)


# ─────────────────────────────────────────────────────────────────────────────
# Per-source `Last-Modified` persistence
# ─────────────────────────────────────────────────────────────────────────────


def _load_cached_last_modified(source: TiltSource) -> str | None:
    if not LAST_MODIFIED_FILE.exists():
        return None
    try:
        data = json.loads(LAST_MODIFIED_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data.get(source.value)


def _save_cached_last_modified(source: TiltSource, value: str | None) -> None:
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
    ts = result.run_finished_at_utc or datetime.now(tz=UTC)
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

    # Phase 1c: serialize anchor cross-check results for post-hoc diagnosis.
    if result.anchor_fits:
        payload["anchor_fits"] = [
            {
                "source_name": f.source_name,
                "ran": f.ran,
                "overlap_buckets": f.overlap_buckets,
                "a": f.a,
                "b": f.b,
                "residual_std_microrad": f.residual_std_microrad,
                "warning": f.warning,
                "note": f.note,
            }
            for f in result.anchor_fits
        ]

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
                "a": s.a,
                "b": s.b,
                "offset_microrad": s.offset_microrad,
                "overlap_buckets": s.overlap_buckets,
                "pairs_used": s.pairs_used,
                "effective_resolution_microrad_per_pixel": (
                    s.effective_resolution_microrad_per_pixel
                ),
                "rows_mad_rejected": s.rows_mad_rejected,
                "is_anchor": s.is_anchor,
                "note": s.note,
                "rows_proximity_dropped": s.rows_proximity_dropped,
                "piecewise_residuals": dict(s.piecewise_residuals),
            }
            for s in rep.sources
        ],
        "pairs": [
            {
                "source_i": p.source_i,
                "source_j": p.source_j,
                "alpha": p.alpha,
                "beta": p.beta,
                "overlap_buckets": p.overlap_buckets,
                "residual_std_microrad": p.residual_std_microrad,
            }
            for p in rep.pairs
        ],
        "winner_counts": dict(rep.winner_counts),
        "transcription_failures_top": [
            {
                "bucket_utc": _dt_str(f.bucket),
                "source": f.source,
                "value_corrected": f.value_corrected,
                "bucket_median": f.bucket_median,
                "delta_microrad": f.delta_microrad,
            }
            for f in sorted(
                rep.transcription_failures,
                key=lambda f: abs(f.delta_microrad),
                reverse=True,
            )[:20]
        ],
        "transcription_failures_total": len(rep.transcription_failures),
        "continuity_violations": [
            {
                "bucket_before": _dt_str(v.bucket_before),
                "bucket_after": _dt_str(v.bucket_after),
                "tilt_before": v.tilt_before,
                "tilt_after": v.tilt_after,
                "delta_microrad": v.delta_microrad,
            }
            for v in rep.continuity_violations
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


def _dt_str(value: pd.Timestamp | datetime | None) -> str | None:
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
        return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(value)


def load_latest_run_report() -> IngestRunResult | None:
    """Read the newest `data/run_reports/*.json` and reconstruct the
    `IngestRunResult` it describes.

    Used by the Streamlit app at module scope to keep Pipeline-tab
    diagnostics populated without firing a fresh ingest on every page load.

    Best-effort: returns None on missing directory, no JSON files, or any
    parse error — the caller is expected to fall back to an empty
    `IngestRunResult()`. Lossy fields in the JSON (e.g., transcription
    failures are truncated to top 20 in the on-disk shape) round-trip with
    that same truncation; the diagnostics surfaces only show the top-20
    anyway, so this is fine.
    """
    if not RUN_REPORTS_DIR.exists():
        return None
    files = sorted(RUN_REPORTS_DIR.glob("*.json"))
    if not files:
        return None
    latest = files[-1]
    try:
        payload = json.loads(latest.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return _deserialize_run_report(payload)


def data_age_seconds() -> float:
    """Age in seconds of the freshest on-disk USGS source.

    Reads the per-source `Last-Modified` headers persisted to
    `data/last_modified.json` by `_save_cached_last_modified`. Returns the
    age of the newest of those headers (the most recently-published source
    on USGS's side, not the most-recently-fetched). Returns `math.inf` if
    the file is missing, unparseable, or empty — the caller treats that as
    "stale, please refresh."
    """
    if not LAST_MODIFIED_FILE.exists():
        return math.inf
    try:
        data = json.loads(LAST_MODIFIED_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return math.inf
    newest: datetime | None = None
    for value in (data or {}).values():
        if not value:
            continue
        try:
            ts = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if newest is None or ts > newest:
            newest = ts
    if newest is None:
        return math.inf
    return (datetime.now(tz=UTC) - newest).total_seconds()


# Refresh-cooldown gating moved to ``kilauea_tracker.state.RefreshStore``
# (see ``src/kilauea_tracker/state/refresh_store.py``). The store owns
# both the cooldown lock and the richer ``running / current_stage /
# finished_utc`` schema the topbar fragment subscribes to. The legacy
# ``data/last_refresh.json`` file is obsolete; the store reads/writes
# ``data/refresh_status.json`` instead.


def _deserialize_run_report(payload: dict) -> IngestRunResult:
    """Inverse of `_write_run_report` — best-effort reconstruction."""
    result = IngestRunResult()
    result.run_started_at_utc = _parse_dt(payload.get("run_started_at_utc"))
    result.run_finished_at_utc = _parse_dt(payload.get("run_finished_at_utc"))
    result.per_source = [
        _deserialize_source_report(p) for p in payload.get("per_source") or []
    ]
    if payload.get("reconcile"):
        result.reconcile = _deserialize_reconcile(payload["reconcile"])
    if payload.get("archive"):
        a = payload["archive"]
        result.archive = ArchivePromotionReport(
            rows_in_archive_before=int(a.get("rows_in_archive_before") or 0),
            rows_in_archive_after=int(a.get("rows_in_archive_after") or 0),
            rows_promoted=int(a.get("rows_promoted") or 0),
            rows_already_archived=int(a.get("rows_already_archived") or 0),
            rows_deferred_by_quorum=int(a.get("rows_deferred_by_quorum") or 0),
            warnings=list(a.get("warnings") or []),
        )
    for f in payload.get("anchor_fits") or []:
        result.anchor_fits.append(
            AnchorFitResult(
                source_name=f.get("source_name") or "",
                ran=bool(f.get("ran")),
                overlap_buckets=int(f.get("overlap_buckets") or 0),
                a=float(f.get("a") or 1.0),
                b=float(f.get("b") or 0.0),
                residual_std_microrad=float(f.get("residual_std_microrad") or 0.0),
                warning=f.get("warning"),
                note=f.get("note") or "",
            )
        )
    return result


def _deserialize_source_report(p: dict) -> IngestReport:
    r = IngestReport(source=None, source_name=p.get("source_name") or "")
    r.fetched = bool(p.get("fetched"))
    r.rows_raw = int(p.get("rows_raw") or 0)
    r.rows_outlier_dropped = int(p.get("rows_outlier_dropped") or 0)
    r.rows_traced = int(p.get("rows_traced") or 0)
    r.rows_appended = int(p.get("rows_appended") or 0)
    r.frame_offset_microrad = float(p.get("frame_offset_microrad") or 0.0)
    r.frame_overlap_buckets = int(p.get("frame_overlap_buckets") or 0)
    r.last_modified = p.get("last_modified")
    r.title_psm_used = p.get("title_psm_used")
    r.title_raw_text = p.get("title_raw_text")
    r.warnings = list(p.get("warnings") or [])
    r.error = p.get("error")
    return r


def _deserialize_reconcile(p: dict) -> ReconcileReport:
    from ..reconcile import (
        ContinuityViolation,
        PairwiseFit,
        ReconcileConflict,
        SourceAlignment,
        TranscriptionFailure,
    )
    rep = ReconcileReport()
    rep.rows_out = int(p.get("rows_out") or 0)
    rep.warnings = list(p.get("warnings") or [])
    rep.winner_counts = dict(p.get("winner_counts") or {})
    for s in p.get("sources") or []:
        rep.sources.append(
            SourceAlignment(
                name=s.get("name") or "",
                rows_in=int(s.get("rows_in") or 0),
                a=float(s.get("a") or 1.0),
                b=float(s.get("b") or 0.0),
                pairs_used=int(s.get("pairs_used") or 0),
                is_anchor=bool(s.get("is_anchor")),
                note=s.get("note"),
                rows_mad_rejected=int(s.get("rows_mad_rejected") or 0),
                effective_resolution_microrad_per_pixel=float(
                    s.get("effective_resolution_microrad_per_pixel") or 0.0
                ),
                offset_microrad=s.get("offset_microrad"),
                overlap_buckets=int(s.get("overlap_buckets") or 0),
                rows_proximity_dropped=int(s.get("rows_proximity_dropped") or 0),
                piecewise_residuals=dict(s.get("piecewise_residuals") or {}),
            )
        )
    for f in p.get("pairs") or []:
        rep.pairs.append(
            PairwiseFit(
                source_i=f.get("source_i") or "",
                source_j=f.get("source_j") or "",
                alpha=float(f.get("alpha") or 1.0),
                beta=float(f.get("beta") or 0.0),
                overlap_buckets=int(f.get("overlap_buckets") or 0),
                residual_std_microrad=float(f.get("residual_std_microrad") or 0.0),
            )
        )
    for tf in p.get("transcription_failures_top") or []:
        bucket = _parse_dt(tf.get("bucket_utc"))
        rep.transcription_failures.append(
            TranscriptionFailure(
                bucket=pd.Timestamp(bucket) if bucket else pd.Timestamp(0),
                source=tf.get("source") or "",
                value_corrected=float(tf.get("value_corrected") or 0.0),
                bucket_median=float(tf.get("bucket_median") or 0.0),
                delta_microrad=float(tf.get("delta_microrad") or 0.0),
            )
        )
    for cv in p.get("continuity_violations") or []:
        rep.continuity_violations.append(
            ContinuityViolation(
                bucket_before=pd.Timestamp(_parse_dt(cv.get("bucket_before")) or 0),
                bucket_after=pd.Timestamp(_parse_dt(cv.get("bucket_after")) or 0),
                tilt_before=float(cv.get("tilt_before") or 0.0),
                tilt_after=float(cv.get("tilt_after") or 0.0),
                delta_microrad=float(cv.get("delta_microrad") or 0.0),
            )
        )
    for c in p.get("conflicts_top") or []:
        rep.conflicts.append(
            ReconcileConflict(
                bucket=pd.Timestamp(_parse_dt(c.get("bucket_utc")) or 0),
                winning_source=c.get("winning_source") or "",
                losing_source=c.get("losing_source") or "",
                winning_tilt=float(c.get("winning_tilt") or 0.0),
                losing_tilt=float(c.get("losing_tilt") or 0.0),
                delta=float(c.get("delta") or 0.0),
            )
        )
    return rep


def _parse_dt(value: str | datetime | None) -> datetime | None:
    """Inverse of `_dt_str` — tolerant of trailing 'Z' and missing tz."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


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
        with contextlib.suppress(OSError):
            old.unlink()


if __name__ == "__main__":
    raise SystemExit(_cli_main())
