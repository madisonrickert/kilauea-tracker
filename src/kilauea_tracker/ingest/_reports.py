"""Dataclass definitions for ingest run reports.

Split out of `pipeline.py` so the JSON serde layer (`_runreport_serde.py`)
can import the dataclasses without creating a circular import with the
orchestration entrypoints (`ingest`, `ingest_all`) that consume them.

Public surface — all three dataclasses are re-exported from
`pipeline.py` so existing callers keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    import pandas as pd

    from ..archive import ArchivePromotionReport
    from ..config import TiltSource
    from ..reconcile import ReconcileReport
    from .calibrate import AnchorFitResult, AxisCalibration


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


@dataclass(frozen=True)
class ModelPredictionRecord:
    """Per-model prediction snapshot persisted into each run report.

    One ``ModelPredictionRecord`` per registered model is written on every
    cron run. Long-term, this lets us grade each model against the actual
    peak that arrives later — the evidence any future ensemble decision
    will rest on. ``None`` fields mean the model couldn't fit on the
    just-ingested data; ``diagnostics`` carries the per-model failure
    detail when that happens.
    """

    model_id: str
    next_event_date_utc: str | None       # ISO timestamp string, or None on failure
    band_lo_utc: str | None
    band_hi_utc: str | None
    headline_text: str | None
    diagnostics: dict


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
    # Per-model prediction snapshots — one entry per registered model,
    # appended after reconciliation by ``_compute_model_predictions``.
    # The eventual evaluation surface joins these against actual peaks
    # to compute per-model accuracy.
    predictions: list[ModelPredictionRecord] = field(default_factory=list)
