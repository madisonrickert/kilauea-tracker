"""JSON schema layer for ingest run reports.

Serialize and deserialize `IngestRunResult` instances against the
`data/run_reports/*.json` shape. Lives in its own module so the schema
is reviewable as a unit and the orchestration in `pipeline.py` doesn't
get diluted by 300+ lines of dict-shaping code.

**Schema version**: writing emits `"schema_version": 2` (introduced
2026-04-25). The deserializer tolerates v1 reports — the dropped fields
(`SourceAlignment.a`, `SourceAlignment.rows_proximity_dropped`,
`PairwiseFit.alpha`, `conflicts_top`/`conflicts_total`) are simply
ignored when present, so older committed reports under
`data/run_reports/` still round-trip.

**On `dataclasses.asdict()`**: a pure asdict() roundtrip would require
renaming the dataclass fields to match the JSON keys (e.g.,
`AxisCalibration.x_start` → `x_start_utc`, `TranscriptionFailure.bucket`
→ `bucket_utc`). That rename touches every consumer of the dataclasses,
not just the serde, so this module keeps the explicit dict
comprehensions instead — same correctness, smaller blast radius. If the
JSON keys ever align with the field names, switch to asdict() then.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ..archive import ArchivePromotionReport
from ..reconcile import ReconcileReport
from ._reports import IngestReport, IngestRunResult, ModelPredictionRecord
from .calibrate import AnchorFitResult

# JSON schema version emitted by `serialize_run_report`. v2 dropped four
# always-constant back-compat fields from the reconcile section; the
# deserializer continues to accept v1 payloads (no version key).
_SCHEMA_VERSION = 2


def serialize_run_report(result: IngestRunResult) -> dict:
    """Serialize an `IngestRunResult` to the on-disk JSON shape."""
    payload: dict = {
        "schema_version": _SCHEMA_VERSION,
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

    # Per-model prediction snapshots. Always emitted — even on runs
    # where every model failed — so the evaluation tab can later see
    # which runs had no usable predictions and which models were
    # responsible.
    if result.predictions:
        payload["predictions"] = [
            {
                "model_id": p.model_id,
                "next_event_date_utc": p.next_event_date_utc,
                "band_lo_utc": p.band_lo_utc,
                "band_hi_utc": p.band_hi_utc,
                "headline_text": p.headline_text,
                "diagnostics": p.diagnostics,
            }
            for p in result.predictions
        ]
    return payload


def deserialize_run_report(payload: dict) -> IngestRunResult:
    """Inverse of `serialize_run_report` — best-effort reconstruction.

    Tolerant of v1 payloads (no `schema_version` key); dropped fields
    are simply ignored.
    """
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
    for p in payload.get("predictions") or []:
        result.predictions.append(
            ModelPredictionRecord(
                model_id=p.get("model_id") or "",
                next_event_date_utc=p.get("next_event_date_utc"),
                band_lo_utc=p.get("band_lo_utc"),
                band_hi_utc=p.get("band_hi_utc"),
                headline_text=p.get("headline_text"),
                diagnostics=dict(p.get("diagnostics") or {}),
            )
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-section serializers
# ─────────────────────────────────────────────────────────────────────────────


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
                "piecewise_residuals": dict(s.piecewise_residuals),
            }
            for s in rep.sources
        ],
        "pairs": [
            {
                "source_i": p.source_i,
                "source_j": p.source_j,
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
        "warnings": list(rep.warnings),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-section deserializers
# ─────────────────────────────────────────────────────────────────────────────


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
    """Reconstruct a `ReconcileReport` from a JSON payload.

    Tolerant of v1 reports (those that still carry the dropped fields
    `a`, `alpha`, `rows_proximity_dropped`, `conflicts_top`,
    `conflicts_total`): the unknown keys are simply ignored.
    """
    from ..reconcile import (
        ContinuityViolation,
        PairwiseFit,
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
                piecewise_residuals=dict(s.get("piecewise_residuals") or {}),
            )
        )
    for f in p.get("pairs") or []:
        rep.pairs.append(
            PairwiseFit(
                source_i=f.get("source_i") or "",
                source_j=f.get("source_j") or "",
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
    return rep


# ─────────────────────────────────────────────────────────────────────────────
# Datetime helpers
# ─────────────────────────────────────────────────────────────────────────────


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
