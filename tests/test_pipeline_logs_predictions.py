"""Tests for prediction logging in the ingest pipeline.

Each registered model's prediction is captured into the run report on
every cron run. This is the data the eventual evaluation tab will join
against actual peak arrivals to grade per-model accuracy. Without this
log we'd have no historical record of what any model said on any given
day.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

from kilauea_tracker.ingest.pipeline import (
    IngestRunResult,
    ModelPredictionRecord,
    _compute_model_predictions,
    _write_run_report,
)
from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.models import registry


def _synthetic_history() -> pd.DataFrame:
    """A minimal sawtooth — ~60 days of data spanning enough peaks to
    let every registered model produce a prediction."""
    base = pd.Timestamp("2026-01-01")
    n_samples = 800
    days = np.linspace(0, 60, n_samples)
    # Six sawtooth cycles: rises 0→8 µrad linearly over each segment,
    # then drops back to 0. Detect_peaks should find ~6 peaks.
    period = 60.0 / 6.0
    saw = (days % period) / period * 8.0
    return pd.DataFrame(
        {
            DATE_COL: [base + pd.Timedelta(days=d) for d in days],
            TILT_COL: saw,
        }
    )


def test_compute_model_predictions_returns_one_record_per_registered_model():
    history = _synthetic_history()
    records = _compute_model_predictions(history)
    registered_ids = {m.id for m in registry.list_models()}
    record_ids = {r.model_id for r in records}
    assert record_ids == registered_ids


def test_each_record_carries_required_fields():
    history = _synthetic_history()
    records = _compute_model_predictions(history)
    for r in records:
        assert isinstance(r, ModelPredictionRecord)
        assert isinstance(r.model_id, str) and r.model_id
        # next_event_date_utc / band_*_utc may be None when fit failed,
        # but if non-None they must be ISO strings (round-trip parseable).
        if r.next_event_date_utc is not None:
            pd.Timestamp(r.next_event_date_utc)
        if r.band_lo_utc is not None:
            pd.Timestamp(r.band_lo_utc)
        if r.band_hi_utc is not None:
            pd.Timestamp(r.band_hi_utc)
        assert isinstance(r.diagnostics, dict)


def test_compute_with_empty_history_returns_records_with_null_predictions():
    """No data to fit → every model returns an empty ModelOutput, but the
    log should still record one entry per model with the failure
    diagnostics. Empty history must not skip the whole logging step."""
    empty = pd.DataFrame({DATE_COL: pd.to_datetime([]), TILT_COL: []})
    records = _compute_model_predictions(empty)
    assert len(records) == len(registry.list_models())
    for r in records:
        assert r.next_event_date_utc is None
        assert r.band_lo_utc is None
        assert r.band_hi_utc is None


def test_write_run_report_serializes_predictions(tmp_path: Path, monkeypatch):
    """End-to-end: a populated IngestRunResult.predictions makes it into
    the JSON written to disk under a top-level ``predictions`` key."""
    from kilauea_tracker.ingest import pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "RUN_REPORTS_DIR", tmp_path)

    result = IngestRunResult(
        run_started_at_utc=pd.Timestamp("2026-04-25T12:00:00Z").to_pydatetime(),
        run_finished_at_utc=pd.Timestamp("2026-04-25T12:01:00Z").to_pydatetime(),
        predictions=[
            ModelPredictionRecord(
                model_id="trendline_exp",
                next_event_date_utc="2026-05-08T00:00:00",
                band_lo_utc="2026-05-05T00:00:00",
                band_hi_utc="2026-05-11T00:00:00",
                headline_text="trendline × exp intersection",
                diagnostics={"trendline_slope_per_day": -0.01},
            ),
            ModelPredictionRecord(
                model_id="interval_median",
                next_event_date_utc="2026-05-09T00:00:00",
                band_lo_utc=None,
                band_hi_utc=None,
                headline_text="median 22d cycle",
                diagnostics={"median_peak_interval_days": 22.0},
            ),
        ],
    )
    out_path = _write_run_report(result)

    payload = json.loads(out_path.read_text())
    assert "predictions" in payload
    pred_block = payload["predictions"]
    assert len(pred_block) == 2
    by_id = {p["model_id"]: p for p in pred_block}
    assert by_id["trendline_exp"]["next_event_date_utc"] == "2026-05-08T00:00:00"
    assert by_id["trendline_exp"]["band_lo_utc"] == "2026-05-05T00:00:00"
    assert by_id["interval_median"]["band_lo_utc"] is None
    assert by_id["interval_median"]["headline_text"] == "median 22d cycle"
    assert by_id["interval_median"]["diagnostics"]["median_peak_interval_days"] == 22.0
