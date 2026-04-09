"""Tests for `kilauea_tracker.cache`.

Cover the cases that matter:
- load_history reads an existing CSV and returns the canonical schema
- load_history returns an empty DataFrame (not a crash) when the file is missing
- append_history adds new rows, preserves existing rows, and writes back
- Dedupe keeps the newest row when timestamps round to the same bucket
- Conflict detection flags drifted re-traces (within a single source CSV)
- Empty input is a no-op

The bootstrap-from-legacy logic that used to live in load_history was
removed when the per-source storage layer landed in `reconcile.py` —
legacy is now one of the inputs to `reconcile_sources()` rather than a
fallback for an empty cache.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kilauea_tracker.cache import (
    CONFLICT_THRESHOLD,
    AppendReport,
    append_history,
    load_history,
)
from kilauea_tracker.model import DATE_COL, TILT_COL


@pytest.fixture
def tmp_history(tmp_path: Path) -> Path:
    return tmp_path / "tilt_history.csv"


def _seed(path: Path, dates: list[str], tilts: list[float]) -> None:
    df = pd.DataFrame({DATE_COL: pd.to_datetime(dates), TILT_COL: tilts})
    df.to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# load_history
# ─────────────────────────────────────────────────────────────────────────────


def test_load_history_returns_empty_when_file_missing(tmp_path: Path):
    """No file → empty DataFrame with the canonical schema, no crash.

    The Streamlit app's `load_tilt()` runs `ingest_all()` on first call,
    which always populates `tilt_history.csv`. Returning empty here is the
    sane behavior for the brief window between app start and first ingest.
    """
    history_csv = tmp_path / "tilt_history.csv"
    df = load_history(history_csv)
    assert len(df) == 0
    assert list(df.columns) == [DATE_COL, TILT_COL]


def test_load_history_round_trip(tmp_history: Path):
    _seed(
        tmp_history,
        ["2026-01-01 00:00:00", "2026-01-01 12:00:00"],
        [9.0, 9.5],
    )
    df = load_history(tmp_history)
    assert len(df) == 2
    assert df[TILT_COL].tolist() == [9.0, 9.5]


# ─────────────────────────────────────────────────────────────────────────────
# append_history
# ─────────────────────────────────────────────────────────────────────────────


def test_append_adds_new_rows(tmp_history: Path):
    _seed(
        tmp_history,
        ["2026-01-01 00:00:00", "2026-01-01 12:00:00"],
        [9.0, 9.5],
    )
    new = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(["2026-01-02 00:00:00", "2026-01-02 12:00:00"]),
            TILT_COL: [10.0, 10.5],
        }
    )
    report = append_history(new, tmp_history)
    assert isinstance(report, AppendReport)
    assert report.rows_added == 2
    assert report.conflicts == []

    df = load_history(tmp_history)
    assert len(df) == 4
    assert df[TILT_COL].tolist() == [9.0, 9.5, 10.0, 10.5]


def test_append_dedupes_overlapping_buckets(tmp_history: Path):
    """Two timestamps within 15 minutes round to the same bucket → kept once."""
    _seed(
        tmp_history,
        ["2026-01-01 00:00:00"],
        [9.0],
    )
    # +5 minutes — same 15-minute bucket → should overwrite
    new = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(["2026-01-01 00:05:00"]),
            TILT_COL: [9.05],
        }
    )
    report = append_history(new, tmp_history)
    df = load_history(tmp_history)
    assert len(df) == 1
    assert df[TILT_COL].iloc[0] == 9.05  # newest wins
    # Tilt delta is 0.05 < CONFLICT_THRESHOLD → no conflict
    assert report.conflicts == []


def test_append_flags_conflict_above_threshold(tmp_history: Path):
    """Same bucket, tilts differ by >1 µrad → conflict warning."""
    _seed(
        tmp_history,
        ["2026-01-01 00:00:00"],
        [9.0],
    )
    new = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(["2026-01-01 00:05:00"]),
            TILT_COL: [9.0 + CONFLICT_THRESHOLD + 0.5],
        }
    )
    report = append_history(new, tmp_history)
    assert len(report.conflicts) == 1
    conflict = report.conflicts[0]
    assert conflict["existing_tilt"] == 9.0
    assert abs(conflict["delta"] - (CONFLICT_THRESHOLD + 0.5)) < 1e-9


def test_append_empty_is_noop(tmp_history: Path):
    _seed(tmp_history, ["2026-01-01 00:00:00"], [9.0])
    empty = pd.DataFrame({DATE_COL: pd.to_datetime([]), TILT_COL: []})
    report = append_history(empty, tmp_history)
    assert report.rows_added == 0
    assert report.conflicts == []
    df = load_history(tmp_history)
    assert len(df) == 1


def test_append_to_nonexistent_creates_history(tmp_path: Path):
    """append_history on a path that doesn't exist yet creates the file."""
    history_csv = tmp_path / "tilt_history.csv"
    new = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 12:00:00"]),
            TILT_COL: [9.0, 9.5],
        }
    )
    report = append_history(new, history_csv)
    assert report.rows_added == 2
    df = load_history(history_csv)
    assert len(df) == 2


def test_append_report_correct_with_intra_batch_dedupe(tmp_history: Path):
    """Regression for an overcount bug in the rows_added/rows_updated maths.

    Setup: existing has 1 row in bucket A. New batch has 3 rows in bucket A
    (intra-batch dedupe — only the last survives) plus 2 rows in fresh
    buckets B and C. After dedupe the cache has 3 rows total: the new B, the
    new C, and the new A row that displaced the existing A.

    Expected report: rows_added=2 (B and C are new buckets), rows_updated=1
    (A was an existing bucket whose row got overwritten). Pre-fix arithmetic
    (`len(new_rows) - rows_added`) gave rows_updated = 5 - 2 = 3, which
    overcounted by 2 because it conflated intra-batch dedupes with updates.
    """
    # All three timestamps below are within ±7.5 minutes of 00:00, so they
    # all round to the same 15-minute bucket as the existing row.
    _seed(tmp_history, ["2026-01-01 00:00:00"], [9.0])
    new = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2026-01-01 00:01:00",  # bucket A — intra-batch (loses)
                    "2026-01-01 00:03:00",  # bucket A — intra-batch (loses)
                    "2026-01-01 00:07:00",  # bucket A — survives, displaces existing
                    "2026-02-01 00:00:00",  # bucket B — new
                    "2026-03-01 00:00:00",  # bucket C — new
                ]
            ),
            TILT_COL: [10.0, 10.1, 10.2, 11.0, 12.0],
        }
    )
    report = append_history(new, tmp_history)
    assert report.rows_added == 2, f"expected 2 added, got {report.rows_added}"
    assert report.rows_updated == 1, f"expected 1 updated, got {report.rows_updated}"

    df = load_history(tmp_history)
    assert len(df) == 3
    # Bucket A's surviving row is the latest one (10.2), not 10.0 or 10.1
    a_row = df[df[DATE_COL] < pd.Timestamp("2026-02-01")]
    assert len(a_row) == 1
    assert a_row[TILT_COL].iloc[0] == 10.2


def test_history_stays_sorted_after_append(tmp_history: Path):
    _seed(tmp_history, ["2026-01-05 00:00:00"], [9.0])
    new = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(["2026-01-01 00:00:00", "2026-01-10 00:00:00"]),
            TILT_COL: [8.0, 10.0],
        }
    )
    append_history(new, tmp_history)
    df = load_history(tmp_history)
    dates = df[DATE_COL].to_numpy()
    assert (dates[1:] >= dates[:-1]).all()
