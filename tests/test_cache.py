"""Tests for `kilauea_tracker.cache`.

Cover the cases that matter:
- Bootstrap from legacy CSV when history doesn't exist
- Append new rows preserves existing rows
- Dedupe keeps the newest row when timestamps round to the same bucket
- Conflict detection flags drifted re-traces
- Empty input is a no-op
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
from kilauea_tracker.config import LEGACY_CSV
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


def test_load_history_bootstraps_from_legacy(tmp_path: Path):
    """If the history file doesn't exist, bootstrap from legacy CSV.

    The legacy CSV is the raw v1.0 Google Sheet export and contains ~770
    blank trailing rows; load_history drops them via `dropna`. We assert that
    bootstrap produces a non-trivial DataFrame and writes a cache file.
    """
    history_csv = tmp_path / "tilt_history.csv"
    assert not history_csv.exists()
    df = load_history(history_csv)
    assert len(df) > 100, "bootstrap should preserve all real rows from legacy"
    assert df[TILT_COL].notna().all(), "no NaN tilts after canonicalization"
    assert df[DATE_COL].notna().all(), "no NaT dates after canonicalization"
    assert history_csv.exists()  # bootstrap wrote a copy
    # Re-loading from the cached file gives the same data
    df2 = load_history(history_csv)
    assert len(df2) == len(df)


def test_bootstrap_respects_legacy_cutoff(tmp_path: Path):
    """The bootstrap drops legacy rows older than `LEGACY_BOOTSTRAP_CUTOFF`.

    Verifies the trim that lets DEC2024_TO_NOW provide denser coverage of the
    pre-July-2025 range than the manually-digitized legacy data.
    """
    from kilauea_tracker.config import LEGACY_BOOTSTRAP_CUTOFF

    history_csv = tmp_path / "tilt_history.csv"
    df = load_history(history_csv)
    # Every bootstrapped row must be at or after the cutoff
    assert (df[DATE_COL] >= LEGACY_BOOTSTRAP_CUTOFF).all(), (
        f"bootstrap returned rows older than cutoff {LEGACY_BOOTSTRAP_CUTOFF}: "
        f"min date is {df[DATE_COL].min()}"
    )
    # And the cutoff actually drops something — sanity check that the legacy
    # CSV genuinely has pre-cutoff data we'd be dropping.
    raw_legacy = pd.read_csv(LEGACY_CSV)
    raw_legacy[DATE_COL] = pd.to_datetime(
        raw_legacy[DATE_COL], format="mixed", dayfirst=False
    )
    raw_legacy = raw_legacy.dropna()
    pre_cutoff = (raw_legacy[DATE_COL] < LEGACY_BOOTSTRAP_CUTOFF).sum()
    assert pre_cutoff > 0, (
        "expected the legacy file to have some pre-cutoff rows; if this "
        "fails the test is now meaningless"
    )
    assert len(df) < len(raw_legacy.dropna()), "trim should drop something"


def test_bootstrap_cutoff_only_applies_at_bootstrap(tmp_path: Path):
    """Reading an existing cache file does NOT apply the cutoff retroactively.

    The cutoff is a bootstrap-time policy. Existing caches that already have
    pre-cutoff rows (e.g. from a manual ingest) should be returned unchanged.
    """
    from kilauea_tracker.config import LEGACY_BOOTSTRAP_CUTOFF

    history_csv = tmp_path / "tilt_history.csv"
    # Seed the cache with rows that straddle the cutoff
    pre_date = LEGACY_BOOTSTRAP_CUTOFF - pd.Timedelta(days=30)
    post_date = LEGACY_BOOTSTRAP_CUTOFF + pd.Timedelta(days=30)
    _seed(
        history_csv,
        [str(pre_date), str(post_date)],
        [9.0, 10.0],
    )
    df = load_history(history_csv)
    assert len(df) == 2, "load_history of existing cache should not trim"
    assert (df[DATE_COL] == pd.Timestamp(pre_date)).any()
    assert (df[DATE_COL] == pd.Timestamp(post_date)).any()


def test_load_history_returns_empty_when_no_legacy(tmp_path: Path, monkeypatch):
    """No history, no legacy → empty DataFrame, no crash."""
    fake_legacy = tmp_path / "nonexistent.csv"
    monkeypatch.setattr("kilauea_tracker.cache.LEGACY_CSV", fake_legacy)
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


def test_append_to_nonexistent_creates_history(tmp_path: Path, monkeypatch):
    """Cache file doesn't exist AND legacy CSV is fake → new history is created."""
    fake_legacy = tmp_path / "no-legacy.csv"
    monkeypatch.setattr("kilauea_tracker.cache.LEGACY_CSV", fake_legacy)
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
