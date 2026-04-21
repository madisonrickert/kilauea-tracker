"""Tests for `kilauea_tracker.archive` and the archive's role inside the
ingest+reconcile pipeline.

These tests are the regression fence around the 2026-04 belt-and-suspenders
fix for intra-source frame drift. The archive is a frozen append-only
record of "what we observed at this timestamp the first time we ever
observed it"; once a row lands there, it must be immune to subsequent
ingest runs that might otherwise re-overwrite history.

Coverage:
  1. Empty archive + first reconcile → archive populated
  2. Existing archive + reconcile producing only new timestamps → grows monotonically
  3. Existing archive + reconcile with conflicting values → keeps first
  4. Archive registered correctly in SOURCE_PRIORITY (just below digital)
  5. Archive wins priority contest over per-source CSV at the same timestamp
  6. End-to-end: 3 simulated ingest runs with synthetic per-source drift →
     archive values for the first run's timestamps are unchanged
  7. Schema stability — no extra columns leak in across promote rounds
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kilauea_tracker.archive import (
    ArchivePromotionReport,
    load_archive,
    promote_to_archive,
)
from kilauea_tracker.config import ARCHIVE_SOURCE_NAME, SOURCE_PRIORITY
from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.reconcile import reconcile_sources


@pytest.fixture
def tmp_archive(tmp_path: Path) -> Path:
    return tmp_path / "archive.csv"


def _df(dates: list[str], tilts: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {DATE_COL: pd.to_datetime(dates), TILT_COL: tilts}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Empty archive + first reconcile → archive populated
# ─────────────────────────────────────────────────────────────────────────────


def test_promote_to_empty_archive_populates_it(tmp_archive: Path):
    """First-ever promote on a fresh checkout writes every merged row."""
    merged = _df(
        [
            "2026-01-01 00:00:00",
            "2026-01-01 12:00:00",
            "2026-01-02 00:00:00",
        ],
        [9.0, 9.5, 10.0],
    )

    report = promote_to_archive(merged, tmp_archive)

    assert isinstance(report, ArchivePromotionReport)
    assert report.rows_in_archive_before == 0
    assert report.rows_promoted == 3
    assert report.rows_already_archived == 0
    assert report.rows_in_archive_after == 3
    assert tmp_archive.exists()

    on_disk = load_archive(tmp_archive)
    assert len(on_disk) == 3
    assert on_disk[TILT_COL].tolist() == [9.0, 9.5, 10.0]


def test_promote_empty_merged_is_noop(tmp_archive: Path):
    """An empty merged view doesn't create the archive file."""
    empty = _df([], [])
    report = promote_to_archive(empty, tmp_archive)
    assert report.rows_promoted == 0
    assert report.rows_in_archive_after == 0
    assert not tmp_archive.exists()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Existing archive + new timestamps → grows monotonically
# ─────────────────────────────────────────────────────────────────────────────


def test_promote_growing_only_new_timestamps_appends(tmp_archive: Path):
    """Promote, promote again with disjoint timestamps → archive grows."""
    first = _df(["2026-01-01 00:00:00"], [9.0])
    promote_to_archive(first, tmp_archive)

    second = _df(
        ["2026-01-01 12:00:00", "2026-01-02 00:00:00"], [9.5, 10.0]
    )
    report = promote_to_archive(second, tmp_archive)

    assert report.rows_in_archive_before == 1
    assert report.rows_promoted == 2
    assert report.rows_already_archived == 0
    assert report.rows_in_archive_after == 3

    on_disk = load_archive(tmp_archive)
    assert len(on_disk) == 3
    assert on_disk[TILT_COL].tolist() == [9.0, 9.5, 10.0]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Conflicting values → archive keeps the first observation
# ─────────────────────────────────────────────────────────────────────────────


def test_promote_keeps_first_observation_on_conflict(tmp_archive: Path):
    """The promotion is keep-first, NOT keep-latest. This is the load-bearing
    contract of the archive — once a value is in there, no future
    reconcile run can rewrite it.
    """
    first = _df(
        ["2026-01-01 00:00:00", "2026-01-02 00:00:00"], [9.0, 10.0]
    )
    promote_to_archive(first, tmp_archive)

    # Same timestamps, different (drifted) values from a later reconcile.
    second = _df(
        ["2026-01-01 00:00:00", "2026-01-02 00:00:00"], [9.7, 10.7]
    )
    report = promote_to_archive(second, tmp_archive)

    assert report.rows_promoted == 0
    assert report.rows_already_archived == 2

    on_disk = load_archive(tmp_archive)
    # Original frozen values preserved.
    assert on_disk[TILT_COL].tolist() == [9.0, 10.0]


def test_promote_partial_conflict_appends_only_new_buckets(tmp_archive: Path):
    """If the second batch contains a mix of existing-bucket and new-bucket
    rows, only the new buckets are appended; existing-bucket rows are
    silently skipped.
    """
    first = _df(["2026-01-01 00:00:00"], [9.0])
    promote_to_archive(first, tmp_archive)

    second = _df(
        ["2026-01-01 00:05:00", "2026-01-02 00:00:00"], [9.7, 10.0]
    )
    report = promote_to_archive(second, tmp_archive)

    assert report.rows_promoted == 1  # only the 2026-01-02 row
    assert report.rows_already_archived == 1  # the 00:05 row collapses to existing bucket

    on_disk = load_archive(tmp_archive)
    # First-frame value preserved at the conflict bucket.
    jan1 = on_disk[on_disk[DATE_COL] < pd.Timestamp("2026-01-02")]
    assert len(jan1) == 1
    assert jan1[TILT_COL].iloc[0] == 9.0
    # New row appended.
    jan2 = on_disk[on_disk[DATE_COL] >= pd.Timestamp("2026-01-02")]
    assert len(jan2) == 1
    assert jan2[TILT_COL].iloc[0] == 10.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Archive registered in SOURCE_PRIORITY in the right slot
# ─────────────────────────────────────────────────────────────────────────────


def test_archive_priority_just_below_digital():
    """The archive must be the second-highest priority source so it wins
    priority contests against everything except the digital reference.
    """
    assert ARCHIVE_SOURCE_NAME == "archive"
    assert SOURCE_PRIORITY[0] == "digital"
    assert SOURCE_PRIORITY[1] == "archive"
    # Everything else comes after.
    assert "two_day" in SOURCE_PRIORITY
    assert SOURCE_PRIORITY.index("archive") < SOURCE_PRIORITY.index("two_day")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Archive wins priority contest over per-source CSV at the same timestamp
# ─────────────────────────────────────────────────────────────────────────────


def test_archive_wins_priority_contest_over_drifted_per_source(tmp_archive: Path):
    """End-to-end check that the reconciler sources archived timestamps
    from the archive, even when a live per-source CSV has a different
    (drifted) value at the same timestamp.

    Setup: an archive with a known value at T1. A `two_day` per-source
    CSV with a drifted value at the same T1 (and an extra fresh row at
    T2). After reconcile, the merged view should match the archive at
    T1 and the two_day source at T2.
    """
    archive_df = _df(["2026-01-01 00:00:00"], [9.0])
    two_day_df = _df(
        ["2026-01-01 00:00:00", "2026-01-01 12:00:00"], [9.7, 10.5]
    )

    sources = {"archive": archive_df, "two_day": two_day_df}
    merged, report = reconcile_sources(sources, proximity_minutes=0)

    # T1 should match the archive (9.0), not the drifted two_day (9.7).
    t1_row = merged[merged[DATE_COL] == pd.Timestamp("2026-01-01 00:00:00")]
    assert len(t1_row) == 1
    # The exact value depends on whether two_day got aligned against the
    # archive (it should have — anchor=archive, two_day aligns to it).
    # After alignment, two_day's frame matches archive's frame, and the
    # archive wins the priority contest at T1.
    assert t1_row[TILT_COL].iloc[0] == pytest.approx(9.0)

    # T2 should be present (only two_day has it), in the archive's frame.
    t2_row = merged[merged[DATE_COL] == pd.Timestamp("2026-01-01 12:00:00")]
    assert len(t2_row) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. End-to-end: 3 ingest runs with synthetic per-source drift
# ─────────────────────────────────────────────────────────────────────────────


def test_archive_immune_to_per_source_drift_across_multiple_runs(
    tmp_archive: Path,
):
    """Simulates 3 consecutive `ingest_all()`-style runs where the
    per-source CSVs drift between runs. The archive must contain the
    FIRST observation of each timestamp, unchanged across all 3 runs.

    Run 1: per-source two_day in frame F0 → reconcile → archive captures F0.
    Run 2: per-source two_day in frame F1 (drifted +0.5) → reconcile.
           Archive's F0 values for the original timestamps should be
           preserved; new timestamps in this run get archived in F1.
    Run 3: per-source two_day in frame F2 (drifted +1.0) → reconcile.
           Archive's F0 and F1 values still preserved.
    """
    # Run 1
    run1_two_day = _df(
        [
            "2026-01-01 00:00:00",
            "2026-01-01 06:00:00",
            "2026-01-01 12:00:00",
        ],
        [9.0, 9.2, 9.4],
    )
    merged1, _ = reconcile_sources({"two_day": run1_two_day}, proximity_minutes=0)
    promote_to_archive(merged1, tmp_archive)
    archive_after_1 = load_archive(tmp_archive)
    assert archive_after_1[TILT_COL].tolist() == [9.0, 9.2, 9.4]

    # Run 2: existing timestamps drifted, plus new timestamps
    run2_two_day = _df(
        [
            "2026-01-01 00:00:00",  # existing — drifted
            "2026-01-01 06:00:00",  # existing — drifted
            "2026-01-01 12:00:00",  # existing — drifted
            "2026-01-01 18:00:00",  # new
            "2026-01-02 00:00:00",  # new
        ],
        [9.5, 9.7, 9.9, 10.1, 10.3],  # all +0.5 from F0
    )
    archive_input = load_archive(tmp_archive)
    merged2, _ = reconcile_sources(
        {"two_day": run2_two_day, "archive": archive_input},
        proximity_minutes=0,
    )
    promote_to_archive(merged2, tmp_archive)
    archive_after_2 = load_archive(tmp_archive)

    # The first 3 rows must still be in the original F0 frame.
    first_three = archive_after_2[
        archive_after_2[DATE_COL] <= pd.Timestamp("2026-01-01 12:00:00")
    ]
    assert first_three[TILT_COL].tolist() == [9.0, 9.2, 9.4], (
        "archive's first observations must be immutable across runs"
    )
    # The 2 new timestamps got captured this run. They were aligned
    # against the archive (since archive is the anchor in this synthetic
    # setup), so they should land in the archive's frame.
    assert len(archive_after_2) == 5

    # Run 3: more drift, more new rows
    run3_two_day = _df(
        [
            "2026-01-01 00:00:00",  # original — even more drifted
            "2026-01-01 06:00:00",
            "2026-01-01 12:00:00",
            "2026-01-01 18:00:00",
            "2026-01-02 00:00:00",
            "2026-01-02 06:00:00",  # new this run
        ],
        [10.0, 10.2, 10.4, 10.6, 10.8, 11.0],  # all +1.0 from F0
    )
    archive_input = load_archive(tmp_archive)
    merged3, _ = reconcile_sources(
        {"two_day": run3_two_day, "archive": archive_input},
        proximity_minutes=0,
    )
    promote_to_archive(merged3, tmp_archive)
    archive_after_3 = load_archive(tmp_archive)

    # Original 3 rows STILL in F0.
    first_three = archive_after_3[
        archive_after_3[DATE_COL] <= pd.Timestamp("2026-01-01 12:00:00")
    ]
    assert first_three[TILT_COL].tolist() == [9.0, 9.2, 9.4]
    # 6 total rows now (the new 2026-01-02 06:00 row was added).
    assert len(archive_after_3) == 6


# ─────────────────────────────────────────────────────────────────────────────
# 7. Schema stability — no extra columns leak in
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 8. Quorum gate — the 2026-04 contamination fix
# ─────────────────────────────────────────────────────────────────────────────


def test_quorum_gate_defers_single_source_buckets(tmp_archive: Path):
    """A bucket that's present in only ONE source with no nearby archive
    neighbour must be deferred — exactly the failure mode that let the
    week-PNG phantom spikes poison the archive.
    """
    merged = _df(["2026-06-01 00:00:00"], [9.0])
    sources = {"week": merged}  # only one source contributes

    report = promote_to_archive(merged, tmp_archive, sources=sources)

    assert report.rows_promoted == 0
    assert report.rows_deferred_by_quorum == 1
    assert not tmp_archive.exists()


def test_quorum_gate_admits_multi_source_buckets(tmp_archive: Path):
    """When ≥2 sources contribute to the same 15-min bucket, the row is
    corroborated and the gate lets it through. This is the common healthy
    case (week + month + two_day frequently all cover recent buckets).
    """
    merged = _df(["2026-06-01 00:00:00"], [9.0])
    sources = {
        "week": _df(["2026-06-01 00:00:00"], [9.0]),
        "month": _df(["2026-06-01 00:05:00"], [9.1]),
    }
    report = promote_to_archive(merged, tmp_archive, sources=sources)
    assert report.rows_promoted == 1
    assert report.rows_deferred_by_quorum == 0


def test_quorum_gate_admits_rows_with_trusted_archive_neighbour(tmp_archive: Path):
    """A single-source row that sits close in time AND value to an already-
    archived row rides in on the neighbour's provenance. This keeps the
    gate from blocking the normal forward march of single-source recent
    rows once the archive has a continuous history.
    """
    # Seed archive with a good row.
    promote_to_archive(_df(["2026-06-01 00:00:00"], [9.0]), tmp_archive)

    # New single-source row, 10 min later, 0.5 µrad higher → within gates.
    new = _df(["2026-06-01 00:10:00"], [9.5])
    sources = {"week": new}
    report = promote_to_archive(new, tmp_archive, sources=sources)
    assert report.rows_promoted == 1
    assert report.rows_deferred_by_quorum == 0


def test_quorum_gate_defers_single_source_row_that_disagrees_with_neighbour(
    tmp_archive: Path,
):
    """A single-source row that's close in time to an archived neighbour
    but wildly different in value is exactly the contamination pattern —
    defer it until a second source confirms.
    """
    promote_to_archive(_df(["2026-06-01 00:00:00"], [9.0]), tmp_archive)

    # Single-source row 10 min later with a -10 µrad spike — classic
    # phantom. Neighbour is close in time but way off in value → gate
    # defers.
    new = _df(["2026-06-01 00:10:00"], [-1.0])
    sources = {"week": new}
    report = promote_to_archive(new, tmp_archive, sources=sources)
    assert report.rows_promoted == 0
    assert report.rows_deferred_by_quorum == 1


def test_quorum_gate_ignores_archive_source_in_count(tmp_archive: Path):
    """The archive feeds itself back into reconcile as a source. The gate
    must NOT count that self-contribution toward quorum — otherwise every
    already-archived bucket would vacuously 're-corroborate' itself and
    the gate is inert.
    """
    # Seed archive.
    promote_to_archive(_df(["2026-06-01 00:00:00"], [9.0]), tmp_archive)

    # New bucket; only sources are archive (from last run) + week (one live).
    # Counts: archive excluded, week = 1 source → below quorum, and the
    # new bucket has no nearby archive neighbour (archive's only row is at
    # 00:00, new row is at 01:00 — 60 min away).
    new = _df(["2026-06-01 01:00:00"], [9.2])
    archive_df = load_archive(tmp_archive)
    sources = {"archive": archive_df, "week": new}
    report = promote_to_archive(new, tmp_archive, sources=sources)
    assert report.rows_deferred_by_quorum == 1


def test_archive_schema_stability_after_multiple_promotes(tmp_archive: Path):
    """Promote a few rounds and verify the on-disk archive only has
    [Date, Tilt (microradians)] — no leaked _bucket, _source, etc.
    """
    promote_to_archive(_df(["2026-01-01 00:00:00"], [9.0]), tmp_archive)
    promote_to_archive(_df(["2026-01-02 00:00:00"], [9.5]), tmp_archive)
    promote_to_archive(_df(["2026-01-03 00:00:00"], [10.0]), tmp_archive)

    raw = pd.read_csv(tmp_archive)
    assert list(raw.columns) == [DATE_COL, TILT_COL]
