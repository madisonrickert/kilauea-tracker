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
    """Two timestamps within 15 minutes round to the same bucket → kept once.

    After the frame-alignment refactor (2026-04), the new row is shifted
    by the median (new − existing) delta in the overlap region BEFORE
    dedupe, so the surviving value lives in the existing CSV's frame.
    Here the single-overlap-bucket case means the offset is exactly the
    delta itself (+0.05) and the new row gets shifted back to 9.0. The
    test still verifies (a) dedupe collapses the two timestamps into one
    bucket, (b) no spurious conflict is reported.
    """
    _seed(
        tmp_history,
        ["2026-01-01 00:00:00"],
        [9.0],
    )
    # +5 minutes — same 15-minute bucket → frame-alignment makes both
    # rows agree, dedupe collapses to one row.
    new = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(["2026-01-01 00:05:00"]),
            TILT_COL: [9.05],
        }
    )
    report = append_history(new, tmp_history)
    df = load_history(tmp_history)
    assert len(df) == 1
    # Frame-aligned value: 9.05 − median(0.05) = 9.0
    assert df[TILT_COL].iloc[0] == pytest.approx(9.0)
    # The 0.05 µrad delta was absorbed by frame alignment; nothing should
    # be flagged as a per-row conflict.
    assert report.conflicts == []
    assert report.frame_offset_microrad == pytest.approx(0.05)
    assert report.frame_overlap_buckets == 1


def test_append_large_frame_shift_is_absorbed_not_flagged_as_conflict(
    tmp_history: Path,
):
    """A bulk frame shift (one bucket overlap, large delta) is absorbed
    by the alignment step, not flagged as a per-row conflict.

    This is the central behavior change in the 2026-04 refactor. Before
    frame alignment, a 1.5 µrad delta in a single overlap bucket would
    trigger a conflict warning even though the cause was a calibration
    rescale, not a real data anomaly. Now the median offset (which is
    the delta itself when there's only one overlap row) is applied to
    the new rows, they line up with existing, and the conflict detector
    has nothing to report. The fact that the offset was applied is still
    visible in `report.frame_offset_microrad` so the operator knows the
    correction happened.
    """
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

    # No leftover per-row conflict — alignment removed the delta entirely.
    assert report.conflicts == []
    # The shift was visible in the report.
    assert report.frame_offset_microrad == pytest.approx(CONFLICT_THRESHOLD + 0.5)
    assert report.frame_overlap_buckets == 1
    # And the resulting CSV value is in the existing frame.
    df = load_history(tmp_history)
    assert df[TILT_COL].iloc[0] == pytest.approx(9.0)


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
    # Frame-offset computation uses nearest-neighbour pairing (v4
    # rewrite) rather than bucket-match, so all three intra-batch rows
    # (10.0, 10.1, 10.2) pair to the one existing row (9.0). Deltas are
    # [1.0, 1.1, 1.2]; median = 1.1. Every new row then shifts by 1.1.
    expected_offset = 1.1
    a_row = df[df[DATE_COL] < pd.Timestamp("2026-02-01")]
    assert len(a_row) == 1
    assert a_row[TILT_COL].iloc[0] == pytest.approx(10.2 - expected_offset)
    b_row = df[
        (df[DATE_COL] >= pd.Timestamp("2026-02-01"))
        & (df[DATE_COL] < pd.Timestamp("2026-03-01"))
    ]
    assert b_row[TILT_COL].iloc[0] == pytest.approx(11.0 - expected_offset)
    c_row = df[df[DATE_COL] >= pd.Timestamp("2026-03-01")]
    assert c_row[TILT_COL].iloc[0] == pytest.approx(12.0 - expected_offset)


# ─────────────────────────────────────────────────────────────────────────────
# Frame alignment (the 2026-04 refactor — these tests are the regression
# fence around the intra-source drift fix described in cache.py's module
# docstring)
# ─────────────────────────────────────────────────────────────────────────────


def test_frame_alignment_no_overlap_keeps_rows_verbatim_with_warning(
    tmp_history: Path,
):
    """When new rows are entirely outside the existing CSV's date range,
    there is no anchor and we cannot compute a frame offset. Behavior:
    append in raw frame, log a warning, frame_offset reported as 0.0,
    overlap_buckets reported as 0. The warning is the contract — without
    it, a future operator could miss that this fetch may have introduced
    drift.
    """
    _seed(tmp_history, ["2026-01-01 00:00:00", "2026-01-01 12:00:00"], [9.0, 9.5])
    new = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                ["2026-03-01 00:00:00", "2026-03-01 12:00:00"]
            ),
            TILT_COL: [10.0, 10.5],
        }
    )
    report = append_history(new, tmp_history)

    assert report.frame_offset_microrad == 0.0
    assert report.frame_overlap_buckets == 0
    assert any("no temporal overlap" in w for w in report.warnings)

    df = load_history(tmp_history)
    assert len(df) == 4
    # Raw values preserved (no shift applied).
    assert df[TILT_COL].tolist() == [9.0, 9.5, 10.0, 10.5]


def test_frame_alignment_idempotency_no_drift_on_repeat_fetch(
    tmp_history: Path,
):
    """Appending the same fetch twice produces zero net drift.

    Round 1: empty CSV + 10 rows → CSV is exactly the fetch (frame F0).
    Round 2: existing CSV (F0) + same 10 rows again → median delta is 0.0
    over all 10 overlap buckets → no shift → CSV unchanged.

    This is the simplest possible regression check that frame alignment
    doesn't introduce drift when none exists.
    """
    dates = pd.date_range("2026-01-01 00:00:00", periods=10, freq="1h")
    tilts = [9.0 + 0.1 * i for i in range(10)]
    new = pd.DataFrame({DATE_COL: dates, TILT_COL: tilts})

    append_history(new, tmp_history)
    df_round1 = load_history(tmp_history)

    report2 = append_history(new, tmp_history)
    df_round2 = load_history(tmp_history)

    assert report2.frame_offset_microrad == pytest.approx(0.0)
    assert report2.frame_overlap_buckets == 10
    pd.testing.assert_frame_equal(
        df_round1.reset_index(drop=True),
        df_round2.reset_index(drop=True),
    )


def test_frame_alignment_corrects_synthetic_drift(tmp_history: Path):
    """Existing CSV in frame F0; new fetch is the SAME data shifted by
    +0.7 µrad (simulating a USGS y-axis rescale). Median delta = +0.7,
    new rows shifted back by -0.7, CSV is unchanged after append.

    This is the core "drift correction works" test. Without frame
    alignment the keep-latest dedupe would overwrite the F0 rows in the
    overlap with F0+0.7 rows from the new fetch, contaminating the CSV.
    """
    dates = pd.date_range("2026-01-01 00:00:00", periods=10, freq="1h")
    tilts_f0 = [9.0 + 0.1 * i for i in range(10)]
    drift = 0.7
    tilts_drifted = [t + drift for t in tilts_f0]

    _seed(tmp_history, [d.isoformat() for d in dates], tilts_f0)
    new = pd.DataFrame({DATE_COL: dates, TILT_COL: tilts_drifted})

    report = append_history(new, tmp_history)

    assert report.frame_offset_microrad == pytest.approx(drift)
    assert report.frame_overlap_buckets == 10

    df = load_history(tmp_history)
    # After alignment the CSV should be unchanged from the F0 seed.
    assert df[TILT_COL].tolist() == pytest.approx(tilts_f0)


def test_frame_alignment_single_overlap_row_logs_low_confidence(
    tmp_history: Path,
):
    """One overlap bucket → median is just that one delta. We still apply
    the offset (better than nothing) but log a low-confidence warning so
    the operator knows the alignment is fragile."""
    _seed(tmp_history, ["2026-01-01 00:00:00"], [9.0])
    new = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                ["2026-01-01 00:05:00", "2026-01-02 00:00:00"]
            ),
            TILT_COL: [9.3, 10.3],
        }
    )
    report = append_history(new, tmp_history)

    assert report.frame_offset_microrad == pytest.approx(0.3)
    assert report.frame_overlap_buckets == 1
    assert any("low confidence" in w for w in report.warnings)

    df = load_history(tmp_history)
    # Both new values shifted by -0.3.
    assert df[TILT_COL].iloc[0] == pytest.approx(9.0)  # bucket A (overlap)
    assert df[TILT_COL].iloc[1] == pytest.approx(10.0)  # day-2 (new only)


def test_frame_alignment_offset_from_overlap_applies_to_all_new_rows(
    tmp_history: Path,
):
    """The median offset is computed FROM the overlap region but APPLIES
    to every new row, including the ones that are outside the overlap.
    This is what makes the alignment work for rolling-window sources:
    the freshly-traced PNG always overlaps a few hours of existing
    history, the median over that overlap fixes the frame, and then the
    new tail-end rows (the ones we actually care about) get shifted
    correctly.
    """
    # Seed: 6 hourly samples at 9.0–9.5
    seed_dates = pd.date_range("2026-01-01 00:00:00", periods=6, freq="1h")
    seed_tilts = [9.0 + 0.1 * i for i in range(6)]
    _seed(tmp_history, [d.isoformat() for d in seed_dates], seed_tilts)

    # New: 3 overlapping hours (in F1, +0.5 µrad shift) + 3 fresh hours
    # extending into the future, also in F1.
    overlap_dates = seed_dates[3:6]
    fresh_dates = pd.date_range("2026-01-01 06:00:00", periods=3, freq="1h")
    new_overlap_tilts = [t + 0.5 for t in seed_tilts[3:6]]
    new_fresh_tilts = [9.6 + 0.5, 9.7 + 0.5, 9.8 + 0.5]  # F1-frame future rows
    new = pd.DataFrame(
        {
            DATE_COL: list(overlap_dates) + list(fresh_dates),
            TILT_COL: new_overlap_tilts + new_fresh_tilts,
        }
    )

    report = append_history(new, tmp_history)

    assert report.frame_offset_microrad == pytest.approx(0.5)
    assert report.frame_overlap_buckets == 3

    df = load_history(tmp_history)
    # Overlap region: existing values preserved (alignment cancels the shift).
    df_overlap = df[df[DATE_COL].isin(overlap_dates)]
    assert df_overlap[TILT_COL].tolist() == pytest.approx(seed_tilts[3:6])
    # Fresh region: shifted back into F0 (-0.5).
    df_fresh = df[df[DATE_COL].isin(fresh_dates)]
    assert df_fresh[TILT_COL].tolist() == pytest.approx([9.6, 9.7, 9.8])


def test_frame_alignment_median_absorbs_outlier_in_overlap(tmp_history: Path):
    """Overlap region contains 9 well-aligned rows + 1 wild outlier.
    Median (not mean) absorbs the outlier; the offset matches the bulk
    of the overlap region.
    """
    seed_dates = pd.date_range("2026-01-01 00:00:00", periods=10, freq="1h")
    seed_tilts = [9.0] * 10
    _seed(tmp_history, [d.isoformat() for d in seed_dates], seed_tilts)

    # 9 rows shifted by +0.4 µrad, 1 row at +50 µrad (an OCR misread).
    new_tilts = [9.4] * 9 + [59.0]
    new = pd.DataFrame({DATE_COL: seed_dates, TILT_COL: new_tilts})

    report = append_history(new, tmp_history)

    # Median of [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 50.0] = 0.4
    # (the outlier sits at one extreme of the sorted list).
    assert report.frame_offset_microrad == pytest.approx(0.4)
    assert report.frame_overlap_buckets == 10


def test_frame_alignment_residual_conflict_still_flagged(tmp_history: Path):
    """After alignment, a single row that's STILL wildly off the median
    is a real per-row data anomaly (data spike, partial OCR misread on
    that one row, etc.) and should still be flagged as a conflict.

    Without this guarantee, the alignment step would silently swallow
    real anomalies — defeating the diagnostic purpose of the conflict
    detector.
    """
    seed_dates = pd.date_range("2026-01-01 00:00:00", periods=10, freq="1h")
    seed_tilts = [9.0] * 10
    _seed(tmp_history, [d.isoformat() for d in seed_dates], seed_tilts)

    # 9 rows shifted by +0.2 µrad (bulk frame), 1 outlier at +5 µrad
    # (a real spike). After alignment removes the 0.2 median, the
    # outlier still sits at +4.8 µrad above existing — well over the
    # 1.0 µrad CONFLICT_THRESHOLD.
    new_tilts = [9.2] * 9 + [14.0]
    new = pd.DataFrame({DATE_COL: seed_dates, TILT_COL: new_tilts})

    report = append_history(new, tmp_history)

    assert report.frame_offset_microrad == pytest.approx(0.2)
    assert len(report.conflicts) == 1
    flagged = report.conflicts[0]
    # The conflict is on the last row, which after alignment shifts to
    # 14.0 - 0.2 = 13.8. Existing was 9.0. Delta = 4.8 µrad.
    assert flagged["delta"] == pytest.approx(4.8)


def test_frame_alignment_warns_on_large_offset(tmp_history: Path):
    """A frame offset above LARGE_FRAME_OFFSET_MICRORAD (5.0) is absorbed
    by alignment but logged as a warning so an operator can eyeball the
    source plot. Real eruption-induced rescales rarely exceed ~3 µrad;
    bigger than that suggests something has gone wrong with calibration.
    """
    seed_dates = pd.date_range("2026-01-01 00:00:00", periods=5, freq="1h")
    seed_tilts = [9.0] * 5
    _seed(tmp_history, [d.isoformat() for d in seed_dates], seed_tilts)

    # New rows shifted by +6 µrad — above the warning threshold.
    new_tilts = [15.0] * 5
    new = pd.DataFrame({DATE_COL: seed_dates, TILT_COL: new_tilts})

    report = append_history(new, tmp_history)

    assert report.frame_offset_microrad == pytest.approx(6.0)
    assert any("large intra-source frame shift" in w for w in report.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Sort stability (existing — kept after frame alignment refactor)
# ─────────────────────────────────────────────────────────────────────────────


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
