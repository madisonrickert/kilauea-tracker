"""Tests for `kilauea_tracker.reconcile`.

Reconciliation is the load-bearing replacement for the old cascading-cache
ingest path. The tests below cover:

  - Anchor selection in priority order
  - Single-pass alignment of a non-anchor source against the anchor
  - Refusal to align when overlap is too small
  - Refusal to apply implausibly large offsets
  - Priority-based per-bucket merge
  - Conflict detection between two aligned sources
  - Determinism: re-running with the same inputs gives the same output
  - Empty / missing sources are no-ops
  - Order independence: presenting sources in a different order or omitting
    sources doesn't change the result for the sources that ARE present
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kilauea_tracker.config import SOURCE_PRIORITY
from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.reconcile import (
    CONFLICT_THRESHOLD_MICRORAD,
    MAX_TRUSTED_OFFSET_MICRORAD,
    MIN_OVERLAP_BUCKETS,
    PROXIMITY_GATE_MINUTES,
    reconcile_sources,
)


def _series(start: str, n: int, base: float, freq: str = "1h") -> pd.DataFrame:
    """Tiny constant-tilt series — perfect for offset detection tests."""
    return pd.DataFrame(
        {
            DATE_COL: pd.date_range(start, periods=n, freq=freq),
            TILT_COL: np.full(n, base, dtype=float),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# Anchor selection
# ─────────────────────────────────────────────────────────────────────────────


def test_first_priority_source_with_data_becomes_anchor():
    """Digital is the highest-priority source — when present it anchors."""
    sources = {
        "digital": _series("2025-03-01", n=48, base=10.0),
        "two_day": _series("2025-03-01", n=48, base=16.0),
    }
    merged, report = reconcile_sources(sources)

    digital_record = next(s for s in report.sources if s.name == "digital")
    two_day_record = next(s for s in report.sources if s.name == "two_day")

    assert digital_record.is_anchor is True
    assert digital_record.offset_microrad == 0.0
    assert two_day_record.is_anchor is False
    assert two_day_record.offset_microrad is not None
    assert abs(two_day_record.offset_microrad - 6.0) < 1e-9


def test_anchor_falls_to_next_alignment_order_when_digital_missing():
    """Alignment order is (digital, dec2024_to_now, legacy, three_month,
    month, week, two_day). If digital is missing, the first source in
    ALIGNMENT_ORDER that DOES have data becomes the anchor — that's `month`
    when only month and two_day are present.

    Note this is intentionally different from SOURCE_PRIORITY: priority is
    used by the per-bucket merge step (where two_day still wins because
    it's higher quality), but alignment is topological so that each source
    can find overlap with the running union of already-aligned sources.
    """
    sources = {
        "two_day": _series("2025-03-01", n=48, base=10.0),
        "month": _series("2025-03-01", n=48, base=12.0),
    }
    merged, report = reconcile_sources(sources)

    anchor = next(s for s in report.sources if s.is_anchor)
    assert anchor.name == "month"
    # two_day was at 10.0, month at 12.0 → two_day's offset against month
    # is -2.0, so shifted two_day = 12.0. Both sources now sit at 12.0 in
    # the aligned frame. The per-bucket merge picks two_day for shared
    # buckets (higher priority), but the value is the same.
    two_day_record = next(s for s in report.sources if s.name == "two_day")
    assert two_day_record.offset_microrad is not None
    assert abs(two_day_record.offset_microrad - (-2.0)) < 1e-9
    assert (merged[TILT_COL] == 12.0).all()


def test_empty_input_returns_empty_history():
    merged, report = reconcile_sources({})
    assert len(merged) == 0
    assert list(merged.columns) == [DATE_COL, TILT_COL]
    assert report.rows_out == 0
    assert report.sources == []


def test_all_empty_dataframes_returns_empty_history():
    """Sources present but with empty DataFrames is the same as no sources."""
    sources = {
        "digital": pd.DataFrame({DATE_COL: pd.Series(dtype="datetime64[ns]"), TILT_COL: []}),
        "two_day": pd.DataFrame({DATE_COL: pd.Series(dtype="datetime64[ns]"), TILT_COL: []}),
    }
    merged, report = reconcile_sources(sources)
    assert len(merged) == 0
    assert report.rows_out == 0


# ─────────────────────────────────────────────────────────────────────────────
# Alignment math
# ─────────────────────────────────────────────────────────────────────────────


def test_systematic_offset_is_subtracted_from_lower_priority_source():
    """The bug case from v1: a lower-priority source sits +6 µrad above the
    anchor. After reconciliation its values should be shifted DOWN by 6.0
    so they line up with the anchor's frame.
    """
    sources = {
        "digital": _series("2025-03-01", n=48, base=10.0),
        "two_day": _series("2025-03-01", n=48, base=16.0),
    }
    merged, report = reconcile_sources(sources)
    # The merged history should be all 10.0 µrad — digital wins every bucket
    # by priority. The two_day rows still got aligned (offset detected) but
    # they lose every contest.
    assert (merged[TILT_COL] == 10.0).all()
    two_day_record = next(s for s in report.sources if s.name == "two_day")
    assert abs(two_day_record.offset_microrad - 6.0) < 1e-9


def test_alignment_uses_median_not_mean_to_resist_outliers():
    """A handful of outlier buckets shouldn't pull the offset.

    Setup: anchor is flat at 10 µrad. Lower-priority source has 44 buckets
    at +6 µrad and 4 buckets at +30 µrad (the outliers). Mean delta would
    be 8; median delta is 6. We want 6.
    """
    cache = _series("2025-03-01", n=48, base=10.0)
    new_values = np.full(48, 16.0)
    new_values[[5, 12, 25, 40]] = 40.0
    new = pd.DataFrame(
        {DATE_COL: cache[DATE_COL].copy(), TILT_COL: new_values}
    )
    sources = {"digital": cache, "two_day": new}
    _, report = reconcile_sources(sources)
    two_day = next(s for s in report.sources if s.name == "two_day")
    assert abs(two_day.offset_microrad - 6.0) < 1e-9


def test_insufficient_overlap_leaves_source_unaligned():
    """Tiny overlap (<MIN_OVERLAP_BUCKETS) — refuse to align, warn, but still
    include the raw rows in the merge so the user sees something."""
    cache = _series("2025-03-01", n=24, base=10.0, freq="1h")
    # New source overlaps the last (MIN_OVERLAP_BUCKETS - 1) hours only.
    new_start = cache[DATE_COL].iloc[-MIN_OVERLAP_BUCKETS + 1]
    new = _series(str(new_start), n=10, base=15.0, freq="1h")
    sources = {"digital": cache, "two_day": new}

    merged, report = reconcile_sources(sources)
    two_day = next(s for s in report.sources if s.name == "two_day")
    assert two_day.offset_microrad is None
    assert two_day.overlap_buckets < MIN_OVERLAP_BUCKETS
    assert any("two_day" in w for w in report.warnings)


def test_implausibly_large_offset_is_refused():
    """A 50-µrad delta is a calibration bug, not drift — must NOT auto-correct."""
    cache = _series("2025-03-01", n=48, base=10.0)
    new = _series("2025-03-01", n=48, base=10.0 + MAX_TRUSTED_OFFSET_MICRORAD + 5)
    sources = {"digital": cache, "two_day": new}

    _, report = reconcile_sources(sources)
    two_day = next(s for s in report.sources if s.name == "two_day")
    assert two_day.offset_microrad is None
    # We did find overlap; we just refused the offset
    assert two_day.overlap_buckets >= MIN_OVERLAP_BUCKETS
    assert any("two_day" in w and "implausibly" in w for w in report.warnings)


# ─────────────────────────────────────────────────────────────────────────────
# Priority merge
# ─────────────────────────────────────────────────────────────────────────────


def test_higher_priority_source_wins_overlapping_buckets():
    """Two sources, both perfectly aligned. The higher-priority one wins
    every bucket — but the lower-priority one still contributes its
    non-overlapping buckets.

    Disable proximity gating because the test deliberately creates
    overlapping samples to exercise the merge step.
    """
    digital = _series("2025-03-01", n=24, base=10.0, freq="1h")
    # two_day overlaps the second half of digital and extends past it
    two_day = _series("2025-03-01 12:00:00", n=24, base=10.0, freq="1h")
    sources = {"digital": digital, "two_day": two_day}

    merged, _ = reconcile_sources(sources, proximity_minutes=0)

    # Total unique 15-min buckets across both = 36 hourly buckets with no gaps
    # (digital 0-23, two_day 12-35) = 36 distinct buckets
    assert len(merged) == 36
    # All values are 10.0 since neither source needed an offset
    assert (merged[TILT_COL] == 10.0).all()


def test_low_priority_source_only_fills_gaps():
    """Lowest-priority source only contributes where higher-priority sources
    have nothing — exactly the behavior that used to require GAP_FILL_SOURCES.

    Disable proximity gating because the test deliberately creates
    overlapping hourly samples; with the gate enabled the lower-priority
    source's overlap-region samples would be dropped before the merge.
    """
    digital = _series("2025-03-01", n=12, base=10.0, freq="1h")
    # three_month overlaps digital but ALSO extends earlier (the "gap")
    three_month = _series("2025-02-28 12:00:00", n=24, base=10.0, freq="1h")
    sources = {"digital": digital, "three_month": three_month}

    merged, _ = reconcile_sources(sources, proximity_minutes=0)

    # Earliest bucket should come from three_month (digital doesn't cover it)
    earliest = merged[DATE_COL].iloc[0]
    assert earliest == pd.Timestamp("2025-02-28 12:00:00")
    # But the buckets where they overlap should come from digital (no offset
    # was applied since both are 10.0, so we can't visually distinguish — but
    # the count should be union, not duplicated)
    assert len(merged) == 24  # union of 12+24 with 12-bucket overlap


def test_proximity_gate_drops_low_priority_spike_between_high_priority():
    """The bug case from the user observation on 2025-04-20: dec2024_to_now
    has an isolated sample at 18:07:35 with a slightly off y-frame, sandwiched
    between two digital samples at 18:00 and 18:30. Without the gate, the
    18:07:35 sample wins its 15-min bucket (which digital doesn't cover) and
    renders as a visible spike in the chart. With the gate, the dec2024_to_now
    sample is dropped because both digital samples are within 60 min of it.
    """
    # digital: hourly samples that don't have a row at 18:07
    digital_dates = pd.to_datetime([
        "2025-04-20 17:30:00",
        "2025-04-20 18:00:00",
        "2025-04-20 18:30:00",
        "2025-04-20 19:00:00",
    ])
    digital = pd.DataFrame({DATE_COL: digital_dates, TILT_COL: [11.36, 11.36, 11.33, 11.31]})
    # The dec2024_to_now spike sample
    spike = pd.DataFrame(
        {DATE_COL: pd.to_datetime(["2025-04-20 18:07:35"]), TILT_COL: [14.43]}
    )

    merged, report = reconcile_sources({"digital": digital, "dec2024_to_now": spike})

    # The spike should not appear in the merged history at all
    assert (merged[TILT_COL].between(11.0, 12.0)).all(), (
        f"unexpected high values in merged: {merged[TILT_COL].tolist()}"
    )
    # And the report should record that one row was proximity-gated
    d2n_record = next(s for s in report.sources if s.name == "dec2024_to_now")
    assert d2n_record.rows_proximity_dropped == 1


def test_proximity_gate_keeps_low_priority_in_genuine_gaps():
    """The flip side of the gate: a low-priority sample that's NOT within
    proximity of any higher-priority sample should be retained — that's the
    whole point of having a low-priority source as a gap-filler.

    Setup: digital has samples on 2025-04-20 only. dec2024_to_now has a
    sample on 2025-04-22 (>2 days from any digital sample). The gate should
    not touch the 2025-04-22 sample.
    """
    digital = _series("2025-04-20", n=8, base=10.0, freq="3h")
    fill = pd.DataFrame(
        {DATE_COL: pd.to_datetime(["2025-04-22 12:00:00"]), TILT_COL: [12.0]}
    )

    merged, report = reconcile_sources({"digital": digital, "dec2024_to_now": fill})

    # The fill sample is far from any digital sample, so it should survive
    assert (merged[DATE_COL] == pd.Timestamp("2025-04-22 12:00:00")).any()
    d2n_record = next(s for s in report.sources if s.name == "dec2024_to_now")
    assert d2n_record.rows_proximity_dropped == 0


def test_piecewise_realign_runs_for_dec2024_to_now_and_records_residuals():
    """The piecewise pass runs against each higher-priority source that
    dec2024_to_now overlaps with and records a residual offset on the
    SourceAlignment record.

    Caveat: this test only verifies the MECHANICS — that the piecewise
    pass runs, finds overlaps, and stores residuals. The residual VALUE
    against digital comes out as ~0 by construction (digital dominated
    the first-pass median offset), but for sources that ALSO align
    indirectly via dec2024_to_now the residual lands at 0 too because
    they were already shifted into dec2024_to_now's frame in the
    alignment loop (chicken-and-egg). The mechanics are still useful as
    a regression check; the per-source proximity gate override
    (PROXIMITY_GATE_OVERRIDES_MINUTES["dec2024_to_now"] = 12h) is what
    actually mitigates the chicken-and-egg in production.
    """
    digital = _series("2025-01-01", n=180, base=0.0, freq="1D")
    d2n_dates = list(pd.date_range("2025-01-01", periods=60, freq="3D")) + list(
        pd.date_range("2025-07-02", periods=60, freq="2D")
    )
    d2n_tilts = [6.928] * 60 + [12.928] * 60
    d2n = pd.DataFrame({DATE_COL: d2n_dates, TILT_COL: d2n_tilts})

    sources = {
        "digital": digital,
        "dec2024_to_now": d2n,
    }
    _, report = reconcile_sources(sources, proximity_minutes=0)

    d2n_record = next(s for s in report.sources if s.name == "dec2024_to_now")
    # First-pass offset is positive (median of d2n - digital ≈ +6.928).
    assert d2n_record.offset_microrad is not None
    assert abs(d2n_record.offset_microrad - 6.928) < 0.5

    # The piecewise pass ran and recorded a residual against digital.
    assert "digital" in d2n_record.piecewise_residuals


def test_piecewise_realign_skips_sources_not_in_PIECEWISE_REALIGN_SOURCES():
    """The realignment is targeted: it only runs for sources listed in
    PIECEWISE_REALIGN_SOURCES (today: just dec2024_to_now). Other sources
    keep their single-offset alignment from the first pass.
    """
    digital = _series("2025-01-01", n=48, base=10.0, freq="1h")
    two_day = _series("2025-01-01", n=48, base=15.0, freq="1h")
    sources = {"digital": digital, "two_day": two_day}

    _, report = reconcile_sources(sources, proximity_minutes=0)
    two_day_record = next(s for s in report.sources if s.name == "two_day")
    # two_day is NOT in PIECEWISE_REALIGN_SOURCES, so its piecewise_residuals
    # dict should be empty.
    assert two_day_record.piecewise_residuals == {}


def test_proximity_gate_can_be_disabled():
    """proximity_minutes=0 disables the gate — useful for tests of the merge
    step that intentionally construct overlapping sources.
    """
    digital = _series("2025-04-20", n=4, base=10.0, freq="1h")
    spike = pd.DataFrame(
        {DATE_COL: pd.to_datetime(["2025-04-20 00:30:00"]), TILT_COL: [15.0]}
    )

    # With gate at default 60min: spike is dropped
    merged_gated, _ = reconcile_sources({"digital": digital, "dec2024_to_now": spike})
    # With gate disabled: spike is kept (and wins its 15-min bucket)
    merged_open, _ = reconcile_sources(
        {"digital": digital, "dec2024_to_now": spike}, proximity_minutes=0
    )

    assert 15.0 not in merged_gated[TILT_COL].tolist()
    assert 15.0 in merged_open[TILT_COL].tolist()


def test_conflict_recorded_when_aligned_sources_disagree():
    """After alignment, if the winning source's value differs from the
    losing source's value at the SAME bucket by more than CONFLICT_THRESHOLD,
    it shows up on the report with both source names and both values.

    Disable proximity gating — the test puts both sources at the same
    timestamps to force a same-bucket disagreement, which the gate would
    otherwise short-circuit by dropping the lower-priority source's row.
    """
    # Anchor is flat. Lower-priority source mostly agrees but one bucket
    # is +5 µrad off (the conflict).
    anchor = _series("2025-03-01", n=24, base=10.0, freq="1h")
    new_vals = np.full(24, 10.0)
    new_vals[10] = 15.0  # the conflict bucket
    new = pd.DataFrame({DATE_COL: anchor[DATE_COL].copy(), TILT_COL: new_vals})
    sources = {"digital": anchor, "two_day": new}

    _, report = reconcile_sources(sources, proximity_minutes=0)

    # The median delta is 0 (most buckets agree), so the offset for two_day
    # is ~0 — meaning the aligned values are still ~10 except for one at ~15.
    # That single bucket is the conflict.
    assert len(report.conflicts) == 1
    c = report.conflicts[0]
    assert c.winning_source == "digital"
    assert c.losing_source == "two_day"
    assert abs(c.delta - (-5.0)) < 1e-9 or abs(c.delta - 5.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Determinism + order independence
# ─────────────────────────────────────────────────────────────────────────────


def test_reconcile_is_deterministic():
    """Same inputs → identical outputs across repeated calls."""
    sources = {
        "digital": _series("2025-03-01", n=24, base=10.0),
        "two_day": _series("2025-03-01 06:00:00", n=24, base=11.0),
        "month": _series("2025-03-01 12:00:00", n=24, base=13.0),
    }
    merged_a, report_a = reconcile_sources(sources)
    merged_b, report_b = reconcile_sources(sources)

    pd.testing.assert_frame_equal(merged_a, merged_b)
    assert [s.name for s in report_a.sources] == [s.name for s in report_b.sources]


def test_dict_iteration_order_does_not_affect_result():
    """The dict's insertion order shouldn't matter — SOURCE_PRIORITY does."""
    a = {
        "digital": _series("2025-03-01", n=24, base=10.0),
        "two_day": _series("2025-03-01 06:00:00", n=24, base=11.0),
    }
    b = {
        "two_day": _series("2025-03-01 06:00:00", n=24, base=11.0),
        "digital": _series("2025-03-01", n=24, base=10.0),
    }
    merged_a, _ = reconcile_sources(a)
    merged_b, _ = reconcile_sources(b)
    pd.testing.assert_frame_equal(merged_a, merged_b)


def test_input_dataframes_are_not_mutated():
    """Reconciliation must work on copies — we don't want to surprise callers
    by changing the DataFrames they passed in."""
    digital = _series("2025-03-01", n=24, base=10.0)
    two_day = _series("2025-03-01", n=24, base=16.0)
    digital_orig = digital.copy()
    two_day_orig = two_day.copy()

    reconcile_sources({"digital": digital, "two_day": two_day})

    pd.testing.assert_frame_equal(digital, digital_orig)
    pd.testing.assert_frame_equal(two_day, two_day_orig)


def test_unknown_source_names_are_ignored():
    """If a caller passes a source name not in SOURCE_PRIORITY, it's silently
    skipped (not added to the merge). The known sources still produce output.
    """
    sources = {
        "digital": _series("2025-03-01", n=24, base=10.0),
        "totally_made_up_source": _series("2025-03-01", n=24, base=99.0),
    }
    merged, report = reconcile_sources(sources)
    assert len(merged) > 0
    assert (merged[TILT_COL] == 10.0).all()
    assert all(s.name in SOURCE_PRIORITY for s in report.sources)


# ─────────────────────────────────────────────────────────────────────────────
# Schema invariants
# ─────────────────────────────────────────────────────────────────────────────


def test_merged_history_has_canonical_schema():
    sources = {"digital": _series("2025-03-01", n=10, base=10.0)}
    merged, _ = reconcile_sources(sources)
    assert list(merged.columns) == [DATE_COL, TILT_COL]
    assert merged[DATE_COL].dtype.kind == "M"
    assert merged[TILT_COL].dtype.kind == "f"


def test_merged_history_is_sorted_chronologically():
    """The merge must produce a chronologically-sorted DataFrame regardless
    of how the input sources are ordered internally."""
    digital = _series("2025-03-15", n=24, base=10.0)
    two_day = _series("2025-03-01", n=24, base=10.0)  # earlier than digital
    sources = {"digital": digital, "two_day": two_day}
    merged, _ = reconcile_sources(sources)
    dates = merged[DATE_COL].to_numpy()
    assert (dates[1:] >= dates[:-1]).all()
