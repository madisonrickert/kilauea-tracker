"""Tests for `kilauea_tracker.reconcile` (pairwise scalar-offset model).

The v4 rewrite collapsed the earlier slope+intercept model `y_i =
a_i · true + b_i` down to scalar-offset-only: `y_i = true + b_i`.
Rationale is in the module docstring. These tests cover:

  - Pair fits correctly recover the injected scalar offset.
  - Digital is pinned to (a=1, b=0) by the joint solve (`a` is a schema
    field retained for back-compat; it's always 1.0 in the new model).
  - A source with no direct digital overlap is recovered via the chain
    of pairwise constraints through intermediate sources.
  - MAD outlier rejection is diagnostic-only — it does not remove a
    source from winner candidacy.
  - Archive is a pure gap-filler: it contributes only for buckets where
    no live source has coverage.
  - Continuity audit surfaces adjacent-bucket steps into the report.
  - Determinism and order-independence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kilauea_tracker.config import (
    CONTINUITY_WARNING_THRESHOLD_MICRORAD,
    PAIRWISE_MIN_OVERLAP_BUCKETS,
    SOURCE_PRIORITY,
)
from kilauea_tracker.model import DATE_COL, TILT_COL
from kilauea_tracker.reconcile import reconcile_sources


def _series(
    start: str,
    n: int,
    base: float = 0.0,
    freq: str = "1h",
    amplitude: float = 0.0,
    period_hours: float = 24.0,
) -> pd.DataFrame:
    """Synthetic tilt series. `amplitude` > 0 gives sinusoidal variation so
    pairwise OLS has enough spread to resolve slope from intercept.
    """
    dates = pd.date_range(start, periods=n, freq=freq)
    t = np.arange(n, dtype=float)
    values = base + amplitude * np.sin(2 * np.pi * t / period_hours)
    return pd.DataFrame({DATE_COL: dates, TILT_COL: values})


def _apply_linear(df: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    """Inject y = a · tilt + b into a truth-frame series."""
    out = df.copy()
    out[TILT_COL] = a * out[TILT_COL] + b
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise fit recovery
# ─────────────────────────────────────────────────────────────────────────────


def test_digital_is_pinned_to_identity():
    """When digital is present, its (a, b) = (1, 0) by construction."""
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 2
    truth = _series("2025-03-01", n=n, base=0.0, amplitude=10.0)
    sources = {
        "digital": truth,
        "dec2024_to_now": _apply_linear(truth, 1.25, 3.0),
    }
    merged, report = reconcile_sources(sources)
    digital_record = next(s for s in report.sources if s.name == "digital")
    assert digital_record.is_anchor is True
    assert abs(digital_record.a - 1.0) < 1e-9
    assert abs(digital_record.b) < 1e-9


def test_pairwise_fit_recovers_injected_scalar_offset():
    """A source with an injected scalar offset `b=3.0` against digital
    must be solved to `a=1.0, b=3.0`. The applied correction brings it
    back to digital's frame exactly in the scalar-offset model.
    """
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 3
    truth = _series("2025-03-01", n=n, base=5.0, amplitude=8.0)
    injected_b = 3.0

    sources = {
        "digital": truth,
        "dec2024_to_now": _apply_linear(truth, 1.0, injected_b),
    }
    merged, report = reconcile_sources(sources)

    dec_record = next(s for s in report.sources if s.name == "dec2024_to_now")
    assert dec_record.a == 1.0, (
        f"fit.a = {dec_record.a:.6f}; b-only model keeps a at exactly 1.0"
    )
    assert abs(dec_record.b - injected_b) < 1e-6, (
        f"fit.b = {dec_record.b:.6f}, expected {injected_b}"
    )
    # With digital's higher effective resolution it wins every overlap bucket.
    # The merged curve must match truth to numerical precision.
    merged_vs_truth = merged.merge(truth, on=DATE_COL, suffixes=("_m", "_t"))
    # In the densified merge, interpolated points at the ends of each
    # source's window may round-trip with tiny timestamp drift. Match
    # strictly on the original truth timestamps only.
    max_err = float(
        (merged_vs_truth[f"{TILT_COL}_m"] - merged_vs_truth[f"{TILT_COL}_t"]).abs().max()
    )
    assert max_err < 1e-6, f"merged disagrees with truth by up to {max_err}"


def test_pairwise_chain_propagates_through_intermediate_source():
    """Source A and source C have NO direct temporal overlap, but both
    overlap with source B. The pairwise calibration must still recover
    A's scalar offset relative to C via the chain A↔B and B↔C.

    Concretely: digital covers Jan-Jun 2025; two_day covers only the last
    48 h; dec2024_to_now covers both. The solve should still produce
    sensible b-offsets for two_day via its overlap with dec2024_to_now.
    """
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 3
    digital_truth = _series("2025-03-01", n=n, base=0.0, amplitude=10.0)
    dec_truth_early = _series("2025-03-01", n=n, base=0.0, amplitude=10.0)
    dec_truth_late = _series("2026-04-20 00:00:00", n=n, base=-5.0, amplitude=6.0)
    two_day_truth = _series("2026-04-20 00:00:00", n=n, base=-5.0, amplitude=6.0)

    # Inject scalar offsets only (b-only model has no slope to recover).
    dec_injected_b = 2.0
    two_day_injected_b = -1.5
    sources = {
        "digital": digital_truth,
        "dec2024_to_now": pd.concat(
            [_apply_linear(dec_truth_early, 1.0, dec_injected_b),
             _apply_linear(dec_truth_late, 1.0, dec_injected_b)],
            ignore_index=True,
        ),
        "two_day": _apply_linear(two_day_truth, 1.0, two_day_injected_b),
    }
    merged, report = reconcile_sources(sources)

    two_day_record = next(s for s in report.sources if s.name == "two_day")
    dec_record = next(s for s in report.sources if s.name == "dec2024_to_now")
    assert dec_record.a == 1.0 and two_day_record.a == 1.0
    assert abs(dec_record.b - dec_injected_b) < 0.2, (
        f"dec.b = {dec_record.b:.4f}, expected ~{dec_injected_b}"
    )
    assert abs(two_day_record.b - two_day_injected_b) < 0.3, (
        f"two_day.b = {two_day_record.b:.4f}, expected ~{two_day_injected_b}"
    )


def test_pairwise_fits_recorded_in_report():
    """Every pair with ≥ PAIRWISE_MIN_OVERLAP_BUCKETS overlap contributes
    one `PairwiseFit` entry to the report.
    """
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 2
    truth = _series("2025-03-01", n=n, base=0.0, amplitude=5.0)
    sources = {
        "digital": truth,
        "dec2024_to_now": _apply_linear(truth, 1.1, 2.0),
        "three_month": _apply_linear(truth, 0.95, -1.0),
    }
    _, report = reconcile_sources(sources)
    # 3 pairs: (digital, dec), (digital, three_month), (dec, three_month)
    assert len(report.pairs) == 3


def test_pair_skipped_when_overlap_too_small():
    """A pair with fewer than PAIRWISE_MIN_OVERLAP_BUCKETS shared buckets
    is skipped from the joint solve.
    """
    small = PAIRWISE_MIN_OVERLAP_BUCKETS - 10  # below threshold
    truth = _series("2025-03-01", n=small, base=0.0, amplitude=5.0)
    sources = {
        "digital": truth,
        "dec2024_to_now": _apply_linear(truth, 1.1, 2.0),
    }
    _, report = reconcile_sources(sources)
    assert len(report.pairs) == 0
    # dec_record still emitted but with no pair constraints — keeps (1, 0).
    dec_record = next(s for s in report.sources if s.name == "dec2024_to_now")
    assert dec_record.pairs_used == 0


# ─────────────────────────────────────────────────────────────────────────────
# MAD outlier rejection
# ─────────────────────────────────────────────────────────────────────────────


def test_mad_outlier_drops_single_bucket_spike():
    """A catastrophically-wrong value at one bucket is rejected by the
    MAD gate when K ≥ 3 sources disagree — the MAD statistic needs at
    least 3 points to meaningfully distinguish an outlier. The source
    itself keeps contributing at its other (non-spiked) buckets.
    """
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 2
    truth = _series("2025-03-01", n=n, base=0.0, amplitude=5.0)
    bad = truth.copy()
    bad_row = n // 2
    bad.loc[bad_row, TILT_COL] = 50.0
    sources = {
        "digital": truth,
        "dec2024_to_now": bad,
        "three_month": truth.copy(),
    }
    merged, report = reconcile_sources(sources)
    # The bad bucket should see dec's value rejected from the merge.
    assert len(report.transcription_failures) >= 1
    dec_record = next(s for s in report.sources if s.name == "dec2024_to_now")
    assert dec_record.rows_mad_rejected >= 1
    # Merged output at the bad bucket tracks truth (digital wins after MAD drops dec).
    truth_at_bad = float(truth.loc[bad_row, TILT_COL])
    bad_bucket_ts = bad.loc[bad_row, DATE_COL]
    merged_at_bucket = merged[merged[DATE_COL] == bad_bucket_ts]
    assert len(merged_at_bucket) >= 1
    assert (
        abs(float(merged_at_bucket.iloc[0][TILT_COL]) - truth_at_bad) < 0.5
    ), (
        f"merged={float(merged_at_bucket.iloc[0][TILT_COL]):.3f}, "
        f"truth={truth_at_bad:.3f}"
    )


def test_k2_disagreement_output_is_correct_even_without_mad_flag():
    """With only K=2 sources disagreeing at a bucket, MAD statistics
    can't resolve which is wrong — but best-effective-resolution still
    picks digital, so the merged output is correct. The MAD gate does
    not emit a transcription failure in this case (that's fine — the
    diagnostic goal of MAD is K ≥ 3).
    """
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 2
    truth = _series("2025-03-01", n=n, base=0.0, amplitude=5.0)
    bad = truth.copy()
    bad.loc[n // 2, TILT_COL] = 50.0
    sources = {"digital": truth, "dec2024_to_now": bad}
    merged, _ = reconcile_sources(sources)
    truth_at_bad = float(truth.loc[n // 2, TILT_COL])
    bad_bucket_ts = bad.loc[n // 2, DATE_COL]
    merged_at_bucket = merged[merged[DATE_COL] == bad_bucket_ts]
    assert len(merged_at_bucket) >= 1
    # digital (effective_resolution=0.001) beats dec2024_to_now (0.62), so
    # truth value wins. The spike is effectively suppressed — just not by MAD.
    assert (
        abs(float(merged_at_bucket.iloc[0][TILT_COL]) - truth_at_bad) < 0.5
    )


# ─────────────────────────────────────────────────────────────────────────────
# Archive as gap-filler only
# ─────────────────────────────────────────────────────────────────────────────


def test_archive_fills_gaps_where_no_live_source():
    """Archive rows whose buckets are not covered by any live source MUST
    appear in the merged output (gap-fill role).
    """
    live = _series("2025-03-01", n=48, base=5.0, freq="15min")
    archive = _series("2024-01-01", n=48, base=20.0, freq="1h")  # older than any live
    sources = {
        "digital": live,
        "archive": archive,
    }
    merged, _ = reconcile_sources(sources)
    # Every live bucket preserved + every archive bucket added (no overlap).
    assert len(merged) == len(live) + len(archive)
    # The 2024 range comes from archive.
    archive_slice = merged[merged[DATE_COL] < pd.Timestamp("2025-01-01")]
    assert len(archive_slice) == len(archive)
    assert abs(archive_slice[TILT_COL].mean() - 20.0) < 1e-9


def test_archive_loses_to_live_source_at_overlapping_bucket():
    """When archive and a live source both cover the same bucket, the live
    source's (corrected) value wins — archive never contributes in the
    contested region.
    """
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 2
    truth = _series("2025-03-01", n=n, base=0.0, amplitude=5.0, freq="1h")
    # Archive has a DIFFERENT value for the same buckets (simulates stale data).
    stale_archive = truth.copy()
    stale_archive[TILT_COL] += 10.0  # intentionally wrong
    sources = {
        "digital": truth,
        "archive": stale_archive,
    }
    merged, _ = reconcile_sources(sources)
    # Densification interpolates digital onto the 15-min grid, so the merged
    # output has more rows than `truth`. Compare on matched timestamps to
    # verify digital (not archive) wins everywhere it covers.
    matched = merged.merge(truth, on=DATE_COL, suffixes=("_m", "_t"))
    max_err = (
        matched[f"{TILT_COL}_m"] - matched[f"{TILT_COL}_t"]
    ).abs().max()
    assert max_err < 1e-6, (
        f"merged diverges from truth by {max_err} — archive may be winning"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Continuity audit
# ─────────────────────────────────────────────────────────────────────────────


def test_continuity_violation_recorded_when_adjacent_buckets_step():
    """A deliberate 10 µrad step in the merged output is surfaced as a
    continuity violation in the report.
    """
    n = 48
    # Two back-to-back series with a step between them.
    first = _series("2025-03-01 00:00:00", n=n, base=0.0, freq="15min")
    second = _series("2025-03-01 12:00:00", n=n, base=10.0, freq="15min")
    # Use digital so the merge passes them through unchanged.
    sources = {
        "digital": pd.concat([first, second], ignore_index=True),
    }
    _, report = reconcile_sources(sources)
    # The report should flag the step at the transition.
    assert any(
        abs(v.delta_microrad) >= CONTINUITY_WARNING_THRESHOLD_MICRORAD
        for v in report.continuity_violations
    )


# ─────────────────────────────────────────────────────────────────────────────
# Determinism & order-independence
# ─────────────────────────────────────────────────────────────────────────────


def test_reconcile_is_deterministic():
    """Same inputs → same outputs."""
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 2
    truth = _series("2025-03-01", n=n, base=0.0, amplitude=5.0)
    sources = {
        "digital": truth,
        "dec2024_to_now": _apply_linear(truth, 1.1, 2.0),
    }
    m1, r1 = reconcile_sources(sources)
    m2, r2 = reconcile_sources(sources)
    pd.testing.assert_frame_equal(
        m1.reset_index(drop=True), m2.reset_index(drop=True)
    )
    assert r1.rows_out == r2.rows_out


def test_reconcile_is_order_independent():
    """Dict insertion order doesn't change the merged output."""
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 2
    truth = _series("2025-03-01", n=n, base=0.0, amplitude=5.0)
    sources_a = {
        "digital": truth,
        "dec2024_to_now": _apply_linear(truth, 1.1, 2.0),
    }
    sources_b = {
        "dec2024_to_now": _apply_linear(truth, 1.1, 2.0),
        "digital": truth,
    }
    m_a, _ = reconcile_sources(sources_a)
    m_b, _ = reconcile_sources(sources_b)
    pd.testing.assert_frame_equal(
        m_a.reset_index(drop=True), m_b.reset_index(drop=True)
    )


def test_empty_sources_returns_empty_history():
    merged, report = reconcile_sources({})
    assert len(merged) == 0
    assert list(merged.columns) == [DATE_COL, TILT_COL]
    assert report.rows_out == 0


# ─────────────────────────────────────────────────────────────────────────────
# v4 scalar-offset-only regression guards
# ─────────────────────────────────────────────────────────────────────────────


def test_digital_hard_pin_exact():
    """With many pair constraints, the digital pin MUST come out exactly
    (a=1.0, b=0.0), not approximately.

    Regression guard for the Phase 2 soft-pin bug where the joint lstsq
    treated the pin as one equal-weight row and returned
    `a_digital = 0.9622` in 2026-04 production data. Under the v4
    b-only model `a` is always exactly 1.0, but the test also exercises
    the `b=0.0` invariant, which is still load-bearing.
    """
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 3
    truth = _series("2025-03-01", n=n, base=0.0, amplitude=10.0)
    sources = {
        "digital": truth,
        "dec2024_to_now": _apply_linear(truth, 1.0, 2.0),
        "three_month": _apply_linear(truth, 1.0, -1.5),
        "month": _apply_linear(truth, 1.0, 0.5),
        "week": _apply_linear(truth, 1.0, 0.8),
        "two_day": _apply_linear(truth, 1.0, -2.0),
    }
    _, report = reconcile_sources(sources)
    digital = next(s for s in report.sources if s.name == "digital")
    assert digital.a == 1.0, f"a_digital = {digital.a}, expected exactly 1.0"
    assert digital.b == 0.0, f"b_digital = {digital.b}, expected exactly 0.0"
    assert digital.is_anchor is True


def test_every_source_has_a_equals_one_under_b_only_model():
    """Schema invariant under the v4 model: `a` is always exactly 1.0
    regardless of what the input looks like. Protects against a future
    regression that reintroduces slope fitting without auditing
    downstream consumers (pipeline JSON serializer, Streamlit).
    """
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 3
    truth = _series("2025-03-01", n=n, base=0.0, amplitude=10.0)
    # Inject data that would produce a meaningful OLS slope if we fit
    # for one (actual slope = 1.5). Under b-only, `a` must stay at 1.
    sources = {
        "digital": truth,
        "dec2024_to_now": _apply_linear(truth, 1.5, 4.0),
    }
    _, report = reconcile_sources(sources)
    for record in report.sources:
        assert record.a == 1.0, (
            f"{record.name}.a = {record.a}; b-only model forbids slope correction"
        )


def test_per_region_winner_deterministic_under_mad_pressure():
    """Phase 4 Commit 4: within a region of stable coverage, the same
    source wins every bucket, even when MAD flags it as an outlier.

    Constructs a pathological scenario: digital and dec2024_to_now both
    cover the same range, but dec has an intermittent 4-5 µrad jitter
    that pre-Commit-4 would cause MAD to reject dec at some buckets,
    flipping the winner to... (no alternate with just 2 sources, but
    the test still asserts no source-change mid-region).
    """
    from kilauea_tracker.reconcile import reconcile_sources as _rs

    n = 40
    # Use an irregular signal so MAD has teeth.
    t = np.arange(n)
    rng = np.random.default_rng(0)
    signal = -20 + 3 * np.sin(t / 2) + rng.normal(0, 0.3, n)
    digital_df = pd.DataFrame({
        DATE_COL: pd.date_range("2025-03-01", periods=n, freq="15min"),
        TILT_COL: signal,
    })
    # dec2024_to_now: same truth + bigger per-bucket noise, biased +2 µrad
    dec_df = digital_df.copy()
    dec_df[TILT_COL] = dec_df[TILT_COL] + 2.0 + rng.normal(0, 1.5, n)

    merged, report = _rs({"digital": digital_df, "dec2024_to_now": dec_df})
    # digital has best effective resolution and should win EVERY bucket
    # in this region — both sources cover every bucket, so the source
    # set is constant.
    assert report.winner_counts.get("digital", 0) == len(merged), (
        f"digital should win all {len(merged)} buckets, got "
        f"{report.winner_counts.get('digital', 0)}"
    )
    # dec2024_to_now should never win here.
    assert report.winner_counts.get("dec2024_to_now", 0) == 0


def test_mad_does_not_remove_from_merge_phase4_commit4():
    """Phase 4 Commit 4 demoted the MAD gate to diagnostic-only.

    An outlier at a bucket MUST still appear as a TranscriptionFailure
    and be counted in `rows_mad_rejected`, but the underlying source
    is NOT removed from winner candidacy — winner is still decided by
    best-effective-resolution across all contributing sources.

    Here we arrange for the outlier to be the BEST-effective-resolution
    source (`digital`). Pre-Commit-4 MAD would have dropped digital and
    the merge would have picked dec's "correct" value. Post-Commit-4
    digital wins anyway because the merge is deterministic.
    """
    from kilauea_tracker.reconcile import reconcile_sources as _rs

    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 2
    truth = _series("2025-03-01", n=n, base=0.0, amplitude=5.0, freq="15min")
    # digital: good everywhere except ONE bucket spiked up
    digital_spiked = truth.copy()
    spike_idx = n // 2
    digital_spiked.loc[spike_idx, TILT_COL] = 50.0  # 50 µrad spike
    dec_ok = _apply_linear(truth, 1.05, 0.5)  # slightly miscalibrated

    sources = {"digital": digital_spiked, "dec2024_to_now": dec_ok.copy()}
    sources["three_month"] = _apply_linear(truth, 0.98, 0.1)  # need K≥3 for MAD
    merged, report = _rs(sources)

    # MAD flagged digital at the spike — logged but didn't remove from winner.
    assert report.transcription_failures, (
        "MAD should still log the spike as a transcription_failure"
    )
    # digital still won the spike bucket (best effective resolution).
    spike_ts = truth.loc[spike_idx, DATE_COL]
    merged_at_spike = merged[merged[DATE_COL] == spike_ts]
    assert len(merged_at_spike) == 1
    # The merged value at the spike IS the spike (50 µrad) because digital
    # won and we no longer remove it. Upstream rolling-median filter is
    # responsible for catching spikes before they reach reconcile.
    assert abs(merged_at_spike.iloc[0][TILT_COL] - 50.0) < 1.0


def test_mad_floor_does_not_reject_well_calibrated_outlier():
    """When multiple miscalibrated sources agree at one value and one
    well-calibrated source sits several µrad away, the MAD floor MUST
    tolerate the well-calibrated source's delta.

    Regression guard for the 2026-04 sawtooth: with `MAD_FLOOR = 2.0`,
    two_day's legitimate 2.5-3 µrad offset from a miscalibrated median
    was alternately rejected bucket-to-bucket, producing a visible
    zigzag. Floor raised to 5 µrad in Phase 4 Commit 3 fixes this while
    still catching genuine OCR-glitch outliers (typically 10+ µrad off).
    """
    from kilauea_tracker.reconcile import reconcile_sources as _rs
    from kilauea_tracker.config import MAD_OUTLIER_SIGMA_FLOOR_MICRORAD

    assert MAD_OUTLIER_SIGMA_FLOOR_MICRORAD >= 3.0, (
        "MAD floor regression: Phase 4 raised this to 5 to stop sawtooth"
    )

    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 2
    # 3 miscalibrated sources agree at -28; one well-calibrated at -25.
    # Delta of 3 µrad would be kicked by floor=2 but not by floor=5.
    truth = _series("2025-03-01", n=n, base=-25.0, amplitude=0.5, freq="15min")
    miscalibrated = truth.copy()
    miscalibrated[TILT_COL] = miscalibrated[TILT_COL] - 3.0  # shifted by -3 µrad
    sources = {
        "digital": truth,  # well-calibrated, best resolution (0.001)
        "week": miscalibrated.copy(),
        "month": miscalibrated.copy(),
        "three_month": miscalibrated.copy(),
    }
    merged, _ = _rs(sources)
    # digital should win every bucket. If MAD kicked it (the 2026-04 bug),
    # we'd see week/month/three_month's value (-28) in the merged output.
    merged_mean = merged[TILT_COL].mean()
    assert abs(merged_mean - (-25.0)) < 0.5, (
        f"merged mean {merged_mean:.2f} suggests digital got MAD-rejected; "
        f"the well-calibrated source should win over miscalibrated consensus"
    )


def test_winner_counts_populated():
    """_merge_best_resolution records which source won each bucket so
    the Streamlit diagnostics panel and JSON run report can display the
    distribution.
    """
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 2
    truth = _series("2025-03-01", n=n, base=0.0, amplitude=5.0, freq="15min")
    sources = {
        "digital": truth,
        "dec2024_to_now": _apply_linear(truth, 1.1, 2.0),
    }
    merged, report = reconcile_sources(sources)
    assert isinstance(report.winner_counts, dict)
    assert sum(report.winner_counts.values()) == len(merged)
    # digital has tighter effective resolution, so it wins every bucket.
    assert report.winner_counts.get("digital", 0) == len(merged)


def test_median_offset_robust_to_gross_outliers():
    """Pair fits use `β = median(y_i - y_j)`, which is robust to a small
    fraction of gross outliers. A mean-based estimator on the same data
    would be dragged several µrad by ±30 µrad spikes.
    """
    from kilauea_tracker.reconcile import _compute_pairwise_fits, ReconcileReport

    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 4
    rng = np.random.default_rng(seed=42)
    truth = _series("2025-03-01", n=n, base=0.0, amplitude=10.0)
    source = _apply_linear(truth, 1.0, 0.0).copy()

    # Inject 3% of samples as ±30 µrad gross outliers.
    n_outliers = max(3, int(n * 0.03))
    outlier_idx = rng.choice(n, size=n_outliers, replace=False)
    source.loc[outlier_idx, TILT_COL] += rng.choice([-30.0, 30.0], size=n_outliers)

    report = ReconcileReport()
    fits = _compute_pairwise_fits(
        {"digital": truth, "dec2024_to_now": source}, report
    )
    assert len(fits) == 1
    fit = fits[0]
    # True relationship: source = digital + 0 (+ outliers). `alpha` is
    # always 1.0 by schema; `beta` must stay near 0.
    assert fit.alpha == 1.0
    assert abs(fit.beta) < 1.0, (
        f"median β={fit.beta} should be near 0 despite {n_outliers} outliers"
    )


def test_disconnected_component_defaults_to_zero_offset():
    """A source with no path to the pin in the pair graph cannot be
    solved — the joint system is under-determined for its variable.
    The fallback is `b=0` with a warning on the report, so downstream
    merge policy can still use the source (best-effective-resolution
    may prefer it) but the diagnostics panel flags the miscalibration.
    """
    # Two disjoint temporal regions. `digital` covers Jan-Jun 2025;
    # `month` covers Apr 2026. Their overlap is zero, so no pair
    # constraint connects month to the pin.
    digital = _series("2025-03-01", n=PAIRWISE_MIN_OVERLAP_BUCKETS * 3,
                      base=0.0, amplitude=8.0)
    month = _series("2026-04-01", n=PAIRWISE_MIN_OVERLAP_BUCKETS * 3,
                    base=-10.0, amplitude=4.0)
    _, report = reconcile_sources({"digital": digital, "month": month})
    month_rec = next(s for s in report.sources if s.name == "month")
    assert month_rec.a == 1.0
    assert month_rec.b == 0.0
    assert month_rec.note is not None and "no path to anchor" in month_rec.note


def test_sources_unknown_to_priority_tuple_silently_ignored_for_fallback_resolution():
    """A source name not in EFFECTIVE_RESOLUTION_FALLBACK_MICRORAD_PER_PIXEL
    gets the default 1.0 µrad/px resolution — it still participates in the
    pairwise fit but loses ties to any named source.
    """
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 2
    truth = _series("2025-03-01", n=n, base=0.0, amplitude=5.0)
    sources = {
        "digital": truth,
        "experimental_source": _apply_linear(truth, 1.0, 0.0),
    }
    merged, report = reconcile_sources(sources)
    # digital wins every bucket (it's in the priority tuple with tight res).
    matched = merged.merge(truth, on=DATE_COL, suffixes=("_m", "_t"))
    max_err = (
        matched[f"{TILT_COL}_m"] - matched[f"{TILT_COL}_t"]
    ).abs().max()
    assert max_err < 1e-6, (
        f"merged diverges from truth by {max_err}; experimental_source may "
        f"be winning due to a priority-tuple regression"
    )


def test_archive_in_source_priority_enumeration():
    """Archive is last in SOURCE_PRIORITY under Phase 2 because it only
    wins buckets no live source covers. This is a rule-enforcement test
    for the Phase 2 config layout.
    """
    assert SOURCE_PRIORITY[-1] == "archive"
    assert SOURCE_PRIORITY[0] == "digital"
