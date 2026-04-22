"""Tests for `kilauea_tracker.reconcile` (Phase 2 pairwise calibration).

The v3 alignment rewrite replaced the priority-merge + scalar-offset stack
with pairwise self-consistency calibration. These tests cover:

  - Pairwise OLS fits correctly recover injected (a_i, b_i) under the
    model y_i = a_i · true + b_i.
  - Digital is pinned to (a=1, b=0) by the joint solve.
  - A source with no direct digital overlap is recovered via the chain
    of pairwise constraints through intermediate sources.
  - MAD outlier rejection drops a catastrophically-wrong sample at a
    single bucket without dropping the entire source.
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


def test_pairwise_fit_recovers_injected_slope_and_intercept():
    """A source with injected (a=1.25, b=3.0) against digital must be
    solved to (a=1.25, b=3.0). The applied correction brings it back to
    digital's frame exactly.
    """
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 3
    truth = _series("2025-03-01", n=n, base=5.0, amplitude=8.0)
    injected_a, injected_b = 1.25, 3.0

    sources = {
        "digital": truth,
        "dec2024_to_now": _apply_linear(truth, injected_a, injected_b),
    }
    merged, report = reconcile_sources(sources)

    dec_record = next(s for s in report.sources if s.name == "dec2024_to_now")
    assert abs(dec_record.a - injected_a) < 1e-6, (
        f"fit.a = {dec_record.a:.6f}, expected {injected_a}"
    )
    assert abs(dec_record.b - injected_b) < 1e-6, (
        f"fit.b = {dec_record.b:.6f}, expected {injected_b}"
    )
    # With digital's higher effective resolution it wins every overlap bucket.
    # The merged curve must match truth to numerical precision.
    merged_vs_truth = merged.merge(truth, on=DATE_COL, suffixes=("_m", "_t"))
    max_err = float(
        (merged_vs_truth[f"{TILT_COL}_m"] - merged_vs_truth[f"{TILT_COL}_t"]).abs().max()
    )
    assert max_err < 1e-6, f"merged disagrees with truth by up to {max_err}"


def test_pairwise_chain_propagates_through_intermediate_source():
    """Source A and source C have NO direct temporal overlap, but both
    overlap with source B. The pairwise calibration must still recover
    their (a, b) via the chain A↔B and B↔C.

    Concretely: digital covers Jan-Jun 2025; two_day covers only the last
    48 h; dec2024_to_now covers both. The solve should still produce
    sensible corrections for two_day via its overlap with dec2024_to_now.
    """
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 3
    digital_truth = _series("2025-03-01", n=n, base=0.0, amplitude=10.0)
    dec_truth_early = _series("2025-03-01", n=n, base=0.0, amplitude=10.0)
    dec_truth_late = _series("2026-04-20 00:00:00", n=n, base=-5.0, amplitude=6.0)
    two_day_truth = _series("2026-04-20 00:00:00", n=n, base=-5.0, amplitude=6.0)

    sources = {
        "digital": digital_truth,
        # dec2024_to_now spans both epochs — same truth in both regions
        "dec2024_to_now": pd.concat(
            [_apply_linear(dec_truth_early, 1.2, 2.0),
             _apply_linear(dec_truth_late, 1.2, 2.0)],
            ignore_index=True,
        ),
        # two_day overlaps dec2024_to_now but not digital
        "two_day": _apply_linear(two_day_truth, 0.9, -1.5),
    }
    merged, report = reconcile_sources(sources)

    two_day_record = next(s for s in report.sources if s.name == "two_day")
    dec_record = next(s for s in report.sources if s.name == "dec2024_to_now")
    # Chain recovery: both should converge toward their injected values.
    assert abs(dec_record.a - 1.2) < 0.01
    assert abs(dec_record.b - 2.0) < 0.2
    assert abs(two_day_record.a - 0.9) < 0.02, (
        f"two_day.a = {two_day_record.a:.4f}, expected 0.9"
    )
    assert abs(two_day_record.b - (-1.5)) < 0.3, (
        f"two_day.b = {two_day_record.b:.4f}, expected -1.5"
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
    # Every merged value should come from digital, not archive.
    assert abs(merged[TILT_COL].mean() - truth[TILT_COL].mean()) < 1e-6


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
# Phase 4 Commit 1: hard-pin digital + pathological-a reset
# ─────────────────────────────────────────────────────────────────────────────


def test_digital_hard_pin_exact():
    """With many pair constraints, the digital pin MUST come out exactly
    (1.0, 0.0), not approximately.

    Regression guard for the Phase 2 soft-pin bug where the joint lstsq
    treated the pin as one equal-weight row and returned
    `a_digital = 0.9622` in 2026-04 production data.
    """
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 3
    truth = _series("2025-03-01", n=n, base=0.0, amplitude=10.0)
    sources = {
        "digital": truth,
        "dec2024_to_now": _apply_linear(truth, 1.15, 2.0),
        "three_month": _apply_linear(truth, 0.85, -1.5),
        "month": _apply_linear(truth, 1.1, 0.5),
        "week": _apply_linear(truth, 0.92, 0.8),
        "two_day": _apply_linear(truth, 1.05, -2.0),
    }
    _, report = reconcile_sources(sources)
    digital = next(s for s in report.sources if s.name == "digital")
    assert digital.a == 1.0, f"a_digital = {digital.a}, expected exactly 1.0"
    assert digital.b == 0.0, f"b_digital = {digital.b}, expected exactly 0.0"
    assert digital.is_anchor is True


def test_pathological_a_resets_to_identity_with_scalar_b():
    """A pair fit that returns pathological `a` (e.g. because the pair's
    true signal is near-flat) must reset to `(a=1, b=median_offset)`
    instead of applying the bad scaling. Amplifying noise by a factor
    of 2-3 is worse than a constant offset.
    """
    # Build a dataset where `month` disagrees with dec2024_to_now by a
    # HUGE slope — exactly the 2026-04 prod pathology. The reset must
    # fire and we expect the source to land at identity with a scalar
    # offset estimated vs dec2024_to_now.
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 3
    truth = _series("2025-03-01", n=n, base=-10.0, amplitude=8.0)
    pathological_a = 0.3  # well outside PAIRWISE_MAX_A_DEVIATION_FRACTION (0.5)
    pathological_b = 6.0
    sources = {
        "digital": truth,
        "dec2024_to_now": _apply_linear(truth, 1.02, 0.1),
        "month": _apply_linear(truth, pathological_a, pathological_b),
    }
    _, report = reconcile_sources(sources)
    month = next(s for s in report.sources if s.name == "month")
    assert month.a == 1.0, f"a_month should reset to 1.0, got {month.a}"
    assert month.note is not None and "pathological" in month.note
    # b should be finite (not inf/NaN) and representative of the scalar
    # offset the identity-reset mode tries to preserve.
    assert np.isfinite(month.b)


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


def test_huber_pair_fit_robust_to_outlier():
    """Pairwise fits use Huber-robust regression so a small fraction of
    gross-outlier samples doesn't swing the recovered slope.

    OLS on the same data would be dragged several % off the true slope
    by the outliers; Huber should stay within 1-2% of truth.
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
    # True relationship: source = 1.0 · digital + 0.0 (+ outliers).
    # Huber should land near (1.0, 0.0).
    assert abs(fit.alpha - 1.0) < 0.02, (
        f"Huber α={fit.alpha} should be within 2% of 1.0 despite outliers"
    )
    assert abs(fit.beta) < 1.0, (
        f"Huber β={fit.beta} should be near 0 despite outliers"
    )


def test_pathological_reset_uses_dec2024_to_now_when_digital_absent_in_overlap():
    """When the pin is `digital` but the pathological source doesn't
    temporally overlap digital, the reset's scalar offset comes from
    the best available overlapping reference (typically
    `dec2024_to_now`, which covers both digital's epoch and the rolling
    sources').
    """
    n = PAIRWISE_MIN_OVERLAP_BUCKETS * 3
    # Digital covers Jan-Jun 2025.
    digital_truth = _series("2025-03-01", n=n, base=0.0, amplitude=8.0)
    # dec2024_to_now covers BOTH Jan-Jun 2025 AND 2026.
    dec_jan = _apply_linear(digital_truth, 1.0, 0.0)
    dec_2026 = _series("2026-04-01", n=n, base=-15.0, amplitude=4.0)
    dec = pd.concat([dec_jan, dec_2026], ignore_index=True)
    # `month` only covers 2026 (no digital overlap). Inject pathological a.
    month_truth = _series("2026-04-01", n=n, base=-15.0, amplitude=4.0)
    month = _apply_linear(month_truth, 0.2, 5.0)  # very pathological a
    sources = {
        "digital": digital_truth,
        "dec2024_to_now": dec,
        "month": month,
    }
    _, report = reconcile_sources(sources)
    month_rec = next(s for s in report.sources if s.name == "month")
    assert month_rec.a == 1.0
    # b should be roughly median(month - dec) ≈ median(0.2·truth+5 - truth)
    # ≈ 5 + 0.2·median(truth) - median(truth). For the 2026 segment
    # median(truth) ≈ -15, so expected b ≈ 5 + 0.2·(-15) - (-15) = 17.
    # Loose bound — exact value depends on bucket boundaries.
    assert 10 < month_rec.b < 25, f"reset b = {month_rec.b}, expected ~15-20"


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
    assert abs(merged[TILT_COL].mean() - truth[TILT_COL].mean()) < 1e-6


def test_archive_in_source_priority_enumeration():
    """Archive is last in SOURCE_PRIORITY under Phase 2 because it only
    wins buckets no live source covers. This is a rule-enforcement test
    for the Phase 2 config layout.
    """
    assert SOURCE_PRIORITY[-1] == "archive"
    assert SOURCE_PRIORITY[0] == "digital"
