"""Per-source reconciliation via pairwise self-consistency calibration.

All five USGS PNG sources (`two_day`, `week`, `month`, `three_month`,
`dec2024_to_now`) are different time-window renderings of the SAME
underlying tilt signal (station UWD, Az 300°). `digital` is the same
signal in authoritative research-release CSV form for Jan-Jun 2025. They
should agree exactly modulo transcription error (OCR mis-reads of the
y-axis labels, sub-pixel trace ambiguity, gridline bleed).

The v3 alignment rewrite models each source as a linear image of a
single underlying truth:

    y_i(t) = a_i · true(t) + b_i + noise_i(t)

where `a_i` absorbs y-slope calibration error (OCR'd y-axis labels
yielding a slope off by a few %) and `b_i` absorbs y-intercept error
(y-range shifts between USGS re-renders). Under this model, for any
pair (i, j) with temporal overlap we can measure

    α_ij = a_i / a_j
    β_ij = b_i - α_ij · b_j

via ordinary least squares on bucket-aligned paired samples. With
digital pinned to (a=1, b=0) by convention, a single joint least-squares
solve over all pair measurements recovers every other source's
calibration factors. Each source is then corrected pointwise via
    corrected_i(t) = (y_i(t) - b_i) / a_i
and merged by best effective resolution at 15-min granularity.

Why this replaces the previous architecture
-------------------------------------------

The predecessor stack — scalar median offset per source, piecewise
residuals by nearest higher-priority reference, proximity gating,
archive-age demotion — tried to approximate a time-varying bias with a
succession of constants and kept producing visible step discontinuities
at source-handoff boundaries. That's because the actual bias is a
y-slope error (e.g. the dec2024_to_now PNG traces to values satisfying
`digital = 1.38 · dec + 5.3`), and a scalar offset can only hide a
slope error at one tilt level; the error resurfaces everywhere else.

The pairwise-fit model recovers the slope correction directly. It is
order-independent, handles sources that have no direct overlap with
digital (chain-propagated through intermediate sources), and produces a
single self-consistent set of calibrations each run.

Merge policy
------------

For each 15-min bucket, the corrected value from the source with the
smallest effective resolution (a_i · µrad/px) wins. Archive is a pure
gap-filler: it contributes only when no live source has a sample in the
bucket (pre-Dec-2024 historical data). MAD outlier rejection per-bucket
discards any source whose corrected value exceeds 3·σ_MAD from the
unweighted median across sources in that bucket; the discard is recorded
as a transcription-failure candidate for post-hoc investigation.

At K≥2 → K=1 transitions (canonical case: day-90 boundary where
three_month's window ends and only dec2024_to_now extends further back)
a ±6h linear blend between the K≥2 consensus and the K=1 corrected
value prevents a visible step where the set of contributing sources
shrinks abruptly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    ARCHIVE_SOURCE_NAME,
    CONTINUITY_WARNING_THRESHOLD_MICRORAD,
    DIGITAL_SOURCE_NAME,
    EFFECTIVE_RESOLUTION_FALLBACK_MICRORAD_PER_PIXEL,
    K1_HANDOFF_BLEND_HOURS,
    MAD_OUTLIER_SIGMA_FLOOR_MICRORAD,
    MAD_OUTLIER_SIGMA_MULTIPLIER,
    PAIRWISE_MAX_A_DEVIATION_FRACTION,
    PAIRWISE_MAX_B_MICRORAD,
    PAIRWISE_MIN_OVERLAP_BUCKETS,
)
from .model import DATE_COL, TILT_COL

# Bucket size for pairwise OLS fits. Coarser than the 15-min merge bucket
# because per-source sampling is variable and coarser buckets yield
# denser paired measurements with less intra-bucket noise.
ALIGNMENT_BUCKET = "1h"

# Bucket size for the final merged output. One row per 15-min bin, fine
# enough to preserve real samples and coarse enough to absorb column-level
# jitter from re-traced overlapping captures.
MERGE_BUCKET = "15min"


@dataclass
class SourceAlignment:
    """Per-source outcome of pairwise self-consistency calibration."""

    name: str
    rows_in: int = 0
    a: float = 1.0                         # recovered y-slope correction
    b: float = 0.0                         # recovered y-intercept correction
    pairs_used: int = 0                    # pair constraints involving this source
    is_anchor: bool = False                # True for the pinned source (digital)
    note: Optional[str] = None             # human-readable diagnostic
    rows_mad_rejected: int = 0             # dropped by per-bucket MAD outlier gate
    effective_resolution_microrad_per_pixel: float = 0.0

    # Back-compat fields consumed by `ingest.pipeline._serialize_reconcile`
    # and the legacy CLI print loop. These are populated so the existing
    # run-report JSON structure keeps working.
    offset_microrad: Optional[float] = None  # = b for back-compat display
    overlap_buckets: int = 0
    rows_proximity_dropped: int = 0  # always 0 — proximity gate was removed
    piecewise_residuals: dict[str, float] = field(default_factory=dict)


@dataclass
class PairwiseFit:
    """One pairwise OLS fit y_i = α_ij · y_j + β_ij over overlapping buckets."""

    source_i: str
    source_j: str
    alpha: float
    beta: float
    overlap_buckets: int
    residual_std_microrad: float


@dataclass
class TranscriptionFailure:
    """A bucket-source where the MAD outlier gate rejected a value."""

    bucket: pd.Timestamp
    source: str
    value_corrected: float
    bucket_median: float
    delta_microrad: float


@dataclass
class ContinuityViolation:
    """Adjacent merged buckets stepping by more than the warning threshold."""

    bucket_before: pd.Timestamp
    bucket_after: pd.Timestamp
    tilt_before: float
    tilt_after: float
    delta_microrad: float


@dataclass
class ReconcileConflict:
    """Back-compat shim for `_serialize_reconcile`. The new algorithm records
    transcription_failures instead of conflicts; this struct exists so the
    JSON serializer keeps working."""

    bucket: pd.Timestamp
    winning_source: str
    losing_source: str
    winning_tilt: float
    losing_tilt: float
    delta: float


@dataclass
class ReconcileReport:
    rows_out: int = 0
    sources: list[SourceAlignment] = field(default_factory=list)
    pairs: list[PairwiseFit] = field(default_factory=list)
    transcription_failures: list[TranscriptionFailure] = field(default_factory=list)
    continuity_violations: list[ContinuityViolation] = field(default_factory=list)
    conflicts: list[ReconcileConflict] = field(default_factory=list)  # always empty
    warnings: list[str] = field(default_factory=list)


def reconcile_sources(
    sources: dict[str, pd.DataFrame],
    *,
    proximity_minutes: int = 0,  # legacy-signature no-op, preserved so callers don't change
) -> tuple[pd.DataFrame, ReconcileReport]:
    """Merge raw per-source tilt data via pairwise self-consistency
    calibration into a single deterministic tilt history.

    Args:
        sources: dict mapping source name (canonical reconcile name) to a
            raw DataFrame with columns `[Date, Tilt (microradians)]`. The
            archive source (if present) is used ONLY for gap-filling buckets
            where no live source has coverage.
        proximity_minutes: legacy parameter kept for call-site compatibility;
            the new algorithm does not use a proximity gate.

    Returns:
        `(merged_history_df, report)`. The merged frame follows the
        `data/tilt_history.csv` schema. The report carries the per-source
        (a, b) corrections, pairwise fits, MAD outliers, and continuity
        violations for the JSON run report.
    """
    del proximity_minutes  # intentionally unused; kept in signature
    report = ReconcileReport()

    # Normalize every input DataFrame and split off the archive.
    live: dict[str, pd.DataFrame] = {}
    archive_df: Optional[pd.DataFrame] = None
    for name, raw in sources.items():
        if raw is None or len(raw) == 0:
            continue
        df = raw[[DATE_COL, TILT_COL]].copy()
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.dropna().sort_values(DATE_COL).reset_index(drop=True)
        if len(df) == 0:
            continue
        if name == ARCHIVE_SOURCE_NAME:
            archive_df = df
        else:
            live[name] = df

    if not live:
        # Archive-only fallback: write out whatever archive we have.
        if archive_df is not None and len(archive_df) > 0:
            report.rows_out = len(archive_df)
            report.warnings.append(
                "no live sources present; merged history is archive-only"
            )
            return archive_df.sort_values(DATE_COL).reset_index(drop=True), report
        return _empty_history_df(), report

    # 1. Pairwise OLS fits over overlapping buckets.
    pair_fits = _compute_pairwise_fits(live, report)

    # 2. Joint least-squares solve for {a_i, b_i}.
    alignments = _solve_pairwise_calibration(live, pair_fits, report)

    # 3. Apply corrections per source.
    corrected = _apply_ab_corrections(live, alignments)

    # 4. Merge by best effective resolution with MAD outlier rejection.
    merged = _merge_best_resolution(corrected, alignments, report, archive_df)

    # 5. Post-merge continuity audit.
    _audit_continuity(merged, report)

    report.rows_out = len(merged)
    return merged, report


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise OLS fits
# ─────────────────────────────────────────────────────────────────────────────


def _compute_pairwise_fits(
    live: dict[str, pd.DataFrame],
    report: ReconcileReport,
    *,
    bucket_freq: str = ALIGNMENT_BUCKET,
) -> list[PairwiseFit]:
    """Return one `PairwiseFit` per ordered pair (i, j) with
    `≥ PAIRWISE_MIN_OVERLAP_BUCKETS` shared `bucket_freq` buckets.

    Each pair is fit via **Huber-robust** regression (via
    `scipy.optimize.least_squares(loss='huber')`) rather than plain OLS.
    Real data has `σ(residual) ≈ 2-4 µrad` against `σ(x) ≈ 3-8 µrad`,
    which gives OLS slope uncertainty of ±0.15-0.2 even at 300+ samples
    — the joint solve then compounds noise across 11 pairs into the
    pathological `a` values observed in 2026-04 production. Huber
    down-weights tail residuals so the recovered slope reflects the
    bulk of the signal, not the outliers.

    Mirrors `calibrate.recalibrate_by_anchor_fit`, which already uses
    Huber successfully for the source-vs-digital anchor fit.

    Only ordered pairs where `i < j` lexically are computed — the
    constraint `a_j = α_ji · a_i` is redundant with `a_i = α_ij · a_j`
    (`α_ji = 1/α_ij`), so one per unordered pair is sufficient.
    """
    try:
        from scipy.optimize import least_squares  # type: ignore
    except ImportError:  # pragma: no cover — scipy is pinned
        least_squares = None

    names = sorted(live.keys())
    buckets_per_source: dict[str, pd.Series] = {}
    for name in names:
        df = live[name]
        buckets_per_source[name] = (
            df.assign(_b=lambda d: d[DATE_COL].dt.floor(bucket_freq))
            .groupby("_b")[TILT_COL]
            .mean()
        )

    fits: list[PairwiseFit] = []
    for i, name_i in enumerate(names):
        for name_j in names[i + 1:]:
            bi = buckets_per_source[name_i]
            bj = buckets_per_source[name_j]
            overlap = bi.index.intersection(bj.index)
            if len(overlap) < PAIRWISE_MIN_OVERLAP_BUCKETS:
                continue
            x = bj.loc[overlap].to_numpy(dtype=float)
            y = bi.loc[overlap].to_numpy(dtype=float)

            if least_squares is not None:
                # Huber-robust: residuals > f_scale µrad are down-weighted
                # to L1, smaller residuals stay L2. f_scale = 1.0 is
                # conservative — most real inter-source disagreement is
                # under 1 µrad, outliers are several µrad.
                def _resid(p, x=x, y=y):
                    return y - (p[0] * x + p[1])
                fit_result = least_squares(
                    _resid,
                    x0=np.array([1.0, 0.0]),
                    loss="huber",
                    f_scale=1.0,
                    max_nfev=200,
                )
                alpha = float(fit_result.x[0])
                beta = float(fit_result.x[1])
            else:  # scipy unavailable: OLS fallback
                A = np.vstack([x, np.ones_like(x)]).T
                solution, *_ = np.linalg.lstsq(A, y, rcond=None)
                alpha, beta = float(solution[0]), float(solution[1])

            residuals = y - (alpha * x + beta)
            residual_std = float(np.std(residuals))
            fit = PairwiseFit(
                source_i=name_i,
                source_j=name_j,
                alpha=alpha,
                beta=beta,
                overlap_buckets=len(overlap),
                residual_std_microrad=residual_std,
            )
            fits.append(fit)
            report.pairs.append(fit)
    return fits


# ─────────────────────────────────────────────────────────────────────────────
# Joint solve for {a_i, b_i}
# ─────────────────────────────────────────────────────────────────────────────


def _solve_pairwise_calibration(
    live: dict[str, pd.DataFrame],
    fits: list[PairwiseFit],
    report: ReconcileReport,
) -> dict[str, SourceAlignment]:
    """Solve two decoupled linear systems for {a_i} and {b_i} jointly.

    The anchor source (`digital` if present, else the first source
    alphabetically) is HARD-pinned at `a=1, b=0` by eliminating its
    variables from the system entirely, not by adding a soft pin row.
    A soft pin is just one equal-weight row in the lstsq, and with many
    pair constraints it gets out-voted — in 2026-04 production data the
    pin solved to `a_digital = 0.9622` instead of 1, propagating a 4%
    scale error to every downstream source.

    Substituted pair constraints (for α_ij from `y_i = α_ij · y_j + β_ij`):
      - both sources pinned → constraint involves no unknowns; skip.
      - only `i` pinned      → `a_j = 1/α_ij`, `b_j = -β_ij/α_ij`.
      - only `j` pinned      → `a_i = α_ij`,   `b_i = β_ij`.
      - neither pinned       → `a_i - α_ij·a_j = 0`,
                               `b_i - α_ij·b_j = β_ij`.

    After the solve, any source whose recovered `|a-1|` exceeds
    PAIRWISE_MAX_A_DEVIATION_FRACTION is flagged AND reset to identity
    (a=1, b=median-offset-vs-anchor) — applying a pathological slope
    correction is worse than applying none, because the division
    amplifies the underlying transcription noise by 2-3×.
    """
    names = sorted(live.keys())
    n = len(names)
    pin_name = DIGITAL_SOURCE_NAME if DIGITAL_SOURCE_NAME in names else names[0]

    # Non-pinned variables are the unknowns we solve for.
    unknowns = [n_ for n_ in names if n_ != pin_name]
    uidx = {name: i for i, name in enumerate(unknowns)}

    pair_counts: dict[str, int] = {name: 0 for name in names}
    for f in fits:
        pair_counts[f.source_i] += 1
        pair_counts[f.source_j] += 1

    if len(unknowns) == 0:
        # Only pin is present — nothing to solve.
        alignments: dict[str, SourceAlignment] = {
            pin_name: SourceAlignment(
                name=pin_name,
                rows_in=len(live[pin_name]),
                a=1.0,
                b=0.0,
                pairs_used=pair_counts[pin_name],
                is_anchor=True,
                effective_resolution_microrad_per_pixel=(
                    EFFECTIVE_RESOLUTION_FALLBACK_MICRORAD_PER_PIXEL.get(pin_name, 1.0)
                ),
                offset_microrad=0.0,
                overlap_buckets=pair_counts[pin_name],
            )
        }
        report.sources.append(alignments[pin_name])
        return alignments

    # Build both systems with pin variables substituted out.
    a_rows: list[list[float]] = []
    a_rhs: list[float] = []
    b_rows: list[list[float]] = []
    b_rhs: list[float] = []
    m = len(unknowns)
    for f in fits:
        i_pin = (f.source_i == pin_name)
        j_pin = (f.source_j == pin_name)
        if i_pin and j_pin:
            continue  # trivial
        row_a = [0.0] * m
        row_b = [0.0] * m
        if not i_pin and not j_pin:
            # a_i - α·a_j = 0,  b_i - α·b_j = β
            row_a[uidx[f.source_i]] = 1.0
            row_a[uidx[f.source_j]] = -f.alpha
            row_b[uidx[f.source_i]] = 1.0
            row_b[uidx[f.source_j]] = -f.alpha
            a_rhs.append(0.0)
            b_rhs.append(f.beta)
        elif j_pin:
            # a_i = α,  b_i = β
            row_a[uidx[f.source_i]] = 1.0
            row_b[uidx[f.source_i]] = 1.0
            a_rhs.append(f.alpha)
            b_rhs.append(f.beta)
        else:  # i_pin
            # a_j = 1/α,  b_j = -β/α  (assuming α is non-zero; it is by
            # construction since α comes from a non-trivial linear fit)
            if abs(f.alpha) < 1e-9:
                continue
            row_a[uidx[f.source_j]] = 1.0
            row_b[uidx[f.source_j]] = 1.0
            a_rhs.append(1.0 / f.alpha)
            b_rhs.append(-f.beta / f.alpha)
        a_rows.append(row_a)
        b_rows.append(row_b)

    if len(a_rows) == 0:
        # No pair constraints left unknowns — everyone defaults to identity.
        a_vec = np.ones(m)
        b_vec = np.zeros(m)
    else:
        a_vec, *_ = np.linalg.lstsq(
            np.array(a_rows), np.array(a_rhs), rcond=None
        )
        b_vec, *_ = np.linalg.lstsq(
            np.array(b_rows), np.array(b_rhs), rcond=None
        )

    alignments: dict[str, SourceAlignment] = {}

    # Pin record: exact identity, no solver noise.
    alignments[pin_name] = SourceAlignment(
        name=pin_name,
        rows_in=len(live[pin_name]),
        a=1.0,
        b=0.0,
        pairs_used=pair_counts[pin_name],
        is_anchor=True,
        effective_resolution_microrad_per_pixel=(
            EFFECTIVE_RESOLUTION_FALLBACK_MICRORAD_PER_PIXEL.get(pin_name, 1.0)
        ),
        offset_microrad=0.0,
        overlap_buckets=pair_counts[pin_name],
    )
    report.sources.append(alignments[pin_name])

    for name in unknowns:
        a = float(a_vec[uidx[name]])
        b = float(b_vec[uidx[name]])
        record = SourceAlignment(
            name=name,
            rows_in=len(live[name]),
            a=a,
            b=b,
            pairs_used=pair_counts[name],
            is_anchor=False,
            effective_resolution_microrad_per_pixel=(
                abs(a) * EFFECTIVE_RESOLUTION_FALLBACK_MICRORAD_PER_PIXEL.get(name, 1.0)
            ),
            offset_microrad=b,
            overlap_buckets=pair_counts[name],  # proxy for the old field
        )

        # If the recovered slope is pathological, applying `(y - b)/a`
        # amplifies transcription noise. Reset to a=1 and estimate b
        # separately via median offset vs the best available reference.
        # Prefer anchor-corrected `dec2024_to_now` (which always overlaps
        # the rolling sources temporally) over `digital` (which covers
        # only Jan-Jun 2025 and rarely overlaps rolling-source windows).
        if abs(a - 1.0) > PAIRWISE_MAX_A_DEVIATION_FRACTION:
            reference = (
                "dec2024_to_now"
                if "dec2024_to_now" in live
                and name != "dec2024_to_now"
                and alignments.get("dec2024_to_now") is not None
                and alignments["dec2024_to_now"].note is None
                else pin_name
            )
            reset_b = _estimate_scalar_offset(live, name, reference)
            msg = (
                f"{name}: pairwise fit is pathological — "
                f"a={a:.4f} (|a-1|={abs(a-1)*100:.1f}% > "
                f"{PAIRWISE_MAX_A_DEVIATION_FRACTION*100:.0f}%); "
                f"reset to a=1.0, b={reset_b:+.2f} µrad"
            )
            record.a = 1.0
            record.b = reset_b
            record.offset_microrad = reset_b
            record.effective_resolution_microrad_per_pixel = (
                EFFECTIVE_RESOLUTION_FALLBACK_MICRORAD_PER_PIXEL.get(name, 1.0)
            )
            record.note = msg
            report.warnings.append(msg)
        elif abs(b) > PAIRWISE_MAX_B_MICRORAD:
            msg = (
                f"{name}: pairwise fit produces large intercept — "
                f"b={b:+.2f} µrad (> {PAIRWISE_MAX_B_MICRORAD} µrad)"
            )
            record.note = msg
            report.warnings.append(msg)

        alignments[name] = record
        report.sources.append(record)
    return alignments


def _estimate_scalar_offset(
    live: dict[str, pd.DataFrame],
    source_name: str,
    reference_name: str,
    *,
    bucket_freq: str = ALIGNMENT_BUCKET,
) -> float:
    """Median bucket-aligned offset of `source` minus `reference`.

    Called when a source's pairwise slope is pathological and we fall
    back to a=1 + scalar-b correction. Returns 0.0 if there's no
    overlap, which defers alignment to downstream handoff logic rather
    than pick a value out of thin air.
    """
    if source_name not in live or reference_name not in live:
        return 0.0
    src = live[source_name]
    ref = live[reference_name]
    src_b = src.assign(_b=src[DATE_COL].dt.floor(bucket_freq)).groupby("_b")[TILT_COL].mean()
    ref_b = ref.assign(_b=ref[DATE_COL].dt.floor(bucket_freq)).groupby("_b")[TILT_COL].mean()
    overlap = src_b.index.intersection(ref_b.index)
    if len(overlap) == 0:
        return 0.0
    return float(np.median(src_b.loc[overlap] - ref_b.loc[overlap]))


# ─────────────────────────────────────────────────────────────────────────────
# Apply corrections
# ─────────────────────────────────────────────────────────────────────────────


def _apply_ab_corrections(
    live: dict[str, pd.DataFrame],
    alignments: dict[str, SourceAlignment],
) -> dict[str, pd.DataFrame]:
    """Return each source's DataFrame with `tilt ← (tilt - b) / a` applied.

    The correction inverts the model `y_i = a_i · true + b_i` to recover
    `true`. A near-zero `a_i` would indicate a pathological solve; clamp
    to 1.0 in that case to avoid div-by-zero and let the continuity check
    surface the problem downstream.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, df in live.items():
        align = alignments.get(name)
        if align is None:
            out[name] = df.copy()
            continue
        a = align.a if abs(align.a) > 1e-9 else 1.0
        corrected = df.copy()
        corrected[TILT_COL] = (corrected[TILT_COL] - align.b) / a
        out[name] = corrected
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Merge by best effective resolution
# ─────────────────────────────────────────────────────────────────────────────


def _merge_best_resolution(
    corrected: dict[str, pd.DataFrame],
    alignments: dict[str, SourceAlignment],
    report: ReconcileReport,
    archive_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Walk every 15-min bucket in the union of corrected sources and emit
    one row per bucket, chosen by best-effective-resolution among sources
    passing the MAD outlier gate.

    Algorithm (per bucket):
      1. Collect every live source's corrected value for the bucket.
      2. Compute unweighted median `m` and MAD `σ = 1.4826 · median(|v-m|)`.
      3. Drop any source where `|v - m| > max(FLOOR, MULTIPLIER · σ)`;
         record each drop as a TranscriptionFailure.
      4. Among survivors, pick the one with the smallest effective
         resolution (= |a_i| · µrad/px for source i).
      5. If no live source survives, fall back to archive (gap-fill).

    K≥2 → K=1 handoff blending is applied as a post-pass: within
    ±K1_HANDOFF_BLEND_HOURS of a transition where the set of sources
    shrinks to one, blend the K=1 source's value toward the last-K≥2
    consensus so the curve is continuous.
    """
    resolutions = {
        name: align.effective_resolution_microrad_per_pixel
        for name, align in alignments.items()
    }

    # Tag every corrected row with its source + merge bucket + value.
    tagged: list[pd.DataFrame] = []
    for name, df in corrected.items():
        d = df[[DATE_COL, TILT_COL]].copy()
        d["_source"] = name
        d["_bucket"] = d[DATE_COL].dt.round(MERGE_BUCKET)
        d["_res"] = resolutions.get(name, 1.0)
        tagged.append(d)

    combined = pd.concat(tagged, ignore_index=True)
    combined = combined.dropna(subset=[TILT_COL, "_bucket"])
    if len(combined) == 0:
        if archive_df is not None and len(archive_df) > 0:
            return archive_df.sort_values(DATE_COL).reset_index(drop=True)
        return _empty_history_df()

    # Aggregate per bucket: keep one row per (bucket, source) — the
    # latest-dated one if multiple rows of the same source fall in the
    # same 15-min bucket.
    combined = combined.sort_values([DATE_COL]).drop_duplicates(
        subset=["_bucket", "_source"], keep="last"
    )

    emitted_rows: list[tuple[pd.Timestamp, float, str]] = []
    for bucket, group in combined.groupby("_bucket", sort=True):
        values = group[TILT_COL].to_numpy(dtype=float)
        sources = group["_source"].to_numpy(dtype=str)
        resolutions_here = group["_res"].to_numpy(dtype=float)
        dates = group[DATE_COL].to_numpy()

        if len(values) == 0:
            continue

        # ── MAD outlier gate ────────────────────────────────────────────────
        if len(values) >= 2:
            m = float(np.median(values))
            mad = float(np.median(np.abs(values - m)))
            sigma = 1.4826 * mad
            threshold = max(
                MAD_OUTLIER_SIGMA_FLOOR_MICRORAD,
                MAD_OUTLIER_SIGMA_MULTIPLIER * sigma,
            )
            keep_mask = np.abs(values - m) <= threshold
            if not keep_mask.all():
                for v, s in zip(values[~keep_mask], sources[~keep_mask]):
                    report.transcription_failures.append(
                        TranscriptionFailure(
                            bucket=bucket,
                            source=str(s),
                            value_corrected=float(v),
                            bucket_median=m,
                            delta_microrad=float(v) - m,
                        )
                    )
                    # count the reject on the source's alignment record
                    align = alignments.get(str(s))
                    if align is not None:
                        align.rows_mad_rejected += 1
                values = values[keep_mask]
                sources = sources[keep_mask]
                resolutions_here = resolutions_here[keep_mask]
                dates = dates[keep_mask]

        if len(values) == 0:
            continue

        # ── Best-effective-resolution wins ──────────────────────────────────
        winner_idx = int(np.argmin(resolutions_here))
        emitted_rows.append(
            (pd.Timestamp(dates[winner_idx]), float(values[winner_idx]), str(sources[winner_idx]))
        )

    merged = pd.DataFrame(
        emitted_rows, columns=[DATE_COL, TILT_COL, "_source"]
    )

    # ── K=1 handoff blending ────────────────────────────────────────────────
    if len(merged) >= 2:
        merged = _blend_k1_handoffs(merged, combined)

    # ── Archive gap-fill for K=0 buckets ────────────────────────────────────
    if archive_df is not None and len(archive_df) > 0:
        merged = _fill_archive_gaps(merged, archive_df)

    merged = merged[[DATE_COL, TILT_COL]].sort_values(DATE_COL).reset_index(drop=True)
    return merged


def _blend_k1_handoffs(
    merged: pd.DataFrame, combined: pd.DataFrame
) -> pd.DataFrame:
    """Taper the merge across K≥2 → K=1 transitions.

    Identifies 15-min buckets where the count of source contributors
    drops from ≥2 to 1 (or rises back) and linearly interpolates the
    merged value across ±K1_HANDOFF_BLEND_HOURS so no step appears at
    the boundary.

    Why this matters: even after pairwise calibration, each source's
    corrected values carry residual transcription noise (typically 1-3
    µrad). Inside a region where ≥2 sources contribute, the best-
    effective-resolution winner may sit a µrad off from another source's
    value; when that other source winks out at a window boundary and the
    winner is suddenly the only one, whatever residual offset it carried
    becomes a visible step — unless we smooth across the transition.
    """
    # Count sources per bucket from the combined-table view.
    counts = (
        combined.groupby("_bucket")["_source"].nunique().rename("_k")
    )
    merged_with = merged.assign(
        _bucket=lambda d: d[DATE_COL].dt.round(MERGE_BUCKET)
    ).merge(counts, left_on="_bucket", right_index=True, how="left")
    merged_with["_k"] = merged_with["_k"].fillna(1).astype(int)

    if not (merged_with["_k"] >= 2).any() or not (merged_with["_k"] == 1).any():
        return merged  # nothing to blend

    # Find transitions: indices where _k crosses between ≥2 and 1.
    k = merged_with["_k"].to_numpy()
    transitions = []
    for i in range(1, len(k)):
        if (k[i - 1] >= 2) != (k[i] >= 2):
            transitions.append(i)

    if not transitions:
        return merged

    blend_delta = pd.Timedelta(hours=K1_HANDOFF_BLEND_HOURS)
    tilts = merged_with[TILT_COL].to_numpy(dtype=float).copy()
    dates = merged_with[DATE_COL].to_numpy()
    # At each transition, smooth the values within ±blend_delta so the
    # curve tapers from the last "high-K" value into the K=1 regime.
    for ti in transitions:
        t_boundary = pd.Timestamp(dates[ti])
        t_start = t_boundary - blend_delta
        t_end = t_boundary + blend_delta
        # Find the last high-K value before ti and the first K=1 value
        # after (or vice versa).
        left_k2_idx = ti - 1
        right_k1_idx = ti
        if k[left_k2_idx] < 2 and k[right_k1_idx] < 2:
            continue  # not a 2→1 boundary
        if k[left_k2_idx] >= 2 and k[right_k1_idx] >= 2:
            continue
        high_side, low_side = (
            (left_k2_idx, right_k1_idx)
            if k[left_k2_idx] >= 2
            else (right_k1_idx, left_k2_idx)
        )
        step = float(tilts[low_side] - tilts[high_side])
        if abs(step) <= CONTINUITY_WARNING_THRESHOLD_MICRORAD:
            continue  # step is small; no blend needed
        # Distribute the step correction linearly across the blend zone on
        # the low-side. The high-K side is treated as the reference.
        window_mask = (pd.to_datetime(dates) >= t_start) & (
            pd.to_datetime(dates) <= t_end
        )
        window_idx = np.where(window_mask)[0]
        if len(window_idx) < 2:
            continue
        for idx in window_idx:
            dist_hours = abs(
                (pd.Timestamp(dates[idx]) - t_boundary).total_seconds() / 3600.0
            )
            # Weight ramps from 1.0 at the boundary to 0 at blend edge.
            w = max(0.0, 1.0 - dist_hours / K1_HANDOFF_BLEND_HOURS)
            # Only adjust the low-K side (where the spurious step lives).
            if k[idx] < 2:
                tilts[idx] -= step * w * 0.5
            else:
                tilts[idx] += step * w * 0.5

    merged_with[TILT_COL] = tilts
    return merged_with[[DATE_COL, TILT_COL, "_source"]].copy()


def _fill_archive_gaps(
    merged: pd.DataFrame, archive_df: pd.DataFrame
) -> pd.DataFrame:
    """Append archive rows whose 15-min buckets are not already in `merged`.

    The archive is a pure gap-filler in Phase 2 — it contributes only for
    buckets no live source covers. Typical use: pre-Dec-2024 history
    (older than dec2024_to_now's window).
    """
    live_buckets = set(
        pd.to_datetime(merged[DATE_COL]).dt.round(MERGE_BUCKET)
    )
    arch = archive_df[[DATE_COL, TILT_COL]].copy()
    arch[DATE_COL] = pd.to_datetime(arch[DATE_COL])
    arch["_bucket"] = arch[DATE_COL].dt.round(MERGE_BUCKET)
    # Keep one archive row per bucket (latest).
    arch = arch.sort_values(DATE_COL).drop_duplicates(subset="_bucket", keep="last")
    fill = arch[~arch["_bucket"].isin(live_buckets)][[DATE_COL, TILT_COL]].copy()
    if len(fill) == 0:
        return merged
    fill["_source"] = ARCHIVE_SOURCE_NAME
    return pd.concat([merged, fill], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Continuity audit
# ─────────────────────────────────────────────────────────────────────────────


def _audit_continuity(merged: pd.DataFrame, report: ReconcileReport) -> None:
    """Flag adjacent-bucket steps exceeding the warning threshold.

    Not a hard fail — during real eruption transitions the signal can step
    by 5+ µrad across a 15-min bucket — but a surge in violations is an
    actionable regression signal. The list is committed to the per-run
    JSON for post-hoc investigation.
    """
    if len(merged) < 2:
        return
    df = merged.sort_values(DATE_COL).reset_index(drop=True)
    deltas = df[TILT_COL].diff().abs()
    violations = deltas > CONTINUITY_WARNING_THRESHOLD_MICRORAD
    for i in np.where(violations.to_numpy())[0]:
        report.continuity_violations.append(
            ContinuityViolation(
                bucket_before=pd.Timestamp(df[DATE_COL].iloc[int(i) - 1]),
                bucket_after=pd.Timestamp(df[DATE_COL].iloc[int(i)]),
                tilt_before=float(df[TILT_COL].iloc[int(i) - 1]),
                tilt_after=float(df[TILT_COL].iloc[int(i)]),
                delta_microrad=float(deltas.iloc[int(i)]),
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            DATE_COL: pd.Series(dtype="datetime64[ns]"),
            TILT_COL: pd.Series(dtype="float64"),
        }
    )
