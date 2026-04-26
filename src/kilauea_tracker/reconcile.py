"""Per-source reconciliation via pairwise scalar-offset alignment.

All five USGS PNG sources (`two_day`, `week`, `month`, `three_month`,
`dec2024_to_now`) are different time-window renderings of the SAME
underlying tilt signal (station UWD, Az 300°). `digital` is the same
signal in authoritative research-release CSV form for Jan-Jun 2025.

The ingest pipeline's per-source y-axis calibration (`calibrate.py`)
already converts pixels → µrad using OCR'd tick labels, so each source
arrives at reconcile in the correct µrad scale. What remains is a
per-source SCALAR OFFSET: USGS re-baselines the y-axis between renders,
so the same sample can appear at y=-25 µrad on today's `week` plot and
y=-22 µrad on the `month` plot of the same moment. That's a pure shift,
not a scale difference.

Model:

    y_i(t) = true(t) + b_i + noise_i(t)

`b_i` is the one free parameter per source. Digital is pinned at b=0.
For each pair (i, j) with temporal overlap we measure

    β_ij = median(y_i - y_j)    over bucket-aligned samples

(median, not OLS, because the distribution of inter-source deltas is
heavy-tailed — OCR glitches and trace drop-outs produce ±20+ µrad
outliers that would drag a mean-based estimator off the signal). The
`{b_i}` are then recovered by a single linear-least-squares solve over
the system `b_i - b_j = β_ij`, with `b_digital = 0` substituted in.

Why this replaces the v3 slope+intercept model
----------------------------------------------

The v3 model (pre-2026-04-22) added a per-source slope `a_i` so pair
fits were OLS lines, not medians. The intent was to absorb any residual
y-scale miscalibration that survived ingest's OCR pass. In practice the
slope recovery was noise-dominated: with pair `σ(residual) ≈ 2-4 µrad`
against `σ(x) ≈ 3-8 µrad`, OLS slope uncertainty was ±0.15-0.2 even at
300+ samples. The joint solve then compounded that noise across multiple
pairs and routinely produced `a_i ≈ 0` for disconnected sources (the
min-norm trivial solution to homogeneous `a_i - α·a_j = 0` rows), which
triggered a pathological-reset cascade that left every rolling source at
b = offset-vs-miscalibrated-dec2024_to_now — i.e. wildly wrong.

Dropping `a_i` removes two failure modes:
  - `(y - b)/a` noise amplification when `a` solves to something far
    from 1.
  - Min-norm collapse of disconnected subgraphs (homogeneous equations
    have no natural anchor to 1; offset equations don't).

If a source has a genuine y-scale miscalibration surviving ingest (e.g.
the 2026-04 `dec2024_to_now` plot traces to values compressing the real
~40 µrad span into ~29 µrad), a scalar offset can't fully correct it —
but neither could the v3 model reliably, and digital still wins every
Jan-Jun 2025 bucket by effective-resolution priority, so the damage is
bounded to the pre-Jan-2025 / post-Jun-2025 windows where there's no
ground truth to disagree with anyway. Surface the disagreement in the
diagnostics panel; fix the trace where possible; don't let the reconcile
layer AMPLIFY it.

Merge policy
------------

For each 15-min bucket, the corrected value from the source with the
smallest effective resolution (µrad/px) wins. Archive is a pure
gap-filler: it contributes only when no live source has a sample in the
bucket (pre-Dec-2024 historical data). MAD outlier rejection per-bucket
is diagnostic-only: it logs outliers as `TranscriptionFailure`s but does
NOT remove them from winner selection, because per-source deterministic
winner preserves continuity and upstream rolling-median filtering
(`trace._filter_rolling_median_outliers`) catches OCR glitches before
they reach reconcile.

At K≥2 → K=1 transitions (canonical case: day-90 boundary where
three_month's window ends and only dec2024_to_now extends further back)
a ±6h linear blend between the K≥2 consensus and the K=1 corrected
value prevents a visible step where the set of contributing sources
shrinks abruptly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

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
    PAIRWISE_MAX_B_MICRORAD,
    PAIRWISE_MIN_OVERLAP_BUCKETS,
)
from .model import DATE_COL, TILT_COL

logger = logging.getLogger(__name__)

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
    """Per-source outcome of pairwise scalar-offset alignment."""

    name: str
    rows_in: int = 0
    b: float = 0.0                         # recovered scalar offset
    pairs_used: int = 0                    # pair constraints involving this source
    is_anchor: bool = False                # True for the pinned source (digital)
    note: str | None = None             # human-readable diagnostic
    rows_mad_rejected: int = 0             # dropped by per-bucket MAD outlier gate
    effective_resolution_microrad_per_pixel: float = 0.0
    offset_microrad: float | None = None  # = b, surfaced in diagnostics panel
    overlap_buckets: int = 0
    piecewise_residuals: dict[str, float] = field(default_factory=dict)


@dataclass
class PairwiseFit:
    """One pairwise median-offset measurement y_i - y_j = β_ij over
    overlapping buckets."""

    source_i: str
    source_j: str
    beta: float                             # median(y_i - y_j) over overlap
    overlap_buckets: int
    residual_std_microrad: float            # std(y_i - y_j - β_ij)


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
class ReconcileReport:
    rows_out: int = 0
    sources: list[SourceAlignment] = field(default_factory=list)
    pairs: list[PairwiseFit] = field(default_factory=list)
    transcription_failures: list[TranscriptionFailure] = field(default_factory=list)
    continuity_violations: list[ContinuityViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Per-source count of 15-min buckets that source won at merge time.
    # Populated by `_merge_best_resolution`. Used by the Streamlit
    # diagnostics panel and surfaced in the JSON run report.
    winner_counts: dict[str, int] = field(default_factory=dict)


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
    archive_df: pd.DataFrame | None = None
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
            logger.warning("reconcile: no live sources present; merged history is archive-only")
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

    Each pair's scalar offset is estimated as `β_ij = median(y_i - y_j)`
    across bucket-aligned samples. Median rather than mean because the
    distribution of inter-source deltas is heavy-tailed — OCR glitches
    and trace drop-outs produce 20+ µrad outliers that would drag a
    mean off the signal. `residual_std_microrad` is the post-offset
    std, useful as a quality gauge: a pair with `residual_std > 4 µrad`
    is a strong hint that one of the two sources has a real y-scale
    problem that a scalar can't fix, and the diagnostics panel surfaces
    it so the user can intervene upstream.

    Only ordered pairs where `i < j` lexically are computed —
    `β_ji = -β_ij`, so one per unordered pair is sufficient.
    """
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
            deltas = (
                bi.loc[overlap].to_numpy(dtype=float)
                - bj.loc[overlap].to_numpy(dtype=float)
            )
            beta = float(np.median(deltas))
            residual_std = float(np.std(deltas - beta))
            fit = PairwiseFit(
                source_i=name_i,
                source_j=name_j,
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
    """Solve the linear system `b_i - b_j = β_ij` for `{b_i}`.

    The anchor source (`digital` if present, else the first source
    alphabetically) is HARD-pinned at `b=0` by eliminating its variable
    from the system — not by adding a soft pin row that the lstsq could
    out-vote.

    Sources in a connected component of the pair graph that contains
    the pin are resolved by the joint solve. Sources in a component
    disconnected from the pin default to `b=0` with a warning: a
    disconnected component's equations are internally-consistent but
    have no external anchor, so the min-norm lstsq solution picks zero
    for every variable, which by coincidence is the right default (no
    offset) — we just need to know it was an uncalibrated default and
    not a measurement. In the happy path with
    PAIRWISE_MIN_OVERLAP_BUCKETS = 20 and the usual 5-source coverage,
    every source reaches digital through ≤ 2 hops via dec2024_to_now,
    so the disconnected case is a diagnostic, not a regular occurrence.
    """
    names = sorted(live.keys())
    pin_name = DIGITAL_SOURCE_NAME if DIGITAL_SOURCE_NAME in names else names[0]

    unknowns = [n_ for n_ in names if n_ != pin_name]
    uidx = {name: i for i, name in enumerate(unknowns)}
    m = len(unknowns)

    pair_counts: dict[str, int] = dict.fromkeys(names, 0)
    for f in fits:
        pair_counts[f.source_i] += 1
        pair_counts[f.source_j] += 1

    # ── Graph-connectedness check (BFS from pin) ────────────────────────────
    # Build an adjacency list across the fit graph, then walk from the
    # pin. Sources not reached are in a disconnected component and
    # won't be resolvable by the lstsq.
    adj: dict[str, set[str]] = {name: set() for name in names}
    for f in fits:
        adj[f.source_i].add(f.source_j)
        adj[f.source_j].add(f.source_i)

    connected_to_pin: set[str] = {pin_name}
    queue = [pin_name]
    while queue:
        current = queue.pop()
        for neighbor in adj[current]:
            if neighbor not in connected_to_pin:
                connected_to_pin.add(neighbor)
                queue.append(neighbor)

    alignments: dict[str, SourceAlignment] = {}

    # Pin record: exact identity.
    alignments[pin_name] = SourceAlignment(
        name=pin_name,
        rows_in=len(live[pin_name]),
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

    if m == 0:
        return alignments

    # ── Build and solve the lstsq for connected unknowns ─────────────────────
    # One row per fit: `b_i - b_j = β_ij`, with the pin's coefficient
    # dropped when it appears (since b_pin = 0).
    rows: list[list[float]] = []
    rhs: list[float] = []
    for f in fits:
        # Skip fits between two non-connected unknowns — their constraint
        # exists internally in a disconnected component, but without an
        # anchor the whole component underdetermines itself. Including the
        # constraint would bias connected-side variables.
        if (
            f.source_i not in connected_to_pin
            or f.source_j not in connected_to_pin
        ):
            continue
        i_pin = (f.source_i == pin_name)
        j_pin = (f.source_j == pin_name)
        if i_pin and j_pin:
            continue
        row = [0.0] * m
        if not i_pin:
            row[uidx[f.source_i]] = 1.0
        if not j_pin:
            row[uidx[f.source_j]] = -1.0
        rows.append(row)
        rhs.append(f.beta)

    if rows:
        b_vec, *_ = np.linalg.lstsq(
            np.array(rows), np.array(rhs), rcond=None
        )
    else:
        b_vec = np.zeros(m)

    # ── Emit alignment records ───────────────────────────────────────────────
    for name in unknowns:
        if name in connected_to_pin:
            b = float(b_vec[uidx[name]])
            note: str | None = None
        else:
            # Disconnected from pin: no evidence to set b. Default to 0
            # and let the effective-resolution merge-winner pick it up
            # only when it's the sole source covering a bucket.
            b = 0.0
            note = (
                f"{name}: no path to anchor ({pin_name}) in pair graph "
                f"(pairs_used={pair_counts[name]}); scalar offset "
                f"defaulted to 0 — merged values in this source's "
                f"window may carry a residual offset"
            )
            report.warnings.append(note)
            logger.warning("reconcile disconnected: %s", note)

        record = SourceAlignment(
            name=name,
            rows_in=len(live[name]),
            b=b,
            pairs_used=pair_counts[name],
            is_anchor=False,
            effective_resolution_microrad_per_pixel=(
                EFFECTIVE_RESOLUTION_FALLBACK_MICRORAD_PER_PIXEL.get(name, 1.0)
            ),
            offset_microrad=b,
            overlap_buckets=pair_counts[name],
            note=note,
        )

        if abs(b) > PAIRWISE_MAX_B_MICRORAD and note is None:
            msg = (
                f"{name}: large scalar offset — "
                f"b={b:+.2f} µrad (> {PAIRWISE_MAX_B_MICRORAD} µrad). "
                f"USGS may have rebaselined this source's y-axis, or "
                f"its trace has drifted from the common frame"
            )
            record.note = msg
            report.warnings.append(msg)
            logger.warning("reconcile large-offset: %s", msg)

        alignments[name] = record
        report.sources.append(record)
    return alignments


# ─────────────────────────────────────────────────────────────────────────────
# Apply corrections
# ─────────────────────────────────────────────────────────────────────────────


# Cap on how far to interpolate between samples of the same source.
# Beyond this, the stretch is left as a gap so another source (or
# archive gap-fill) can take over instead of fabricating a straight
# line across a multi-hour fetch failure. 3×stride is wide enough to
# absorb normal jitter and the odd single-bucket drop-out but narrow
# enough that a half-day gap remains visible as a gap.
_DENSIFY_GAP_STRIDE_MULTIPLE = 3.0


def _densify_to_merge_grid(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Interpolate a source's samples onto a regular `freq` grid.

    Bounded to `[df.min, df.max]` — no extrapolation past the source's
    own window. Gaps wider than `_DENSIFY_GAP_STRIDE_MULTIPLE` × the
    source's median inter-sample stride are left as NaN so downstream
    merge policy (best-effective-resolution / archive gap-fill) can
    pick a different source rather than fabricate data across a fetch
    failure.
    """
    if df is None or len(df) == 0:
        return df.copy() if df is not None else pd.DataFrame(
            columns=[DATE_COL, TILT_COL]
        )

    d = df[[DATE_COL, TILT_COL]].copy()
    d[DATE_COL] = pd.to_datetime(d[DATE_COL])
    d = d.dropna().sort_values(DATE_COL).drop_duplicates(
        subset=DATE_COL, keep="last"
    )
    if len(d) < 2:
        return d

    strides = d[DATE_COL].diff().dropna().dt.total_seconds()
    if len(strides) == 0:
        return d
    median_stride_s = float(strides.median())
    max_gap = pd.Timedelta(
        seconds=max(median_stride_s * _DENSIFY_GAP_STRIDE_MULTIPLE, 900.0)
    )

    grid_start = d[DATE_COL].min().ceil(freq)
    grid_end = d[DATE_COL].max().floor(freq)
    if grid_end < grid_start:
        return d
    grid = pd.date_range(grid_start, grid_end, freq=freq)

    series = d.set_index(DATE_COL)[TILT_COL]
    combined_index = series.index.union(grid)
    interp = (
        series.reindex(combined_index)
        .sort_index()
        .interpolate(method="time", limit_direction="both")
    )
    # Mask any grid point whose nearest actual sample on either side is
    # > max_gap away — protects against fabricated data across fetch
    # failures.
    actual_times = series.index.to_numpy()
    grid_times = grid.to_numpy()
    idx_after = np.searchsorted(actual_times, grid_times, side="left")
    idx_before = np.clip(idx_after - 1, 0, len(actual_times) - 1)
    idx_after = np.clip(idx_after, 0, len(actual_times) - 1)
    before_dt = grid_times - actual_times[idx_before]
    after_dt = actual_times[idx_after] - grid_times
    gap_to_nearest = np.minimum(
        np.abs(before_dt.astype("timedelta64[ns]").astype("int64")),
        np.abs(after_dt.astype("timedelta64[ns]").astype("int64")),
    )
    max_gap_ns = max_gap.value
    mask_too_far = gap_to_nearest > max_gap_ns

    result = interp.reindex(grid)
    result.iloc[mask_too_far] = np.nan
    out = pd.DataFrame({DATE_COL: grid, TILT_COL: result.values})
    out = out.dropna(subset=[TILT_COL]).reset_index(drop=True)
    return out


def _apply_ab_corrections(
    live: dict[str, pd.DataFrame],
    alignments: dict[str, SourceAlignment],
) -> dict[str, pd.DataFrame]:
    """Return each source's DataFrame with `tilt ← tilt - b` applied.

    Inverts the scalar-offset model `y_i = true + b_i`. Function name
    retained for call-site compatibility; since `a ≡ 1` there's no
    division to worry about.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, df in live.items():
        align = alignments.get(name)
        if align is None:
            out[name] = df.copy()
            continue
        corrected = df.copy()
        corrected[TILT_COL] = corrected[TILT_COL] - align.b
        out[name] = corrected
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Merge by best effective resolution
# ─────────────────────────────────────────────────────────────────────────────


def _merge_best_resolution(
    corrected: dict[str, pd.DataFrame],
    alignments: dict[str, SourceAlignment],
    report: ReconcileReport,
    archive_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Walk every 15-min bucket in the union of corrected sources and emit
    one row per bucket, chosen by best-effective-resolution over the
    entire set of contributing sources (not only MAD-survivors).

    Phase 4 Commit 4: the MAD gate NO LONGER removes sources from winner
    selection — it only LOGS outliers to `transcription_failures` and
    `rows_mad_rejected` for diagnostic visibility. Deterministic
    per-source-set selection means the same source wins every bucket
    within a region of stable coverage, which eliminates the sawtooth
    that MAD-driven mid-region winner-flipping produced in 2026-04 prod.

    Upstream filtering (`trace._filter_rolling_median_outliers`) catches
    genuine OCR-glitch spikes before they reach reconcile, so the MAD
    gate's removal role was rarely load-bearing; its sole remaining
    purpose is diagnostic recording of residual inter-source
    disagreement.

    Algorithm (per bucket):
      1. Collect every live source's corrected value for the bucket.
      2. Compute median and MAD; record any source beyond
         `max(FLOOR, MULTIPLIER·σ_MAD)` as a TranscriptionFailure (log only).
      3. Pick the source with the smallest effective resolution
         (`|a_i| · µrad/px`) from ALL contributing sources.
      4. If no live source covers the bucket, fall back to archive.

    K≥2 → K=1 handoff blending is applied as a post-pass (unchanged).
    """
    resolutions = {
        name: align.effective_resolution_microrad_per_pixel
        for name, align in alignments.items()
    }

    # Tag every source's rows with both actual and densified-interpolated
    # coverage. Densification is used for WINNER SELECTION only — it lets
    # a low-stride, high-priority source (e.g. three_month every 3h) win
    # every 15-min bucket in its window instead of losing intermittent
    # buckets to a higher-stride, lower-priority peer (e.g.
    # dec2024_to_now every 17h) just because the peer happened to sample
    # closer to that exact bucket. Without this, the per-bucket winner
    # rotated and produced the ~14 µrad Feb 2026 sawtooth between
    # three_month and dec2024_to_now.
    #
    # But we emit only the ACTUAL-sample rows — interpolated rows are
    # used to decide "who should own this bucket if they had a sample"
    # and then the actual sample from the winner (or any other actual
    # sample in the same bucket when the winner had none) carries the
    # value. This keeps tilt_history and the downstream archive at the
    # natural sample cadence of the sources rather than inflating to
    # every 15-min slot across 16 months.
    tagged: list[pd.DataFrame] = []
    for name, df in corrected.items():
        d_actual = df[[DATE_COL, TILT_COL]].copy()
        d_actual[DATE_COL] = pd.to_datetime(d_actual[DATE_COL])
        d_actual = d_actual.dropna().sort_values(DATE_COL).reset_index(drop=True)
        if len(d_actual) == 0:
            continue
        d_actual["_source"] = name
        d_actual["_bucket"] = d_actual[DATE_COL].dt.round(MERGE_BUCKET)
        d_actual["_res"] = resolutions.get(name, 1.0)
        d_actual["_origin"] = "actual"
        tagged.append(d_actual)

        densified = _densify_to_merge_grid(df, MERGE_BUCKET)
        if len(densified) == 0:
            continue
        # Drop grid rows that collide with an actual-sample bucket — we
        # already have the actual for those; the interpolated copy would
        # just duplicate.
        actual_buckets = set(d_actual["_bucket"])
        d_interp = densified.copy()
        d_interp["_source"] = name
        d_interp["_bucket"] = d_interp[DATE_COL].dt.round(MERGE_BUCKET)
        d_interp = d_interp[~d_interp["_bucket"].isin(actual_buckets)].copy()
        if len(d_interp) == 0:
            continue
        d_interp["_res"] = resolutions.get(name, 1.0)
        d_interp["_origin"] = "interp"
        tagged.append(d_interp)

    if not tagged:
        if archive_df is not None and len(archive_df) > 0:
            return archive_df.sort_values(DATE_COL).reset_index(drop=True)
        return _empty_history_df()

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
        sources_arr = group["_source"].to_numpy(dtype=str)
        resolutions_here = group["_res"].to_numpy(dtype=float)
        dates = group[DATE_COL].to_numpy()
        origins = group["_origin"].to_numpy(dtype=str)

        if len(values) == 0:
            continue

        # ── MAD outlier gate (diagnostic-only) ──────────────────────────────
        # Only score actual-sample rows against each other; interpolated
        # rows are driven from adjacent actuals so including them would
        # double-count and deflate MAD.
        actual_mask = origins == "actual"
        if actual_mask.sum() >= 2:
            actual_values = values[actual_mask]
            actual_sources = sources_arr[actual_mask]
            m = float(np.median(actual_values))
            mad = float(np.median(np.abs(actual_values - m)))
            sigma = 1.4826 * mad
            threshold = max(
                MAD_OUTLIER_SIGMA_FLOOR_MICRORAD,
                MAD_OUTLIER_SIGMA_MULTIPLIER * sigma,
            )
            outlier_mask = np.abs(actual_values - m) > threshold
            if outlier_mask.any():
                for v, s in zip(actual_values[outlier_mask], actual_sources[outlier_mask], strict=False):
                    report.transcription_failures.append(
                        TranscriptionFailure(
                            bucket=bucket,
                            source=str(s),
                            value_corrected=float(v),
                            bucket_median=m,
                            delta_microrad=float(v) - m,
                        )
                    )
                    align = alignments.get(str(s))
                    if align is not None:
                        align.rows_mad_rejected += 1

        # ── Winner selection uses ALL rows (actual + interpolated) ──────────
        # The best-resolution source in the coverage set wins the bucket
        # deterministically — including interpolated coverage so a tight-
        # cadence source (three_month) outranks a loose one (dec) even
        # when only the loose one happens to have an actual sample in
        # THIS bucket. The emission step below then picks the winner's
        # actual row if any, else the best other actual row to carry
        # the winner's value.
        winner_idx = int(np.argmin(resolutions_here))
        winner_source = str(sources_arr[winner_idx])
        winner_value = float(values[winner_idx])

        # Emission priority:
        #   1. If the winner has an actual sample here, emit it.
        #   2. Else if SOME source has an actual here, emit the winner's
        #      interpolated value at that actual's timestamp (so we
        #      don't invent a timestamp).
        #   3. Else emit the winner's interpolated value at the bucket
        #      boundary (stops archive gap-fill from re-populating a
        #      bucket that IS covered by a live source).
        actual_rows_mask = origins == "actual"
        winner_actual_mask = actual_rows_mask & (sources_arr == winner_source)
        if winner_actual_mask.any():
            k = int(np.where(winner_actual_mask)[0][0])
            emitted_rows.append(
                (pd.Timestamp(dates[k]), float(values[k]), winner_source)
            )
        elif actual_rows_mask.any():
            k = int(np.where(actual_rows_mask)[0][0])
            emitted_rows.append(
                (pd.Timestamp(dates[k]), winner_value, winner_source)
            )
        else:
            # No actual sample from any source in this bucket; emit the
            # winner's interp at the bucket boundary.
            emitted_rows.append(
                (pd.Timestamp(bucket), winner_value, winner_source)
            )
        report.winner_counts[winner_source] = (
            report.winner_counts.get(winner_source, 0) + 1
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
        # Count archive gap-fills as winners so the observability panel
        # shows archive's contribution honestly.
        if ARCHIVE_SOURCE_NAME in merged.get("_source", pd.Series(dtype=str)).values:
            archive_fills = int(
                (merged["_source"] == ARCHIVE_SOURCE_NAME).sum()
            ) if "_source" in merged.columns else 0
            if archive_fills > 0:
                report.winner_counts[ARCHIVE_SOURCE_NAME] = (
                    report.winner_counts.get(ARCHIVE_SOURCE_NAME, 0) + archive_fills
                )

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
