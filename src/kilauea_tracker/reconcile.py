"""Per-source reconciliation: anchor-based alignment + priority merge.

Replaces the old cascading-update logic in `ingest.pipeline`. Under the
previous architecture each source aligned against the cache-as-of-now and
then overwrote overlapping buckets via last-write-wins dedupe — order-
dependent, no provenance, errors compounded as each source's misalignment
got baked into the cache that later sources then aligned against.

This module decouples raw source storage from the merged "view" the model
consumes. Inputs to reconciliation are the raw per-source DataFrames. The
output is a single merged tilt history that is a *deterministic* function
of those inputs and `config.SOURCE_PRIORITY`.

Algorithm
---------

Two orderings live in `config.py` and they are deliberately distinct:

  - `ALIGNMENT_ORDER` is the *topological* order in which sources are aligned.
    Each source's y-offset is computed against the union of all already-
    aligned sources, so the order has to ensure every subsequent source
    has temporal overlap with at least one prior source. Digital anchors
    first (Jan-Jun 2025); DEC2024_TO_NOW comes next because it's the only
    source that overlaps both digital and the recent rolling-window PNGs;
    legacy follows; the rolling-window PNGs last.
  - `SOURCE_PRIORITY` is the *quality* order used by the per-bucket merge.
    When two sources both have data for the same 15-min bucket, the one
    earlier in `SOURCE_PRIORITY` wins. This is digital → two_day → week →
    month → legacy → dec2024_to_now → three_month, putting
    highest-resolution recent data on top.

Steps:

1.  Walk sources in `ALIGNMENT_ORDER`. The first one with data becomes the
    *anchor* — offset 0, defines the global y-frame.
2.  Each subsequent source computes ONE y-offset against the union of all
    already-aligned sources, in their overlap region. The offset is the
    median bucket-level delta at `ALIGNMENT_BUCKET` granularity. If the
    overlap is too small or the offset is implausibly large, the source is
    left unaligned and a warning is recorded — its raw values are still
    included in the merge but will lose per-bucket contests to aligned
    higher-priority sources.
3.  After every source has been processed, walk the union of 15-min buckets
    across all sources. For each bucket, pick the value from the
    highest-priority source (in `SOURCE_PRIORITY` order) that has data
    there. The result is the merged tilt history.

Why this fixes the whack-a-mole
-------------------------------

- **Order-independent.** Every source's offset is computed against a *fixed*
  reference (the union of higher-priority sources), not against a moving
  cache. Re-running gives the same answer.
- **Single-pass alignment.** No source's offset depends on its own previous
  contributions to the cache, so misalignment errors don't compound.
- **Provenance preserved.** Per-source files on disk are untouched by
  reconciliation; only the merged view is recomputed. We can always trace
  any sample in `tilt_history.csv` back to its source by looking up its
  bucket in the per-source files.
- **Conflicts become signal.** A bucket where two aligned sources disagree
  by more than `CONFLICT_THRESHOLD_MICRORAD` is recorded with both values
  and provenance, so the warning is actionable instead of mysterious.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .config import (
    ALIGNMENT_ORDER,
    ARCHIVE_MAX_AGE_FOR_PRIORITY_DEMOTION_DAYS,
    ARCHIVE_SOURCE_NAME,
    SOURCE_PRIORITY,
)
from .model import DATE_COL, TILT_COL

# Bucket size for the per-source y-offset calculation. Coarser than the
# 15-min merge bucket so that sparse sources (legacy at ~6h spacing,
# DEC2024_TO_NOW at ~16h) can find enough overlapping buckets to compute a
# meaningful median against denser higher-priority sources.
ALIGNMENT_BUCKET = "1h"

# Bucket size for the priority-based merge. The merged tilt history retains
# one row per 15-min bucket — fine enough to preserve real samples, coarse
# enough to merge near-duplicate re-traces from the same source.
MERGE_BUCKET = "15min"

# Need at least this many overlapping ALIGNMENT_BUCKETs before we trust the
# computed median offset. Below this we leave the source unaligned and warn.
MIN_OVERLAP_BUCKETS = 5

# Refuse to apply offsets larger than this. A delta this big is a calibration
# bug, not y-axis drift, and silently shifting the data by 15+ µrad would
# mask the real problem.
MAX_TRUSTED_OFFSET_MICRORAD = 15.0

# A merged bucket where two aligned sources disagree by more than this is
# flagged as a conflict in the report. The HIGHER-priority source's value is
# what survives the merge regardless — the conflict is informational.
CONFLICT_THRESHOLD_MICRORAD = 1.0

# Proximity-gating radius. A low-priority source's sample is dropped from the
# merge candidate set if any higher-priority source has a sample within this
# many minutes. The motivation is dec2024_to_now's local y-frame distortion:
# even after the median offset is applied, individual samples can still be a
# few µrad off from the trusted reference. When such a sample lands in a
# 15-min bucket the higher-priority source happens not to cover, it wins the
# bucket by default and renders as a visible "spike" between adjacent
# higher-priority samples. The proximity gate drops these would-be spikes
# without affecting samples in genuine multi-hour gaps where dec2024_to_now
# is providing real coverage.
#
# 60 min was chosen because:
#   - digital has 30-min spacing, so a dec2024_to_now sample within ±60 min
#     of a digital sample is always sandwiched between two digital samples
#     (or right next to one) and adds no information.
#   - legacy is irregular at ~6h spacing, so the gate doesn't drop
#     dec2024_to_now samples that are filling real legacy gaps (>1h from
#     any legacy sample).
PROXIMITY_GATE_MINUTES = 60

# Per-source overrides for the proximity gate radius. Sources NOT listed
# here use PROXIMITY_GATE_MINUTES.
#
# `dec2024_to_now` gets a much larger 12-hour radius because its traced
# curve has non-uniform local y-frame distortion: even after the global
# median offset is applied, individual samples can still sit several µrad
# off from the trusted reference. Within the digital-overlap range a
# 60-minute gate is enough (digital has 30-min spacing). But within the
# legacy-overlap range, legacy is sparse (~14h average spacing), so the
# 60-min gate doesn't drop the dec2024_to_now samples that fall between
# legacy samples — they then render as ~5 µrad sawtooth jumps in the
# chart. A 12-hour gate aligns with legacy's typical spacing and ensures
# dec2024_to_now only contributes in true multi-hour gaps.
#
# In the post-Nov 2025 → Jan 2026 region (no legacy, no digital, no PNG
# coverage) dec2024_to_now is alone and the gate is a no-op — every
# dec2024_to_now sample is far from any higher-priority sample, so they
# all survive. That's the legitimate gap-fill case.
PROXIMITY_GATE_OVERRIDES_MINUTES: dict[str, int] = {
    "dec2024_to_now": 12 * 60,
}

# Sources for which we run a second-pass piecewise realignment after the
# first-pass single-offset alignment. The piecewise pass computes per-source
# residual offsets against each higher-priority source individually so that
# a source with non-uniform local y-frame distortion (today: dec2024_to_now)
# can be locally re-corrected. The realignment is purely additive on top of
# the first-pass global offset, so a residual of 0 means "the global offset
# was already correct in this region." See _piecewise_realign() for details.
PIECEWISE_REALIGN_SOURCES: frozenset = frozenset({"dec2024_to_now"})


@dataclass
class SourceAlignment:
    """How one source was treated during reconciliation."""

    name: str
    rows_in: int = 0                  # raw row count of the input DataFrame
    offset_microrad: Optional[float] = None  # y-offset applied (None = unaligned)
    overlap_buckets: int = 0          # how many alignment buckets the offset was computed over
    is_anchor: bool = False           # True for the first source (offset by definition 0)
    note: Optional[str] = None        # human-readable explanation when offset is None
    rows_proximity_dropped: int = 0   # rows removed by the proximity gate before merge
    piecewise_residuals: dict[str, float] = field(default_factory=dict)
    """Per-higher-priority-source residual offsets applied in the second-
    pass piecewise realignment. Empty unless the source is in
    PIECEWISE_REALIGN_SOURCES."""


@dataclass
class ReconcileConflict:
    """Two sources disagree about the same 15-min bucket post-alignment."""

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
    conflicts: list[ReconcileConflict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def reconcile_sources(
    sources: dict[str, pd.DataFrame],
    *,
    proximity_minutes: int = PROXIMITY_GATE_MINUTES,
) -> tuple[pd.DataFrame, ReconcileReport]:
    """Merge raw per-source tilt data into a single deterministic history.

    Args:
        sources: dict mapping source name (must be a member of
            `config.SOURCE_PRIORITY`) to a raw, unaligned DataFrame with
            columns `[Date, Tilt (microradians)]`. Sources may be missing or
            empty — they're just skipped.
        proximity_minutes: radius for the proximity gate (see
            `PROXIMITY_GATE_MINUTES`). Pass 0 to disable proximity gating
            entirely — useful for tests that intentionally construct
            overlapping sources to exercise the merge step.

    Returns:
        `(merged_history_df, report)`. `merged_history_df` has the same
        canonical schema as `data/tilt_history.csv`. `report` summarizes
        what each source contributed and any conflicts that surfaced.
    """
    report = ReconcileReport()
    aligned: dict[str, pd.DataFrame] = {}

    for name in ALIGNMENT_ORDER:
        df_raw = sources.get(name)
        if df_raw is None or len(df_raw) == 0:
            continue

        # Normalize: copy + select canonical columns + drop NaN.
        df_in = df_raw[[DATE_COL, TILT_COL]].copy()
        df_in[DATE_COL] = pd.to_datetime(df_in[DATE_COL])
        df_in = df_in.dropna().sort_values(DATE_COL).reset_index(drop=True)
        if len(df_in) == 0:
            continue

        record = SourceAlignment(name=name, rows_in=len(df_in))

        if not aligned:
            # First source with data becomes the anchor — defines the y-frame.
            record.is_anchor = True
            record.offset_microrad = 0.0
            aligned[name] = df_in
            report.sources.append(record)
            continue

        reference = pd.concat(aligned.values(), ignore_index=True)
        offset, overlap = _median_bucket_offset(df_in, reference)
        record.overlap_buckets = overlap

        if offset is None:
            if overlap < MIN_OVERLAP_BUCKETS:
                record.note = (
                    f"insufficient overlap with higher-priority sources "
                    f"({overlap} buckets, need ≥{MIN_OVERLAP_BUCKETS})"
                )
            else:
                # We have overlap but the median delta exceeded the trust limit.
                record.note = (
                    f"refused to apply implausibly large offset "
                    f"(>{MAX_TRUSTED_OFFSET_MICRORAD} µrad)"
                )
            report.warnings.append(
                f"{name}: not aligned — {record.note}"
            )
            aligned[name] = df_in  # include unaligned, will lose priority contests
        else:
            shifted = df_in.copy()
            shifted[TILT_COL] = shifted[TILT_COL] - offset
            aligned[name] = shifted
            record.offset_microrad = offset

        report.sources.append(record)

    if not aligned:
        report.rows_out = 0
        return _empty_history_df(), report

    # Second-pass piecewise realignment for sources with non-uniform local
    # y-frame distortion (see PIECEWISE_REALIGN_SOURCES docstring).
    for target_name in PIECEWISE_REALIGN_SOURCES:
        aligned, residuals = _piecewise_realign(aligned, target_name)
        if residuals:
            record = next(
                (s for s in report.sources if s.name == target_name), None
            )
            if record is not None:
                record.piecewise_residuals = residuals

    # Drop low-priority samples that would render as spikes between adjacent
    # higher-priority samples (see PROXIMITY_GATE_MINUTES docstring).
    if proximity_minutes > 0:
        gated, gate_drops = _drop_samples_near_higher_priority(
            aligned, proximity_minutes=proximity_minutes
        )
    else:
        gated, gate_drops = aligned, {}
    for source_name, n_dropped in gate_drops.items():
        record = next(
            (s for s in report.sources if s.name == source_name), None
        )
        if record is not None:
            record.rows_proximity_dropped = n_dropped

    merged = _merge_by_priority(gated, report)
    report.rows_out = len(merged)
    return merged, report


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────


def _median_bucket_offset(
    new_rows: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    bucket_freq: str = ALIGNMENT_BUCKET,
) -> tuple[Optional[float], int]:
    """Compute `median(new - reference)` over their overlapping `bucket_freq`
    buckets. Returns `(offset, overlap_count)`.

    `offset` is `None` when the offset can't be trusted: either the overlap
    is below `MIN_OVERLAP_BUCKETS`, or the absolute offset exceeds
    `MAX_TRUSTED_OFFSET_MICRORAD`. The overlap count is always returned for
    diagnostic purposes.

    Why median (not mean): a few buckets near eruption transitions disagree
    wildly across captures because the curve is moving fast and the
    column-median pixel can land at very different positions. Mean would
    chase those outliers; median ignores them.
    """
    if len(new_rows) == 0 or len(reference) == 0:
        return None, 0

    new_buckets = (
        new_rows[[DATE_COL, TILT_COL]]
        .assign(_b=lambda d: d[DATE_COL].dt.floor(bucket_freq))
        .groupby("_b")[TILT_COL]
        .mean()
    )
    ref_buckets = (
        reference[[DATE_COL, TILT_COL]]
        .assign(_b=lambda d: d[DATE_COL].dt.floor(bucket_freq))
        .groupby("_b")[TILT_COL]
        .mean()
    )
    overlap_index = new_buckets.index.intersection(ref_buckets.index)
    overlap_n = len(overlap_index)

    if overlap_n < MIN_OVERLAP_BUCKETS:
        return None, overlap_n

    deltas = new_buckets.loc[overlap_index] - ref_buckets.loc[overlap_index]
    offset = float(deltas.median())

    if abs(offset) > MAX_TRUSTED_OFFSET_MICRORAD:
        return None, overlap_n

    return offset, overlap_n


def _piecewise_realign(
    aligned: dict[str, pd.DataFrame],
    target_name: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Refine `target_name`'s alignment by computing per-source RESIDUAL
    offsets against each higher-priority source individually, then applying
    the residual associated with whichever higher-priority source is closest
    in time to each target row.

    The first-pass alignment loop produces a single global offset for each
    source — the median delta against the union of higher-priority sources.
    For most sources that's correct, but for sources with non-uniform local
    y-frame distortion (today: dec2024_to_now) the global median is biased
    toward the densest co-existing source's overlap region. Less-dense
    overlap regions are then left with several µrad of residual error.

    This function computes that residual *per higher-priority source* and
    applies it as an additional correction. After the second pass:

      - In the digital-overlap range, dec2024_to_now uses the residual
        relative to digital (≈0 by construction, since digital dominated
        the first-pass median).
      - In the legacy-overlap range, dec2024_to_now uses the residual
        relative to legacy. The saw-tooth artifact at the digital→legacy
        handoff disappears.
      - In recent dense-PNG overlap, dec2024_to_now uses the residual
        relative to whichever PNG source has the closest sample.

    Returns `(new_aligned_dict, residuals_by_source)`. The residuals dict
    is empty when no usable residuals could be computed (e.g. target has
    no overlap with any higher-priority source).
    """
    if target_name not in aligned or len(aligned[target_name]) == 0:
        return aligned, {}

    target_df = aligned[target_name]
    priority_index = {n: i for i, n in enumerate(SOURCE_PRIORITY)}
    target_priority = priority_index.get(target_name, len(SOURCE_PRIORITY))

    # Compute the residual median delta against each higher-priority source
    # individually, using the SAME bucket granularity as the first-pass
    # alignment so the numbers are comparable.
    per_source_residuals: dict[str, float] = {}
    for other_name, other_df in aligned.items():
        if other_name == target_name or len(other_df) == 0:
            continue
        other_priority = priority_index.get(other_name, len(SOURCE_PRIORITY))
        if other_priority >= target_priority:
            continue
        residual, n = _median_bucket_offset(target_df, other_df)
        if residual is not None and n >= MIN_OVERLAP_BUCKETS:
            per_source_residuals[other_name] = residual

    if not per_source_residuals:
        return aligned, {}

    # Build a sorted lookup of (date, source_name) for the higher-priority
    # sources we have residuals for. Each target row gets the residual of
    # whichever source's nearest sample is closest in time.
    rows: list[tuple[pd.Timestamp, str]] = []
    for src_name in per_source_residuals:
        for d in aligned[src_name][DATE_COL]:
            rows.append((pd.Timestamp(d), src_name))
    if not rows:
        return aligned, per_source_residuals

    rows.sort(key=lambda r: r[0])
    nearest_lookup = pd.DataFrame(rows, columns=["_date", "_src"])
    nearest_lookup["_date"] = pd.to_datetime(nearest_lookup["_date"]).astype(
        "datetime64[ns]"
    )

    target_sorted = target_df.sort_values(DATE_COL).reset_index(drop=True).copy()
    target_sorted[DATE_COL] = target_sorted[DATE_COL].astype("datetime64[ns]")

    nearest = pd.merge_asof(
        target_sorted[[DATE_COL]].copy(),
        nearest_lookup,
        left_on=DATE_COL,
        right_on="_date",
        direction="nearest",
    )

    corrections = nearest["_src"].map(per_source_residuals).fillna(0.0)

    new_target = target_sorted.copy()
    new_target[TILT_COL] = new_target[TILT_COL] - corrections.values

    new_aligned = dict(aligned)
    new_aligned[target_name] = new_target
    return new_aligned, per_source_residuals


def _drop_samples_near_higher_priority(
    aligned: dict[str, pd.DataFrame],
    *,
    proximity_minutes: int = PROXIMITY_GATE_MINUTES,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Drop low-priority samples that fall within `proximity_minutes` of any
    higher-priority sample.

    Each source can override the default radius via
    `PROXIMITY_GATE_OVERRIDES_MINUTES`. Sources that opt for a larger
    radius are dropped more aggressively when higher-priority sources are
    nearby — used today for dec2024_to_now whose local y-frame distortion
    means even a "well aligned" sample can render as a sawtooth between
    legacy samples that are 2-12 hours away.

    Why this exists: dec2024_to_now's traced curve has non-uniform local
    y-frame distortion. Even after a global median offset is applied, an
    individual dec2024_to_now sample can still sit a few µrad off from the
    trusted reference. When such a sample lands in a 15-min merge bucket
    that no higher-priority source happens to cover, it wins by default and
    renders as a visible spike between the adjacent higher-priority samples.
    Proximity gating drops the spike-makers BEFORE the merge, leaving
    higher-priority sources to render their natural smooth curves and only
    letting low-priority sources contribute in genuine multi-hour gaps.

    Returns `(gated_aligned, drops_per_source)`. The dict counts how many
    rows were dropped from each source so the report can surface it.
    """
    priority_index = {name: i for i, name in enumerate(SOURCE_PRIORITY)}

    result: dict[str, pd.DataFrame] = {}
    drops: dict[str, int] = {}

    for name, df in aligned.items():
        my_priority = priority_index.get(name, len(SOURCE_PRIORITY))
        # Per-source override takes precedence over the caller-supplied default.
        my_radius = PROXIMITY_GATE_OVERRIDES_MINUTES.get(name, proximity_minutes)
        proximity = pd.Timedelta(minutes=my_radius)
        # Build the union of all higher-priority sources' timestamps. We
        # only care about the dates here; the tilt values don't matter for
        # proximity testing.
        higher_dates = []
        for other_name, other_df in aligned.items():
            other_priority = priority_index.get(other_name, len(SOURCE_PRIORITY))
            if other_priority < my_priority and len(other_df) > 0:
                higher_dates.append(other_df[DATE_COL])

        if not higher_dates:
            # No higher-priority sources to gate against — pass through.
            result[name] = df
            continue

        higher = pd.concat(higher_dates, ignore_index=True)
        # Normalize to nanosecond resolution so merge_asof's dtype check
        # passes — different sources can carry M8[us] vs M8[ns] depending
        # on how they were ingested (PNG traces produce ns; the digital
        # CSV via pandas.read_csv produces us).
        higher = pd.to_datetime(higher).astype("datetime64[ns]")
        higher = higher.sort_values().reset_index(drop=True)

        df_sorted = df.sort_values(DATE_COL).reset_index(drop=True).copy()
        df_sorted[DATE_COL] = df_sorted[DATE_COL].astype("datetime64[ns]")
        higher_df = pd.DataFrame({"_higher_date": higher})

        # merge_asof finds the nearest higher-priority date for each row in
        # df_sorted in O(n log m) time. Both inputs must be sorted, which
        # they are.
        nearest = pd.merge_asof(
            df_sorted,
            higher_df,
            left_on=DATE_COL,
            right_on="_higher_date",
            direction="nearest",
        )
        time_to_higher = (nearest[DATE_COL] - nearest["_higher_date"]).abs()
        keep_mask = time_to_higher > proximity

        kept = df_sorted[keep_mask].reset_index(drop=True)
        result[name] = kept
        n_dropped = int((~keep_mask).sum())
        if n_dropped > 0:
            drops[name] = n_dropped

    return result, drops


def _merge_by_priority(
    aligned: dict[str, pd.DataFrame],
    report: ReconcileReport,
) -> pd.DataFrame:
    """Walk every 15-min bucket in the union of `aligned` sources and pick
    the value from the highest-priority source that has data there.

    Higher priority = lower index in `SOURCE_PRIORITY`. We tag each row with
    its priority index, sort by (bucket, priority), and drop duplicates per
    bucket keeping the FIRST (= highest-priority) row.

    Archive priority demotion: archive rows whose timestamps fall within
    `ARCHIVE_MAX_AGE_FOR_PRIORITY_DEMOTION_DAYS` of the current UTC time
    are temporarily demoted to the lowest priority slot. This lets a live
    source override a freshly-archived bad row on subsequent cron runs —
    the archive's keep-first semantics mean a contaminated row is otherwise
    permanent. Older archive rows keep their priority-2 slot because their
    value as a frozen historical anchor outweighs any risk of stale drift.

    Conflicts — buckets where two aligned sources disagree by more than
    `CONFLICT_THRESHOLD_MICRORAD` — are recorded on `report.conflicts` so the
    UI can show actionable provenance instead of just a count.
    """
    priority_index = {name: i for i, name in enumerate(SOURCE_PRIORITY)}
    lowest_priority = len(SOURCE_PRIORITY)  # one past the end
    now_utc = pd.Timestamp.now("UTC").tz_localize(None)
    demotion_cutoff = now_utc - pd.Timedelta(
        days=ARCHIVE_MAX_AGE_FOR_PRIORITY_DEMOTION_DAYS
    )

    tagged: list[pd.DataFrame] = []
    for name, df in aligned.items():
        d = df[[DATE_COL, TILT_COL]].copy()
        d["_source"] = name
        d["_bucket"] = d[DATE_COL].dt.round(MERGE_BUCKET)
        base_priority = priority_index[name]
        if name == ARCHIVE_SOURCE_NAME:
            # Row-by-row priority: young rows get demoted; old rows keep
            # their priority-2 slot. This is the only source with
            # per-row priority variation.
            is_young = d[DATE_COL] >= demotion_cutoff
            d["_priority"] = base_priority
            d.loc[is_young, "_priority"] = lowest_priority
        else:
            d["_priority"] = base_priority
        tagged.append(d)

    combined = pd.concat(tagged, ignore_index=True)
    combined = combined.sort_values(
        ["_bucket", "_priority", DATE_COL], kind="stable"
    )

    # Conflict detection: any bucket where the winning value (lowest-priority
    # index) and any losing value differ by more than the threshold. We do
    # this in one groupby pass before the dedupe so we can capture both
    # sides of the conflict.
    for bucket, group in combined.groupby("_bucket", sort=False):
        if len(group) < 2:
            continue
        # group is sorted by priority because of the outer sort_values
        winner_row = group.iloc[0]
        winning_tilt = float(winner_row[TILT_COL])
        for _, row in group.iloc[1:].iterrows():
            losing_tilt = float(row[TILT_COL])
            delta = winning_tilt - losing_tilt
            if abs(delta) > CONFLICT_THRESHOLD_MICRORAD:
                report.conflicts.append(
                    ReconcileConflict(
                        bucket=bucket,
                        winning_source=str(winner_row["_source"]),
                        losing_source=str(row["_source"]),
                        winning_tilt=winning_tilt,
                        losing_tilt=losing_tilt,
                        delta=delta,
                    )
                )

    deduped = combined.drop_duplicates(subset="_bucket", keep="first")
    deduped = deduped.drop(columns=["_source", "_priority", "_bucket"])
    deduped = deduped.sort_values(DATE_COL).reset_index(drop=True)
    return deduped


def _empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            DATE_COL: pd.Series(dtype="datetime64[ns]"),
            TILT_COL: pd.Series(dtype="float64"),
        }
    )
