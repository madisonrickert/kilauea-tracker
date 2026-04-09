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

from .config import ALIGNMENT_ORDER, SOURCE_PRIORITY
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


@dataclass
class SourceAlignment:
    """How one source was treated during reconciliation."""

    name: str
    rows_in: int = 0                  # raw row count of the input DataFrame
    offset_microrad: Optional[float] = None  # y-offset applied (None = unaligned)
    overlap_buckets: int = 0          # how many alignment buckets the offset was computed over
    is_anchor: bool = False           # True for the first source (offset by definition 0)
    note: Optional[str] = None        # human-readable explanation when offset is None


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
) -> tuple[pd.DataFrame, ReconcileReport]:
    """Merge raw per-source tilt data into a single deterministic history.

    Args:
        sources: dict mapping source name (must be a member of
            `config.SOURCE_PRIORITY`) to a raw, unaligned DataFrame with
            columns `[Date, Tilt (microradians)]`. Sources may be missing or
            empty — they're just skipped.

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

    merged = _merge_by_priority(aligned, report)
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


def _merge_by_priority(
    aligned: dict[str, pd.DataFrame],
    report: ReconcileReport,
) -> pd.DataFrame:
    """Walk every 15-min bucket in the union of `aligned` sources and pick
    the value from the highest-priority source that has data there.

    Higher priority = lower index in `SOURCE_PRIORITY`. We tag each row with
    its priority index, sort by (bucket, priority), and drop duplicates per
    bucket keeping the FIRST (= highest-priority) row.

    Conflicts — buckets where two aligned sources disagree by more than
    `CONFLICT_THRESHOLD_MICRORAD` — are recorded on `report.conflicts` so the
    UI can show actionable provenance instead of just a count.
    """
    priority_index = {name: i for i, name in enumerate(SOURCE_PRIORITY)}

    tagged: list[pd.DataFrame] = []
    for name, df in aligned.items():
        d = df[[DATE_COL, TILT_COL]].copy()
        d["_source"] = name
        d["_priority"] = priority_index[name]
        d["_bucket"] = d[DATE_COL].dt.round(MERGE_BUCKET)
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
