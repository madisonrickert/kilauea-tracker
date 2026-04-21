"""Append-only canonical archive of observed tilt history.

The archive is a single CSV at `data/archive.csv` whose contents are
NEVER overwritten — only added to. After every `ingest_all()` reconcile
run, `promote_to_archive()` walks the freshly-merged tilt history and
appends any 15-min buckets that aren't already in the archive. Buckets
already present in the archive are left alone, even if the new merged
view has a different value for them. **First observation wins, forever.**

The archive then feeds back into `reconcile.reconcile_sources()` as a
high-priority source (`SOURCE_PRIORITY` slot just below `digital`). Once
a row is in the archive, the reconciler sources it from the archive in
all future runs — immune to drift in the live per-source CSVs.

Why this exists
---------------

Frame alignment in `cache.append_history` (the 2026-04 fix) keeps each
per-source CSV in a single y-frame forever. But that fix only protects
the live per-source files going forward. It doesn't help if:

  - A per-source CSV is wiped and rebuilt from scratch (e.g. an operator
    re-ingests after an OCR bug fix). The new file's first frame may be
    different from the old file's first frame.
  - A future ingest pipeline change inadvertently re-introduces drift in
    a way the regression tests don't catch.

The archive is the durable belt-and-suspenders: a frozen historical
record of "what we observed at this timestamp the first time we ever
observed it." Even if every per-source CSV gets corrupted, the merged
view still produces the right answer for archived timestamps because
the archive wins the reconciler's priority contest.

Why "keep first" instead of "keep best"
---------------------------------------

There is no objective ground truth to score "best" against. Any quality
heuristic is just a deferred bias dressed up as objectivity. Keep-first
is honest: the first time we saw a row, that's what we wrote down, and
that's what we'll always remember. If the first observation turns out
to have been wrong, an operator can manually edit `data/archive.csv` to
correct it — but the system will never silently re-overwrite history.

Schema
------

The archive uses the canonical `[Date, Tilt (microradians)]` schema with
no extra columns. Source provenance lives in `archive.csv`'s commit
history (every commit that touches it shows the diff) — we deliberately
don't tag rows with their original source because the archive is meant
to be source-agnostic by the time it ships. The reconciler sees it as
just another aligned source named `"archive"`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    ARCHIVE_CSV,
    ARCHIVE_QUORUM_MIN_SOURCES,
    ARCHIVE_QUORUM_NEIGHBOUR_MINUTES,
    ARCHIVE_QUORUM_NEIGHBOUR_THRESHOLD_MICRORAD,
)
from .model import DATE_COL, TILT_COL

# Bucket granularity for keep-first dedupe. Same as cache.DEDUPE_BUCKET so
# the archive lines up with the per-source CSVs at the same temporal
# resolution. Imported here as a constant rather than from cache.py to
# avoid a circular import (cache.py never needs to know about the archive).
ARCHIVE_BUCKET = "15min"


@dataclass
class ArchivePromotionReport:
    """Outcome of one `promote_to_archive()` call."""

    rows_in_archive_before: int = 0
    rows_in_archive_after: int = 0
    rows_promoted: int = 0
    rows_already_archived: int = 0
    # Rows that were otherwise promotable but were deferred by the quorum
    # gate — too few source contributors AND no matching archived neighbour.
    # These rows remain eligible on future runs when a second source has
    # caught up to the bucket.
    rows_deferred_by_quorum: int = 0
    warnings: list[str] = field(default_factory=list)


def load_archive(path: Path = ARCHIVE_CSV) -> pd.DataFrame:
    """Read the archive CSV.

    Returns an empty DataFrame (with the canonical schema) if the file
    doesn't exist. The very first `ingest_all()` call after a fresh
    checkout will populate the archive from the merged view.
    """
    if not path.exists():
        return _empty_archive()
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed", dayfirst=False)
    df = df[[DATE_COL, TILT_COL]].dropna()
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def promote_to_archive(
    merged_df: pd.DataFrame,
    path: Path = ARCHIVE_CSV,
    *,
    sources: Optional[dict[str, pd.DataFrame]] = None,
) -> ArchivePromotionReport:
    """Append rows from `merged_df` to the archive with keep-first dedupe
    and an optional multi-source quorum gate.

    Args:
        merged_df: A reconciled tilt history (the output of
            `reconcile.reconcile_sources()`). Schema must be
            `[Date, Tilt (microradians)]`.
        path:      Where to write. Defaults to `config.ARCHIVE_CSV`.
        sources:   Optional — the raw per-source DataFrames that fed the
            reconcile (same dict shape as `reconcile_sources` received).
            When provided, each candidate row must be corroborated by at
            least `ARCHIVE_QUORUM_MIN_SOURCES` contributing sources (at the
            same 15-min bucket) OR sit close enough to an already-archived
            neighbour to piggyback on its provenance. When omitted (the
            legacy call path kept for backward compatibility), no quorum
            gate is applied — all non-duplicate rows are promoted.

    Returns:
        An `ArchivePromotionReport` with before/after row counts, how many
        rows were promoted, how many were deferred by the quorum gate, and
        any free-text warnings.

    Behavior:
      - If `merged_df` is empty, no-op.
      - For each 15-min bucket in `merged_df` not already in the archive:
        apply the quorum gate (if `sources` supplied). Promote if it passes,
        defer otherwise. The `archive` source itself (present because the
        reconciler re-consumes its own archive as a source) never counts
        toward the source quorum — self-corroboration would defeat the
        purpose of the gate.
      - Sort the result by Date and write back to disk.
    """
    report = ArchivePromotionReport()
    if merged_df is None or len(merged_df) == 0:
        return report

    existing = load_archive(path)
    report.rows_in_archive_before = len(existing)

    incoming = merged_df[[DATE_COL, TILT_COL]].copy()
    incoming[DATE_COL] = pd.to_datetime(incoming[DATE_COL])
    incoming = incoming.dropna(subset=[DATE_COL, TILT_COL])

    if len(incoming) == 0:
        report.rows_in_archive_after = report.rows_in_archive_before
        return report

    # Bucket both sides at 15-min granularity. Reduce within-bucket to one
    # representative each (latest by date for incoming; existing is already
    # one-per-bucket because it was archived under these same semantics).
    incoming["_bucket"] = incoming[DATE_COL].dt.round(ARCHIVE_BUCKET)
    incoming_dedup = (
        incoming.sort_values(DATE_COL)
        .drop_duplicates(subset="_bucket", keep="last")
    )

    if len(existing) > 0:
        existing_buckets = existing[DATE_COL].dt.round(ARCHIVE_BUCKET)
        already_archived_mask = incoming_dedup["_bucket"].isin(set(existing_buckets))
        candidate_rows = incoming_dedup[~already_archived_mask]
        report.rows_already_archived = int(already_archived_mask.sum())
    else:
        candidate_rows = incoming_dedup
        report.rows_already_archived = 0

    # Apply the quorum gate BEFORE stripping `_bucket`; the gate needs it.
    if sources is not None and len(candidate_rows) > 0:
        candidate_rows, n_deferred, warn = _apply_quorum_gate(
            candidate_rows, sources=sources, existing=existing
        )
        report.rows_deferred_by_quorum = n_deferred
        if warn:
            report.warnings.append(warn)

    new_rows = candidate_rows.drop(columns=["_bucket"])
    report.rows_promoted = len(new_rows)

    if len(new_rows) == 0:
        # Nothing new — write nothing, archive on disk is already correct.
        report.rows_in_archive_after = report.rows_in_archive_before
        return report

    combined = pd.concat([existing, new_rows[[DATE_COL, TILT_COL]]], ignore_index=True)
    combined = combined.sort_values(DATE_COL).reset_index(drop=True)
    report.rows_in_archive_after = len(combined)

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return report


def _apply_quorum_gate(
    candidate_rows: pd.DataFrame,
    *,
    sources: dict[str, pd.DataFrame],
    existing: pd.DataFrame,
) -> tuple[pd.DataFrame, int, str]:
    """Keep only rows that either:
      (a) received contributions from `ARCHIVE_QUORUM_MIN_SOURCES` different
          sources at the same 15-min bucket, OR
      (b) sit within `ARCHIVE_QUORUM_NEIGHBOUR_MINUTES` of an existing
          archive row and within
          `ARCHIVE_QUORUM_NEIGHBOUR_THRESHOLD_MICRORAD` of its value.

    Returns `(kept_rows, n_deferred, warning_message_or_empty)`.

    The `archive` source itself is excluded from the source count — it would
    otherwise vacuously corroborate every bucket that the reconciler had
    already re-consumed from the archive, defeating the whole point of the
    gate.
    """
    # Source-count per bucket (excluding the archive source — see docstring).
    contributions: dict[pd.Timestamp, int] = {}
    for source_name, df in sources.items():
        if source_name == "archive" or df is None or len(df) == 0:
            continue
        if DATE_COL not in df.columns:
            continue
        buckets = pd.to_datetime(df[DATE_COL]).dt.round(ARCHIVE_BUCKET)
        for b in buckets.dropna().unique():
            contributions[b] = contributions.get(b, 0) + 1

    # Prepare a sorted neighbour lookup against existing archive rows.
    if len(existing) > 0:
        e = existing[[DATE_COL, TILT_COL]].copy()
        e[DATE_COL] = pd.to_datetime(e[DATE_COL]).astype("datetime64[ns]")
        e = e.sort_values(DATE_COL).reset_index(drop=True)
    else:
        e = existing

    candidates = candidate_rows.sort_values(DATE_COL).reset_index(drop=True).copy()
    candidates[DATE_COL] = pd.to_datetime(candidates[DATE_COL]).astype("datetime64[ns]")

    if len(e) > 0:
        nearest = pd.merge_asof(
            candidates[[DATE_COL, TILT_COL, "_bucket"]],
            e.rename(columns={DATE_COL: "_neighbor_date", TILT_COL: "_neighbor_tilt"}),
            left_on=DATE_COL,
            right_on="_neighbor_date",
            direction="nearest",
        )
        time_to_neighbor = (nearest[DATE_COL] - nearest["_neighbor_date"]).abs()
        value_delta = (nearest[TILT_COL] - nearest["_neighbor_tilt"]).abs()
        proximity_radius = pd.Timedelta(minutes=ARCHIVE_QUORUM_NEIGHBOUR_MINUTES)
        has_trusted_neighbor = (
            (time_to_neighbor <= proximity_radius)
            & (value_delta <= ARCHIVE_QUORUM_NEIGHBOUR_THRESHOLD_MICRORAD)
        ).to_numpy()
    else:
        has_trusted_neighbor = [False] * len(candidates)

    source_counts = candidates["_bucket"].map(contributions).fillna(0).astype(int)
    passes_quorum = source_counts >= ARCHIVE_QUORUM_MIN_SOURCES
    keep_mask = passes_quorum.to_numpy() | has_trusted_neighbor

    kept = candidates[keep_mask].reset_index(drop=True)
    n_deferred = int((~keep_mask).sum())
    warn = ""
    if n_deferred > 0:
        warn = (
            f"archive quorum gate deferred {n_deferred} row(s) — fewer than "
            f"{ARCHIVE_QUORUM_MIN_SOURCES} sources contributed and no "
            f"trusted archive neighbour within "
            f"{ARCHIVE_QUORUM_NEIGHBOUR_MINUTES} min / "
            f"{ARCHIVE_QUORUM_NEIGHBOUR_THRESHOLD_MICRORAD} µrad"
        )
    return kept, n_deferred, warn


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────


def _empty_archive() -> pd.DataFrame:
    return pd.DataFrame(
        {
            DATE_COL: pd.Series(dtype="datetime64[ns]"),
            TILT_COL: pd.Series(dtype="float64"),
        }
    )
