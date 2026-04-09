"""Tilt CSV reader and per-source append helper.

This module hosts two responsibilities:

  1. `load_history()` — reads the merged tilt history CSV that the model
     and Streamlit UI consume. The merged history is a derived artifact
     produced by `ingest.pipeline.ingest_all()`; this function just reads
     whatever's on disk and returns an empty DataFrame if nothing is.

  2. `append_history()` — appends new rows to *any* tilt CSV with intra-
     file dedupe at 15-min buckets. Used by `ingest.pipeline.ingest()` to
     write into per-source files at `data/sources/<source>.csv`. Each
     source's file is independent — there is no cross-source merging here;
     that happens later in `reconcile.reconcile_sources()`.

Dedupe strategy in `append_history`:
  1. Concat existing + new rows.
  2. Round each Date to the nearest 15-minute bucket.
  3. Drop duplicates on the rounded key, keeping the LAST occurrence —
     fresh re-traces overwrite older ones because USGS may have re-rendered
     the same time window with slightly different y values.
  4. Detect "conflicts" where the new row's tilt disagrees with the cached
     row by more than `CONFLICT_THRESHOLD` µrad. These are intra-source
     conflicts (one source re-traced its own data inconsistently between
     captures) and indicate calibration drift in that single source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import HISTORY_CSV
from .model import DATE_COL, TILT_COL

# How fine the dedupe key is rounded. Smaller = more rows preserved, larger =
# more aggressive merging.
DEDUPE_BUCKET = "15min"

# Tilt difference (in µrad) above which two rows at the same dedupe-bucket are
# flagged as a "conflict" suggesting calibration drift.
CONFLICT_THRESHOLD = 1.0


@dataclass
class AppendReport:
    rows_added: int = 0
    rows_updated: int = 0
    conflicts: list[dict] = field(default_factory=list)


def load_history(path: Path = HISTORY_CSV) -> pd.DataFrame:
    """Read the merged tilt history CSV.

    Returns an empty DataFrame (with the canonical schema) if the file
    doesn't exist. The merged history is derived from per-source CSVs by
    `ingest.pipeline.ingest_all()`; on a fresh checkout the Streamlit
    app's first call to `load_tilt()` will run the pipeline and populate
    this file before the UI tries to read it.
    """
    if not path.exists():
        return _empty_history()
    return _read_canonical(path)


def append_history(
    new_rows: pd.DataFrame,
    path: Path = HISTORY_CSV,
) -> AppendReport:
    """Merge new rows into the history CSV with dedupe + conflict detection.

    Args:
        new_rows: DataFrame with `[Date, Tilt (microradians)]`.
        path:     Where to write. Defaults to `config.HISTORY_CSV`.

    Returns:
        An AppendReport listing how many rows were added/updated and any
        bucket-level conflicts that were resolved by keeping the new value.
    """
    report = AppendReport()
    if new_rows is None or len(new_rows) == 0:
        return report

    new_rows = new_rows[[DATE_COL, TILT_COL]].copy()
    new_rows[DATE_COL] = pd.to_datetime(new_rows[DATE_COL])
    new_rows = new_rows.dropna(subset=[DATE_COL, TILT_COL])

    existing = load_history(path)

    # Tag rows by source so we can compute conflicts after the merge.
    existing = existing.assign(_source="existing")
    new_rows = new_rows.assign(_source="new")
    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined["_bucket"] = combined[DATE_COL].dt.round(DEDUPE_BUCKET)

    # Conflict detection: any bucket with both an existing and a new sample
    # whose tilts differ by more than CONFLICT_THRESHOLD.
    grouped = combined.groupby("_bucket", sort=False)
    for bucket, group in grouped:
        if len(group) < 2:
            continue
        sources = set(group["_source"])
        if "existing" not in sources or "new" not in sources:
            continue
        old = group[group["_source"] == "existing"][TILT_COL].iloc[-1]
        new = group[group["_source"] == "new"][TILT_COL].iloc[-1]
        if abs(float(new) - float(old)) > CONFLICT_THRESHOLD:
            report.conflicts.append(
                {
                    "bucket": bucket,
                    "existing_tilt": float(old),
                    "new_tilt": float(new),
                    "delta": float(new) - float(old),
                }
            )

    # Compute the bucket-level diff BEFORE deduplication so the report
    # numbers reflect what the user actually cares about: how many new
    # 15-minute buckets appeared, vs. how many existing buckets got
    # overwritten by a fresh re-trace. The previous formulation
    # (`len(deduped) - len(existing)`) overcounted in the presence of
    # intra-batch dedupes — multiple new rows landing in the same bucket
    # were each counted as if they had survived.
    existing_buckets = set(combined.loc[combined["_source"] == "existing", "_bucket"])
    new_buckets = set(combined.loc[combined["_source"] == "new", "_bucket"])
    report.rows_added = len(new_buckets - existing_buckets)
    report.rows_updated = len(new_buckets & existing_buckets)

    # Sort by Date so `keep="last"` keeps the most recently-arriving row per
    # bucket. We push existing rows first by leveraging the concat order plus
    # a stable sort.
    combined = combined.sort_values([DATE_COL, "_source"], kind="stable")
    deduped = combined.drop_duplicates(subset="_bucket", keep="last")
    deduped = deduped.drop(columns=["_bucket", "_source"])
    deduped = deduped.sort_values(DATE_COL).reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    deduped.to_csv(path, index=False)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────


def _read_canonical(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed", dayfirst=False)
    df = df[[DATE_COL, TILT_COL]].dropna()
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(
        {DATE_COL: pd.Series(dtype="datetime64[ns]"), TILT_COL: []}
    )
