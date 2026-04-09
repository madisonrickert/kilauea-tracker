"""Append-only tilt history cache.

The cache is just a CSV at `data/tilt_history.csv` with columns
`Date,Tilt (microradians)` — the same schema v1.0 used, so the bootstrap is
a literal file copy from `legacy/Tiltmeter Data - Sheet1.csv`.

Why CSV instead of parquet/sqlite/etc.: the file is tiny (~30 KB for 11 months
of v1.0 data; <2 MB for 5 years), CSV diffs cleanly in git so the user can
always see what new data the ingest pipeline pulled, and the format never
needs migration. The dedupe logic does all the heavy lifting.

Dedupe strategy:
  1. Concat existing + new rows.
  2. Round each Date to the nearest 15-minute bucket — fine enough to preserve
     real samples (USGS publishes ~hourly), coarse enough to merge near-duplicate
     re-traces.
  3. Drop duplicates on the rounded key, keeping the LAST occurrence — fresh
     re-calibration is allowed to overwrite older y-offsets.
  4. Detect "conflicts" where the new row's tilt disagrees with the cached row
     by more than `CONFLICT_THRESHOLD` µrad and emit a soft warning. The
     warning surfaces in the Streamlit UI so the user can spot calibration drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import HISTORY_CSV, LEGACY_BOOTSTRAP_CUTOFF, LEGACY_CSV
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
    """Read the local tilt history CSV.

    Bootstraps from `legacy/Tiltmeter Data - Sheet1.csv` on first run if the
    history file doesn't exist yet — gives v2.0 immediate parity with v1.0's
    11 months of input.

    The bootstrap respects `config.LEGACY_BOOTSTRAP_CUTOFF`: rows older than
    that date are dropped from the legacy import because the DEC2024_TO_NOW
    source provides denser, more uniform samples for that range. The legacy
    file on disk is left intact.
    """
    if not path.exists():
        if LEGACY_CSV.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            df = _read_canonical(LEGACY_CSV)
            if LEGACY_BOOTSTRAP_CUTOFF is not None:
                df = df[df[DATE_COL] >= LEGACY_BOOTSTRAP_CUTOFF].reset_index(drop=True)
            df.to_csv(path, index=False)
        else:
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
