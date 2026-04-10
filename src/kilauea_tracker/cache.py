"""Tilt CSV reader and per-source append helper.

This module hosts two responsibilities:

  1. `load_history()` — reads the merged tilt history CSV that the model
     and Streamlit UI consume. The merged history is a derived artifact
     produced by `ingest.pipeline.ingest_all()`; this function just reads
     whatever's on disk and returns an empty DataFrame if nothing is.

  2. `append_history()` — appends new rows to *any* tilt CSV with intra-
     file dedupe at 15-min buckets AND intra-source frame alignment.
     Used by `ingest.pipeline.ingest()` to write into per-source files
     at `data/sources/<source>.csv`. Each source's file is independent
     — there is no cross-source merging here; that happens later in
     `reconcile.reconcile_sources()`.

Frame alignment (the load-bearing piece):

  USGS auto-rescales the y-axis on its tilt PNGs whenever the data range
  shifts (e.g. during an active eruption when deflation pushes the plot
  off the previous y-bounds). Each fetch is independently calibrated
  against its own visible tick labels, so consecutive fetches from the
  same source can produce values for the same timestamp that differ by
  a roughly-constant offset — the "frame" of the calibration shifts.

  Without alignment, append_history's keep-latest dedupe would overwrite
  the overlap region with rows in the new frame while leaving rows
  *outside* the new PNG window in their old frame. After many fetches
  the per-source CSV becomes a quilt of N different y-frames stitched
  at fetch boundaries — exactly the drift the user observed in 2026-04
  during episode 44.

  The fix: BEFORE appending, compute the median (new − existing) y-delta
  across the overlap buckets and shift the new rows by that amount, so
  they slot into the existing CSV's y-frame. After alignment, the per-
  source CSV stays in ONE frame forever — the frame established by the
  very first fetch — and every subsequent fetch contributes its NEW
  information without contaminating that frame.

  Conceptually identical to what `reconcile.reconcile_sources()` does
  ACROSS sources; this is the WITHIN-source equivalent applied between
  successive fetches of the same PNG.

Dedupe strategy in `append_history`:
  1. Compute the intra-source frame offset (see above) and shift new rows.
  2. Concat existing + (shifted) new rows.
  3. Round each Date to the nearest 15-minute bucket.
  4. Drop duplicates on the rounded key, keeping the LAST occurrence.
  5. Detect "conflicts" where the new row's tilt disagrees with the
     cached row by more than `CONFLICT_THRESHOLD` µrad AFTER alignment.
     A conflict here is a real per-row anomaly (data spike, partial OCR
     misread for one row, etc.) — bulk frame shifts are absorbed by the
     alignment step and never reach the conflict detector.
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
# flagged as a "conflict" suggesting a real per-row anomaly. (Bulk frame
# shifts are absorbed by the alignment step before conflict detection runs,
# so a residual conflict is genuinely surprising.)
CONFLICT_THRESHOLD = 1.0

# Minimum number of overlap buckets required for the median frame offset
# to be considered well-determined. With 1 or 2 overlap buckets we still
# apply the offset (better than nothing) but flag it as low-confidence.
MIN_OVERLAP_FOR_HIGH_CONFIDENCE = 3

# Frame offset above which we log a warning. A real eruption-induced
# rescale of the USGS PNG y-axis can produce offsets in the 1-5 µrad
# range; anything bigger means the calibration probably drifted hard
# and the operator should eyeball the result.
LARGE_FRAME_OFFSET_MICRORAD = 5.0


@dataclass
class AppendReport:
    rows_added: int = 0
    rows_updated: int = 0
    conflicts: list[dict] = field(default_factory=list)
    # Frame alignment diagnostics. `frame_offset_microrad` is the median
    # (new − existing) y-delta computed across overlap buckets and applied
    # to the new rows BEFORE deduping. `frame_overlap_buckets` is how many
    # 15-min buckets contributed to the median. Together they let the
    # ingest pipeline surface "we corrected for a 0.6 µrad frame shift on
    # this fetch" diagnostics in the Streamlit warnings panel.
    frame_offset_microrad: float = 0.0
    frame_overlap_buckets: int = 0
    warnings: list[str] = field(default_factory=list)


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


def compute_intra_source_frame_offset(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
) -> tuple[float, int]:
    """Median (new − existing) y-delta across the overlap region.

    Both inputs are bucketed to `DEDUPE_BUCKET` and reduced to one row
    per bucket (latest within the bucket, matching what the dedupe step
    will do downstream). Buckets present in BOTH frames contribute one
    delta each; the median across those deltas is the returned offset.

    Returns `(0.0, 0)` when there is no overlap — the caller should
    leave new rows as-is in that case and (probably) log a warning,
    since appending without an anchor is the failure mode this whole
    function exists to prevent.

    Median (not mean) so a single wildly-wrong value in the overlap
    region can't drag the entire frame offset off-true.
    """
    if existing is None or new_rows is None:
        return 0.0, 0
    if len(existing) == 0 or len(new_rows) == 0:
        return 0.0, 0

    e = existing[[DATE_COL, TILT_COL]].copy()
    e[DATE_COL] = pd.to_datetime(e[DATE_COL])
    e = e.dropna(subset=[DATE_COL, TILT_COL])
    e["_bucket"] = e[DATE_COL].dt.round(DEDUPE_BUCKET)
    e = e.sort_values(DATE_COL).drop_duplicates(subset="_bucket", keep="last")

    n = new_rows[[DATE_COL, TILT_COL]].copy()
    n[DATE_COL] = pd.to_datetime(n[DATE_COL])
    n = n.dropna(subset=[DATE_COL, TILT_COL])
    n["_bucket"] = n[DATE_COL].dt.round(DEDUPE_BUCKET)
    n = n.sort_values(DATE_COL).drop_duplicates(subset="_bucket", keep="last")

    merged = e.merge(n, on="_bucket", suffixes=("_existing", "_new"))
    if len(merged) == 0:
        return 0.0, 0

    deltas = merged[f"{TILT_COL}_new"] - merged[f"{TILT_COL}_existing"]
    return float(deltas.median()), len(merged)


def append_history(
    new_rows: pd.DataFrame,
    path: Path = HISTORY_CSV,
) -> AppendReport:
    """Merge new rows into the history CSV with frame alignment + dedupe.

    Args:
        new_rows: DataFrame with `[Date, Tilt (microradians)]`.
        path:     Where to write. Defaults to `config.HISTORY_CSV`.

    Returns:
        An AppendReport listing how many rows were added/updated, the
        intra-source frame offset that was applied, the number of overlap
        buckets that determined it, any post-alignment per-row conflicts
        that survived the median, and free-text warnings (low-confidence
        alignment, no-overlap appends, large frame shifts).
    """
    report = AppendReport()
    if new_rows is None or len(new_rows) == 0:
        return report

    new_rows = new_rows[[DATE_COL, TILT_COL]].copy()
    new_rows[DATE_COL] = pd.to_datetime(new_rows[DATE_COL])
    new_rows = new_rows.dropna(subset=[DATE_COL, TILT_COL])

    existing = load_history(path)

    # ── Frame alignment ─────────────────────────────────────────────────
    # Anchor the new rows to whatever y-frame the existing CSV is already
    # in. This is the load-bearing piece that prevents intra-source drift
    # over many fetches; see the module docstring for the full rationale.
    offset, n_overlap = compute_intra_source_frame_offset(existing, new_rows)
    report.frame_offset_microrad = offset
    report.frame_overlap_buckets = n_overlap

    if n_overlap > 0:
        new_rows = new_rows.copy()
        new_rows[TILT_COL] = new_rows[TILT_COL] - offset

        if n_overlap < MIN_OVERLAP_FOR_HIGH_CONFIDENCE:
            report.warnings.append(
                f"frame offset of {offset:+.3f} µrad determined from only "
                f"{n_overlap} overlap bucket(s) — low confidence; new rows "
                f"applied anyway since the alternative is no anchor at all"
            )
        if abs(offset) >= LARGE_FRAME_OFFSET_MICRORAD:
            report.warnings.append(
                f"large intra-source frame shift of {offset:+.3f} µrad; "
                f"USGS may have rescaled this PNG's y-axis aggressively. "
                f"The shift was absorbed but eyeballing the source plot "
                f"is recommended."
            )
    elif len(existing) > 0:
        # Both sides have rows but no temporal overlap. We can't anchor
        # the new fetch's frame against anything; the only options are
        # (a) append in raw frame (introduces the drift this function
        # exists to prevent), or (b) drop the new rows (data loss). The
        # least-bad choice is (a) plus a loud warning so the operator
        # knows the resulting CSV may be quilted.
        report.warnings.append(
            f"no temporal overlap between {len(new_rows)} new rows and the "
            f"existing CSV at {path.name}; appending in raw frame (drift "
            f"risk — manual inspection recommended)"
        )

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
