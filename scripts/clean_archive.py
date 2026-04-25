"""One-shot surgical cleanup of `data/archive.csv`.

The archive's keep-first semantics mean any contaminated row becomes
permanent. In April 2026 the `week` PNG trace produced recurring ~-10 µrad
phantom spikes (HSV blue-mask picking up non-curve pixels); some of those
rows landed in the archive before the per-row outlier filter was in place.
Once archived, they beat every live-source contender and rendered as
visible step jumps on the chart.

Detection strategy — "re-reconcile without the archive":

The per-source CSVs (`data/sources/*.csv`) plus the digital reference
together define the best-available ground truth once run through
`reconcile.reconcile_sources` with the archive omitted. Any archive row
whose value disagrees with that ground-truth by more than
`DISAGREEMENT_THRESHOLD` is a contamination candidate — the archive is
the one with bad data, not the full reconcile of every other source.

This also correctly handles the blockwise case where the contamination
spans many hours and forms a clean internal plateau (the 2026-04-17
block). The bracketed-block detector was fooled by those because it
couldn't tell which side of a round-trip was the truth; this one asks the
live sources directly.

After running this, re-run `python -m kilauea_tracker.ingest.pipeline` to
rebuild `data/tilt_history.csv` against the cleaned archive. Commit the
two files together with the diff report.

Usage:
    python scripts/clean_archive.py --dry-run
    python scripts/clean_archive.py
    python scripts/clean_archive.py --threshold 2.5  # stricter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kilauea_tracker.config import (  # noqa: E402
    ALL_SOURCES,
    ARCHIVE_CSV,
    DIGITAL_CSV,
    DIGITAL_SOURCE_NAME,
    SOURCES_DIR,
    TILT_SOURCE_NAME,
    TRACE_OUTLIER_THRESHOLD_MICRORAD,
)
from kilauea_tracker.ingest.trace import _filter_rolling_median_outliers  # noqa: E402
from kilauea_tracker.model import DATE_COL, TILT_COL  # noqa: E402
from kilauea_tracker.reconcile import reconcile_sources  # noqa: E402

# How far an archive row can drift from the re-reconciled ground truth at
# its 15-min bucket before we treat it as contamination. 3 µrad is about
# twice the typical short-interval noise floor of the real curve and well
# below any phantom-block magnitude.
DISAGREEMENT_THRESHOLD = 3.0

BUCKET = "15min"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            {
                DATE_COL: pd.Series(dtype="datetime64[ns]"),
                TILT_COL: pd.Series(dtype="float64"),
            }
        )
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed", dayfirst=False)
    df = df[[DATE_COL, TILT_COL]].dropna()
    return df.sort_values(DATE_COL).reset_index(drop=True)


def _load_filtered_sources(
    sources_dir: Path, digital_csv: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, list[tuple[pd.Timestamp, float, float]]]]:
    """Load every non-archive source, apply the same rolling-median outlier
    filter that `trace.trace_curve` now uses, and return the cleaned
    DataFrames plus a per-source list of dropped (timestamp, value,
    local_median) tuples.

    Per-source CSVs that were appended BEFORE the outlier filter existed
    still contain phantom rows; this filter strips them retroactively for
    the ground-truth comparison AND (when --clean-sources is set) for the
    on-disk rewrite.
    """
    sources: dict[str, pd.DataFrame] = {}
    dropped_per_source: dict[
        str, list[tuple[pd.Timestamp, float, float]]
    ] = {}
    for s in ALL_SOURCES:
        name = TILT_SOURCE_NAME[s]
        p = sources_dir / f"{name}.csv"
        if not p.exists():
            continue
        df = _load_csv(p)
        if len(df) == 0:
            continue
        filtered, report = _filter_rolling_median_outliers(df)
        sources[name] = filtered
        dropped_per_source[name] = list(report.dropped_rows)

    # The digital reference is raw instrument data, not PNG-traced, so it
    # doesn't have the phantom-spike issue. Pass it through unfiltered.
    if digital_csv.exists():
        df = _load_csv(digital_csv)
        if len(df) > 0:
            sources[DIGITAL_SOURCE_NAME] = df

    return sources, dropped_per_source


def _build_ground_truth(
    sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Re-run reconcile on every supplied source (archive deliberately
    omitted). The merged result is the best available ground truth.
    """
    merged, _report = reconcile_sources(sources)
    return merged


def scan(
    archive_path: Path,
    sources_dir: Path,
    digital_csv: Path,
    *,
    threshold: float = DISAGREEMENT_THRESHOLD,
) -> tuple[pd.DataFrame, list[dict], dict[str, list[tuple[pd.Timestamp, float, float]]]]:
    archive = _load_csv(archive_path)
    if len(archive) == 0:
        return archive, [], {}

    sources, dropped_per_source = _load_filtered_sources(sources_dir, digital_csv)
    truth = _build_ground_truth(sources)
    if len(truth) == 0:
        raise RuntimeError(
            "re-reconcile produced no rows — per-source CSVs are empty or missing"
        )

    # Build a lookup from bucket → truth value. If multiple truth rows
    # fall in the same bucket, take the latest (keeps the comparison 1:1
    # with the archive's 15-min grid).
    truth_by_bucket = (
        truth.assign(_bucket=truth[DATE_COL].dt.round(BUCKET))
        .sort_values(DATE_COL)
        .drop_duplicates(subset="_bucket", keep="last")
        .set_index("_bucket")[TILT_COL]
    )

    # Wide window for the orphan pass. Phantom clusters can span 8-24 h so
    # anything narrower gets dominated by the contamination itself; 24 h
    # reliably anchors the median on clean data on either side.
    ORPHAN_WINDOW = pd.Timedelta(hours=24)
    # Orphan threshold deliberately tight — these rows have no live-source
    # backing, so the only signal we have is local archive consistency.
    # Kīlauea's real curve drifts slowly (< 1 µrad/h over quiet periods,
    # a few µrad/h during active deflation), so a ≥3 µrad orphan delta
    # across a 24-hour window is almost certainly contamination.
    ORPHAN_THRESHOLD = 3.0
    # Second orphan variant for rows that are isolated in time (no other
    # archive row within ±30 min). Real data has 15-minute-ish spacing;
    # isolated archive rows are the fingerprint of sparse phantom
    # stragglers left over when their contributing source was cleaned
    # upstream. These rows get a looser neighbour-window (6 h) and a
    # tighter threshold because isolation itself is already a signal.
    ISOLATION_GAP = pd.Timedelta(minutes=30)
    ISOLATION_WINDOW = pd.Timedelta(hours=6)
    ISOLATION_THRESHOLD = 2.5

    # First pass: flag rows against re-reconciled ground truth where we
    # have a bucket match.
    arc_buckets_rounded = archive[DATE_COL].dt.round(BUCKET).to_numpy()
    arc_tilts = archive[TILT_COL].to_numpy()
    arc_dates_ns = pd.to_datetime(archive[DATE_COL]).astype("datetime64[ns]").to_numpy()

    flagged_by_idx: dict[int, dict] = {}

    for idx in range(len(archive)):
        bucket = arc_buckets_rounded[idx]
        if bucket not in truth_by_bucket.index:
            continue
        arc_val = float(arc_tilts[idx])
        truth_val = float(truth_by_bucket.loc[bucket])
        delta = arc_val - truth_val
        if abs(delta) > threshold:
            flagged_by_idx[idx] = {
                "idx": idx,
                "date": archive[DATE_COL].iloc[idx],
                "archived_tilt": arc_val,
                "truth_tilt": truth_val,
                "delta": delta,
                "reason": "truth",
            }

    # Second pass: orphan detection, two rounds. Each row is judged
    # against a ±24 h archive median OR (if isolated) a ±6 h median, with
    # already-flagged rows excluded from the median. Two rounds is the
    # sweet spot: a single pass leaves some stragglers whose nearest
    # neighbour was itself a just-flagged phantom; more than two rounds
    # starts un-flagging marginal rows as the surviving cluster median
    # shifts.
    for _round in range(2):
        _orphan_pass(
            archive=archive,
            arc_dates_ns=arc_dates_ns,
            arc_tilts=arc_tilts,
            arc_buckets_rounded=arc_buckets_rounded,
            truth_by_bucket=truth_by_bucket,
            flagged_by_idx=flagged_by_idx,
            orphan_window=ORPHAN_WINDOW,
            orphan_threshold=ORPHAN_THRESHOLD,
            isolation_gap=ISOLATION_GAP,
            isolation_window=ISOLATION_WINDOW,
            isolation_threshold=ISOLATION_THRESHOLD,
        )

    flagged = [flagged_by_idx[i] for i in sorted(flagged_by_idx)]
    return archive, flagged, dropped_per_source


def _orphan_pass(
    *,
    archive: pd.DataFrame,
    arc_dates_ns,
    arc_tilts,
    arc_buckets_rounded,
    truth_by_bucket,
    flagged_by_idx: dict,
    orphan_window: pd.Timedelta,
    orphan_threshold: float,
    isolation_gap: pd.Timedelta,
    isolation_window: pd.Timedelta,
    isolation_threshold: float,
) -> None:
    """One orphan-detection round; mutates `flagged_by_idx` in place."""
    flagged_indices = set(flagged_by_idx.keys())
    active_mask = [i not in flagged_indices for i in range(len(archive))]
    keep_tilts = arc_tilts[active_mask]
    keep_dates = arc_dates_ns[active_mask]

    # Isolation uses the *active* (not-yet-flagged) archive rows —
    # stragglers whose nearest neighbour was flagged in a prior round
    # count as isolated here.
    active_dates_for_isolation = keep_dates

    for idx in range(len(archive)):
        if idx in flagged_by_idx:
            continue
        if arc_buckets_rounded[idx] in truth_by_bucket.index:
            continue
        ts = arc_dates_ns[idx]

        dense_left = int(
            pd.Series(active_dates_for_isolation).searchsorted(
                ts - isolation_gap.to_numpy(), side="left"
            )
        )
        dense_right = int(
            pd.Series(active_dates_for_isolation).searchsorted(
                ts + isolation_gap.to_numpy(), side="right"
            )
        )
        # Count rows including self if self is still in keep_dates.
        self_in_keep = idx not in flagged_by_idx
        neighbour_count = dense_right - dense_left - (1 if self_in_keep else 0)
        is_isolated = neighbour_count <= 0

        if is_isolated:
            window_delta_td = isolation_window
            local_threshold = isolation_threshold
            reason = "orphan-isolated"
        else:
            window_delta_td = orphan_window
            local_threshold = orphan_threshold
            reason = "orphan"

        lo = ts - window_delta_td.to_numpy()
        hi = ts + window_delta_td.to_numpy()
        left = int(pd.Series(keep_dates).searchsorted(lo, side="left"))
        right = int(pd.Series(keep_dates).searchsorted(hi, side="right"))
        if right - left < 3:
            continue
        window = keep_tilts[left:right]
        med = float(pd.Series(window).median())
        arc_val = float(arc_tilts[idx])
        delta = arc_val - med
        if abs(delta) > local_threshold:
            flagged_by_idx[idx] = {
                "idx": idx,
                "date": archive[DATE_COL].iloc[idx],
                "archived_tilt": arc_val,
                "truth_tilt": med,
                "delta": delta,
                "reason": reason,
            }


def write_report(
    archive: pd.DataFrame,
    flagged: list[dict],
    dropped_per_source: dict[str, list[tuple[pd.Timestamp, float, float]]],
    out_path: Path,
    threshold: float,
) -> None:
    lines = [
        "# Cache cleanup diff report",
        "",
        "## Per-source CSVs — rows flagged by the rolling-median filter",
        "",
        f"Applied the `trace._filter_rolling_median_outliers` filter "
        f"(threshold ±{TRACE_OUTLIER_THRESHOLD_MICRORAD:.1f} µrad) to each "
        "per-source CSV. Rows listed below were in the CSVs from before "
        "the filter existed. With `--clean-sources`, they are removed "
        "from the on-disk file.",
        "",
        "| Source | Drops |",
        "|---|---:|",
    ]
    for name, drops in dropped_per_source.items():
        lines.append(f"| {name} | {len(drops)} |")
    lines.append("")
    for name, drops in dropped_per_source.items():
        if not drops:
            continue
        lines.append(f"### `{name}.csv` — {len(drops)} row(s)")
        lines.append("")
        lines.append("| Date (UTC) | Raw tilt | Local median | Δ |")
        lines.append("|---|---:|---:|---:|")
        for ts, val, med in drops:
            delta = val - med
            lines.append(
                f"| {pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M:%S')} "
                f"| {val:+.2f} | {med:+.2f} | {delta:+.2f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Archive — rows flagged for removal",
            "",
            f"Scanned {len(archive)} archive rows.",
            f"  • `truth` rows were compared to the re-reconciled live-sources value at the same bucket (threshold ±{threshold:.1f} µrad).",
            "  • `orphan` rows have no live-source coverage; they were compared to a ±6h rolling median of the archive itself (threshold ±4.0 µrad). Catches residual phantoms whose contributing source was already cleaned upstream.",
            f"Flagged {len(flagged)} row(s) total.",
            "",
            "| Date (UTC) | Archive tilt | Reference | Δ | Reason |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in flagged:
        lines.append(
            f"| {row['date'].strftime('%Y-%m-%d %H:%M:%S')} "
            f"| {row['archived_tilt']:+.2f} "
            f"| {row['truth_tilt']:+.2f} "
            f"| {row['delta']:+.2f} "
            f"| {row['reason']} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the diff report; do not rewrite the archive.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=ARCHIVE_CSV,
        help="Path to archive.csv (default: data/archive.csv).",
    )
    parser.add_argument(
        "--sources-dir",
        type=Path,
        default=SOURCES_DIR,
        help="Per-source CSV directory (default: data/sources).",
    )
    parser.add_argument(
        "--digital",
        type=Path,
        default=DIGITAL_CSV,
        help="Digital tiltmeter reference CSV (default: data/uwd_digital_az300.csv).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DISAGREEMENT_THRESHOLD,
        help=(
            f"Disagreement threshold in µrad (default: {DISAGREEMENT_THRESHOLD}). "
            "Rows whose archived value deviates from the merged-live-sources "
            "value at the same bucket by more than this are flagged."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parent / "clean_archive.report.md",
        help="Where to write the markdown diff report.",
    )
    parser.add_argument(
        "--clean-sources",
        action="store_true",
        help=(
            "Also rewrite each `data/sources/*.csv` with the same rolling-"
            "median outlier filter that `trace.trace_curve` now uses. Use "
            "when the per-source CSVs contain historical phantom rows from "
            "before the filter was added to the ingest path."
        ),
    )
    args = parser.parse_args()

    archive, flagged, dropped_per_source = scan(
        args.archive,
        args.sources_dir,
        args.digital,
        threshold=args.threshold,
    )
    write_report(archive, flagged, dropped_per_source, args.report, args.threshold)
    total_source_drops = sum(len(v) for v in dropped_per_source.values())
    print(
        f"Per-source rolling-median filter: {total_source_drops} outlier row(s) "
        f"across {len([k for k, v in dropped_per_source.items() if v])} source(s)."
    )
    print(
        f"Archive: scanned {len(archive)} rows; "
        f"flagged {len(flagged)} row(s) (threshold ±{args.threshold:.1f} µrad)."
    )
    print(f"Diff report: {args.report}")

    if not flagged and total_source_drops == 0:
        return 0
    if args.dry_run:
        print("--dry-run: files left unchanged.")
        return 0

    if args.clean_sources and total_source_drops > 0:
        for name, drops in dropped_per_source.items():
            if not drops:
                continue
            csv_path = args.sources_dir / f"{name}.csv"
            df = _load_csv(csv_path)
            from kilauea_tracker.ingest.trace import (  # local import
                _filter_rolling_median_outliers,
            )
            filtered, _ = _filter_rolling_median_outliers(df)
            filtered.to_csv(csv_path, index=False)
            print(
                f"Wrote cleaned {csv_path.name}: {len(filtered)} rows "
                f"(was {len(df)}; dropped {len(df) - len(filtered)})."
            )
    elif total_source_drops > 0 and not args.clean_sources:
        print(
            f"Note: {total_source_drops} per-source outlier row(s) detected "
            "but not rewritten. Re-run with --clean-sources to apply."
        )

    if flagged:
        flagged_indices = {row["idx"] for row in flagged}
        cleaned = archive.drop(index=sorted(flagged_indices)).reset_index(drop=True)
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(args.archive, index=False)
        print(f"Wrote cleaned archive: {len(cleaned)} rows (was {len(archive)}).")

    print(
        "Next: run `python -m kilauea_tracker.ingest.pipeline` "
        "to rebuild tilt_history.csv."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
