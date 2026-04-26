"""Backtest harness — CLI wrapper around ``kilauea_tracker.backtest``.

For each detected complete inflation segment in the recent history,
fit each registered model on the first FRACTION of the segment, project
forward, find when each model says the next pulse will land, and
compare to the segment's actual peak time. Prints a per-model error
table at multiple data fractions plus a coverage column (how often
each model produced any prediction at all).

Usage:
    uv run python scripts/compare_models.py
    uv run python scripts/compare_models.py --n-episodes 10
    uv run python scripts/compare_models.py --fractions 0.1 0.3 0.5 0.7 0.9

The pure backtest logic lives in ``src/kilauea_tracker/backtest.py`` so
the same computation feeds the in-app Backtest page.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kilauea_tracker.backtest import (
    DEFAULT_N_SEGMENTS,
    DEFAULT_QUARTILES,
    BacktestResult,
    run_backtest,
)
from kilauea_tracker.config import HISTORY_CSV
from kilauea_tracker.model import DATE_COL


def _print_per_quartile_table(result: BacktestResult) -> None:
    n = len(result.segments)
    print()
    print(f"Backtest across {n} complete inflation segments")
    print(
        f"Quartiles (cumulative data fractions): "
        f"{', '.join(f'{f:.0%}' for f in result.fractions)}"
    )
    print()
    print("Per-quartile detail — median |error| and signed median in HOURS;")
    print("coverage = number of segments where the model produced any prediction.")
    print()

    for f in result.fractions:
        print(f"=== Quartile @ {f:.0%} of inflation phase ===")
        header = (
            f"  {'model':<22} {'cov':>7}  {'med_|err|':>10}  "
            f"{'med_signed':>11}  {'mean_signed':>12}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        rows: list[tuple[float, str]] = []
        for mid in result.model_ids:
            s = result.stats(mid, f)
            cov = f"{s.n_predictions}/{s.n_segments}"
            if s.median_abs_error_h is None:
                row = (
                    f"  {mid:<22} {cov:>7}  "
                    f"{'—':>10}  {'—':>11}  {'—':>12}"
                )
                rank_key = float("inf")
            else:
                row = (
                    f"  {mid:<22} {cov:>7}  "
                    f"{s.median_abs_error_h:>9.1f}h  "
                    f"{s.median_signed_error_h:>+10.1f}h  "
                    f"{s.mean_signed_error_h:>+11.1f}h"
                )
                rank_key = (
                    s.median_abs_error_h if s.coverage >= 0.5 else float("inf")
                )
            rows.append((rank_key, row))
        rows.sort(key=lambda r: r[0])
        for _, row in rows:
            print(row)
        print()


def _print_per_model_table(result: BacktestResult) -> None:
    n = len(result.segments)
    print("=== Per-model trajectory across quartiles ===")
    header = "  " + f"{'model':<22}" + "".join(
        f"{int(f * 100):>7}%" for f in result.fractions
    ) + "  (median |err| in hours; '—' = no prediction)"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for mid in result.model_ids:
        cells = [f"  {mid:<22}"]
        for f in result.fractions:
            s = result.stats(mid, f)
            if s.median_abs_error_h is None:
                cells.append(f"{'—':>7}")
            else:
                cells.append(f"{s.median_abs_error_h:>6.0f}h")
        print("".join(cells))
    print(f"  (across {n} segments)")
    print()


def _print_recommendations(result: BacktestResult) -> None:
    print("=== Recommended model by quartile ===")
    print(
        f"  {'quartile':<10}  {'best model':<22} {'med_|err|':>10}  "
        f"{'coverage':>10}"
    )
    print("  " + "-" * 60)
    for f, best in result.best_per_quartile().items():
        if best is None:
            print(
                f"  {f:>9.0%}   {'(none qualifying)':<22} "
                f"{'—':>10}  {'—':>10}"
            )
            continue
        cov = f"{best.n_predictions}/{best.n_segments}"
        print(
            f"  {f:>9.0%}   {best.model_id:<22} "
            f"{best.median_abs_error_h:>9.1f}h  {cov:>10}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=HISTORY_CSV,
        help=f"Path to tilt history CSV (default: {HISTORY_CSV})",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=DEFAULT_N_SEGMENTS,
        help="How many recent inflation segments to backtest against",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_QUARTILES),
        help="Data fractions (quartiles) to evaluate at",
    )
    args = parser.parse_args()

    tilt = (
        pd.read_csv(args.csv, parse_dates=[DATE_COL])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    print(f"Backtesting against {args.csv}")
    result = run_backtest(
        tilt, n_segments=args.n_episodes, fractions=tuple(args.fractions)
    )
    print(f"Found {len(result.segments)} complete inflation segments:")
    for s in result.segments:
        print(
            f"  {s.trough_date.strftime('%Y-%m-%d %H:%M')} → "
            f"{s.peak_date.strftime('%Y-%m-%d %H:%M')}  "
            f"({s.duration_hours:.1f} h)"
        )

    _print_per_quartile_table(result)
    _print_per_model_table(result)
    _print_recommendations(result)
    print(
        "Sign convention: signed error > 0 ⇒ predicted LATER than the actual "
        "peak; signed error < 0 ⇒ predicted EARLIER. Coverage is "
        "segments-with-prediction / total."
    )


if __name__ == "__main__":
    main()
