"""Plot `dec2024_to_now` (or any PNG source) tilt against `digital` tilt over
their temporal overlap.

Why this exists (Phase 0c of the v3 alignment rewrite): the new anchor
cross-check (`calibrate.recalibrate_by_anchor_fit`) assumes the relationship
between each PNG-traced source and digital is *linear*:
`digital = a * png + b`. That assumption is load-bearing — if it turns out
the PNG has non-linear distortion (e.g., a y-axis scale change mid-window),
the whole calibration-via-regression approach is flawed.

This script pairs up the two series at a shared bucket granularity and
produces a scatter plot plus the fitted line. Eyeballing the scatter is
the cheapest way to confirm linearity before committing to the regression
correction.

Usage:
    uv run python scripts/scatter_anchor_overlap.py
    uv run python scripts/scatter_anchor_overlap.py --source week
    uv run python scripts/scatter_anchor_overlap.py --out /tmp/scatter.png

Output: a PNG at the given path (defaults to `data/anchor_scatter_<src>.png`)
plus a one-line stdout summary with the fitted `(a, b)` and residual std.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kilauea_tracker.config import (
    ANCHOR_FIT_MIN_OVERLAP_BUCKETS,
    ANCHOR_FIT_TRIM_HOURS,
    DATA_DIR,
    DIGITAL_CSV,
    SOURCES_DIR,
)
from kilauea_tracker.ingest.calibrate import recalibrate_by_anchor_fit
from kilauea_tracker.model import DATE_COL, TILT_COL


def _read_canonical_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed", dayfirst=False)
    df = df[[DATE_COL, TILT_COL]].dropna()
    return df.sort_values(DATE_COL).reset_index(drop=True)


def main(
    source: str, out_path: Path | None, bucket_freq: str
) -> int:
    src_csv = SOURCES_DIR / f"{source}.csv"
    if not src_csv.exists():
        print(f"ERROR: source CSV not found: {src_csv}")
        return 2
    if not DIGITAL_CSV.exists():
        print(f"ERROR: digital CSV not found: {DIGITAL_CSV}")
        return 2

    src = _read_canonical_csv(src_csv)
    dig = _read_canonical_csv(DIGITAL_CSV)
    print(
        f"loaded {source}: {len(src):,} rows "
        f"[{src[DATE_COL].min()} → {src[DATE_COL].max()}]"
    )
    print(
        f"loaded digital: {len(dig):,} rows "
        f"[{dig[DATE_COL].min()} → {dig[DATE_COL].max()}]"
    )

    fit = recalibrate_by_anchor_fit(source, src, dig, bucket_freq=bucket_freq)
    if not fit.ran:
        print(f"fit did not run: {fit.note}")
        return 1

    # Rebuild the paired samples the fit used so we can plot them with
    # the Huber regression line overlaid.
    dig_start = dig[DATE_COL].min() + pd.Timedelta(hours=ANCHOR_FIT_TRIM_HOURS)
    dig_end = dig[DATE_COL].max() - pd.Timedelta(hours=ANCHOR_FIT_TRIM_HOURS)
    dig_trim = dig[(dig[DATE_COL] >= dig_start) & (dig[DATE_COL] <= dig_end)]
    src_b = (
        src.assign(_b=lambda d: d[DATE_COL].dt.floor(bucket_freq))
        .groupby("_b")[TILT_COL]
        .mean()
    )
    dig_b = (
        dig_trim.assign(_b=lambda d: d[DATE_COL].dt.floor(bucket_freq))
        .groupby("_b")[TILT_COL]
        .mean()
    )
    overlap = src_b.index.intersection(dig_b.index)
    x = src_b.loc[overlap].to_numpy(dtype=float)
    y = dig_b.loc[overlap].to_numpy(dtype=float)

    print(
        f"\nfit: digital = {fit.a:.5f} · {source} + {fit.b:+.4f}   "
        f"residual_std={fit.residual_std_microrad:.3f} µrad   "
        f"n={fit.overlap_buckets} buckets (≥{ANCHOR_FIT_MIN_OVERLAP_BUCKETS} "
        f"required)"
    )
    if fit.warning:
        print(f"WARN: {fit.warning}")

    # Quadratic fit for comparison — if it's materially better than linear,
    # the PNG has non-linear distortion and the whole anchor-regression
    # approach is suspect.
    poly2 = np.polyfit(x, y, 2)
    linear = np.polyfit(x, y, 1)
    resid_linear = y - np.polyval(linear, x)
    resid_quad = y - np.polyval(poly2, x)
    std_linear = float(np.std(resid_linear))
    std_quad = float(np.std(resid_quad))
    print(
        f"residual std: linear={std_linear:.3f} µrad  "
        f"quadratic={std_quad:.3f} µrad  "
        f"improvement={100 * (std_linear - std_quad) / std_linear:.1f}%"
    )
    if std_quad < 0.6 * std_linear:
        print("⚠️  quadratic fits materially better than linear — PNG may have "
              "non-linear distortion; anchor cross-check assumption is shaky")

    # Plot — try matplotlib, fall back gracefully if unavailable.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available; skipping plot)")
        return 0

    out = out_path or (DATA_DIR / f"anchor_scatter_{source}.png")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=4, alpha=0.4, label="paired buckets")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, fit.a * xs + fit.b, "r-", linewidth=1.5, label=f"huber: y = {fit.a:.4f}x + {fit.b:+.3f}")
    ax.plot(xs, xs, "k--", linewidth=0.8, alpha=0.5, label="identity")
    ax.set_xlabel(f"{source} tilt (µrad)")
    ax.set_ylabel("digital tilt (µrad)")
    ax.set_title(
        f"Anchor overlap: {source} vs digital  "
        f"(n={fit.overlap_buckets}, resid_std={fit.residual_std_microrad:.2f} µrad)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="dec2024_to_now",
        help="source name (default: dec2024_to_now)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output PNG path"
    )
    parser.add_argument(
        "--bucket", default="1h", help="bucket frequency (default: 1h)"
    )
    args = parser.parse_args()
    raise SystemExit(main(args.source, args.out, args.bucket))
