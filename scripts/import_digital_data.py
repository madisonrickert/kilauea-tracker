"""One-shot import of the USGS UWD digital tiltmeter data.

The digital files are 1-minute samples of raw X-Y tilt from the Uēkahuna
LILY borehole tiltmeter for Jan–Jun 2025. They are *much* more accurate
than anything we trace out of the USGS plot images, but they are NOT live —
USGS publishes them as a one-time research release.

Data source (please cite if you republish):
    USGS ScienceBase, item 67ead922d34ed02007f83585
    https://www.sciencebase.gov/catalog/item/67ead922d34ed02007f83585

This script:
  1. Reads the 6 segment CSVs (file boundaries are at instrument relevelings).
  2. Projects raw (X, Y) onto compass azimuth 300° to match the convention
     used by the live USGS plots:
         tilt_300 = Xtilt * sin(300°) + Ytilt * cos(300°)
  3. Resamples each segment to 30-minute means (1-minute spacing is overkill
     for our model and would balloon the committed CSV).
  4. Tags each row with `segment` (1..6) so the reconcile layer can align
     each releveling-bounded segment independently if it ever needs to.
  5. Writes the result to `data/uwd_digital_az300.csv` for the pipeline.

Place the raw `UWD_*.csv` segment files inside `data/raw/uwd_digital/` and
run the script with no arguments. The raw directory is gitignored because
the source files are large and the *processed* CSV (~350 KB) is what
actually feeds the pipeline. If you keep the raw files somewhere else,
pass `--source DIR` to point at them.

Run with:
    uv run python scripts/import_digital_data.py [--source DIR]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# Default to a project-relative path so the script is self-contained — no
# user-specific Downloads paths leaking into the app. The directory is
# gitignored; drop the raw segment files here before running.
DEFAULT_SOURCE = REPO_ROOT / "data" / "raw" / "uwd_digital"
OUTPUT_PATH = REPO_ROOT / "data" / "uwd_digital_az300.csv"

# Resample period — 30 min is fine grained enough to capture the model's
# saturation curve dynamics while keeping the file size sane (~12K rows).
RESAMPLE_PERIOD = "30min"

# Az 300° projection convention. The USGS plots show the tilt component along
# the 300°-from-North direction (clockwise). With X = "East" and Y = "North"
# (which the README's `Az=0` translation equation implies), the projection is:
#     tilt_along_az = X*sin(az) + Y*cos(az)
# (sin/cos swapped from the standard math convention because compass azimuth
# is measured clockwise from North, not counterclockwise from East.)
AZ_300_RADIANS = math.radians(300)
SIN_AZ = math.sin(AZ_300_RADIANS)
COS_AZ = math.cos(AZ_300_RADIANS)


def _project_az_300(df: pd.DataFrame) -> pd.Series:
    return df["Xtilt(microrad)"] * SIN_AZ + df["Ytilt(microrad)"] * COS_AZ


def _read_segment(path: Path, segment_id: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["UTCDateTime"] = pd.to_datetime(df["UTCDateTime"]).dt.tz_localize(None)

    out = pd.DataFrame(
        {
            "Date": df["UTCDateTime"],
            "Tilt (microradians)": _project_az_300(df),
        }
    )

    # Resample to RESAMPLE_PERIOD means
    out = (
        out.set_index("Date")
        .resample(RESAMPLE_PERIOD)
        .mean()
        .dropna()
        .reset_index()
    )
    out["segment"] = segment_id
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=(
            "Directory containing UWD_*.csv segment files. Defaults to the "
            "project-relative path data/raw/uwd_digital/ — drop the raw "
            "files there to keep the import self-contained."
        ),
    )
    args = parser.parse_args()

    source: Path = args.source
    if not source.exists():
        rel = (
            source.relative_to(REPO_ROOT)
            if source.is_absolute() and REPO_ROOT in source.parents
            else source
        )
        print(f"ERROR: source directory does not exist: {rel}")
        print(
            "       Create it and place the UWD_*.csv segment files inside, "
            "or pass --source DIR to point at them."
        )
        return 1

    files = sorted(source.glob("UWD_*.csv"))
    if not files:
        print(f"ERROR: no UWD_*.csv files found in {source}")
        print(
            "       The raw segment files are named UWD_<segment>.csv. "
            "Check that the directory contains them."
        )
        return 1

    print(f"Reading {len(files)} segment file(s) from {source}…")
    segments = []
    for i, f in enumerate(files, start=1):
        seg = _read_segment(f, segment_id=i)
        print(
            f"  segment {i}: {f.name}  "
            f"raw rows ≈ {sum(1 for _ in open(f)) - 1}, "
            f"resampled rows = {len(seg)}, "
            f"range {seg['Date'].min()} → {seg['Date'].max()}, "
            f"tilt(Az300°) {seg['Tilt (microradians)'].min():.2f} → {seg['Tilt (microradians)'].max():.2f}"
        )
        segments.append(seg)

    combined = pd.concat(segments, ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(
        f"\nWrote {len(combined)} rows ({OUTPUT_PATH.stat().st_size if OUTPUT_PATH.exists() else 0} bytes) "
        f"to {OUTPUT_PATH.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
