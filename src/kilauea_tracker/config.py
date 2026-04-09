"""Centralized configuration: USGS source URLs, file paths, and defaults.

The four USGS PNG URLs are intentionally hardcoded rather than templated because
USGS uses an inconsistent naming scheme (`UWD-POC-TILT-*` for week/month, plain
`UWD-TILT-*` for 2day/3month).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
HISTORY_CSV = DATA_DIR / "tilt_history.csv"
LEGACY_CSV = PROJECT_ROOT / "legacy" / "Tiltmeter Data - Sheet1.csv"
LAST_GOOD_CALIBRATION = DATA_DIR / "last_good_calibration.json"
# Pre-processed digital tiltmeter data — produced once by
# `scripts/import_digital_data.py` from USGS's research-release CSVs.
# This is the canonical reference for Jan-Jun 2025; the ingest pipeline
# anchors everything else to it.
DIGITAL_CSV = DATA_DIR / "uwd_digital_az300.csv"

# Per-source raw storage. Each ingest source writes its raw traced rows to
# its own CSV here. The merged tilt_history.csv is then *derived* from these
# files via reconcile.reconcile_sources(). Storing raw inputs separately
# from the merged view means reconciliation is a pure function of the raw
# inputs and can be re-run without re-fetching from USGS.
SOURCES_DIR = DATA_DIR / "sources"


class TiltSource(Enum):
    """The five USGS tilt PNG captures we ingest."""

    TWO_DAY = "2day"
    WEEK = "week"
    MONTH = "month"
    THREE_MONTH = "3month"
    DEC2024_TO_NOW = "dec2024_to_now"


# Hardcoded URLs — naming inconsistency forces this (see module docstring).
USGS_TILT_URLS: dict[TiltSource, str] = {
    TiltSource.TWO_DAY: "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-TILT-2day.png",
    TiltSource.WEEK: "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-POC-TILT-week.png",
    TiltSource.MONTH: "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-POC-TILT-month.png",
    TiltSource.THREE_MONTH: "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-TILT-3month.png",
    TiltSource.DEC2024_TO_NOW: "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-TILT-Dec2024_to_now.png",
}

# Iteration order for `ingest_all`. Order is purely cosmetic now: each source
# writes to its own per-source CSV, and the merged history is rebuilt by
# reconcile.reconcile_sources() afterwards. Reconciliation uses SOURCE_PRIORITY
# (below) — not iteration order — to decide which source wins each bucket.
ALL_SOURCES: tuple[TiltSource, ...] = (
    TiltSource.THREE_MONTH,
    TiltSource.MONTH,
    TiltSource.WEEK,
    TiltSource.TWO_DAY,
    TiltSource.DEC2024_TO_NOW,
)

# Source identifiers for the reconciliation layer. These are strings (not
# TiltSource enum members) because two of the sources — `digital` and `legacy`
# — aren't USGS PNG captures and so don't fit the TiltSource enum. The strings
# are also the file stems for the per-source CSVs in SOURCES_DIR/.
#
# Priority order (highest first): digital is the gold reference because it's
# raw USGS instrument data (100% confidence) but only covers Jan-Jun 2025.
# The dense PNG sources rank by inverse window length — TWO_DAY has the
# highest sample density and the smallest legend overlap risk; THREE_MONTH
# is at the bottom because (a) it's the lowest-resolution image and (b)
# the 3-month plot is the only one whose curve actually passes through the
# legend region. Legacy data is hand-traced (medium confidence). DEC2024_TO_NOW
# is the longest-coverage image source so it slots above THREE_MONTH but
# below the dense recent sources.
#
# This order is consulted by the *merge* step in reconcile.py: when two
# sources both contain data for the same 15-min bucket, the one earlier in
# this tuple wins.
SOURCE_PRIORITY: tuple[str, ...] = (
    "digital",
    "two_day",
    "week",
    "month",
    "legacy",
    "dec2024_to_now",
    "three_month",
)

# Alignment order — DIFFERENT from SOURCE_PRIORITY. Alignment is a topological
# operation: each source's y-offset is computed against the union of *already
# aligned* sources, so we need to process sources in an order where every
# subsequent source has temporal overlap with at least one prior source.
#
# - `digital` first because it's the highest-confidence frame (Jan-Jun 2025).
#   It defines the global y-frame everyone else gets shifted into.
# - `dec2024_to_now` second because it's the only source that overlaps BOTH
#   digital (Jan-Jun 2025) AND the recent dense sources (Apr 2026 today). It
#   bridges the temporal gap between digital and the rolling-window PNGs.
# - `legacy` third because its post-cutoff range (Jul-Nov 2025) overlaps with
#   dec2024_to_now and lets later sources reach back via two hops.
# - The four rolling-window PNG sources last, in any order — they all overlap
#   with dec2024_to_now and so each has the entire chain of prior alignments
#   to anchor against.
#
# Sources whose entire date range falls outside the union of all earlier
# sources are reported as unaligned (their raw values are still merged in).
ALIGNMENT_ORDER: tuple[str, ...] = (
    "digital",
    "dec2024_to_now",
    "legacy",
    "three_month",
    "month",
    "week",
    "two_day",
)

# Maps a TiltSource enum member to its identifier in SOURCE_PRIORITY / its
# per-source CSV filename. Identifiers are snake_case to match Python naming
# conventions; TiltSource.values use a mix that doesn't (e.g. "2day").
TILT_SOURCE_NAME: dict[TiltSource, str] = {
    TiltSource.TWO_DAY: "two_day",
    TiltSource.WEEK: "week",
    TiltSource.MONTH: "month",
    TiltSource.THREE_MONTH: "three_month",
    TiltSource.DEC2024_TO_NOW: "dec2024_to_now",
}

# Identifiers for the two non-PNG sources, kept as constants so callers don't
# have to remember the spelling.
DIGITAL_SOURCE_NAME = "digital"
LEGACY_SOURCE_NAME = "legacy"

# When loading legacy data into its per-source CSV, drop rows older than this
# date. The legacy file's pre-July-2025 coverage is sparse and irregular
# (~2-day median spacing, hand-digitized) and overlaps with the much denser
# DEC2024_TO_NOW source. We keep the post-cutoff portion because that's where
# legacy provides unique coverage (Jul-Nov 2025) that no other source has.
# The legacy CSV on disk is left intact as a historical record.
LEGACY_BOOTSTRAP_CUTOFF = pd.Timestamp("2025-07-01")


def source_csv_path(source_name: str) -> Path:
    """Return the per-source raw CSV path for a reconciliation source."""
    return SOURCES_DIR / f"{source_name}.csv"


@dataclass(frozen=True)
class PeakDetectionDefaults:
    """Defaults for `peaks.detect_peaks`, derived from v1.0's hardcoded peak data
    (`legacy/eruption_projection.py:50-67`): peaks 8.2-11.8 µrad spaced 11-23 days."""

    min_prominence: float = 4.0   # microradians
    min_distance_days: float = 5.0
    min_height: float = 5.0       # microradians


PEAK_DEFAULTS = PeakDetectionDefaults()

# Streamlit cache TTL for the ingest pipeline (seconds). USGS updates these
# plots roughly every 15 minutes, so re-fetching more often is wasted.
INGEST_CACHE_TTL_SECONDS = 15 * 60

# Display timezone — v1.0 labels Pacific/Honolulu Time on the x-axis (line 329).
DISPLAY_TIMEZONE = "Pacific/Honolulu"
