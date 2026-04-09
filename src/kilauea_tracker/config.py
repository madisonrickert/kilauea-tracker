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

# Iteration order for `ingest_all`. Each subsequent source's "new" rows
# displace the previous source's overlapping rows (`keep="last"` in
# append_history). So we ingest from coarsest-recent-resolution to finest:
#   THREE_MONTH (~3h/sample)  → populates the last 3 months
#   MONTH       (~1h/sample)  → overwrites the last month with finer data
#   WEEK        (~13min/sample) → overwrites the last week
#   TWO_DAY     (~4min/sample)  → overwrites the last 2 days
#
# DEC2024_TO_NOW runs LAST in fill-gaps-only mode (see GAP_FILL_SOURCES). Its
# y-coordinate frame is much wider (-60 to 80 µrad) so its raw values are not
# directly comparable to the other sources' (-30 to 20 µrad), but cross-source
# alignment corrects for that in the overlap region. After alignment, only
# DEC2024_TO_NOW samples that fall in 15-min buckets the cache *doesn't*
# already have are appended — filling gaps without overwriting the higher
# resolution recent data. This is the source that fills the ~6-week gap
# between the legacy bootstrap (ends 2025-11-25) and THREE_MONTH's earliest
# sample (~Jan 8 2026).
ALL_SOURCES: tuple[TiltSource, ...] = (
    TiltSource.THREE_MONTH,
    TiltSource.MONTH,
    TiltSource.WEEK,
    TiltSource.TWO_DAY,
    TiltSource.DEC2024_TO_NOW,
)

# Sources whose new rows should only fill gaps in the cache, not overwrite
# existing buckets. Used for low-resolution long-history sources where the
# higher-resolution windows already have better data for their region.
GAP_FILL_SOURCES: frozenset = frozenset({TiltSource.DEC2024_TO_NOW})

# When bootstrapping the cache from `legacy/Tiltmeter Data - Sheet1.csv`,
# drop rows older than this date. The legacy file's pre-July-2025 coverage
# is sparse and irregular (~2-day median spacing, manually digitized) while
# DEC2024_TO_NOW provides ~15h-spaced samples for the same range. By
# trimming the legacy bootstrap here, DEC2024_TO_NOW's gap-fill mode
# naturally populates the pre-July buckets with its denser data on the
# next ingest cycle. The legacy file on disk is left intact as a
# historical record.
LEGACY_BOOTSTRAP_CUTOFF = pd.Timestamp("2025-07-01")


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
