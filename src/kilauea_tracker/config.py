"""Centralized configuration: USGS source URLs, file paths, and defaults.

The four USGS PNG URLs are intentionally hardcoded rather than templated because
USGS uses an inconsistent naming scheme (`UWD-POC-TILT-*` for week/month, plain
`UWD-TILT-*` for 2day/3month).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
HISTORY_CSV = DATA_DIR / "tilt_history.csv"
LEGACY_CSV = PROJECT_ROOT / "legacy" / "Tiltmeter Data - Sheet1.csv"
LAST_GOOD_CALIBRATION = DATA_DIR / "last_good_calibration.json"


class TiltSource(Enum):
    """The four USGS tilt PNG captures we ingest."""

    TWO_DAY = "2day"
    WEEK = "week"
    MONTH = "month"
    THREE_MONTH = "3month"


# Hardcoded URLs — naming inconsistency forces this (see module docstring).
USGS_TILT_URLS: dict[TiltSource, str] = {
    TiltSource.TWO_DAY: "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-TILT-2day.png",
    TiltSource.WEEK: "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-POC-TILT-week.png",
    TiltSource.MONTH: "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-POC-TILT-month.png",
    TiltSource.THREE_MONTH: "https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-TILT-3month.png",
}

# All four sources, ordered shortest-to-longest window. Iteration order matters
# for the cache: shorter windows have higher resolution and are appended first
# so the dedupe (`keep="last"`) prefers them when timestamps overlap.
ALL_SOURCES: tuple[TiltSource, ...] = (
    TiltSource.THREE_MONTH,
    TiltSource.MONTH,
    TiltSource.WEEK,
    TiltSource.TWO_DAY,
)


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
