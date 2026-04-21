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
LAST_GOOD_CALIBRATION = DATA_DIR / "last_good_calibration.json"
# Pre-processed digital tiltmeter data — produced once by
# `scripts/import_digital_data.py` from USGS's research-release CSVs.
# This is the canonical reference for Jan-Jun 2025; the ingest pipeline
# anchors everything else to it.
DIGITAL_CSV = DATA_DIR / "uwd_digital_az300.csv"

# Append-only canonical archive of every reconciled observation. Each
# ingest_all() run promotes new merged-view timestamps into this file
# with keep-first dedupe — once a row is in the archive, it never
# changes. The archive then feeds back into reconcile.reconcile_sources
# as a high-priority source so the merged view sources historical
# timestamps from the frozen archive instead of the (potentially
# drifting) live per-source CSVs. See `archive.py` for the full
# rationale.
ARCHIVE_CSV = DATA_DIR / "archive.csv"

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
# TiltSource enum members) because the digital source isn't a USGS PNG capture
# and so doesn't fit the TiltSource enum. The strings are also the file stems
# for the per-source CSVs in SOURCES_DIR/.
#
# Priority order (highest first): digital is the gold reference because it's
# raw USGS instrument data (100% confidence) but only covers Jan-Jun 2025.
# The dense PNG sources rank by inverse window length — TWO_DAY has the
# highest sample density and the smallest legend overlap risk; THREE_MONTH
# is at the bottom because (a) it's the lowest-resolution image and (b)
# the 3-month plot is the only one whose curve actually passes through the
# legend region. DEC2024_TO_NOW is the longest-coverage image source so it
# slots above THREE_MONTH but below the dense recent sources.
#
# `legacy` (the v1 hand-traced PlotDigitizer CSV from `legacy/Tiltmeter Data
# - Sheet1.csv`) was previously slotted between `month` and `dec2024_to_now`,
# but the user observed in 2026-04 that it was creating systemic ~6 µrad
# offsets in the 6/29-8/21 range — its hand-traced samples don't reliably
# match dec2024_to_now's auto-traced frame and the chicken-and-egg of
# aligning legacy through dec2024_to_now made the offsets noisy. We removed
# legacy entirely; dec2024_to_now covers the same Jul-Nov 2025 range with
# higher density and a single consistent y-frame.
#
# This order is consulted by the *merge* step in reconcile.py: when two
# sources both contain data for the same 15-min bucket, the one earlier in
# this tuple wins.
SOURCE_PRIORITY: tuple[str, ...] = (
    "digital",
    "archive",
    "two_day",
    "week",
    "month",
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
# - The four rolling-window PNG sources last, in any order — they all overlap
#   with dec2024_to_now and so each has the entire chain of prior alignments
#   to anchor against.
#
# Sources whose entire date range falls outside the union of all earlier
# sources are reported as unaligned (their raw values are still merged in).
ALIGNMENT_ORDER: tuple[str, ...] = (
    "digital",
    "archive",
    "dec2024_to_now",
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

# Identifier for the digital reference source, kept as a constant so callers
# don't have to remember the spelling.
DIGITAL_SOURCE_NAME = "digital"

# Identifier for the append-only archive source. Same kind of constant as
# DIGITAL_SOURCE_NAME — used by archive.py and ingest.pipeline so the spelling
# only lives in one place.
ARCHIVE_SOURCE_NAME = "archive"


def source_csv_path(source_name: str) -> Path:
    """Return the per-source raw CSV path for a reconciliation source."""
    return SOURCES_DIR / f"{source_name}.csv"


@dataclass(frozen=True)
class PeakDetectionDefaults:
    """Defaults for `peaks.detect_peaks`.

    Originally these were derived from v1.0's hardcoded peak data
    (`legacy/eruption_projection.py:50-67`): peaks 8.2-11.8 µrad spaced
    11-23 days. The min_height of 5 µrad was fine for that regime.

    Since then Kīlauea has shifted to a phase with much smaller eruption
    cycles — peaks at 2-4 µrad and deflation magnitudes still around
    7-30 µrad. The min_height floor was rejecting valid peaks (e.g.
    2026-01-11 at 2.84 µrad, 2026-03-10 at 2.65 µrad) even though their
    prominence (the drop to the next deflation trough) was 16-30 µrad.

    Setting min_height=None defers entirely to prominence, which is the
    metric that actually distinguishes a real eruption peak from
    background noise. Prominence measures how much the curve rises above
    its surrounding minima, regardless of absolute tilt level — so the
    detector adapts as the volcano cycles through different tilt regimes.
    """

    min_prominence: float = 4.0          # microradians (drop to next trough)
    min_distance_days: float = 5.0       # don't double-count peaks closer than this
    min_height: float | None = None      # no absolute floor; rely on prominence


PEAK_DEFAULTS = PeakDetectionDefaults()

# Streamlit cache TTL for the ingest pipeline (seconds). USGS updates these
# plots roughly every 15 minutes, so re-fetching more often is wasted.
INGEST_CACHE_TTL_SECONDS = 15 * 60

# Display timezone — v1.0 labels Pacific/Honolulu Time on the x-axis (line 329).
DISPLAY_TIMEZONE = "Pacific/Honolulu"


# ─── Ingest validation knobs (added after the 2026-04 archive contamination) ───
#
# A bad `week` PNG trace in April 2026 produced recurring phantom rows ~10 µrad
# below the real curve. Those rows got promoted into the append-only archive
# (keep-first, immutable) and then won every future reconcile contest. The
# three knobs below are the three layers of defense now in place.

# Per-row outlier rejection applied inside `trace.trace_curve`. After the HSV
# mask + column-median step produces the raw rows, we compute a rolling
# 5-neighbour median and drop any row whose tilt deviates from it by more
# than this many µrad. The phantom spikes sit ~10 µrad below the curve; real
# eruption transitions drop gradually over hours and stay well within 4 µrad
# of the local median. 4.0 µrad is a comfortable middle ground.
TRACE_OUTLIER_THRESHOLD_MICRORAD = 4.0

# Below this sample count we skip the outlier filter entirely — we don't
# have enough neighbours to trust the rolling median.
TRACE_OUTLIER_MIN_SAMPLES = 10

# Quorum gate applied inside `archive.promote_to_archive`. A 15-min bucket
# can only be promoted into the archive when at least this many aligned
# sources in the reconcile input contributed rows to that bucket. A single
# flaky source can no longer poison the archive — a second source must
# agree before the row becomes permanent.
#
# Exception: buckets that already have an adjacent archived neighbour within
# ARCHIVE_QUORUM_NEIGHBOUR_MINUTES and whose value is within
# ARCHIVE_QUORUM_NEIGHBOUR_THRESHOLD_MICRORAD of that neighbour skip the
# quorum requirement — continuity with known-good archive state is evidence
# enough.
ARCHIVE_QUORUM_MIN_SOURCES = 2
ARCHIVE_QUORUM_NEIGHBOUR_MINUTES = 30
ARCHIVE_QUORUM_NEIGHBOUR_THRESHOLD_MICRORAD = 3.0

# Archive priority demotion. In `reconcile._merge_by_priority`, archive rows
# younger than this many days are demoted below the live PNG sources so
# that a higher-quality live source can override a recently-archived bad
# row. Older archive rows keep their priority-2 slot unchanged — they are
# the load-bearing historical anchor the archive was designed to be.
#
# 14 days lets the daily cron run ~14 opportunities to replace a bad row
# before it becomes frozen, which is comfortably more than the 1-2 day
# window over which sibling sources typically catch up to a just-archived
# bucket.
ARCHIVE_MAX_AGE_FOR_PRIORITY_DEMOTION_DAYS = 14
