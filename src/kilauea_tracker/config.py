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
# Phase 2 note: reconcile.py no longer uses SOURCE_PRIORITY for merge
# selection — the merge now picks by best effective resolution (product
# of the source's solved y-slope correction `a_i` and a per-source
# µrad/pixel fallback). This tuple is retained because it enumerates
# every valid source name the system supports; tests and helpers still
# reference it as the canonical source list. Archive moved to the end
# because it is now a pure gap-filler (contributes only when no live
# source covers a bucket).
SOURCE_PRIORITY: tuple[str, ...] = (
    "digital",
    "two_day",
    "week",
    "month",
    "dec2024_to_now",
    "three_month",
    "archive",
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


# ─── Phase 1 (2026-04 alignment rewrite): transcription quality guards ─────────
#
# The v3 alignment plan (see `.claude/plans/foamy-yawning-horizon.md`)
# diagnoses the whack-a-mole step discontinuities as primarily y-calibration
# errors at ingest time, with a long tail from per-column tracing artifacts.
# These knobs gate the transcription pipeline so a bad OCR / bad pixel cluster
# can no longer silently propagate through per-source CSVs into the merge.

# Phase 1a. Reject a y-axis calibration whose linear fit residual exceeds
# this many µrad. The current `np.polyfit` on OCR'd y-labels already
# computes a residual but the code never acts on it, so a single mis-OCR'd
# label (e.g. "-5" read as "-4") tilts the fit and every traced value is
# then biased by a tilt-dependent amount.
#
# Why µrad (not pixels): µrad per pixel varies ~20× across sources
# (`two_day` is ~0.01 µrad/px, `dec2024_to_now` is ~0.6 µrad/px), so a
# pixel-based threshold either lets slop through on dense sources or
# spuriously rejects high-resolution ones. A µrad threshold is directly
# meaningful in data space: 0.5 µrad of label-fit bias translates to the
# same amount of per-sample bias regardless of source resolution, and
# the inter-source noise floor itself is ~0.5-1 µrad, so anything at or
# below this can't materially move merged output.
#
# Healthy clean-fixture residuals (observed on committed fixtures): 3-month
# ≈ 0.07 µrad, 2-day ≈ 0.18 µrad, week ≈ 0.01 µrad, month ≈ 0 µrad.
# Live failure mode observed on 2026-04-21 `month` capture: ~6.5 µrad
# residual caused by OCR misreading "0" as "1" on one tick — this threshold
# cleanly catches it while tolerating every clean fixture.
Y_CALIBRATION_MAX_RESIDUAL_MICRORAD = 0.5

# Phase 1a. If a freshly-computed y-slope deviates from the rolling median
# of the last N successful runs by more than this fraction, warn and fall
# back to the last-known-good slope. USGS occasionally shifts the y-range
# but does not change the per-pixel scale between consecutive captures, so
# a sudden slope change is a near-certain OCR bug.
Y_SLOPE_HISTORY_LENGTH = 20
Y_SLOPE_REGRESSION_TOLERANCE_PERCENT = 3.0

# Phase 1b. Expected total time window span per source (hours). The title
# OCR produces `(x_start, x_end)` and we sanity-check the resulting span
# against the advertised window. A title minute/hour misread that slips
# past the regex would otherwise time-shift every sample silently.
# `dec2024_to_now` uses a monotonically-growing window with no fixed
# length; it is range-checked separately (must be ≥ 30 days, ≤ 5 years).
X_WINDOW_EXPECTED_HOURS: dict[str, float] = {
    "two_day": 48.0,
    "week": 168.0,
    "month": 720.0,
    "three_month": 2160.0,
}
X_WINDOW_TOLERANCE_HOURS = 2.0

# Phase 1b (extension). The title OCR can return a valid-span range whose
# ABSOLUTE year is wrong — e.g. today's `two_day` capture was read as
# `2008-04-20 → 2008-04-22` (48-hour span, passes the window check) when
# the real title said `2026-04-20 → 2026-04-22`. Multi-digit year
# misreads slip past the single-digit recovery in `_recover_ocr_year_misread`.
# Catch them with a recency check: `x_end` must be within this many hours
# of `now` (in UTC). USGS occasionally runs behind by several hours, so
# 72 h gives a comfortable margin while still rejecting the 18-year slip.
X_END_MAX_AGE_HOURS = 72.0

# Phase 1c. Anchor-referenced calibration cross-check. Fits
# `digital = a * png + b` via Huber robust regression for any PNG source
# that temporally overlaps digital (Jan-Jun 2025). If the fit produces
# a non-identity correction within these tolerances, we WARN and apply
# the correction to that source's values; beyond these tolerances we
# apply the correction AND flag it as a transcription failure candidate.
#
# As of 2026-04, `dec2024_to_now` is the only live rolling source that
# still overlaps digital — the other PNGs (two_day/week/month/three_month)
# have moved past it by ~10 months. The check is effectively dead code
# for those sources and lives on only for dec2024_to_now.
ANCHOR_FIT_A_WARNING_FRACTION = 0.03          # |a-1| > 3% → warn
ANCHOR_FIT_B_WARNING_MICRORAD = 5.0           # |b|   > 5 µrad → warn
ANCHOR_FIT_TRIM_HOURS = 6                     # trim first/last 6 h of digital overlap
ANCHOR_FIT_MIN_OVERLAP_BUCKETS = 50           # regression not meaningful below this

# Phase 1d. Kīlauea's fastest real DI transitions on record are ~5 µrad/hour
# (see `legacy/Tiltmeter Data - Sheet1.csv` eruption-transition slopes).
# A column-to-column tilt rate exceeding this × 3 is non-physical; the
# trace has almost certainly landed on a gridline, axis tick, or other
# artifact. Reject per-column before the rolling-median pass.
MAX_PHYSICAL_RATE_MICRORAD_PER_HOUR = 15.0

# Phase 1e. The real Az-300 blue curve is typically 1-3 pixels thick in
# the rendered PNG (varies by window). Any column with more than this
# many lit pixels is crossing a gridline, axis label, or is in a JPEG
# artifact cluster. Drop the column and let the downstream rolling-median
# filter interpolate.
CURVE_MAX_COLUMN_WIDTH_PIXELS = 8

# Columns lighting up between [WIDE_COLUMN_THRESHOLD_PIXELS, CURVE_MAX_COLUMN_WIDTH_PIXELS]
# blue pixels get special handling in trace_curve. Stripes this tall are
# near-vertical transitions (eruption drops, rapid rises) where the curve
# sweeps through several y-rows in one column's time step. Median-per-
# column on those stripes lands mid-transition and FLATTENS the true
# extremum. We instead pick the endpoint that continues the direction of
# motion from the previous column, which preserves peaks and troughs at
# the cost of one pixel's worth of timing resolution on that column.
WIDE_COLUMN_THRESHOLD_PIXELS = 3


# ─── Phase 2 (2026-04 alignment rewrite): pairwise self-consistency ────────────
#
# Replaces the old scalar-offset + piecewise-residual + proximity-gate +
# archive-age-demotion stack with a single well-specified algorithm:
#
#   1. For every pair (i, j) of sources with temporal overlap, compute the
#      OLS regression  y_i = α_ij · y_j + β_ij  over bucket-aligned pairs.
#   2. Under the model  y_i(t) = a_i · true(t) + b_i  we have
#      α_ij = a_i / a_j  and  β_ij = b_i − α_ij · b_j.
#   3. Pin (a_digital, b_digital) = (1, 0). Solve two least-squares systems
#      for {a_i} then {b_i}.
#   4. Apply corrections true_i(t) = (y_i(t) − b_i) / a_i per source.
#   5. Merge by best effective resolution per bucket, rejecting outliers
#      via MAD (median absolute deviation) from the per-bucket median.
#
# See `src/kilauea_tracker/reconcile.py` for the full algorithm and
# `.claude/plans/foamy-yawning-horizon.md` for the design rationale.

# Minimum bucket-aligned overlap per pair before its OLS fit is trusted.
# Below this the pair contributes no constraint — too few points to
# resolve slope (α_ij) from intercept (β_ij).
PAIRWISE_MIN_OVERLAP_BUCKETS = 50

# Huber-like cutoff on the cross-source MAD outlier check at merge time.
# A source's corrected value is dropped from the bucket's merge candidate
# set when it exceeds  max(MAD_OUTLIER_SIGMA_FLOOR_MICRORAD,
# MAD_OUTLIER_SIGMA_MULTIPLIER * σ_MAD)  from the bucket's unweighted
# median.  The floor keeps the check meaningful at buckets where all
# sources agree (σ_MAD ≈ 0).
#
# Floor set to 5 µrad (was 2 µrad in Phase 2): the typical inter-source
# calibration disagreement after pairwise correction is 2-3 µrad, so a
# 2 µrad floor would flag well-calibrated sources as outliers when the
# consensus happens to fall close to the miscalibrated side — the exact
# mechanism that produced the 2026-04-22 alternating-lines prod
# regression.  5 µrad still catches real OCR-glitch outliers (typically
# 10+ µrad off) while tolerating legitimate inter-source spread.
MAD_OUTLIER_SIGMA_MULTIPLIER = 3.0
MAD_OUTLIER_SIGMA_FLOOR_MICRORAD = 5.0

# Blending zone for K≥2 → K=1 handoffs (the canonical day-90 boundary where
# three_month ends and only dec2024_to_now covers further back). Inside
# this many hours of the transition, merge uses a linear taper between the
# K≥2 consensus and the K=1 source's corrected value so the curve doesn't
# step. Outside the zone, pure best-effective-resolution wins.
K1_HANDOFF_BLEND_HOURS = 6.0

# Post-merge continuity audit: flag any adjacent-bucket step exceeding this
# many µrad into `ReconcileReport.continuity_violations` for post-hoc
# diagnosis. Not a hard fail — the signal can legitimately step by several
# µrad during DI transitions — but a surge in violations is an actionable
# regression signal.
CONTINUITY_WARNING_THRESHOLD_MICRORAD = 4.0

# Sanity bounds on the joint (a, b) solve. A source whose fit moves it by
# more than these values is almost certainly suffering a calibration bug
# Phase 1 should have caught; apply the correction anyway but flag it as
# a transcription failure candidate.
PAIRWISE_MAX_A_DEVIATION_FRACTION = 0.50
PAIRWISE_MAX_B_MICRORAD = 30.0

# Effective-resolution fallback per source (µrad per plot pixel). Used as
# the merge tie-breaker when two sources both have a corrected value for
# the same 15-min bucket. Tighter = wins. Values reflect the typical
# µrad/px ratio of each USGS PNG's y-axis scale (≈ y-range / plot-height):
#   two_day ≈ 6 µrad / 225 px ≈ 0.027 µrad/px
#   week    ≈ 20 µrad / 225 px ≈ 0.09 µrad/px
#   month   ≈ 30 µrad / 225 px ≈ 0.13 µrad/px
#   three_month ≈ 40 µrad / 225 px ≈ 0.18 µrad/px
#   dec2024_to_now ≈ 140 µrad / 225 px ≈ 0.62 µrad/px
# Digital is the authoritative ground-truth CSV — treat its resolution as
# tight enough to dominate every per-bucket tie when it has coverage.
# Archive is a derived gap-filler — it loses every tie with a live source.
EFFECTIVE_RESOLUTION_FALLBACK_MICRORAD_PER_PIXEL: dict[str, float] = {
    "digital": 0.001,
    "two_day": 0.027,
    "week": 0.09,
    "month": 0.13,
    "three_month": 0.18,
    "dec2024_to_now": 0.62,
    "archive": 1000.0,  # effectively last-place; only wins K=0 gaps
}
