"""Tests for `pipeline._align_to_cache` — cross-source y-offset correction.

The bug this guards against: each USGS PNG re-renders with its own y-axis
labels, and Tesseract introduces small intercept differences that compound
to a systematic ~5-7 µrad offset between captures of the same physical
sensor. Without correction, every overlap bucket flagged as a conflict and
the model fits saw step jumps where one source handed off to another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kilauea_tracker.ingest.pipeline import (
    MAX_TRUSTED_OFFSET_MICRORAD,
    MIN_OVERLAP_BUCKETS_FOR_ALIGN,
    _align_to_cache,
    _filter_to_gap_buckets,
)
from kilauea_tracker.model import DATE_COL, TILT_COL


def _series(start: str, n: int, base_tilt: float, freq: str = "1h") -> pd.DataFrame:
    """Build a tilt DataFrame with constant y for `_align_to_cache` tests."""
    return pd.DataFrame(
        {
            DATE_COL: pd.date_range(start, periods=n, freq=freq),
            TILT_COL: np.full(n, base_tilt, dtype=float),
        }
    )


def test_align_subtracts_systematic_offset():
    """The bug case: new trace is uniformly +6 µrad from the cache.

    Expected: alignment computes +6.0 offset, returns rows shifted DOWN by 6.0
    so they match the cache exactly.
    """
    cache = _series("2026-03-01", n=48, base_tilt=10.0)        # cache at 10 µrad
    new = _series("2026-03-01", n=48, base_tilt=16.0)          # new at 16 µrad
    aligned, offset, overlap = _align_to_cache(new, cache)
    assert offset is not None
    assert abs(offset - 6.0) < 1e-9
    assert overlap == 48
    assert (aligned[TILT_COL] == 10.0).all()


def test_align_with_no_overlap_returns_unchanged():
    """No bucket overlap → no alignment, no offset, original rows."""
    cache = _series("2026-01-01", n=24, base_tilt=10.0)
    new = _series("2026-06-01", n=24, base_tilt=15.0)
    aligned, offset, overlap = _align_to_cache(new, cache)
    assert offset is None
    assert overlap == 0
    assert (aligned[TILT_COL] == 15.0).all()


def test_align_below_overlap_threshold_returns_unchanged():
    """Tiny overlap (<MIN_OVERLAP_BUCKETS_FOR_ALIGN) — refuse to align.

    A 4-bucket overlap is too small to confidently distinguish a real
    systematic offset from per-bucket noise; we leave the trace untouched.
    """
    cache = _series("2026-03-01", n=24, base_tilt=10.0)
    # New trace overlaps with the last 4 hours of the cache only.
    new_start = cache[DATE_COL].iloc[-MIN_OVERLAP_BUCKETS_FOR_ALIGN + 1]
    new = _series(str(new_start), n=10, base_tilt=15.0)
    aligned, offset, overlap = _align_to_cache(new, cache)
    assert offset is None
    assert overlap < MIN_OVERLAP_BUCKETS_FOR_ALIGN
    assert (aligned[TILT_COL] == 15.0).all()


def test_align_refuses_implausibly_large_offset():
    """A 50 µrad delta is a calibration bug, not drift — must NOT auto-correct."""
    cache = _series("2026-03-01", n=48, base_tilt=10.0)
    new = _series("2026-03-01", n=48, base_tilt=10.0 + MAX_TRUSTED_OFFSET_MICRORAD + 5)
    aligned, offset, overlap = _align_to_cache(new, cache)
    assert offset is None  # refused
    assert overlap == 48
    # Original rows untouched
    assert (aligned[TILT_COL] == new[TILT_COL]).all()


def test_align_with_coarser_bucket_finds_more_overlap():
    """The bug case: two sources whose individual sample timestamps never
    quite line up at 15-min granularity, but DO share hourly buckets often.

    Setup: cache has rows every 4 hours offset by 7 minutes (so they don't
    snap to a 15-min boundary nicely). New trace has rows every 4 hours
    offset by 23 minutes. At 15-min bucketing only a few collide; at 1-hour
    bucketing every pair shares an hour bucket.
    """
    base = pd.Timestamp("2025-08-01 00:07:00")
    cache = pd.DataFrame(
        {
            DATE_COL: [base + pd.Timedelta(hours=4 * i) for i in range(40)],
            TILT_COL: [10.0] * 40,
        }
    )
    new_base = pd.Timestamp("2025-08-01 00:23:00")
    new = pd.DataFrame(
        {
            DATE_COL: [new_base + pd.Timedelta(hours=4 * i) for i in range(40)],
            TILT_COL: [16.0] * 40,
        }
    )

    _, _, overlap_15min = _align_to_cache(new, cache, bucket_freq="15min")
    _, offset_1h, overlap_1h = _align_to_cache(new, cache, bucket_freq="1h")

    assert overlap_1h > overlap_15min
    assert overlap_1h >= MIN_OVERLAP_BUCKETS_FOR_ALIGN
    assert offset_1h is not None
    assert abs(offset_1h - 6.0) < 0.5


def test_align_uses_median_not_mean():
    """A few outlier buckets near a transition shouldn't pull the offset.

    Setup: 48 cache buckets at 10 µrad, 48 new buckets where 44 are at +6
    (the systematic offset) and 4 are at +30 (transition outliers). Mean
    delta would be 10; median delta is 6. We want 6.
    """
    cache = _series("2026-03-01", n=48, base_tilt=10.0)
    new_values = np.full(48, 16.0)
    new_values[[5, 12, 25, 40]] = 40.0  # outliers
    new = pd.DataFrame(
        {
            DATE_COL: cache[DATE_COL].copy(),
            TILT_COL: new_values,
        }
    )
    aligned, offset, overlap = _align_to_cache(new, cache)
    assert offset is not None
    assert abs(offset - 6.0) < 1e-9


def test_align_empty_inputs_no_op():
    cache = pd.DataFrame({DATE_COL: pd.Series(dtype="datetime64[ns]"), TILT_COL: []})
    new = _series("2026-01-01", n=10, base_tilt=10.0)
    aligned, offset, overlap = _align_to_cache(new, cache)
    assert offset is None
    assert overlap == 0

    aligned, offset, overlap = _align_to_cache(cache, new)  # other direction
    assert offset is None
    assert overlap == 0


def test_align_does_not_mutate_inputs():
    cache = _series("2026-03-01", n=48, base_tilt=10.0)
    new = _series("2026-03-01", n=48, base_tilt=16.0)
    new_orig_values = new[TILT_COL].copy()
    aligned, _, _ = _align_to_cache(new, cache)
    # Original DataFrame untouched — only the returned copy is shifted.
    assert (new[TILT_COL] == new_orig_values).all()
    assert (aligned[TILT_COL] != new[TILT_COL]).all()


# ─────────────────────────────────────────────────────────────────────────────
# _filter_to_gap_buckets — used by gap-fill sources (DEC2024_TO_NOW)
# ─────────────────────────────────────────────────────────────────────────────


def test_gap_filter_drops_buckets_already_in_cache():
    """Default 1-day bucket: any new row whose calendar day is already
    represented in the cache gets dropped. Existing rows on Jan 1 and Jan 2
    block ALL new rows on those days regardless of hour-of-day.
    """
    cache = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                ["2026-01-01 03:00:00", "2026-01-02 17:00:00"]
            ),
            TILT_COL: [10.0, 11.0],
        }
    )
    new = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2026-01-01 12:00:00",  # same day as cache → DROP
                    "2026-01-02 06:00:00",  # same day as cache → DROP
                    "2026-01-03 00:00:00",  # GAP day → keep
                    "2026-01-04 00:00:00",  # GAP day → keep
                    "2026-01-05 12:00:00",  # GAP day → keep
                ]
            ),
            TILT_COL: [10.5, 9.0, 11.5, 10.0, 13.0],
        }
    )
    filtered, dropped = _filter_to_gap_buckets(new, cache)
    assert dropped == 2
    assert len(filtered) == 3
    kept_days = filtered[DATE_COL].dt.strftime("%Y-%m-%d").tolist()
    assert kept_days == ["2026-01-03", "2026-01-04", "2026-01-05"]


def test_gap_filter_with_finer_bucket_freq():
    """Test the bucket_freq parameter with 15-min bins (the old default).
    Locks in that the function still works at finer granularity."""
    cache = pd.DataFrame(
        {DATE_COL: pd.to_datetime(["2026-01-01 00:00:00"]), TILT_COL: [10.0]}
    )
    new = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(
                [
                    "2026-01-01 00:05:00",  # same 15-min bucket → DROP
                    "2026-01-01 00:30:00",  # different bucket → keep
                    "2026-01-01 06:00:00",  # different bucket → keep
                ]
            ),
            TILT_COL: [10.5, 11.0, 12.0],
        }
    )
    filtered, dropped = _filter_to_gap_buckets(new, cache, bucket_freq="15min")
    assert dropped == 1
    assert len(filtered) == 2


def test_gap_filter_with_empty_cache_keeps_everything():
    cache = pd.DataFrame({DATE_COL: pd.Series(dtype="datetime64[ns]"), TILT_COL: []})
    new = _series("2026-01-01", n=10, base_tilt=10.0)
    filtered, dropped = _filter_to_gap_buckets(new, cache)
    assert dropped == 0
    assert len(filtered) == 10


def test_gap_filter_with_empty_new_returns_empty():
    cache = _series("2026-01-01", n=5, base_tilt=10.0)
    new = pd.DataFrame({DATE_COL: pd.Series(dtype="datetime64[ns]"), TILT_COL: []})
    filtered, dropped = _filter_to_gap_buckets(new, cache)
    assert dropped == 0
    assert len(filtered) == 0


def test_gap_filter_does_not_mutate_inputs():
    cache = _series("2026-01-01", n=5, base_tilt=10.0, freq="1h")
    new = _series("2026-01-01", n=5, base_tilt=11.0, freq="1h")
    new_orig = new.copy()
    cache_orig = cache.copy()
    _filter_to_gap_buckets(new, cache)
    pd.testing.assert_frame_equal(new, new_orig)
    pd.testing.assert_frame_equal(cache, cache_orig)
