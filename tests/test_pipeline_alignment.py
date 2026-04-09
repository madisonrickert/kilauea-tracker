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
