"""Curve × trendline intersection helper.

Lifted from ``models/trendline_exp.py`` (Apr 2026 refactor) so every
prediction model can reuse the same projection-window scan + brentq
sign-bracketed solve. Pure: no I/O, no clock reads, no module-level
mutable state — same inputs always produce the same outputs.

The intersection is solved in float-days-since-epoch (matching
``model.to_days``) for numerical conditioning. Callers convert back to
``pd.Timestamp`` via ``model.from_days`` after the solve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from ..model import from_days

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

# Sanity bounds on the predicted intersection's tilt value. v1.0 used
# (-20, 15); the live data eventually exceeded those, silently dropping
# valid predictions. Widened to (-50, 50) — brentq + the projection
# bracket already guarantee a sensible date range, so this is just a
# "no-NaN" guard.
INTERSECTION_TILT_MIN = -50.0
INTERSECTION_TILT_MAX = 50.0

# Forward projection horizon. We scan this many days past
# ``max(last_current_day, last_peak_day)`` looking for the first sign
# change of (curve - trendline). Beyond 90 days the intersection is
# physically meaningless for Kilauea's monthly cadence.
PROJECTION_WINDOW_DAYS = 90.0

# Number of points in the scan grid. 91 = one per day at 90-day horizon.
# Densifying past this returns diminishing returns — brentq does the
# precise solve once a sign-change interval is found.
_SCAN_GRID_POINTS = 91


def find_intersection(
    f_curve: Callable[[float], float],
    f_lin: Callable[[float], float],
    last_current_day: float,
    last_peak_day: float,
) -> tuple[pd.Timestamp | None, float | None]:
    """Solve ``f_curve(x) == f_lin(x)`` in the future projection window.

    Scans the window for the first sign change in ``f_curve - f_lin``,
    then brentq-refines. Returns ``(None, None)`` when no sign change
    is found in the window (e.g., the curve asymptotes below the
    trendline) or when the resulting tilt value is outside the sanity
    bounds.
    """
    projection_start = max(last_peak_day, last_current_day)
    projection_end = projection_start + PROJECTION_WINDOW_DAYS

    def diff(x: float) -> float:
        return float(f_curve(x) - f_lin(x))

    scan = np.linspace(projection_start, projection_end, _SCAN_GRID_POINTS)
    try:
        diffs = np.array([diff(x) for x in scan])
    except Exception:
        return None, None

    sign_change_idx = None
    for i in range(len(diffs) - 1):
        if np.isnan(diffs[i]) or np.isnan(diffs[i + 1]):
            continue
        if diffs[i] == 0:
            sign_change_idx = i
            break
        if diffs[i] * diffs[i + 1] < 0:
            sign_change_idx = i
            break

    if sign_change_idx is None:
        return None, None

    a, b = scan[sign_change_idx], scan[sign_change_idx + 1]
    if diffs[sign_change_idx] == 0:
        root = a
    else:
        try:
            root = brentq(diff, a, b, xtol=1e-4, maxiter=100)
        except Exception:
            return None, None

    tilt_at_root = float(f_lin(root))
    if not (INTERSECTION_TILT_MIN < tilt_at_root < INTERSECTION_TILT_MAX):
        return None, None

    return from_days(root), tilt_at_root


def find_linear_intersection(
    m_curve: float,
    b_curve: float,
    m_trend: float,
    b_trend: float,
    earliest_day: float,
    latest_day: float,
) -> tuple[pd.Timestamp | None, float | None]:
    """Closed-form intersection of two lines, range-checked.

    For two-linear models (``linear``, ``linear_naive``, ``linear_hist``,
    ``linear_stitched``) we don't need brentq — the intersection is a
    one-divide. Returns ``(None, None)`` when slopes are parallel
    (denominator ~0) or the root is outside the valid projection window.
    """
    denom = m_curve - m_trend
    if abs(denom) < 1e-12:
        return None, None
    root = (b_trend - b_curve) / denom
    if not (earliest_day <= root <= latest_day):
        return None, None
    tilt_at_root = float(m_trend * root + b_trend)
    if not (INTERSECTION_TILT_MIN < tilt_at_root < INTERSECTION_TILT_MAX):
        return None, None
    return from_days(root), tilt_at_root
