"""Phase 1c: anchor-referenced calibration cross-check (digital → source).

Each USGS PNG source carries its own residual y-axis miscalibration that
the OCR axis fit can't fully absorb. When the digital research-release
file is available as ground truth, we can fit
`digital_tilt = a · source_tilt + b` over the temporal overlap and apply
that correction to the source. The Huber-robust regression below is the
machinery for that.

Split out of `calibrate.py` so anchor-fit logic isn't tangled with OCR
axis calibration — they're different concerns that just happened to
share a file. The OCR axis-calibration code stays in `calibrate.py`;
the anchor-fit code lives here. Both are imported through the existing
`calibrate.AnchorFitResult` / `calibrate.recalibrate_by_anchor_fit` /
`calibrate.apply_anchor_fit` re-exports so existing call sites keep
working.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    ANCHOR_FIT_A_WARNING_FRACTION,
    ANCHOR_FIT_B_WARNING_MICRORAD,
    ANCHOR_FIT_MIN_OVERLAP_BUCKETS,
    ANCHOR_FIT_TRIM_HOURS,
)
from ..model import DATE_COL, TILT_COL


@dataclass
class AnchorFitResult:
    """Outcome of `recalibrate_by_anchor_fit()` for one source."""

    source_name: str
    ran: bool = False                      # True iff regression was attempted
    overlap_buckets: int = 0               # after bucketing + trimming
    a: float = 1.0                         # slope correction (identity = 1.0)
    b: float = 0.0                         # intercept correction (identity = 0.0)
    residual_std_microrad: float = 0.0     # Huber-loss-weighted residual spread
    warning: str | None = None          # human-readable reason to warn
    note: str | None = None             # reason regression was skipped


def recalibrate_by_anchor_fit(
    source_name: str,
    source_df: pd.DataFrame,
    digital_df: pd.DataFrame,
    *,
    bucket_freq: str = "1h",
) -> AnchorFitResult:
    """Fit `digital_tilt = a * source_tilt + b` via Huber-robust regression
    over the temporal overlap and return the (a, b) correction.

    The model assumes each PNG source is a linear image of the same
    underlying tilt signal that digital captures authoritatively. Fitting
    against digital recovers a source-specific `(a, b)` that collapses the
    systematic y-slope / y-intercept errors the legacy alignment offset
    couldn't absorb (because it was a single scalar).

    Implementation notes:
      * Buckets both sides at `bucket_freq` (default 1 h) and keeps only
        buckets present in both frames. Below `ANCHOR_FIT_MIN_OVERLAP_BUCKETS`
        the result is not meaningful — caller should leave the source
        untouched.
      * Trims the first and last `ANCHOR_FIT_TRIM_HOURS` of digital's range
        to avoid endpoint pathologies where the research-release file
        starts/ends mid-sample.
      * Uses `scipy.optimize.least_squares` with Huber loss. `np.polyfit`
        would be OLS and pay dearly for the tail outliers this data has.
    """
    result = AnchorFitResult(source_name=source_name)

    if source_df is None or len(source_df) == 0:
        result.note = "source_df empty"
        return result
    if digital_df is None or len(digital_df) == 0:
        result.note = "digital_df empty"
        return result

    src = source_df[[DATE_COL, TILT_COL]].copy()
    dig = digital_df[[DATE_COL, TILT_COL]].copy()
    src[DATE_COL] = pd.to_datetime(src[DATE_COL]).astype("datetime64[ns]")
    dig[DATE_COL] = pd.to_datetime(dig[DATE_COL]).astype("datetime64[ns]")
    src = src.dropna().sort_values(DATE_COL)
    dig = dig.dropna().sort_values(DATE_COL)
    if src.empty or dig.empty:
        result.note = "no rows after cleaning"
        return result

    # Trim digital's endpoints — the research-release file begins/ends
    # mid-sample and the first/last few hours inflate the Huber residual.
    dig_start = dig[DATE_COL].min() + pd.Timedelta(hours=ANCHOR_FIT_TRIM_HOURS)
    dig_end = dig[DATE_COL].max() - pd.Timedelta(hours=ANCHOR_FIT_TRIM_HOURS)
    dig_trimmed = dig[(dig[DATE_COL] >= dig_start) & (dig[DATE_COL] <= dig_end)]
    if len(dig_trimmed) == 0:
        result.note = "digital overlap window empty after trim"
        return result

    # Bucket-align both sides at `bucket_freq`, take the mean in each bucket.
    src_b = src.assign(_b=lambda d: d[DATE_COL].dt.floor(bucket_freq)).groupby("_b")[TILT_COL].mean()
    dig_b = (
        dig_trimmed.assign(_b=lambda d: d[DATE_COL].dt.floor(bucket_freq))
        .groupby("_b")[TILT_COL]
        .mean()
    )
    overlap = src_b.index.intersection(dig_b.index)
    result.overlap_buckets = len(overlap)
    if len(overlap) < ANCHOR_FIT_MIN_OVERLAP_BUCKETS:
        result.note = (
            f"overlap too small ({len(overlap)} buckets, need "
            f"≥{ANCHOR_FIT_MIN_OVERLAP_BUCKETS})"
        )
        return result

    x = src_b.loc[overlap].to_numpy(dtype=float)
    y = dig_b.loc[overlap].to_numpy(dtype=float)

    # Huber-robust regression via scipy.optimize.least_squares. Initial
    # guess (a=1, b=0) is the identity transform — correct when calibration
    # is already clean. `f_scale` sets the Huber elbow: residuals larger
    # than f_scale µrad are treated as outliers (L1); smaller are L2. 1.0
    # is conservative — most real inter-source disagreement should be
    # under it.
    try:
        from scipy.optimize import least_squares  # type: ignore
    except ImportError:  # pragma: no cover — scipy is a pinned dep
        result.note = "scipy not available"
        return result

    def residuals(params: np.ndarray) -> np.ndarray:
        a, b = params
        return y - (a * x + b)

    fit = least_squares(
        residuals,
        x0=np.array([1.0, 0.0]),
        loss="huber",
        f_scale=1.0,
        max_nfev=200,
    )
    a_fit, b_fit = float(fit.x[0]), float(fit.x[1])
    resid = residuals(fit.x)
    resid_std = float(np.sqrt(np.mean(resid**2)))

    result.ran = True
    result.a = a_fit
    result.b = b_fit
    result.residual_std_microrad = resid_std

    drift_a = abs(a_fit - 1.0)
    drift_b = abs(b_fit)
    if drift_a > ANCHOR_FIT_A_WARNING_FRACTION or drift_b > ANCHOR_FIT_B_WARNING_MICRORAD:
        result.warning = (
            f"anchor cross-check flags {source_name}: digital = "
            f"{a_fit:.4f} · png + {b_fit:+.3f}  "
            f"(|a-1|={drift_a:.1%}, |b|={drift_b:.2f} µrad, "
            f"residual_std={resid_std:.2f} µrad over {len(overlap)} buckets)"
        )
    return result


def apply_anchor_fit(df: pd.DataFrame, fit: AnchorFitResult) -> pd.DataFrame:
    """Apply a (a, b) correction from `recalibrate_by_anchor_fit` to a
    source's DataFrame in-place-copy fashion: returns a new DataFrame with
    tilt transformed as `tilt ← a * tilt + b` so it matches digital's frame.

    Identity when `fit.ran` is False or the fit is within tolerance.
    """
    if not fit.ran:
        return df
    if fit.warning is None:
        # Within tolerance — don't nudge the data, but record that the
        # fit ran cleanly.
        return df
    out = df.copy()
    out[TILT_COL] = fit.a * out[TILT_COL] + fit.b
    return out
