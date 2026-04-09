"""Auto-detect episodic tilt peaks.

Replaces the hardcoded `peak_data_for_trendline` literal at
`legacy/eruption_projection.py:50-67`. The shape we're looking for is
characteristic: tilt rises slowly over days, hits a local maximum, then
drops sharply (within hours) as the eruption pulse releases pressure.

`scipy.signal.find_peaks` handles this cleanly when given a uniform-time-step
input — so we resample to a 1-hour grid first.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd
from scipy.signal import find_peaks

from .config import PEAK_DEFAULTS
from .model import DATE_COL, TILT_COL


def detect_peaks(
    tilt_df: pd.DataFrame,
    min_prominence: float | None = None,
    min_distance_days: float | None = None,
    min_height: float | None = None,
) -> pd.DataFrame:
    """Detect episodic tilt peaks.

    Args:
        tilt_df:           DataFrame with ['Date', 'Tilt (microradians)'].
        min_prominence:    Minimum peak prominence in microradians. Defaults
                           to `config.PEAK_DEFAULTS.min_prominence` (4 µrad).
        min_distance_days: Minimum time gap between detected peaks. Defaults
                           to `config.PEAK_DEFAULTS.min_distance_days` (5 days).
        min_height:        Minimum absolute tilt for a sample to count as a
                           candidate peak. Defaults to
                           `config.PEAK_DEFAULTS.min_height` (5 µrad).

    Returns:
        DataFrame with columns ['Date', 'Tilt (microradians)', 'prominence'],
        sorted by Date. Empty if no peaks meet the thresholds.
    """
    defaults = asdict(PEAK_DEFAULTS)
    if min_prominence is None:
        min_prominence = defaults["min_prominence"]
    if min_distance_days is None:
        min_distance_days = defaults["min_distance_days"]
    if min_height is None:
        min_height = defaults["min_height"]

    if len(tilt_df) == 0:
        return _empty_peaks_df()

    # Resample to a uniform 1-hour grid. Required because find_peaks's
    # `distance` parameter is measured in samples, not time. We average
    # duplicate timestamps via mean, then linearly interpolate the gaps.
    df = (
        tilt_df[[DATE_COL, TILT_COL]]
        .dropna()
        .sort_values(DATE_COL)
        .set_index(DATE_COL)
    )
    df = df[~df.index.duplicated(keep="last")]
    if len(df) < 2:
        return _empty_peaks_df()

    resampled = df.resample("1h").mean().interpolate(method="linear")
    if len(resampled) == 0:
        return _empty_peaks_df()

    y = resampled[TILT_COL].to_numpy()
    distance_samples = max(int(round(min_distance_days * 24)), 1)

    indices, props = find_peaks(
        y,
        prominence=min_prominence,
        distance=distance_samples,
        height=min_height,
    )

    if len(indices) == 0:
        return _empty_peaks_df()

    return pd.DataFrame(
        {
            DATE_COL: resampled.index[indices],
            TILT_COL: y[indices],
            "prominence": props["prominences"],
        }
    ).reset_index(drop=True)


def _empty_peaks_df() -> pd.DataFrame:
    return pd.DataFrame(
        {DATE_COL: pd.Series(dtype="datetime64[ns]"), TILT_COL: [], "prominence": []}
    )
