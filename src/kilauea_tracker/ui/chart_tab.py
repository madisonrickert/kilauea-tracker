"""Chart tab — the main Plotly figure plus the model-tuning controls.

Takes pre-computed data (tilt + peaks + prediction + overlays) and renders.
Peak-detection tuning sliders used to live in the sidebar; they're now here,
because they are the scientific controls for the chart and only matter to
users who are actively reading it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..plotting import build_figure

if TYPE_CHECKING:
    import pandas as pd

    from ..model import Prediction


@dataclass
class PeakTuningValues:
    """Values pulled from the model-tuning sliders — passed back to the caller."""

    trendline_window: int
    min_prominence: float
    min_distance_days: int
    min_height: float


def render_peak_tuning(
    *,
    default_trendline_window: int,
    default_min_prominence: float,
    default_min_distance_days: int,
    default_min_height: float,
    max_peaks_available: int,
) -> PeakTuningValues:
    """Render the "Advanced model tuning" expander. Returns the slider values."""
    import streamlit as st

    with st.expander("⚙️ Advanced model tuning", expanded=False):
        st.caption(
            "Move these if you want to see how the fit reacts to different "
            "peak-detection settings. The defaults are tuned for UWD Az 300°."
        )
        trendline_window = st.slider(
            "Trendline window (number of recent peaks)",
            min_value=2,
            max_value=max(max_peaks_available, 2),
            value=min(default_trendline_window, max(max_peaks_available, 2)),
            help="How many recent peaks feed the linear trendline fit.",
        )
        st.markdown("**Peak detection sensitivity**")
        col1, col2, col3 = st.columns(3)
        with col1:
            min_prominence = st.slider(
                "Min prominence (µrad)", 0.5, 10.0, default_min_prominence, 0.1
            )
        with col2:
            min_distance_days = st.slider(
                "Min distance (days)", 3, 30, default_min_distance_days, 1
            )
        with col3:
            min_height = st.slider(
                "Min height (µrad)", -5.0, 30.0, default_min_height, 0.5
            )
    return PeakTuningValues(
        trendline_window=trendline_window,
        min_prominence=min_prominence,
        min_distance_days=min_distance_days,
        min_height=min_height,
    )


def show(
    *,
    tilt_df: pd.DataFrame,
    fit_peaks_df: pd.DataFrame,
    prediction: Prediction | None,
    all_peaks_df: pd.DataFrame | None,
    per_source_overlay: dict[str, pd.DataFrame] | None,
    state: str | None = None,
    show_current_episode: bool = True,
    show_next_event_prediction: bool = True,
) -> None:
    """Render the main chart at full width."""
    import streamlit as st

    fig = build_figure(
        tilt_df,
        fit_peaks_df,
        prediction,
        all_peaks_df=all_peaks_df,
        per_source_overlay=per_source_overlay,
        show_current_episode=show_current_episode,
        show_next_event_prediction=show_next_event_prediction,
        state=state,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Hover for exact values. Drag to zoom. Double-click to reset. "
        "Toggle per-source overlays from the legend."
    )
