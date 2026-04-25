"""Pipeline tab — maintainer surface for the ingest pipeline.

Always visible in the tab bar so curious visitors can peek, but clearly
labeled as "behind-the-scenes" so they understand this is not the user path.

The tab is a thin wrapper: streamlit_app.py passes in the already-built
diagnostic renderers (ingest status, per-source PNGs, inspector overlays,
run reports). We just organize them under sub-sections here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def show(
    *,
    ingest_status_renderer: Callable[[], None] | None = None,
    source_plots_renderer: Callable[[], None] | None = None,
    inspector_overlays_renderer: Callable[[], None] | None = None,
    run_reports_renderer: Callable[[], None] | None = None,
) -> None:
    """Render the Pipeline tab as a sequence of sub-sections."""
    import streamlit as st

    st.caption(
        "**Behind the scenes** — this tab shows the data-ingest pipeline that "
        "fetches the USGS tilt PNGs, OCRs the axis labels, traces the curves, "
        "and reconciles everything into one 15-minute time series. Useful "
        "for sanity-checking the model inputs and for debugging the day the "
        "USGS page changes its plot fonts."
    )

    if ingest_status_renderer is not None:
        with st.expander("📡 Ingest pipeline status", expanded=True):
            ingest_status_renderer()

    if source_plots_renderer is not None:
        with st.expander("🛰 USGS source plots + calibration", expanded=False):
            source_plots_renderer()

    if inspector_overlays_renderer is not None:
        with st.expander("🔍 PNG transcription quality inspector", expanded=False):
            inspector_overlays_renderer()

    if run_reports_renderer is not None:
        with st.expander("📜 Recent refresh runs", expanded=False):
            run_reports_renderer()
