"""Data tab — CSV export + reconciliation diagnostics + detected-peaks table.

Takes the already-loaded tilt DataFrame and the ingest result. Composes the
export controls. Heavy lifting is left to the caller (simple / debug / ZIP
modes all build their bytes from the same tilt_df — this module is the UI
shell around them).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd

from ..model import DATE_COL, TILT_COL


@dataclass
class ExportSelection:
    """What the user asked to export. The caller turns this into bytes."""

    start: date
    end: date
    include_sources: list[str]
    mode: str  # "simple" | "debug" | "zip"


def _date_range_selector(tilt_df: pd.DataFrame) -> tuple[date, date]:
    """Start + end date input pair with sensible defaults from the data."""
    import streamlit as st

    if len(tilt_df) == 0:
        today = pd.Timestamp.utcnow().date()
        return today, today

    min_d = pd.Timestamp(tilt_df[DATE_COL].min()).date()
    max_d = pd.Timestamp(tilt_df[DATE_COL].max()).date()
    default_start = max(min_d, date(max_d.year, 1, 1))

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input(
            "Start date", value=default_start, min_value=min_d, max_value=max_d
        )
    with col2:
        end = st.date_input(
            "End date", value=max_d, min_value=min_d, max_value=max_d
        )
    if start > end:
        st.warning("Start date is after end date — swapping.")
        start, end = end, start
    return start, end


def _sources_checkbox_grid(available_sources: list[str]) -> list[str]:
    """Render the per-source include checkboxes. Returns the selected names."""
    import streamlit as st

    st.markdown("**Sources to include**")
    cols = st.columns(min(len(available_sources), 4) or 1)
    selected: list[str] = []
    for i, src in enumerate(available_sources):
        with cols[i % len(cols)]:
            if st.checkbox(src, value=True, key=f"export-src-{src}"):
                selected.append(src)
    return selected


def show(
    *,
    tilt_df: pd.DataFrame,
    all_peaks_df: Optional[pd.DataFrame],
    available_sources: list[str],
    simple_csv_builder,
    debug_csv_builder,
    zip_builder,
    reconciliation_renderer: Optional[callable] = None,
) -> None:
    """Render the Data tab.

    The three ``*_builder`` callbacks take (start, end, include_sources) and
    return ``(filename, bytes)``. Kept as callbacks so this module doesn't
    need to import the reconciliation / archive modules directly.
    """
    import streamlit as st

    st.markdown("### Export tilt history as CSV")
    start, end = _date_range_selector(tilt_df)
    include_sources = _sources_checkbox_grid(available_sources)

    if not include_sources:
        st.info("Select at least one source to export.")
    else:
        n_rows = len(
            tilt_df[
                (tilt_df[DATE_COL].dt.date >= start)
                & (tilt_df[DATE_COL].dt.date <= end)
            ]
        )
        st.caption(f"~{n_rows:,} rows in the selected range.")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Simple CSV", use_container_width=True):
                name, payload = simple_csv_builder(start, end, include_sources)
                st.download_button("Download " + name, payload, file_name=name,
                                   mime="text/csv", use_container_width=True)
        with col2:
            if st.button("Debug CSV", use_container_width=True):
                name, payload = debug_csv_builder(start, end, include_sources)
                st.download_button("Download " + name, payload, file_name=name,
                                   mime="text/csv", use_container_width=True)
        with col3:
            if st.button("Full ZIP (+run reports)", use_container_width=True):
                name, payload = zip_builder(start, end, include_sources)
                st.download_button("Download " + name, payload, file_name=name,
                                   mime="application/zip", use_container_width=True)

    if reconciliation_renderer is not None:
        st.markdown("---")
        st.markdown("### Reconciliation diagnostics")
        reconciliation_renderer()

    if all_peaks_df is not None and len(all_peaks_df) > 0:
        st.markdown("---")
        st.markdown(f"### Detected peaks ({len(all_peaks_df)} total)")
        st.dataframe(
            all_peaks_df.sort_values(DATE_COL, ascending=False),
            width="stretch",
            hide_index=True,
        )
