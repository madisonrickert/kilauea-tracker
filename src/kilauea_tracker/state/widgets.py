"""Widget-state defaults and typed read accessors.

Streamlit's widget→key auto-binding still owns the writes — every
slider/selectbox/toggle in the app continues to declare ``key="tw_..."``
or ``key="adv_..."`` and Streamlit copies the user's value into
``st.session_state[key]`` on each interaction. This module is the
*read* side: pages call ``widget_snapshot()`` once per rerun and
receive a frozen ``WidgetSnapshot`` typed dataclass instead of
poking ``st.session_state["adv_min_prominence"]`` directly.

Single source of truth for the keys + their defaults lives below in
``WIDGET_DEFAULTS``. The shell calls ``init_widget_defaults()`` once
near the top of ``streamlit_app.py`` to seed every key with its
default; the rest of the app reads through the typed snapshot.
"""

from __future__ import annotations

from ..config import PEAK_DEFAULTS
from .snapshot import ChartWidgets, OverlayLayers, PeakWidgets, WidgetSnapshot

# Default values for every widget key the app uses. The shape of this
# dict is *the* declaration of the widget surface — adding a new
# session_state-backed widget means adding it here AND extending the
# corresponding sub-dataclass in snapshot.py.
WIDGET_DEFAULTS: dict[str, object] = {
    # Topbar / chart-shell
    "tw_n_peaks_for_fit": 6,
    "tw_show_per_source": False,
    "tw_timezone_choice": "HST (Pacific/Honolulu)",
    # Advanced peak detection (Chart tab)
    "adv_min_prominence": PEAK_DEFAULTS.min_prominence,
    "adv_min_distance_days": PEAK_DEFAULTS.min_distance_days,
    "adv_min_height": -10.0,  # legacy floor; UI lets users override
    # Inspector overlay layers (Pipeline tab) — twelve toggles
    "ovl_dots": True,
    "ovl_bbox": False,
    "ovl_yticks": False,
    "ovl_ygrid": False,
    "ovl_corners": False,
    "ovl_blue": False,
    "ovl_legend": False,
    "ovl_dropcols": False,
    "ovl_outliers": False,
    "ovl_now": False,
    "ovl_green": False,
    "ovl_csv": False,
}


def init_widget_defaults() -> None:
    """Seed every widget key in ``st.session_state`` with its default.

    Idempotent — uses ``setdefault``, so re-runs after the user has
    interacted don't clobber their selections. Call once near the top
    of ``streamlit_app.py`` before any page body renders.
    """
    import streamlit as st

    for key, default in WIDGET_DEFAULTS.items():
        st.session_state.setdefault(key, default)


def widget_snapshot() -> WidgetSnapshot:
    """Return the current widget values as a frozen ``WidgetSnapshot``.

    Idempotently seeds ``WIDGET_DEFAULTS`` first — Streamlit's v1
    multipage auto-discovery (any ``pages/`` directory) runs each page
    as an independent script, so we can't assume ``streamlit_app.py``
    has already called ``init_widget_defaults()``. Calling
    ``setdefault`` on already-seeded keys is a no-op, so paying for
    this on every snapshot is cheaper than the alternative
    (KeyError if a user deep-links to ``/chart`` and the entry script
    didn't run first).
    """
    import streamlit as st

    init_widget_defaults()
    return WidgetSnapshot(
        chart=ChartWidgets(
            n_peaks_for_fit=int(st.session_state["tw_n_peaks_for_fit"]),
            show_per_source=bool(st.session_state["tw_show_per_source"]),
            timezone_choice=str(st.session_state["tw_timezone_choice"]),
        ),
        peaks=PeakWidgets(
            min_prominence=float(st.session_state["adv_min_prominence"]),
            min_distance_days=float(st.session_state["adv_min_distance_days"]),
            min_height=float(st.session_state["adv_min_height"]),
        ),
        overlays=OverlayLayers(
            dots=bool(st.session_state["ovl_dots"]),
            bbox=bool(st.session_state["ovl_bbox"]),
            yticks=bool(st.session_state["ovl_yticks"]),
            ygrid=bool(st.session_state["ovl_ygrid"]),
            corners=bool(st.session_state["ovl_corners"]),
            blue=bool(st.session_state["ovl_blue"]),
            legend=bool(st.session_state["ovl_legend"]),
            dropcols=bool(st.session_state["ovl_dropcols"]),
            outliers=bool(st.session_state["ovl_outliers"]),
            now=bool(st.session_state["ovl_now"]),
            green=bool(st.session_state["ovl_green"]),
            csv=bool(st.session_state["ovl_csv"]),
        ),
    )
