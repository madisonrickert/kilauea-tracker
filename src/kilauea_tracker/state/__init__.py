"""Unified app-state module.

Hybrid storage by design:

  * ``RefreshStore`` (singleton via ``@st.cache_resource``, persisted
    to ``data/refresh_status.json``) owns cross-thread refresh state.
    Both the manual click handler and the page-load background daemon
    write through the same ``start() / advance() / complete()`` API,
    so the topbar fragment has a single source of truth for "is a
    fetch in flight?".

  * ``WidgetSnapshot`` is a typed read over ``st.session_state`` for
    the per-tab, in-memory widget keys (``tw_*``, ``adv_*``, ``ovl_*``).
    Streamlit's normal widget→key auto-binding still owns the writes;
    we just give pages a dataclass to read instead of stringly-typed
    dictionary lookups.

The two surfaces are unified behind ``get_state()`` which returns a
frozen ``AppState`` snapshot. Pages call ``get_state()`` once at the
top of their script and read fields off the resulting dataclass.
"""

from __future__ import annotations

from .accessor import get_state
from .refresh_store import REFRESH_STATUS_FILE, RefreshStore, get_refresh_store
from .snapshot import (
    AppState,
    ChartWidgets,
    OverlayLayers,
    PeakWidgets,
    RefreshSnapshot,
    RefreshSource,
    WidgetSnapshot,
)
from .widgets import WIDGET_DEFAULTS, init_widget_defaults, widget_snapshot

__all__ = [
    "REFRESH_STATUS_FILE",
    "WIDGET_DEFAULTS",
    "AppState",
    "ChartWidgets",
    "OverlayLayers",
    "PeakWidgets",
    "RefreshSnapshot",
    "RefreshSource",
    "RefreshStore",
    "WidgetSnapshot",
    "get_refresh_store",
    "get_state",
    "init_widget_defaults",
    "widget_snapshot",
]
