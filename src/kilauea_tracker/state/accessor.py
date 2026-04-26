"""``get_state()`` — assemble the unified ``AppState`` snapshot.

One call per script rerun. Every field is either a session_state
read, a stat() call, or a snapshot of the cache_resource singleton.
Cheap by design — pages can call this freely.
"""

from __future__ import annotations

from ..config import HISTORY_CSV
from .refresh_store import get_refresh_store
from .snapshot import AppState
from .widgets import widget_snapshot


def get_state() -> AppState:
    """Return a frozen view of every read-side state in the app.

    Components: refresh-subsystem snapshot (cross-thread),
    widget snapshot (per-tab), the session-bound ``last_ingest_at``,
    and the ``tilt_history.csv`` mtime that drives mtime-keyed caches.
    """
    import streamlit as st

    return AppState(
        refresh=get_refresh_store().snapshot(),
        widgets=widget_snapshot(),
        last_ingest_at=st.session_state.get("last_ingest_at"),
        history_mtime=(
            HISTORY_CSV.stat().st_mtime if HISTORY_CSV.exists() else 0.0
        ),
    )
