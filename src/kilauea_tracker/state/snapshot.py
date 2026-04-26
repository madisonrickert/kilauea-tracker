"""Frozen-dataclass view types for the unified `AppState`.

These types are the *read surface* every page and view binds to. They
carry no logic — assembly happens in `state.accessor.get_state()`,
mutation happens through explicit store methods or Streamlit's normal
widget→key auto-binding.

Kept dependency-free (stdlib only) so any layer of the app — including
pure compute layers — can depend on these types without pulling in
Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime

RefreshSource = Literal["manual", "background"]


@dataclass(frozen=True)
class RefreshSnapshot:
    """The current state of the ingest-refresh subsystem.

    Assembled by ``RefreshStore.snapshot()`` from the persisted JSON
    file plus the in-memory singleton. Cross-thread, cross-tab, and
    cross-restart durable.
    """

    running: bool
    current_stage: str | None
    started_utc: datetime | None
    finished_utc: datetime | None
    last_error: str | None
    source: RefreshSource | None


@dataclass(frozen=True)
class ChartWidgets:
    """Topbar / chart widget state — keys ``tw_*`` in session_state."""

    n_peaks_for_fit: int
    show_per_source: bool
    timezone_choice: str


@dataclass(frozen=True)
class PeakWidgets:
    """Advanced peak-detection sliders — keys ``adv_*`` in session_state."""

    min_prominence: float
    min_distance_days: float
    min_height: float


@dataclass(frozen=True)
class ModelWidgets:
    """Prediction-model selector — key ``adv_active_model_id`` in session_state.

    Controls which registered model drives the Chart page's hero, banner,
    state classification, and overlay curves. The Now page intentionally
    does NOT read this; it always uses ``DEFAULT_MODEL_ID`` so non-
    technical visitors see a stable headline.
    """

    active_id: str


@dataclass(frozen=True)
class OverlayLayers:
    """Pipeline-tab inspector overlay toggles — keys ``ovl_*`` in session_state."""

    dots: bool
    bbox: bool
    yticks: bool
    ygrid: bool
    corners: bool
    blue: bool
    legend: bool
    dropcols: bool
    outliers: bool
    now: bool
    green: bool
    csv: bool


@dataclass(frozen=True)
class WidgetSnapshot:
    """Typed read of every widget value the app cares about.

    Streamlit's widget→key auto-binding still owns the writes; this
    snapshot is read-only. Assembled per rerun by
    ``state.widgets.widget_snapshot()``.
    """

    chart: ChartWidgets
    peaks: PeakWidgets
    overlays: OverlayLayers
    model: ModelWidgets


@dataclass(frozen=True)
class AppState:
    """The unified read view returned by ``get_state()``.

    One instance per script rerun. Reading it is cheap — every field
    is either a session_state lookup or a stat call.
    """

    refresh: RefreshSnapshot
    widgets: WidgetSnapshot
    last_ingest_at: datetime | None
    history_mtime: float
