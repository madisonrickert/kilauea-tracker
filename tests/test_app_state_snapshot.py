"""Tests for the widget-snapshot read surface.

The frozen-dataclass snapshot is just a typed view over
``st.session_state``; these tests exercise that the keys, defaults,
and types match between ``WIDGET_DEFAULTS``, ``init_widget_defaults``,
and ``widget_snapshot``.
"""

from __future__ import annotations

import pytest

st = pytest.importorskip("streamlit")

from kilauea_tracker.state.snapshot import (  # noqa: E402
    ChartWidgets,
    OverlayLayers,
    PeakWidgets,
    WidgetSnapshot,
)
from kilauea_tracker.state.widgets import (  # noqa: E402
    WIDGET_DEFAULTS,
    init_widget_defaults,
    widget_snapshot,
)


@pytest.fixture(autouse=True)
def clear_session_state():
    """Each test starts with a clean session_state dict."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    yield
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def test_init_widget_defaults_seeds_every_key():
    init_widget_defaults()
    for key, expected in WIDGET_DEFAULTS.items():
        assert key in st.session_state, f"missing key after init: {key}"
        assert st.session_state[key] == expected


def test_init_widget_defaults_is_idempotent_does_not_clobber():
    init_widget_defaults()
    st.session_state["adv_min_prominence"] = 0.42
    init_widget_defaults()  # second call must not reset
    assert st.session_state["adv_min_prominence"] == 0.42


def test_widget_snapshot_returns_typed_dataclasses():
    init_widget_defaults()
    snap = widget_snapshot()
    assert isinstance(snap, WidgetSnapshot)
    assert isinstance(snap.chart, ChartWidgets)
    assert isinstance(snap.peaks, PeakWidgets)
    assert isinstance(snap.overlays, OverlayLayers)


def test_widget_snapshot_reflects_current_session_state():
    init_widget_defaults()
    st.session_state["tw_n_peaks_for_fit"] = 12
    st.session_state["adv_min_prominence"] = 7.5
    st.session_state["ovl_bbox"] = True

    snap = widget_snapshot()
    assert snap.chart.n_peaks_for_fit == 12
    assert snap.peaks.min_prominence == 7.5
    assert snap.overlays.bbox is True
    # Untouched defaults still pass through
    assert snap.chart.timezone_choice == "HST (Pacific/Honolulu)"
    assert snap.overlays.dots is True


def test_widget_snapshot_is_frozen():
    from dataclasses import FrozenInstanceError

    init_widget_defaults()
    snap = widget_snapshot()
    with pytest.raises(FrozenInstanceError):
        snap.chart.n_peaks_for_fit = 99  # type: ignore[misc]


def test_widget_defaults_keys_match_snapshot_fields():
    """Every key in WIDGET_DEFAULTS must be readable in the snapshot."""
    init_widget_defaults()
    # If a key is in WIDGET_DEFAULTS but widget_snapshot doesn't read
    # it, this test is fine (the snapshot is a *subset* by design,
    # though today it's a 1:1 match). The thing we want to catch is
    # the reverse: snapshot reading a key that init_widget_defaults
    # didn't seed → KeyError. Since widget_snapshot indexes
    # st.session_state directly, calling it after init_widget_defaults
    # is the test.
    snap = widget_snapshot()
    assert snap is not None  # didn't raise
