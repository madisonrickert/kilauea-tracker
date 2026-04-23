"""Smoke tests for the UI tab modules.

We don't spin up Streamlit — these tests verify the modules import cleanly,
their public symbols exist, and simple pure helpers behave.
"""

from __future__ import annotations

import pytest

from kilauea_tracker.ui import about_tab, chart_tab, data_tab, pipeline_tab


def test_about_tab_module_has_show():
    assert callable(about_tab.show)


def test_about_tab_overview_mentions_key_anchors():
    """The About copy must mention the core concepts every visitor will ask about."""
    body = about_tab._OVERVIEW_MARKDOWN
    for anchor in ("UWD", "azimuth 300°", "USGS", "trendline", "exponential"):
        assert anchor in body, f"About copy missing anchor {anchor!r}"


def test_chart_tab_public_surface():
    assert callable(chart_tab.show)
    assert callable(chart_tab.render_peak_tuning)
    # Dataclass must be constructible with the four expected fields.
    v = chart_tab.PeakTuningValues(
        trendline_window=5, min_prominence=3.0,
        min_distance_days=7, min_height=5.0,
    )
    assert v.trendline_window == 5


def test_data_tab_public_surface():
    assert callable(data_tab.show)
    sel = data_tab.ExportSelection(
        start=__import__("datetime").date(2026, 1, 1),
        end=__import__("datetime").date(2026, 4, 22),
        include_sources=["week"],
        mode="simple",
    )
    assert sel.mode == "simple"


def test_pipeline_tab_public_surface():
    assert callable(pipeline_tab.show)


def test_pipeline_tab_show_accepts_all_none(monkeypatch):
    """Passing no renderers must not raise — the caption still renders."""
    calls = []

    class _StSpy:
        def caption(self, *a, **k): calls.append(("caption", a))
        def markdown(self, *a, **k): calls.append(("markdown", a))
        def expander(self, *a, **k):
            class _Cm:
                def __enter__(self_): return None
                def __exit__(self_, *exc): return False
            return _Cm()

    import sys
    monkeypatch.setitem(sys.modules, "streamlit", _StSpy())
    pipeline_tab.show()
    # caption ran; no errors.
    assert any(call[0] == "caption" for call in calls)
