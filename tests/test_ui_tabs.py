"""Smoke tests for the surviving UI modules.

We don't spin up Streamlit — these tests verify the modules import
cleanly, their public symbols exist, and simple pure helpers behave.

The legacy ``chart_tab``, ``data_tab``, and ``pipeline_tab`` modules
were removed in the st.navigation migration; their per-page logic
moved into ``pages/*.py`` and is covered by
``tests/test_streamlit_app_structure.py``.
"""

from __future__ import annotations

from kilauea_tracker.ui import about_tab


def test_about_tab_module_has_show():
    assert callable(about_tab.show)


def test_about_tab_overview_mentions_key_anchors():
    """The About copy must mention the core concepts every visitor will ask about."""
    body = (
        about_tab._ABOUT_PROJECT_MARKDOWN
        + about_tab._HOW_IT_WORKS_MARKDOWN
        + about_tab._AUTHOR_MARKDOWN
    )
    for anchor in ("UWD", "azimuth 300°", "USGS", "trendline", "exponential"):
        assert anchor in body, f"About copy missing anchor {anchor!r}"


def test_about_tab_introduces_project_and_author():
    """Feedback-locked: the About tab must cover the project story AND the
    author, not just the technical explanation of the model."""
    body = (
        about_tab._ABOUT_PROJECT_MARKDOWN
        + about_tab._HOW_IT_WORKS_MARKDOWN
        + about_tab._AUTHOR_MARKDOWN
    )
    assert "About this project" in body
    assert "About the author" in body
    assert "Madison Rickert" in body
    assert "github.com/madisonrickert" in body
