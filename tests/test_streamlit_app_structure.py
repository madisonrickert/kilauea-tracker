"""Lock the top-level Streamlit app structure.

We can't import ``streamlit_app.py`` in tests (it runs Streamlit code at
module scope), but we can verify its structural invariants by parsing the
AST and scanning the source. These tests catch accidental deletion of a
tab, a mismatched tab label, or a syntax regression before boot.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.fixture(scope="module")
def app_source() -> str:
    return APP_PATH.read_text()


@pytest.fixture(scope="module")
def app_tree(app_source: str) -> ast.Module:
    return ast.parse(app_source)


def test_app_parses(app_tree: ast.Module):
    """If this fails, streamlit_app.py has a syntax error."""
    assert isinstance(app_tree, ast.Module)


def test_app_declares_exactly_six_tabs(app_source: str):
    """One st.tabs() call with six labels: Now/Chart/Cameras/Data/Pipeline/About."""
    assert app_source.count("st.tabs([") == 1, (
        "expected exactly one st.tabs() call in streamlit_app.py"
    )
    for label in ("Now", "Chart", "Cameras", "Data", "Pipeline", "About"):
        assert f'"{label}"' in app_source, f"tab label {label!r} missing"


def test_every_tab_has_a_with_block(app_source: str):
    """Every tab declared must have a matching `with tab_X:` body."""
    for tab in (
        "tab_now", "tab_chart", "tab_cameras",
        "tab_data", "tab_pipeline", "tab_about",
    ):
        assert f"with {tab}:" in app_source, f"no `with {tab}:` block"


def test_about_tab_delegates_to_ui_module(app_source: str):
    """The About tab must render via ui.about_tab, not inline markdown."""
    # Find the `with tab_about:` block and confirm it calls about_tab.show().
    lines = app_source.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.strip() == "with tab_about:")
    body = "\n".join(lines[idx : idx + 10])
    assert "about_tab.show()" in body


def test_cameras_tab_delegates_to_ui_module(app_source: str):
    """The Cameras tab must render via ui.cameras.show_grid()."""
    lines = app_source.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.strip() == "with tab_cameras:")
    body = "\n".join(lines[idx : idx + 10])
    assert "cameras.show_grid()" in body


def test_no_legacy_inline_webcams(app_source: str):
    """The 2-col inline webcam expander must be gone (cameras module owns it)."""
    assert "USGS_WEBCAMS" not in app_source, (
        "legacy USGS_WEBCAMS constant still present — migrate to ui.cameras"
    )
    assert "USGS Kīlauea summit webcams" not in app_source, (
        "legacy webcam expander header still present"
    )


def test_sidebar_no_longer_holds_advanced_peaks(app_source: str):
    """Advanced peak-detection sliders must have moved to the Chart tab."""
    # The sidebar block should no longer reference the slider keys inline —
    # the keys are set via session_state and the actual widgets live in
    # tab_chart. Quick proxy: locate the sidebar block and confirm it does
    # NOT contain a slider for "Minimum prominence (µrad)".
    sidebar_start = app_source.index("with st.sidebar:")
    sidebar_end = app_source.index("# Session-state defaults for widgets", sidebar_start)
    sidebar_body = app_source[sidebar_start:sidebar_end]
    assert "Minimum prominence (µrad)" not in sidebar_body
    assert "Minimum spacing (days)" not in sidebar_body


def test_sidebar_no_longer_holds_inspector_overlays(app_source: str):
    sidebar_start = app_source.index("with st.sidebar:")
    sidebar_end = app_source.index("# Session-state defaults for widgets", sidebar_start)
    sidebar_body = app_source[sidebar_start:sidebar_end]
    assert 'key="ovl_dots"' not in sidebar_body
    assert "Inspector overlays" not in sidebar_body


def test_hero_state_banner_cameras_imports(app_source: str):
    """The top-level imports must include the new UI modules."""
    for symbol in ("about_tab", "cameras", "hero", "state_banner"):
        assert symbol in app_source, f"missing import for ui.{symbol}"
    assert "build_style_block" in app_source
