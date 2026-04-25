"""Lock the top-level Streamlit app structure.

We can't import the page scripts in tests (each runs Streamlit code at
module scope), but we can verify their structural invariants by parsing
the AST and scanning the source. These tests catch accidental deletion
of a page, a mismatched page label, a syntax regression, or a re-
introduction of legacy patterns (the retired sidebar, an above-the-fold
ingest panel) before boot.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "streamlit_app.py"
PAGES_DIR = REPO_ROOT / "pages"


@pytest.fixture(scope="module")
def app_source() -> str:
    return APP_PATH.read_text()


@pytest.fixture(scope="module")
def app_tree(app_source: str) -> ast.Module:
    return ast.parse(app_source)


def _read_page(name: str) -> str:
    return (PAGES_DIR / name).read_text()


def test_app_parses(app_tree: ast.Module):
    """If this fails, streamlit_app.py has a syntax error."""
    assert isinstance(app_tree, ast.Module)


def test_app_declares_exactly_five_pages(app_source: str):
    """One ``st.navigation([...])`` call registering five ``st.Page(...)``
    pages: now / chart / cameras / pipeline / about."""
    assert app_source.count("st.navigation(") == 1, (
        "expected exactly one st.navigation() call in streamlit_app.py"
    )
    # Each page declaration appears as st.Page("pages/<name>.py", ...).
    expected = ("now", "chart", "cameras", "pipeline", "about")
    for name in expected:
        assert f'st.Page("pages/{name}.py"' in app_source, (
            f"missing st.Page declaration for pages/{name}.py"
        )
    assert "tab_data" not in app_source, (
        "Data tab was merged into Pipeline; no tab_data reference should remain"
    )


def test_every_page_file_exists():
    """Each registered page must have a corresponding pages/<name>.py file."""
    for name in ("now.py", "chart.py", "cameras.py", "pipeline.py", "about.py"):
        assert (PAGES_DIR / name).exists(), f"missing page file: pages/{name}"


def test_about_page_delegates_to_ui_module():
    """The About page must render via ui.about_tab, not inline markdown."""
    body = _read_page("about.py")
    assert "about_tab.show()" in body


def test_cameras_page_delegates_to_ui_module():
    """The Cameras page must render via ui.cameras.show_grid()."""
    body = _read_page("cameras.py")
    assert "cameras.show_grid()" in body


def test_no_legacy_inline_webcams(app_source: str):
    """The 2-col inline webcam expander must be gone (cameras module owns
    it). Check both the entrypoint and pages/now.py."""
    now_body = _read_page("now.py")
    for source, name in ((app_source, "streamlit_app.py"), (now_body, "pages/now.py")):
        assert "USGS_WEBCAMS" not in source, (
            f"legacy USGS_WEBCAMS constant still present in {name}"
        )
        assert "USGS Kīlauea summit webcams" not in source, (
            f"legacy webcam expander header still present in {name}"
        )


def test_sidebar_retired(app_source: str):
    """The old ``with st.sidebar:`` block was retired in favour of a compact
    top bar + per-page controls. Its reappearance would mean a regression."""
    assert "with st.sidebar:" not in app_source, (
        "st.sidebar block is back — controls should live in the top bar + pages"
    )


def test_initial_sidebar_collapsed(app_source: str):
    """With the sidebar retired, ``initial_sidebar_state`` should not be
    ``expanded`` — otherwise Streamlit renders an empty sidebar by default."""
    assert 'initial_sidebar_state="collapsed"' in app_source


def test_top_bar_holds_refresh_and_tz(app_source: str):
    """The top bar, rendered above ``st.navigation``, must contain the
    Refresh button and the timezone selector that previously lived in the
    sidebar — both stay in the entrypoint so they're persistent across
    page changes."""
    assert '"🔄 Refresh"' in app_source
    assert 'key="tw_timezone_choice"' in app_source


def test_chart_page_holds_trendline_and_per_source():
    """Trendline window slider and per-source toggle must live on the
    Chart page (co-located with the chart they affect)."""
    body = _read_page("chart.py")
    assert 'key="tw_n_peaks_for_fit"' in body, (
        "trendline window slider must live in pages/chart.py"
    )
    assert 'key="tw_show_per_source"' in body, (
        "per-source toggle must live in pages/chart.py"
    )


def test_chart_page_holds_csv_export():
    """CSV export must live on the Chart page so highlight-to-export is
    a single-page workflow."""
    body = _read_page("chart.py")
    assert '"📤 Export data as CSV"' in body, (
        "CSV export expander must live in pages/chart.py"
    )
    # Detected peaks moved here too — it depends on the Chart page's sliders.
    assert "📍 Detected peaks" in body, (
        "Detected peaks table must live in pages/chart.py, near its sliders"
    )


def test_now_page_owns_hero_banner_cameras():
    """Hero, state banner, and camera strip render inside pages/now.py so
    navigating between pages doesn't double-up on them."""
    body = _read_page("now.py")
    assert "hero.show(" in body
    assert "state_banner.show(" in body
    assert "cameras.show_strip()" in body


def test_no_ingest_status_above_the_nav(app_source: str):
    """The noisy ingest-pipeline-status expander must not render above
    the navigation in the entrypoint. It still lives inside the Pipeline
    page."""
    lines = app_source.splitlines()
    nav_idx = next(i for i, ln in enumerate(lines) if "st.navigation(" in ln)
    prelude = "\n".join(lines[:nav_idx])
    assert '"📡 Ingest pipeline status' not in prelude, (
        "ingest status should live in pages/pipeline.py, not above the nav"
    )


def test_pipeline_page_owns_inspector():
    """The PNG transcription inspector + reconcile diagnostics must live
    on the Pipeline page."""
    body = _read_page("pipeline.py")
    assert "Transcription quality inspector" in body
    assert "Reconcile diagnostics" in body
    assert "Inspector overlay layers" in body


def test_now_page_uses_page_link_ctas():
    """The Now-page CTAs that jump to Chart / Cameras must be st.page_link
    widgets pointing at the canonical page paths — not the legacy
    st.button + on_click pattern from the URL-router fix."""
    body = _read_page("now.py")
    assert "st.page_link(" in body, "Now page must use st.page_link for CTAs"
    assert '"pages/chart.py"' in body, "Now page must link to /chart"
    assert '"pages/cameras.py"' in body, "Now page must link to /cameras"
