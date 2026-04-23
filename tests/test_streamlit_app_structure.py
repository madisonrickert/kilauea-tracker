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


def test_app_declares_exactly_five_tabs(app_source: str):
    """One st.tabs() call with five labels: Now/Chart/Cameras/Pipeline/About.
    The old "Data" tab was merged into Pipeline — its contents (reconcile +
    model diagnostics) are all behind-the-scenes instrumentation."""
    assert app_source.count("st.tabs([") == 1, (
        "expected exactly one st.tabs() call in streamlit_app.py"
    )
    # Extract the argument list to st.tabs([...]) and check labels appear there.
    start = app_source.index("st.tabs([")
    end = app_source.index("])", start)
    tabs_decl = app_source[start:end + 2]
    for label in ("Now", "Chart", "Cameras", "Pipeline", "About"):
        assert f'"{label}"' in tabs_decl, f"tab label {label!r} missing from st.tabs(...)"
    assert '"Data"' not in tabs_decl, (
        'the "Data" tab label still appears in st.tabs(...) — it was merged into Pipeline'
    )


def test_every_tab_has_a_with_block(app_source: str):
    """Every tab declared must have a matching `with tab_X:` body."""
    for tab in (
        "tab_now", "tab_chart", "tab_cameras",
        "tab_pipeline", "tab_about",
    ):
        assert f"with {tab}:" in app_source, f"no `with {tab}:` block"
    # Data tab is retired — no bare `with tab_data:` should remain.
    assert "with tab_data:" not in app_source, (
        "Data tab was merged into Pipeline; `with tab_data:` should not exist"
    )


def test_about_tab_delegates_to_ui_module(app_source: str):
    """The About tab must render via ui.about_tab, not inline markdown."""
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


def test_sidebar_retired(app_source: str):
    """The old ``with st.sidebar:`` block was retired in favour of a compact
    top bar + per-tab controls. Its reappearance would mean a regression."""
    assert "with st.sidebar:" not in app_source, (
        "st.sidebar block is back — controls should live in the top bar + tabs"
    )


def test_initial_sidebar_collapsed(app_source: str):
    """With the sidebar retired, ``initial_sidebar_state`` should not be
    ``expanded`` — otherwise Streamlit renders an empty sidebar by default."""
    assert 'initial_sidebar_state="collapsed"' in app_source


def test_top_bar_holds_refresh_and_tz(app_source: str):
    """The top bar, rendered below the title, must contain the Refresh
    button and the timezone selector that previously lived in the sidebar."""
    assert '"🔄 Refresh"' in app_source
    assert 'key="tw_timezone_choice"' in app_source


def test_chart_tab_holds_trendline_and_per_source(app_source: str):
    """Trendline window slider and per-source toggle must live in the
    Chart tab (co-located with the chart they affect)."""
    lines = app_source.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "with tab_chart:")
    end = next(
        i
        for i, ln in enumerate(lines[start + 1 :], start + 1)
        if ln.startswith("with tab_")
    )
    body = "\n".join(lines[start:end])
    assert 'key="tw_n_peaks_for_fit"' in body, (
        "trendline window slider must live in the Chart tab"
    )
    assert 'key="tw_show_per_source"' in body, (
        "per-source toggle must live in the Chart tab"
    )


def test_chart_tab_holds_csv_export(app_source: str):
    """CSV export must live in the Chart tab so highlight-to-export is
    a single-tab workflow (regression-prevention for feedback #10)."""
    lines = app_source.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "with tab_chart:")
    end = next(
        i
        for i, ln in enumerate(lines[start + 1 :], start + 1)
        if ln.startswith("with tab_")
    )
    body = "\n".join(lines[start:end])
    assert '"📤 Export data as CSV"' in body, (
        "CSV export expander must live in the Chart tab"
    )
    # Detected peaks moved here too — it depends on the Chart tab's sliders.
    assert "📍 Detected peaks" in body, (
        "Detected peaks table must live in the Chart tab, near its sliders"
    )


def test_now_tab_owns_hero_banner_cameras(app_source: str):
    """Hero, state banner, and camera strip render inside tab_now so
    navigating between tabs doesn't double-up on the hero/banner above."""
    lines = app_source.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "with tab_now:")
    end = next(
        i
        for i, ln in enumerate(lines[start + 1 :], start + 1)
        if ln.startswith("with tab_")
    )
    body = "\n".join(lines[start:end])
    assert "hero.show(" in body
    assert "state_banner.show(" in body
    assert "cameras.show_strip()" in body


def test_no_ingest_status_above_the_fold(app_source: str):
    """The noisy ingest-pipeline-status expander must not render above the
    tabs on the front page. It still lives inside the Pipeline tab."""
    lines = app_source.splitlines()
    tabs_idx = next(i for i, ln in enumerate(lines) if "st.tabs([" in ln)
    prelude = "\n".join(lines[:tabs_idx])
    assert '"ⓘ Ingest pipeline status' not in prelude, (
        "ingest status should live inside the Pipeline tab, not above it"
    )


def test_hero_state_banner_cameras_imports(app_source: str):
    """The top-level imports must include the new UI modules."""
    for symbol in ("about_tab", "cameras", "hero", "state_banner"):
        assert symbol in app_source, f"missing import for ui.{symbol}"
    assert "build_style_block" in app_source
