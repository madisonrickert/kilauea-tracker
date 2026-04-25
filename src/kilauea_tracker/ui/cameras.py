"""USGS HVO webcams — the eight live still-capture endpoints.

Two renderers:
    - ``show_strip(n=4)`` — compact 4-wide grid for the Now tab (hero context).
    - ``show_grid()`` — full 8-camera grid for the Cameras tab.

The webcam table is the single source of truth (was duplicated in
``streamlit_app.py`` prior to the overhaul). Each entry carries the USGS URL
and a human description that doubles as the ``alt`` attribute on the image
for screen readers.

URLs last HEAD-checked 2026-04-09 against the USGS HVO site.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Webcam:
    name: str
    description: str
    url: str


WEBCAMS: tuple[Webcam, ...] = (
    Webcam(
        "K2cam",
        "Caldera view from Uēkahuna bluff observation tower",
        "https://volcanoes.usgs.gov/observatories/hvo/cams/K2cam/images/M.jpg",
    ),
    Webcam(
        "V1cam",
        "West Halemaʻumaʻu crater from northwest rim",
        "https://volcanoes.usgs.gov/observatories/hvo/cams/V1cam/images/M.jpg",
    ),
    Webcam(
        "V2cam",
        "East Halemaʻumaʻu crater from northeast rim",
        "https://volcanoes.usgs.gov/cams/V2cam/images/M.jpg",
    ),
    Webcam(
        "V3cam",
        "South Halemaʻumaʻu crater from south rim",
        "https://volcanoes.usgs.gov/cams/V3cam/images/M.jpg",
    ),
    Webcam(
        "B1cam",
        "Caldera down-dropped block from east rim",
        "https://volcanoes.usgs.gov/observatories/hvo/cams/B1cam/images/M.jpg",
    ),
    Webcam(
        "KWcam",
        "Halemaʻumaʻu panorama from west rim",
        "https://volcanoes.usgs.gov/observatories/hvo/cams/KWcam/images/M.jpg",
    ),
    Webcam(
        "F1cam",
        "Thermal imagery from west rim",
        "https://volcanoes.usgs.gov/observatories/hvo/cams/F1cam/images/M.jpg",
    ),
    Webcam(
        "KPcam",
        "Summit view from Mauna Loa Strip Road",
        "https://volcanoes.usgs.gov/cams/KPcam/images/M.jpg",
    ),
)

# Cameras surfaced in the compact strip on the Now tab. Chosen for coverage —
# Uēkahuna (the tilt station's bluff), west + east Halema‘uma‘u, and the
# Mauna Loa overview. Anything not in the strip is still on the Cameras tab.
STRIP_NAMES: tuple[str, ...] = ("K2cam", "V1cam", "V2cam", "KPcam")


def strip_cameras() -> tuple[Webcam, ...]:
    """Return the subset of WEBCAMS that belongs in the Now-tab strip."""
    by_name = {w.name: w for w in WEBCAMS}
    return tuple(by_name[n] for n in STRIP_NAMES)


def _render_one(cam: Webcam) -> None:
    """Render a single camera tile: image + caption. Imported lazily."""
    import streamlit as st

    try:
        # caption doubles as the alt attribute in Streamlit's image renderer.
        st.image(cam.url, width="stretch", caption=cam.description)
    except Exception as e:  # pragma: no cover — network glitch only
        st.caption(f"⚠️ could not load {cam.name}: {e}")
    st.markdown(f"**[{cam.name}]({cam.url})**")


def show_strip() -> None:
    """Compact 4-wide strip for the Now tab."""
    import streamlit as st

    cams = strip_cameras()
    cols = st.columns(len(cams))
    for col, cam in zip(cols, cams, strict=False):
        with col:
            _render_one(cam)


def show_grid() -> None:
    """Full 8-camera grid for the Cameras tab, with overview copy."""
    import streamlit as st

    st.markdown(
        "Live still captures from the eight USGS HVO webcams watching "
        "Kīlauea's summit caldera and Halemaʻumaʻu crater. Click any "
        "thumbnail to open the full-resolution image. The [USGS webcams page]"
        "(https://www.usgs.gov/volcanoes/kilauea/summit-webcams) has the "
        "live time-lapse feeds and a camera location map."
    )
    # 2-column grid.
    rows = [WEBCAMS[i : i + 2] for i in range(0, len(WEBCAMS), 2)]
    for pair in rows:
        cols = st.columns(2)
        for col, cam in zip(cols, pair, strict=False):
            with col:
                _render_one(cam)
