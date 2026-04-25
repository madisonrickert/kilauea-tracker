"""Cameras module — lock the webcam table shape."""

from __future__ import annotations

import re

from kilauea_tracker.ui.cameras import STRIP_NAMES, WEBCAMS, strip_cameras


def test_eight_webcams_present():
    """USGS HVO has eight cams on the summit page — the app must list them all."""
    assert len(WEBCAMS) == 8


def test_camera_names_are_unique():
    names = [w.name for w in WEBCAMS]
    assert len(names) == len(set(names))


def test_camera_urls_are_unique():
    urls = [w.url for w in WEBCAMS]
    assert len(urls) == len(set(urls))


def test_every_camera_has_nonempty_description():
    """Description doubles as the image alt attr — non-empty for a11y."""
    for cam in WEBCAMS:
        assert cam.description, f"{cam.name}: empty description"
        assert cam.description.strip() == cam.description


def test_every_url_matches_usgs_volcanoes_pattern():
    pattern = re.compile(
        r"^https://volcanoes\.usgs\.gov/"
        r"(observatories/hvo/)?cams/[A-Z0-9]+cam/images/M\.jpg$"
    )
    for cam in WEBCAMS:
        assert pattern.match(cam.url), f"{cam.name}: URL does not match pattern: {cam.url}"


def test_strip_selects_four_known_cameras():
    cams = strip_cameras()
    assert len(cams) == 4
    assert [c.name for c in cams] == list(STRIP_NAMES)


def test_every_strip_name_exists_in_webcams():
    names = {w.name for w in WEBCAMS}
    for sn in STRIP_NAMES:
        assert sn in names, f"strip-selected {sn!r} is not in WEBCAMS"
