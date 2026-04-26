"""Tests for `kilauea_tracker.safety_alerts`.

Mocks the requests.get calls so the suite never touches the network in
CI. Each test patches the module-level `requests.get` to return a
canned response and asserts that the parser/filter behaves correctly
on that input.

Coverage:
  - USGS HANS parsing: extracts the Kīlauea record, normalizes color/
    alert level, parses the sent timestamp, returns None when Kīlauea
    is not in the elevated list (i.e. NORMAL status).
  - NWS keyword filter: keeps unambiguous volcano vocabulary anywhere
    in Hawaii (Tier 1), keeps "Kilauea" mentions only when zones cover
    the Big Island (Tier 2), keeps Big Island wind/air-quality alerts
    even when text doesn't mention the volcano (Tier 3), drops
    everything else.
  - Disambiguation guard: a Kauai Flash Flood Warning whose description
    mentions the town of Kīlauea must NOT be returned. This is the
    real false positive observed against the live API on 2026-04-09.
  - Failure handling: HTTP errors on either source are recorded in
    `summary.errors` without raising and the other source still flows.
  - ISO timestamp parsing: tolerates "Z" suffix and offset-bearing
    forms; returns None on garbage input.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

from kilauea_tracker.safety_alerts import (
    SafetyAlertSummary,
    USGSVolcanoStatus,
    _is_volcano_relevant,
    _parse_iso_utc,
    _parse_usgs_record,
    fetch_safety_alerts,
)
from kilauea_tracker.safety_alerts import _fetch as _safety_alerts_fetch

# ─────────────────────────────────────────────────────────────────────────────
# ISO timestamp parser
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_iso_utc_handles_z_suffix():
    out = _parse_iso_utc("2026-04-09T19:30:00Z")
    assert out == datetime(2026, 4, 9, 19, 30, tzinfo=UTC)


def test_parse_iso_utc_handles_offset():
    out = _parse_iso_utc("2026-04-09T09:30:00-10:00")
    assert out == datetime(2026, 4, 9, 19, 30, tzinfo=UTC)


def test_parse_iso_utc_returns_none_on_garbage():
    assert _parse_iso_utc(None) is None
    assert _parse_iso_utc("") is None
    assert _parse_iso_utc("not a date") is None


# ─────────────────────────────────────────────────────────────────────────────
# USGS HANS record parsing
# ─────────────────────────────────────────────────────────────────────────────


def test_parse_usgs_record_normalizes_fields():
    raw = {
        "volcano_name": "Kilauea",
        "obs_fullname": "Hawaiian Volcano Observatory",
        "color_code": "red",
        "alert_level": "warning",
        "sent_utc": "2026-04-10T00:27:56Z",
        "notice_url": "https://example.test/notice",
        "notice_type_cd": "VONA",
        "notice_identifier": "DOI-USGS-HVO-2026-04-09",
    }
    status = _parse_usgs_record(raw)
    assert isinstance(status, USGSVolcanoStatus)
    assert status.volcano_name == "Kilauea"
    assert status.color_code == "RED"
    assert status.alert_level == "WARNING"
    assert status.sent_utc == datetime(2026, 4, 10, 0, 27, 56, tzinfo=UTC)
    assert status.notice_url == "https://example.test/notice"


# ─────────────────────────────────────────────────────────────────────────────
# NWS volcano-relevance filter
# ─────────────────────────────────────────────────────────────────────────────


def _props(
    *,
    event: str = "",
    headline: str = "",
    description: str = "",
    zones: list[str] | None = None,
) -> dict:
    return {
        "event": event,
        "headline": headline,
        "description": description,
        "geocode": {"UGC": zones or []},
    }


def test_filter_keeps_tier1_unambiguous_volcano_keyword_anywhere():
    """An ashfall advisory ANYWHERE in Hawaii — even on Kauai zones —
    is kept. The keyword `ashfall` is a volcano-specific term that
    doesn't collide with non-volcanic place names.
    """
    p = _props(
        event="Ashfall Advisory",
        description="Light ashfall expected downwind of the summit vent.",
        zones=["HIZ001"],  # Kauai zone, irrelevant to this filter tier
    )
    assert _is_volcano_relevant(p) is True


def test_filter_keeps_tier2_kilauea_mention_with_big_island_zone():
    """A Special Weather Statement that mentions Kīlauea AND covers a
    Big Island zone is the canonical case for Tier 2.
    """
    p = _props(
        event="Special Weather Statement",
        description="Kilauea Episode 44 precursory activity continues.",
        zones=["HIZ051"],  # Big Island Summit zone
    )
    assert _is_volcano_relevant(p) is True


def test_filter_drops_kauai_alert_that_mentions_kilauea_town():
    """The 2026-04-09 false positive: a Kauai Flash Flood Warning whose
    description listed affected areas including the town of Kīlauea on
    Kauai. Tier 2 must reject this because the alert covers ZERO Big
    Island zones.
    """
    p = _props(
        event="Flash Flood Warning",
        description=(
            "Locations expected to experience flooding include "
            "Kekaha, Princeville, Kilauea, Alakai swamp trails, "
            "Kokee state park."
        ),
        zones=["HIC007"],  # Kauai county code, NOT a Big Island zone
    )
    assert _is_volcano_relevant(p) is False


def test_filter_keeps_tier3_big_island_wind_advisory_without_text_match():
    """A Wind Advisory on the Big Island Summit doesn't mention the
    volcano in its text, but wind direction over the summit is the
    dominant factor for downwind tephra exposure during a fountaining
    episode. Tier 3 keeps it via zone + event-type match.
    """
    p = _props(
        event="Wind Advisory",
        description="Northeast winds 25 to 35 mph with higher gusts.",
        zones=["HIZ051"],  # Big Island Summit
    )
    assert _is_volcano_relevant(p) is True


def test_filter_drops_unrelated_kauai_high_surf_advisory():
    """A High Surf Advisory on Kauai with no volcano keywords and no
    Big Island zones must be filtered out — it's noise relative to the
    eruption tracker's purpose.
    """
    p = _props(
        event="High Surf Advisory",
        description="Surf 8 to 12 feet along south-facing shores.",
        zones=["HIZ001"],
    )
    assert _is_volcano_relevant(p) is False


def test_filter_drops_big_island_alert_with_irrelevant_event_type():
    """A Big Island event whose type is NOT in the relevant list and
    whose text doesn't mention the volcano is dropped. Example: a
    coastal flood advisory on the Big Island east shore is irrelevant
    to summit tephra exposure.
    """
    p = _props(
        event="Coastal Flood Advisory",
        description="Higher than normal tides expected.",
        zones=["HIZ023"],
    )
    assert _is_volcano_relevant(p) is False


# ─────────────────────────────────────────────────────────────────────────────
# Top-level fetch with mocked HTTP
# ─────────────────────────────────────────────────────────────────────────────


class _MockResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _fake_get(url: str, **kwargs):
    if "hans-public" in url:
        return _MockResponse(
            [
                {
                    "volcano_name": "Kilauea",
                    "obs_fullname": "Hawaiian Volcano Observatory",
                    "color_code": "RED",
                    "alert_level": "WARNING",
                    "sent_utc": "2026-04-10T00:27:56Z",
                    "notice_url": "https://example.test/notice",
                    "notice_type_cd": "VONA",
                    "notice_identifier": "test-1",
                },
                {
                    "volcano_name": "Mauna Loa",
                    "obs_fullname": "Hawaiian Volcano Observatory",
                    "color_code": "YELLOW",
                    "alert_level": "ADVISORY",
                    "sent_utc": "2026-03-01T00:00:00Z",
                    "notice_url": "https://example.test/mauna",
                },
            ]
        )
    if "weather.gov/alerts" in url:
        return _MockResponse(
            {
                "features": [
                    # Tier 1: ashfall — kept regardless of zones
                    {
                        "properties": _props(
                            event="Ashfall Advisory",
                            headline="Light ashfall downwind of vent",
                            description="Light ashfall expected.",
                            zones=["HIZ001"],
                        )
                        | {
                            "severity": "Moderate",
                            "urgency": "Expected",
                            "areaDesc": "Big Island Southeast",
                            "sent": "2026-04-09T20:00:00Z",
                            "expires": "2026-04-10T08:00:00Z",
                            "senderName": "NWS Honolulu HI",
                            "@id": "https://example.test/alert/ashfall",
                        }
                    },
                    # Tier 2: kilauea + Big Island zone — kept
                    {
                        "properties": _props(
                            event="Special Weather Statement",
                            headline="Kilauea Episode 44 update",
                            description="Kilauea Episode 44 fountaining ongoing.",
                            zones=["HIZ051"],
                        )
                        | {
                            "severity": "Moderate",
                            "urgency": "Expected",
                            "areaDesc": "Big Island Summit",
                            "sent": "2026-04-09T22:00:00Z",
                            "expires": "2026-04-10T22:00:00Z",
                            "senderName": "NWS Honolulu HI",
                            "@id": "https://example.test/alert/kilauea",
                        }
                    },
                    # The 2026-04-09 false positive: Kauai flood mentioning
                    # Kīlauea town. Tier 2 must drop this.
                    {
                        "properties": _props(
                            event="Flash Flood Warning",
                            headline="Kauai flood",
                            description=(
                                "Locations include Princeville, Kilauea, "
                                "and Hanalei."
                            ),
                            zones=["HIC007"],
                        )
                        | {
                            "severity": "Severe",
                            "urgency": "Immediate",
                            "areaDesc": "Kauai, HI",
                            "sent": "2026-04-09T19:00:00Z",
                            "expires": "2026-04-09T22:00:00Z",
                            "senderName": "NWS Honolulu HI",
                            "@id": "https://example.test/alert/kauai-flood",
                        }
                    },
                    # Unrelated noise: high surf on Oahu — dropped
                    {
                        "properties": _props(
                            event="High Surf Advisory",
                            headline="Oahu surf",
                            description="Surf 8 to 12 feet.",
                            zones=["HIZ002"],
                        )
                        | {
                            "severity": "Minor",
                            "urgency": "Expected",
                            "areaDesc": "Oahu",
                            "sent": "2026-04-09T21:00:00Z",
                            "expires": "2026-04-10T06:00:00Z",
                            "senderName": "NWS Honolulu HI",
                            "@id": "https://example.test/alert/surf",
                        }
                    },
                ]
            }
        )
    raise AssertionError(f"unexpected URL in test: {url}")


def test_fetch_safety_alerts_end_to_end_with_mocked_http():
    with patch.object(_safety_alerts_fetch.requests, "get", side_effect=_fake_get):
        summary = fetch_safety_alerts()

    assert isinstance(summary, SafetyAlertSummary)
    assert summary.errors == []

    assert summary.usgs_status is not None
    assert summary.usgs_status.color_code == "RED"
    assert summary.usgs_status.alert_level == "WARNING"

    # Two alerts pass filtering: ashfall (Tier 1) and Kīlauea Episode 44
    # (Tier 2). The Kauai flood and Oahu surf are dropped.
    assert len(summary.nws_alerts) == 2
    events = {a.event for a in summary.nws_alerts}
    assert events == {"Ashfall Advisory", "Special Weather Statement"}

    # Sorted newest-first.
    assert summary.nws_alerts[0].sent >= summary.nws_alerts[1].sent


def test_fetch_safety_alerts_records_errors_without_raising():
    """If both sources fail, the function returns a summary with errors
    rather than raising. The Streamlit layer can render a quiet caption.
    """
    def boom(url, **kwargs):
        raise RuntimeError(f"network down: {url}")

    with patch.object(_safety_alerts_fetch.requests, "get", side_effect=boom):
        summary = fetch_safety_alerts()

    assert summary.usgs_status is None
    assert summary.nws_alerts == []
    assert len(summary.errors) == 2  # one per source
    assert any("HANS" in e for e in summary.errors)
    assert any("NWS" in e for e in summary.errors)


def test_fetch_safety_alerts_one_source_failure_doesnt_block_other():
    """If USGS HANS is down but NWS works, we still get NWS alerts."""
    def half_broken(url, **kwargs):
        if "hans-public" in url:
            raise RuntimeError("hans down")
        return _fake_get(url, **kwargs)

    with patch.object(_safety_alerts_fetch.requests, "get", side_effect=half_broken):
        summary = fetch_safety_alerts()

    assert summary.usgs_status is None
    assert len(summary.nws_alerts) == 2
    assert len(summary.errors) == 1
    assert "HANS" in summary.errors[0]
