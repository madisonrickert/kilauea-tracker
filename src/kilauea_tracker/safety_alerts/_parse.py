"""Pure parsing + filtering for safety-alert payloads.

Everything here takes a `dict` (the JSON-decoded payload from one of the
upstream APIs) or a `str` and returns a typed dataclass or a bool. No
network, no clock, no module-level mutable state — same input always
produces the same output. The fetch layer (`./_fetch.py`) calls into
this module after retrieving the raw payloads.

Splitting fetch from parse lets the parsers be exercised by tests
without any HTTP mocking and keeps the purity contract honest:
`safety_alerts/_parse.py` is what CLAUDE.md's "pure compute layers"
list refers to once the package layout lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

# Volcano-relevance keyword filter applied to NWS alert event +
# headline + description fields. Case-insensitive substring match.
#
# Two tiers:
#
# - `_GEOGRAPHIC_VOLCANO_KEYWORDS` are unambiguous volcano-event terms.
#   Any alert anywhere in Hawaii whose text contains one of these is
#   kept regardless of which zones it covers, because these words don't
#   collide with non-volcanic place names.
#
# - `_AMBIGUOUS_VOLCANO_KEYWORDS` includes the volcano's name itself,
#   which is also the name of a town on Kauai. An alert matching one
#   of these is kept ONLY when at least one of its affected zones is
#   on the Big Island — otherwise we drop the false positive (the
#   Kauai-town Kīlauea is in a different alert area entirely).
_GEOGRAPHIC_VOLCANO_KEYWORDS = (
    "volcan",
    "ashfall",
    "tephra",
    " so2",
    " so 2",
    "sulfur dioxide",
    "vog",
    "fountaining",
    "fissure",
    "halemaumau",
    "halemaʻumaʻu",
)

_AMBIGUOUS_VOLCANO_KEYWORDS = (
    "kilauea",
    "kīlauea",
)

# NWS area zone codes for the Big Island. Wind / air-quality alerts on
# these zones are kept even if their text doesn't mention the volcano,
# because wind direction over the summit drives tephra dispersion and
# is operationally relevant during a fountaining episode.
#
# Codes verified empirically against the live NWS API on 2026-04-09 by
# inspecting the zones attached to a "Special Weather Statement" about
# Kīlauea Episode 44. The zones below cover the inhabited and summit
# areas of the Big Island; new zone codes may need to be added if NWS
# reorganizes their zone map (rare, ~once a decade).
_BIG_ISLAND_ZONES = frozenset(
    {
        "HIZ023",  # Big Island North
        "HIZ026",
        "HIZ027",
        "HIZ028",
        "HIZ051",  # Big Island Summit (Mauna Loa / Mauna Kea)
        "HIZ052",  # Kona
        "HIZ053",  # Kohala
        "HIZ054",  # Big Island Interior / Volcano area
    }
)

# NWS event types that we consider operationally relevant on the Big
# Island during an eruption even if the text doesn't mention the volcano.
# Wind direction is the dominant factor for downwind tephra exposure.
_BIG_ISLAND_RELEVANT_EVENTS = frozenset(
    {
        "Wind Advisory",
        "High Wind Warning",
        "High Wind Watch",
        "Air Quality Alert",
        "Air Stagnation Advisory",
        "Special Weather Statement",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class USGSVolcanoStatus:
    """One USGS HANS record for an elevated volcano."""

    volcano_name: str
    observatory: str            # e.g. "Hawaiian Volcano Observatory"
    color_code: str             # GREEN/YELLOW/ORANGE/RED
    alert_level: str            # NORMAL/ADVISORY/WATCH/WARNING
    sent_utc: datetime | None
    notice_url: str | None
    notice_type_code: str | None = None
    notice_identifier: str | None = None


@dataclass
class NWSAlert:
    """One NWS active-alert record, normalized."""

    event: str                  # "Ashfall Advisory" / "Special Weather Statement" / ...
    headline: str
    description: str
    severity: str               # Extreme/Severe/Moderate/Minor/Unknown
    urgency: str                # Immediate/Expected/Future/Past/Unknown
    area_desc: str              # human-readable affected areas
    sent: datetime | None
    expires: datetime | None
    sender_name: str
    web_url: str | None      # link to the alert page on weather.gov
    affected_zones: list[str] = field(default_factory=list)


@dataclass
class SafetyAlertSummary:
    """Combined output of one fetch round."""

    usgs_status: USGSVolcanoStatus | None = None
    nws_alerts: list[NWSAlert] = field(default_factory=list)
    fetched_at: datetime | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return self.usgs_status is not None or len(self.nws_alerts) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Parsers + filter (pure)
# ─────────────────────────────────────────────────────────────────────────────


def _parse_usgs_record(record: dict) -> USGSVolcanoStatus:
    return USGSVolcanoStatus(
        volcano_name=record.get("volcano_name", "Unknown"),
        observatory=record.get("obs_fullname")
        or record.get("obs_abbr")
        or "Unknown",
        color_code=(record.get("color_code") or "UNSPECIFIED").upper(),
        alert_level=(record.get("alert_level") or "UNSPECIFIED").upper(),
        sent_utc=_parse_iso_utc(record.get("sent_utc")),
        notice_url=record.get("notice_url"),
        notice_type_code=record.get("notice_type_cd"),
        notice_identifier=record.get("notice_identifier"),
    )


def _parse_nws_record(props: dict) -> NWSAlert:
    return NWSAlert(
        event=str(props.get("event") or ""),
        headline=str(props.get("headline") or ""),
        description=str(props.get("description") or ""),
        severity=str(props.get("severity") or "Unknown"),
        urgency=str(props.get("urgency") or "Unknown"),
        area_desc=str(props.get("areaDesc") or ""),
        sent=_parse_iso_utc(props.get("sent")),
        expires=_parse_iso_utc(props.get("expires")),
        sender_name=str(props.get("senderName") or ""),
        web_url=props.get("@id"),
        affected_zones=_extract_zone_codes(props),
    )


def _is_volcano_relevant(props: dict) -> bool:
    text_blob = " ".join(
        str(props.get(k) or "")
        for k in ("event", "headline", "description", "instruction")
    ).lower()
    zone_codes = _extract_zone_codes(props)
    on_big_island = any(z in _BIG_ISLAND_ZONES for z in zone_codes)

    # Tier 1: unambiguous volcano vocabulary — keep regardless of zones.
    if any(kw in text_blob for kw in _GEOGRAPHIC_VOLCANO_KEYWORDS):
        return True

    # Tier 2: ambiguous "Kīlauea" mentions — only keep if the alert
    # actually covers a Big Island zone. Disambiguates from the town
    # named Kīlauea on Kauai, which appears in unrelated Kauai flood/
    # surf alerts.
    if any(kw in text_blob for kw in _AMBIGUOUS_VOLCANO_KEYWORDS) and on_big_island:
        return True

    # Tier 3: zone-only — Big Island wind/air-quality alerts even when
    # the text doesn't mention the volcano, since wind direction over
    # the summit drives downwind tephra exposure.
    event = (props.get("event") or "").strip()
    return bool(on_big_island and event in _BIG_ISLAND_RELEVANT_EVENTS)


def _extract_zone_codes(props: dict) -> list[str]:
    """NWS alert features list affected zones under properties.geocode.UGC.

    Returns the raw UGC strings (e.g. "HIZ016"). When the field is
    absent or malformed, returns [].
    """
    geocode = props.get("geocode") or {}
    if not isinstance(geocode, dict):
        return []
    ugc = geocode.get("UGC") or []
    if not isinstance(ugc, list):
        return []
    return [str(z) for z in ugc]


def _parse_iso_utc(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp into a UTC datetime, tolerantly.

    Both NWS and HANS use offset-aware ISO strings (e.g.
    `2026-04-09T19:30:00-10:00` or `2026-04-09T19:30:00Z`). We
    normalize everything to UTC. Returns None on any parse failure
    rather than raising — alerts with missing timestamps still render.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    s = str(value).strip()
    if not s:
        return None
    # Python <3.11 doesn't accept "Z" suffix in fromisoformat; normalize.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)
