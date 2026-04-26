"""HTTP fetch layer for safety alerts (USGS HANS + NWS).

This is the only side-effecting half of the safety-alerts package.
Everything that touches the network or reads the wall clock lives here;
parsing, filtering, and dataclasses live in `./_parse.py`. Tests that
need to bypass the network patch ``safety_alerts._fetch.requests``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import requests

from ._parse import (
    NWSAlert,
    SafetyAlertSummary,
    USGSVolcanoStatus,
    _is_volcano_relevant,
    _parse_nws_record,
    _parse_usgs_record,
)

# NWS API requires a descriptive User-Agent per
# https://www.weather.gov/documentation/services-web-api so they can
# contact maintainers about misbehaving clients. Identify the project
# and a contact URL.
_USER_AGENT = (
    "kilauea-tracker (https://github.com/madisonrickert/kilauea-tracker)"
)

USGS_HANS_URL = (
    "https://volcanoes.usgs.gov/hans-public/api/volcano/getElevatedVolcanoes"
)
NWS_HI_ALERTS_URL = "https://api.weather.gov/alerts/active?area=HI"

# Network timeouts. Conservative because the Streamlit container is
# blocking on these and a slow alert fetch would delay the page render.
HTTP_TIMEOUT_SECONDS = 10


def fetch_safety_alerts(
    *,
    volcano_name: str = "Kilauea",
    timeout: int = HTTP_TIMEOUT_SECONDS,
) -> SafetyAlertSummary:
    """Fetch USGS volcano status + NWS alerts and return a normalized summary.

    Best-effort: if one source fails the other still runs, and the
    failure is recorded in `summary.errors`. Never raises.
    """
    summary = SafetyAlertSummary(fetched_at=datetime.now(tz=UTC))

    try:
        summary.usgs_status = _fetch_usgs_volcano_status(
            volcano_name=volcano_name, timeout=timeout
        )
    except Exception as e:
        summary.errors.append(f"USGS HANS fetch failed: {e}")

    try:
        summary.nws_alerts = _fetch_nws_volcano_relevant_alerts(timeout=timeout)
    except Exception as e:
        summary.errors.append(f"NWS alerts fetch failed: {e}")

    return summary


def _fetch_usgs_volcano_status(
    *, volcano_name: str, timeout: int
) -> USGSVolcanoStatus | None:
    """Fetch the USGS HANS elevated-volcanoes feed and pick out one volcano.

    Returns None when the requested volcano is at NORMAL status (HANS
    only lists *elevated* volcanoes — absence from the list means the
    volcano is at GREEN/NORMAL or has no current notice).
    """
    resp = requests.get(
        USGS_HANS_URL,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"unexpected HANS payload type: {type(data).__name__}")

    target = volcano_name.lower()
    for record in data:
        name = (record.get("volcano_name") or "").lower()
        if target not in name:
            continue
        return _parse_usgs_record(record)
    return None


def _fetch_nws_volcano_relevant_alerts(*, timeout: int) -> list[NWSAlert]:
    """Fetch active NWS alerts in Hawaii and filter for volcano relevance.

    Two filter passes are OR'd together:
      1. Keyword match against the volcano vocabulary on the alert text.
      2. Big Island zone code + an event type that affects tephra/vog
         exposure (wind, air quality, etc.) — even if the text doesn't
         mention the volcano explicitly.

    Returns the deduplicated, sorted (most recent first) result.
    """
    resp = requests.get(
        NWS_HI_ALERTS_URL,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/geo+json",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    features = payload.get("features", [])
    if not isinstance(features, list):
        return []

    alerts: list[NWSAlert] = []
    for feature in features:
        props = feature.get("properties") or {}
        if not isinstance(props, dict):
            continue
        if not _is_volcano_relevant(props):
            continue
        alerts.append(_parse_nws_record(props))

    # Sort newest-sent first so the most recent operationally-relevant
    # advisory is at the top of the list.
    alerts.sort(key=lambda a: a.sent or datetime.min.replace(tzinfo=UTC), reverse=True)
    return alerts
