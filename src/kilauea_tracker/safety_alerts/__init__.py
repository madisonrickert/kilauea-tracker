"""Fetch and normalize public safety alerts relevant to a Kīlauea eruption.

Two independent sources are queried:

1. **USGS HANS API** — `volcanoes.usgs.gov/hans-public/api/volcano/getElevatedVolcanoes`
   The Hazard Alert Notification System publishes the official aviation
   color code (GREEN/YELLOW/ORANGE/RED) and ground-based alert level
   (NORMAL/ADVISORY/WATCH/WARNING) for every volcano whose status is
   currently elevated above NORMAL. We filter the response to Kīlauea.

2. **NWS API** — `api.weather.gov/alerts/active?area=HI`
   The National Weather Service issues "Special Weather Statements,"
   "Ashfall Advisories," and other products that go out during a
   fountaining episode (tephra fallout, vog drift, wind direction
   advisories for the inhabited areas downwind of the vent). Volcano-
   related products use the standard NWS alert schema; we filter the
   Hawaii feed by keyword + Big Island geography.

Both sources are public, free, no auth required, and update in roughly
real time as new advisories are issued. The Streamlit layer wraps the
top-level `fetch_safety_alerts()` in `@st.cache_data(ttl=900)` so we
don't hammer the APIs on every rerun.

Failure handling: each fetch is independently best-effort. If one source
is unreachable, the returned `SafetyAlertSummary` records the error in
its `errors` list and the other source's data still flows through. The
Streamlit layer can render whatever's available without blocking the
main UI.

Package layout:
  - `_parse.py` — pure parsers, filters, dataclasses (no I/O, no clock).
  - `_fetch.py` — HTTP fetchers and the `fetch_safety_alerts` orchestrator.
"""

from __future__ import annotations

from ._fetch import HTTP_TIMEOUT_SECONDS as HTTP_TIMEOUT_SECONDS
from ._fetch import NWS_HI_ALERTS_URL as NWS_HI_ALERTS_URL
from ._fetch import USGS_HANS_URL as USGS_HANS_URL
from ._fetch import fetch_safety_alerts as fetch_safety_alerts
from ._parse import NWSAlert as NWSAlert
from ._parse import SafetyAlertSummary as SafetyAlertSummary
from ._parse import USGSVolcanoStatus as USGSVolcanoStatus

# Internals re-exported so existing test imports continue to work without
# poking the underscore submodules directly.
from ._parse import _is_volcano_relevant as _is_volcano_relevant
from ._parse import _parse_iso_utc as _parse_iso_utc
from ._parse import _parse_usgs_record as _parse_usgs_record
