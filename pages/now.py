"""Now — landing page.

Hero block (state chip + sparkline + headline), public safety alerts
(USGS HANS aviation color code + NWS Hawaii advisories), state banner,
live camera strip, and the two big CTAs that jump to Chart / Cameras.

This is the page non-technical visitors land on. Everything below is the
"single answer at a glance" layout: there's enough on this page to know
what's happening at Kīlauea right now without reading anything else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from kilauea_tracker import app_state
from kilauea_tracker.state import get_state
from kilauea_tracker.ui import cameras, hero, state_banner

if TYPE_CHECKING:
    from kilauea_tracker.safety_alerts import NWSAlert, USGSVolcanoStatus

# One snapshot read per rerun — every widget value the page needs.
state = get_state()
DISPLAY_TZ = (
    "Pacific/Honolulu"
    if state.widgets.chart.timezone_choice.startswith("HST")
    else "UTC"
)
TZ_LABEL = "HST" if DISPLAY_TZ == "Pacific/Honolulu" else "UTC"


# ─────────────────────────────────────────────────────────────────────────────
# Public safety alerts (USGS HANS aviation color code + NWS Hawaii alerts)
# ─────────────────────────────────────────────────────────────────────────────
#
# Independent of the tilt model — these come from official channels (USGS
# HANS for the volcano alert level / aviation color code, NWS for tephra/
# wind/SO2 advisories). Rendered immediately below the eruption lifecycle
# banner because that's where the user is already looking when the volcano
# is doing something interesting.


def _render_usgs_color_badge(status: USGSVolcanoStatus) -> None:
    """Render the USGS aviation color code as a colored markdown badge."""
    color_map = {
        "GREEN": "#1e8e3e",
        "YELLOW": "#f9ab00",
        "ORANGE": "#e8710a",
        "RED": "#d93025",
    }
    bg = color_map.get(status.color_code, "#5f6368")
    sent_str = ""
    if status.sent_utc is not None:
        sent_local = pd.Timestamp(status.sent_utc).tz_convert(DISPLAY_TZ)
        sent_str = f" · issued {sent_local.strftime('%b %-d, %-I:%M %p')} {TZ_LABEL}"
    # Inside a raw-HTML <div>, Streamlit doesn't re-parse markdown, so
    # `[label](url)` would render as literal text. Emit an <a> tag.
    notice_link = (
        f' &nbsp;<a href="{status.notice_url}" target="_blank" rel="noopener"'
        f' style="color: inherit; text-decoration: underline;">full notice ↗</a>'
        if status.notice_url
        else ""
    )
    st.markdown(
        f"<div style='padding: 0.6rem 0.9rem; border-radius: 6px; "
        f"background: {bg}; color: white; font-weight: 600; "
        f"margin-bottom: 0.5rem;'>"
        f"USGS volcano alert: <b>{status.color_code}</b> aviation color · "
        f"<b>{status.alert_level}</b> ground alert"
        f"<span style='font-weight: 400; opacity: 0.9;'>{sent_str}</span>"
        f"</div>"
        f"<div style='font-size: 0.85rem; color: #aaa; margin-bottom: 1rem;'>"
        f"{status.observatory}{notice_link}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_nws_alert(alert: NWSAlert) -> None:
    """Render one filtered NWS alert as a compact info card."""
    severity_icon = {
        "Extreme": "🛑",
        "Severe": "⚠️",
        "Moderate": "🟠",
        "Minor": "🟡",
    }.get(alert.severity, "ⓘ")
    expires_str = ""
    if alert.expires is not None:
        exp_local = pd.Timestamp(alert.expires).tz_convert(DISPLAY_TZ)
        expires_str = f" · expires {exp_local.strftime('%b %-d, %-I:%M %p')} {TZ_LABEL}"
    st.markdown(
        f"**{severity_icon} {alert.event}** &nbsp; "
        f"<span style='color:#888'>· {alert.area_desc}{expires_str}</span>",
        unsafe_allow_html=True,
    )
    if alert.headline and alert.headline != alert.event:
        st.caption(alert.headline)
    if alert.description:
        with st.expander("Full advisory text"):
            st.text(alert.description)


tilt_df = app_state.load_tilt_df()
all_peaks = app_state.get_peaks(
    tilt_df,
    min_prominence=state.widgets.peaks.min_prominence,
    min_distance_days=state.widgets.peaks.min_distance_days,
    min_height=state.widgets.peaks.min_height,
)
recent_peaks = app_state.get_recent_peaks(all_peaks, state.widgets.chart.n_peaks_for_fit)
prediction = app_state.get_prediction(tilt_df, recent_peaks)
eruption_state, eruption_state_info = app_state.get_eruption_state(tilt_df, prediction)
safety = app_state.get_safety_alerts()


if safety.usgs_status is not None or safety.nws_alerts:
    if safety.usgs_status:
        _render_usgs_color_badge(safety.usgs_status)
    if safety.nws_alerts:
        st.markdown(
            f"**Active NWS advisories ({len(safety.nws_alerts)})** "
            f"<span style='color:#888; font-size:0.85rem'>"
            f"· filtered for Kīlauea / Big Island relevance</span>",
            unsafe_allow_html=True,
        )
        for alert in safety.nws_alerts:
            _render_nws_alert(alert)
        st.caption(
            "Source: USGS Hazard Alert Notification System + NWS Honolulu. "
            "Updated every 15 minutes; click the top-bar Refresh button to "
            "force a fresh fetch."
        )
elif safety.errors:
    # Both sources failed. Show a single quiet caption rather than a loud red
    # banner — alerts are auxiliary, not load-bearing.
    st.caption(
        "Safety alerts unavailable right now "
        f"({len(safety.errors)} source(s) returned an error)."
    )

# ── Hero block: one dramatic answer + last-30-days sparkline ───────────
# The sparkline reads as a visual fingerprint of recent activity rather
# than a readable chart — axes are suppressed, hover disabled.
hero.show(eruption_state, prediction, tilt_df)

# CTA: jump to the full prediction model chart. st.page_link is a real
# multipage-aware navigation primitive, so the URL changes to /chart and
# the browser back button does the right thing. Styling lives in styles.py
# under the .st-key-cta_chart wrapper.
st.page_link(
    "pages/chart.py",
    label="📈 View full prediction model →",
)

# ── State banner ──────────────────────────────────────────────────────
# Consistent 3-part banner (icon + headline, plain explainer, guidance)
# rendered from the state_copy table. `calm` skips rendering.
state_banner.show(eruption_state, eruption_state_info)

# ── Live camera strip ─────────────────────────────────────────────────
# Four-camera strip below the hero — seeing the volcano is the second-
# most emotionally valuable thing on the page after the prediction. The
# full 8-camera grid lives on /cameras.
st.markdown("#### 📷 Live cameras")
cameras.show_strip()

st.page_link(
    "pages/cameras.py",
    label="📷 View all cameras →",
)
