"""Structured, three-part state banner.

Replaces the per-state wall-of-text HTML that lived inline in ``streamlit_app.py``.
Every state renders the same shape: icon + bold headline, plain-English
explainer, collapsible "what this means for you" guidance.

The HTML is emitted as a single markdown block so Streamlit renders it in one
container, with ``role="status"`` and ``aria-live="polite"`` for screen readers
since the banner's content changes when the model re-classifies.
"""

from __future__ import annotations

from typing import Mapping

from .palette import STATE_COLOR
from .state_copy import get as get_copy


def render_html(state: str, info: Mapping | None = None) -> str:
    """Return the banner HTML for a given state.

    Pure — no Streamlit calls — so it's testable and reusable. The Streamlit
    layer (``show``) wraps this in an ``st.markdown(..., unsafe_allow_html=True)``.

    Args:
        state: one of calm/starting/imminent/overdue/active.
        info:  the diagnostics dict from ``_eruption_state`` (optional).
    """
    info = info or {}
    copy = get_copy(state)
    accent = STATE_COLOR[state]
    explanation = copy.explainer(info)

    # Markdown inside HTML doesn't auto-render, so we pre-strip the simplest
    # inline ** → <strong> for the numeric values embedded by state_copy.
    explanation_html = _inline_bold(explanation)

    return (
        f'<div class="kt-banner" '
        f'style="--banner-accent: {accent};" '
        f'role="status" aria-live="polite">'
        f'<p class="kt-banner__headline">'
        f'<span aria-hidden="true">{copy.icon}</span> {copy.headline}'
        f'</p>'
        f'<p class="kt-banner__explanation">{explanation_html}</p>'
        f'</div>'
    )


def _inline_bold(text: str) -> str:
    """Convert ``**x**`` markdown spans to ``<strong>x</strong>``.

    Lightweight — we only need this one inline style since state_copy only
    bolds numeric values. Avoiding the full markdown path keeps the test
    surface and output HTML predictable.
    """
    parts = text.split("**")
    # Odd indices are bolded spans (between pairs of **).
    out: list[str] = []
    for i, chunk in enumerate(parts):
        if i % 2 == 1:
            out.append(f"<strong>{chunk}</strong>")
        else:
            out.append(chunk)
    return "".join(out)


def show(state: str, info: Mapping | None = None) -> None:
    """Render the banner in the current Streamlit container.

    The ``calm`` state intentionally renders nothing — the hero block above
    already communicates "nothing's happening right now" through the chip
    and the countdown; doubling it would add noise.
    """
    if state == "calm":
        return
    import streamlit as st

    st.markdown(render_html(state, info), unsafe_allow_html=True)

    # The old banner additionally embedded a small webcam alongside for
    # active/starting states. That now lives on the Now tab as a 4-cam strip
    # immediately below the hero, so we don't duplicate it here.
