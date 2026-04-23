"""Volcano palette — dark-only, hue-monotonic state ramp.

Tuned via WCAG AA audit on ``--obsidian`` and an OKLCH monotonicity check:
hue marches steadily from cool neutral toward red, and ``--flame`` owns the
highest chroma so the ``active`` state wins the eye. See the design plan for
the full color-theory rationale.

Consumers:
    - ``ui.styles`` injects these as CSS custom properties.
    - ``plotting.py`` uses the hex values directly for Plotly traces.
    - ``ui.state_banner`` maps state names to state tokens via ``STATE_COLOR``.
"""

from __future__ import annotations

# Surface tokens
OBSIDIAN = "#0f1419"        # page background (deepest)
BASALT = "#1e2537"          # elevated surface
STEAM = "#e2e8f0"           # primary text on dark

# State ramp — hue monotonically decreasing 250° → 25°, chroma climbing to FLAME
ASH = "#64748b"             # calm
EMBER = "#b98a2e"           # starting
MAGMA = "#e07a2a"           # imminent
RUST = "#b55a2e"            # overdue
FLAME = "#e0372a"           # active

# Brand accent — locked, not a state
LAVA = "#ff6b35"

# Transparent lava for glow rings, gradient fills, selection halos
HALO = "rgba(255, 107, 53, 0.18)"

# All named tokens, in order. Used by styles.py and by palette-contrast tests.
ALL_TOKENS: dict[str, str] = {
    "obsidian": OBSIDIAN,
    "basalt": BASALT,
    "ash": ASH,
    "ember": EMBER,
    "magma": MAGMA,
    "lava": LAVA,
    "rust": RUST,
    "flame": FLAME,
    "steam": STEAM,
    "halo": HALO,
}

# State name → token hex. Keep stable — tests + state_banner depend on it.
STATE_COLOR: dict[str, str] = {
    "calm": ASH,
    "starting": EMBER,
    "imminent": MAGMA,
    "overdue": RUST,
    "active": FLAME,
}

# Ordering used by the state-ramp monotonicity tests. Canonical urgency order.
STATE_RAMP_ORDER: tuple[str, ...] = ("calm", "starting", "imminent", "overdue", "active")
