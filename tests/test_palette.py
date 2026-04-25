"""Palette + styles regression tests.

Lock the color-theory audit from the UX overhaul into CI:
    - every state chip passes WCAG 3:1 against ``--obsidian`` (UI component minimum);
    - every text-class token (steam, ember, magma) passes AA 4.5:1 against obsidian;
    - the state ramp's OKLCH hue is monotonically decreasing calm → active;
    - ``--flame`` has the highest chroma in the ramp (owns the eye);
    - ``ui.styles`` emits every palette token + the Inter @import.

If a future palette tweak breaks any of these, CI catches it before merge.
"""

from __future__ import annotations

import itertools
import math

import pytest

from kilauea_tracker.ui.palette import (
    ALL_TOKENS,
    FLAME,
    OBSIDIAN,
    STATE_COLOR,
    STATE_RAMP_ORDER,
)
from kilauea_tracker.ui.styles import build_style_block

# ─────────────────────────────────────────────────────────────────────────────
# Color math (kept local — pulling in `colour` or `coloraide` for two formulas
# isn't worth the dep). Both functions follow published reference formulas.
# ─────────────────────────────────────────────────────────────────────────────


def _hex_to_rgb_01(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance(hex_color: str) -> float:
    """WCAG 2.x relative luminance Y."""
    r, g, b = (_srgb_to_linear(c) for c in _hex_to_rgb_01(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(fg: str, bg: str) -> float:
    yf, yb = _relative_luminance(fg), _relative_luminance(bg)
    hi, lo = max(yf, yb), min(yf, yb)
    return (hi + 0.05) / (lo + 0.05)


def _rgb_to_oklab(r: float, g: float, b: float) -> tuple[float, float, float]:
    """sRGB (0-1) → OKLab. Reference: Ottosson, bottosson.github.io."""
    r_lin, g_lin, b_lin = (_srgb_to_linear(c) for c in (r, g, b))
    # `l` is the LMS-long-cone intermediate per Ottosson's published reference;
    # ruff's E741 flags the lowercase name but renaming would break the canonical
    # `l/m/s` LMS naming used throughout the OKLab literature.
    l = 0.4122214708 * r_lin + 0.5363325363 * g_lin + 0.0514459929 * b_lin  # noqa: E741
    m = 0.2119034982 * r_lin + 0.6806995451 * g_lin + 0.1073969566 * b_lin
    s = 0.0883024619 * r_lin + 0.2817188376 * g_lin + 0.6299787005 * b_lin
    l_, m_, s_ = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return (L, a, bb)


def _hex_to_oklch(hex_color: str) -> tuple[float, float, float]:
    L, a, b = _rgb_to_oklab(*_hex_to_rgb_01(hex_color))
    chroma = math.sqrt(a * a + b * b)
    hue_deg = math.degrees(math.atan2(b, a)) % 360
    return (L, chroma, hue_deg)


# ─────────────────────────────────────────────────────────────────────────────
# WCAG contrast — each state chip + text tokens against --obsidian
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("state,hex_value", list(STATE_COLOR.items()))
def test_state_chip_passes_wcag_3_to_1_against_obsidian(state: str, hex_value: str):
    """Every state chip must be distinguishable on the page bg (3:1 UI minimum)."""
    ratio = _contrast(hex_value, OBSIDIAN)
    assert ratio >= 3.0, (
        f"state {state!r} ({hex_value}) vs obsidian ({OBSIDIAN}) = {ratio:.2f}:1; "
        "WCAG UI-component threshold is 3:1"
    )


@pytest.mark.parametrize(
    "token_name",
    ["steam", "ember", "magma"],
    ids=["steam-text", "ember-text", "magma-text"],
)
def test_text_tokens_pass_aa_against_obsidian(token_name: str):
    """Tokens that may render as plain text on the page bg need AA 4.5:1."""
    hex_value = ALL_TOKENS[token_name]
    ratio = _contrast(hex_value, OBSIDIAN)
    assert ratio >= 4.5, (
        f"{token_name} ({hex_value}) vs obsidian = {ratio:.2f}:1; AA text is 4.5:1"
    )


def test_steam_is_aaa_on_obsidian():
    """Body text should be comfortably AAA, not just barely AA."""
    ratio = _contrast(ALL_TOKENS["steam"], OBSIDIAN)
    assert ratio >= 7.0, f"steam on obsidian is only {ratio:.2f}:1; expected AAA ≥ 7"


# ─────────────────────────────────────────────────────────────────────────────
# OKLCH state-ramp monotonicity — hue must advance steadily toward red
# ─────────────────────────────────────────────────────────────────────────────


def test_state_ramp_hue_is_monotonic_toward_red():
    """Hue should march from cool-neutral (calm) to red (active) without reversals.

    A yellow "overdue" chip between orange "imminent" and red "active" would
    be a hue reversal — that's the v2.1 bug this test prevents regressing.
    """
    hues = [_hex_to_oklch(STATE_COLOR[s])[2] for s in STATE_RAMP_ORDER]
    # Calm is blueish (~250°); once we cross into the warm half (hue ≤ 80°),
    # every subsequent step must bring hue closer to 0°/red.
    warm_hues = [h for h in hues if h <= 180]
    for prev, nxt in itertools.pairwise(warm_hues):
        assert nxt <= prev, (
            f"hue reversal in state ramp: {warm_hues} (expect monotonic toward 0°)"
        )


def test_flame_has_highest_chroma_in_ramp():
    """Active state wins the eye via chroma, not luminance."""
    chromas = {s: _hex_to_oklch(STATE_COLOR[s])[1] for s in STATE_RAMP_ORDER}
    top_state = max(chromas, key=chromas.get)
    assert top_state == "active", (
        f"expected 'active' to hold highest chroma; got {top_state} "
        f"(chromas: {chromas})"
    )


def test_flame_hex_is_locked():
    """Sanity — if someone renames FLAME they should rename it here too."""
    assert STATE_COLOR["active"] == FLAME


# ─────────────────────────────────────────────────────────────────────────────
# styles.py — token completeness
# ─────────────────────────────────────────────────────────────────────────────


def test_style_block_contains_every_palette_token():
    css = build_style_block()
    for name in ALL_TOKENS:
        assert f"--{name}:" in css, f"missing CSS variable --{name} in <style> block"


def test_style_block_imports_inter():
    css = build_style_block()
    assert "Inter" in css
    assert "@import" in css


def test_style_block_is_wrapped_in_style_tags():
    css = build_style_block()
    assert css.startswith("<style>")
    assert css.rstrip().endswith("</style>")


def test_every_state_has_a_color():
    """No state may be missing from the STATE_COLOR map."""
    for state in STATE_RAMP_ORDER:
        assert state in STATE_COLOR, f"state {state!r} not mapped to a color"
        assert STATE_COLOR[state].startswith("#")
