"""Teaching tints for model diagnostics.

Maps a raw model-diagnostic value (trendline slope, exp-fit k, etc.) to a
volcano-palette color + a short verdict + a one-line expected-range note.
Lets the Pipeline tab display numbers next to a visible classification
instead of just printing raw floats — a visitor can tell at a glance
whether a reading is typical, elevated, or extreme for the current
eruptive phase.

Range thresholds are calibrated on the current (Dec 2024 → now) Kīlauea
fountain phase. They're deliberately loose — the volcano's behavior shifts
over time and we'd rather tint an unusual-but-valid reading as "elevated"
than hide it behind a hard cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass

from .palette import ASH, EMBER, FLAME, STEAM


@dataclass(frozen=True)
class Tint:
    """One diagnostic classification."""
    color: str   # palette hex token
    label: str   # short verdict chip text
    note: str    # one-line expected-range caption


# Reusable tint shapes.
_TYPICAL = Tint(STEAM, "typical", "within the normal range for this eruptive phase")
_ELEVATED = Tint(EMBER, "elevated", "outside typical — worth noticing")
_EXTREME = Tint(FLAME, "extreme", "well outside normal — unusual regime")


def trendline_slope_tint(slope_per_day: float) -> Tint:
    """Peaks-over-time slope.

    * < 0.05 µrad/day — essentially flat, normal background drift.
    * < 0.3 µrad/day — trending; each episode's peak height is shifting.
    * < 1.0 µrad/day — strong trend; episodes are notably escalating (or tapering).
    * ≥ 1.0 µrad/day — exceptional rate of change.
    """
    a = abs(slope_per_day)
    if a < 0.05:
        return Tint(ASH, "flat", "peaks roughly constant over the window")
    if a < 0.3:
        return Tint(STEAM, "trending", "peak heights are drifting with time")
    if a < 1.0:
        return Tint(EMBER, "strong trend", "each episode's peak notably higher or lower")
    return Tint(FLAME, "extreme trend", "exceptional rate of change across peaks")


def episode_samples_tint(n: int) -> Tint:
    """How many tilt readings fed the exp saturation fit."""
    if n < 4:
        return Tint(FLAME, "too few", "exp fit needs ≥ 4 samples to converge")
    if n < 12:
        return Tint(EMBER, "undersampled", "fit converges but with wider uncertainty")
    return _TYPICAL


def exp_k_tint(k: float) -> Tint:
    """Rise-rate parameter k of ``tilt = A·(1 − exp(−k·t)) + C``.

    The time constant is ``1/k``. For Kīlauea UWD we typically see
    ``k ≈ 0.05–0.2 /day`` during active inflation (5–20 day time constants).
    """
    if k <= 0:
        return Tint(FLAME, "invalid", "k must be positive for a rising saturation curve")
    if 0.02 <= k <= 0.3:
        return _TYPICAL
    if k < 0.02:
        return Tint(EMBER, "slow rise", "time constant > 50 days — unusually slow")
    return Tint(EMBER, "rapid rise", "saturating quickly — episode may be short")


def exp_amplitude_tint(A: float) -> Tint:
    """Amplitude A of the exp fit — total rise this episode would gain at saturation."""
    if A <= 0:
        return Tint(FLAME, "invalid", "A must be positive")
    if A < 5:
        return Tint(EMBER, "small", "below typical fountain amplitude (15–35 µrad)")
    if A <= 50:
        return _TYPICAL
    return Tint(EMBER, "large", "above typical fountain amplitude (>50 µrad)")


def render_chip_html(
    label: str,
    value: str,
    unit: str,
    tint: Tint,
) -> str:
    """HTML for one teaching-tool diagnostic chip.

    Separate from the Streamlit ``st.metric`` widget so we can color-tint
    the number, attach a verdict badge, and include a one-line range note —
    all of which Streamlit's built-in metric doesn't support.
    """
    # Convert hex to an rgba fade for the verdict badge background.
    c = tint.color.lstrip("#")
    if len(c) == 6:
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        badge_bg = f"rgba({r}, {g}, {b}, 0.18)"
    else:
        badge_bg = tint.color
    return (
        '<div class="kt-diag-chip">'
        f'<div class="kt-diag-chip__label">{label}</div>'
        f'<div class="kt-diag-chip__value" style="color: {tint.color};">'
        f'{value}<span class="kt-diag-chip__unit">{unit}</span>'
        '</div>'
        f'<div class="kt-diag-chip__verdict" style="background: {badge_bg}; color: {tint.color};">'
        f'{tint.label}</div>'
        f'<div class="kt-diag-chip__note">{tint.note}</div>'
        '</div>'
    )
