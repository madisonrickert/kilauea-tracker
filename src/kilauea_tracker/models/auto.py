"""Phase-aware ensemble — picks ``linear`` or ``linear_naive`` per phase.

The 7-segment / 4-quartile backtest (see ``backtest.py`` and the
report committed to the planning notes) shows a clean phase
crossover: ``linear`` (late-window fit) wins at quartiles 25% and 50%,
while ``linear_naive`` (full-window fit) wins at 75% and 100%. The
crossover is roughly at the halfway mark — the same threshold the
phase estimator surfaces as ``LATE_PHASE_THRESHOLD = 0.5``.

This model uses the phase estimator to pick which underlying base
model to delegate to, returning that model's ``ModelOutput`` verbatim
plus a diagnostic indicating which base was picked and why.

Pure: no I/O, no clock reads. The phase estimator's ``now`` defaults
to the latest tilt sample's timestamp, which keeps this deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .linear import LinearModel, LinearNaiveModel

if TYPE_CHECKING:
    import pandas as pd

    from .output import ModelOutput


@dataclass(frozen=True)
class AutoModel:
    """Picks ``linear`` (early phase) or ``linear_naive`` (late phase)
    based on the phase estimator. Best of both models per the backtest.
    """

    id: str = "auto"
    label: str = "Auto (phase-aware: linear → linear_naive)"
    description: str = (
        "Switches between the two best within-cycle linear models based "
        "on how far through the inflation phase we are. Uses `linear` "
        "below the halfway mark (best at quartiles 25% and 50% in the "
        "backtest), `linear_naive` at and after the halfway mark (best "
        "at quartiles 75% and 100%). Halfway = elapsed-since-trough "
        "÷ median historical inflation duration. Falls back to `linear` "
        "if the phase can't be estimated."
    )

    def predict(
        self, tilt_df: pd.DataFrame, peaks_df: pd.DataFrame
    ) -> ModelOutput:
        # Lazy import — ``phase`` reaches into ``models._episodes``,
        # which would trigger this package's __init__ during import
        # cycles if loaded at module top.
        from ..phase import LATE_PHASE_THRESHOLD, estimate_phase

        phase = estimate_phase(tilt_df, peaks_df)
        if phase.fraction is not None and phase.fraction >= LATE_PHASE_THRESHOLD:
            chosen_label = "linear_naive"
            out = LinearNaiveModel().predict(tilt_df, peaks_df)
        else:
            chosen_label = "linear"
            out = LinearModel().predict(tilt_df, peaks_df)

        # Annotate the output with phase context. We can't mutate the
        # frozen dataclass, so we return a copy with the merged
        # diagnostics; everything else passes through unchanged.
        merged_diag = dict(out.diagnostics)
        merged_diag.update({
            "auto_chosen_base": chosen_label,
            "auto_phase_fraction": phase.fraction,
            "auto_elapsed_hours": phase.elapsed_hours,
            "auto_median_duration_hours": phase.median_duration_hours,
            "auto_n_historical_episodes": phase.n_historical_episodes,
            "auto_threshold": LATE_PHASE_THRESHOLD,
        })
        new_headline = None
        if out.headline_text is not None:
            new_headline = (
                f"{out.headline_text} · auto picked {chosen_label}"
                f" (phase ~{phase.fraction:.0%})"
                if phase.fraction is not None
                else f"{out.headline_text} · auto picked {chosen_label}"
            )

        # Return a ModelOutput with the merged diagnostics + new headline.
        from .output import ModelOutput as _MO

        return _MO(
            next_event_date=out.next_event_date,
            confidence_band=out.confidence_band,
            headline_text=new_headline,
            curves=out.curves,
            diagnostics=merged_diag,
            next_event_tilt=out.next_event_tilt,
        )
