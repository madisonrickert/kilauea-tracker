"""Prediction-model package.

Each model is a small module exporting a ``Model``-conformant object.
This ``__init__`` imports every concrete model module and registers its
instance so the shared registry is populated by the time any consumer
calls ``models.registry.list_models()``.

Adding a new model means: (1) write a new module exposing a frozen
dataclass with ``id``/``label``/``description``/``predict``, (2) add
two lines below to import + register it. No other file in the codebase
needs to change for the new model to appear in the chart's selector,
the cron's run-report log, and the homepage's default-model fallback.
"""

from __future__ import annotations

import contextlib

from . import registry
from .auto import AutoModel
from .ffm_voight import FFMVoightModel
from .interval_median import IntervalMedianModel
from .linear import LinearModel, LinearNaiveModel
from .linear_hist import LinearHistModel
from .linear_stitched import LinearStitchedModel
from .output import ModelOutput, NamedCurve
from .power_law import PowerLawModel
from .power_law_hist import PowerLawHistModel
from .protocol import Model
from .trendline_exp import TrendlineExpModel

__all__ = [
    "AutoModel",
    "FFMVoightModel",
    "IntervalMedianModel",
    "LinearHistModel",
    "LinearModel",
    "LinearNaiveModel",
    "LinearStitchedModel",
    "Model",
    "ModelOutput",
    "NamedCurve",
    "PowerLawHistModel",
    "PowerLawModel",
    "TrendlineExpModel",
    "registry",
]


def _register_default_models() -> None:
    """Populate the registry. Idempotent — re-imports leave the registry
    in the same state because ``register`` raises on duplicates and we
    suppress that here. (Streamlit reruns can re-import this package.)

    Selector order: ``auto`` first because it's the configured default
    (the phase-aware ensemble that picks the best base model for the
    current inflation phase). Then the within-cycle base models, then
    the cross-cycle models, then the experimental FFM, then the
    interval-median baseline last as an always-available sanity check.
    ``trendline_exp`` is kept registered for back-compat and historical
    comparison but slotted near the bottom — the per-quartile backtest
    showed it's the worst performer in the current regime.
    """
    models_in_order = (
        AutoModel(),
        LinearModel(),
        LinearNaiveModel(),
        LinearHistModel(),
        LinearStitchedModel(),
        PowerLawModel(),
        PowerLawHistModel(),
        TrendlineExpModel(),
        FFMVoightModel(),
        IntervalMedianModel(),
    )
    for model in models_in_order:
        with contextlib.suppress(ValueError):
            registry.register(model)


_register_default_models()
