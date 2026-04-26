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
from .interval_median import IntervalMedianModel
from .output import ModelOutput, NamedCurve
from .protocol import Model
from .trendline_exp import TrendlineExpModel

__all__ = [
    "IntervalMedianModel",
    "Model",
    "ModelOutput",
    "NamedCurve",
    "TrendlineExpModel",
    "registry",
]


def _register_default_models() -> None:
    """Populate the registry. Idempotent — re-imports leave the registry
    in the same state because ``register`` raises on duplicates and we
    suppress that here. (Streamlit reruns can re-import this package.)"""
    for model in (TrendlineExpModel(), IntervalMedianModel()):
        with contextlib.suppress(ValueError):
            registry.register(model)


_register_default_models()
