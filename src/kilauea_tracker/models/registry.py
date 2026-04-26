"""Module-level registry of prediction models.

Models register themselves on import (typically from
``models/__init__.py``). Consumers ask the registry by id; the registry
preserves insertion order so the selector dropdown renders in a stable
sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import Model

REGISTRY: dict[str, Model] = {}

# The model the homepage and the cron pipeline use when no explicit id
# is requested. Changing this changes what non-technical visitors see
# on the Now page — coordinate with the visible-curves story before
# touching it.
DEFAULT_MODEL_ID: str = "trendline_exp"


def register(model: Model) -> None:
    """Add ``model`` to the registry under its ``id``.

    Raises ``ValueError`` if a model with the same id is already registered.
    The duplicate guard is intentional — silent overwrites would let a
    stale import order silently swap which model the Now page renders.
    """
    if model.id in REGISTRY:
        raise ValueError(f"model id already registered: {model.id!r}")
    REGISTRY[model.id] = model


def get(model_id: str) -> Model:
    """Look up a model by id. Raises ``KeyError`` if not registered."""
    if model_id not in REGISTRY:
        raise KeyError(f"unknown model id: {model_id!r}")
    return REGISTRY[model_id]


def list_models() -> list[Model]:
    """Every registered model, in insertion order."""
    return list(REGISTRY.values())
