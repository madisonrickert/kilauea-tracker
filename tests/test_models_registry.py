"""Tests for the model registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from kilauea_tracker.models import registry
from kilauea_tracker.models.output import ModelOutput

if TYPE_CHECKING:
    import pandas as pd


class _FakeModel:
    id = "fake"
    label = "Fake model"
    description = "Always returns None for tests."

    def predict(self, tilt_df: pd.DataFrame, peaks_df: pd.DataFrame) -> ModelOutput:
        return ModelOutput(
            next_event_date=None,
            confidence_band=None,
            headline_text=None,
            curves=[],
            diagnostics={},
        )


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot and restore the module-level registry around each test."""
    saved = dict(registry.REGISTRY)
    yield
    registry.REGISTRY.clear()
    registry.REGISTRY.update(saved)


def test_register_then_get_round_trips():
    fake = _FakeModel()
    registry.register(fake)
    got = registry.get("fake")
    assert got is fake


def test_get_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        registry.get("does-not-exist")


def test_register_duplicate_id_raises():
    fake = _FakeModel()
    registry.register(fake)
    with pytest.raises(ValueError):
        registry.register(fake)


def test_list_models_returns_registered_in_insertion_order():
    fake_a = _FakeModel()
    fake_b = _FakeModel()
    fake_b.id = "fake_b"  # type: ignore[misc]
    registry.register(fake_a)
    registry.register(fake_b)
    listed_ids = [m.id for m in registry.list_models()]
    # Both should appear; insertion order preserved (registry is a dict).
    assert "fake" in listed_ids
    assert "fake_b" in listed_ids
    assert listed_ids.index("fake") < listed_ids.index("fake_b")


def test_default_model_id_is_trendline_exp():
    assert registry.DEFAULT_MODEL_ID == "trendline_exp"
