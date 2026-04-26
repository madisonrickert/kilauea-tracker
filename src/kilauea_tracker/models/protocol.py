"""Protocol every prediction model satisfies.

Models are duck-typed against this protocol — they don't subclass it.
A typical model module exposes a small frozen-dataclass-like object
with the four attributes here and registers itself in the registry on
import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd

    from .output import ModelOutput


@runtime_checkable
class Model(Protocol):
    """The interface every registered prediction model satisfies."""

    id: str
    """Stable string identifier — used as the dict key in the registry,
    persisted in run reports, and stored in ``adv_active_model_id``
    widget state. Don't rename without a migration."""

    label: str
    """Human-readable name shown in the model selector dropdown."""

    description: str
    """One-sentence tooltip shown beside the selector to help the user
    pick. Surfaces in ``st.selectbox(help=...)`` on the Chart page."""

    def predict(
        self, tilt_df: pd.DataFrame, peaks_df: pd.DataFrame
    ) -> ModelOutput:
        """Return a ``ModelOutput`` for the given tilt history + peaks.

        Pure: same inputs → same outputs, no I/O, no clock reads.
        Never raises — return a ``ModelOutput`` with ``None`` fields and
        a ``diagnostics`` entry describing the failure if the fit can't
        be computed.
        """
        ...
