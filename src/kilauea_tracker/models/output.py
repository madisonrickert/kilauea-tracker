"""Common-contract dataclasses every prediction model returns.

Pure data — no Plotly types, no Streamlit, no I/O. The chart's renderer
turns ``NamedCurve`` instances into Plotly traces; the model layer only
declares *what* to draw, never *how*. This keeps every model module free
of plotting dependencies (CLAUDE.md purity rule for ``model.py`` and its
siblings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


# Style hints the chart's renderer maps to Plotly trace properties.
# ``primary`` curves are the model's main visual statement (drawn solid,
# accent color). ``secondary`` is a supporting line (dimmer, often dashed).
# ``ribbon`` curves carry ``band_lo`` / ``band_hi`` arrays and render as a
# filled band rather than a single line.
CurveStyle = Literal["solid", "dashed", "dotted"]
ColorRole = Literal["primary", "secondary", "ribbon"]


@dataclass(frozen=True)
class NamedCurve:
    """A single overlay curve a model declares for the chart to draw.

    The ``days`` array is float days since the Unix epoch (matching the
    convention in ``model.to_days``); ``values`` is in microradians. Set
    ``band_lo`` and ``band_hi`` (both same shape as ``values``) to render
    a filled ribbon between them — ``values`` becomes the centerline.
    """

    label: str
    days: np.ndarray
    values: np.ndarray
    style: CurveStyle = "solid"
    color_role: ColorRole = "primary"
    band_lo: np.ndarray | None = None
    band_hi: np.ndarray | None = None


@dataclass(frozen=True)
class ModelOutput:
    """The uniform return type of every registered prediction model.

    Any field can be ``None`` if the underlying fit failed — the function
    never raises. ``curves`` is always a list (possibly empty); models
    without curve-shaped outputs (e.g. interval-only baselines) return
    ``curves=[]`` and rely on ``next_event_date`` + ``confidence_band``
    alone. ``next_event_tilt`` is the y-coordinate (µrad) at which to
    pin the predicted-event marker on the chart — only models that fit
    a curve through tilt-vs-time can supply it; interval-only models
    leave it ``None`` and the chart picks a reasonable fallback.
    """

    next_event_date: pd.Timestamp | None
    confidence_band: tuple[pd.Timestamp, pd.Timestamp] | None
    headline_text: str | None
    curves: list[NamedCurve]
    diagnostics: dict
    next_event_tilt: float | None = None
