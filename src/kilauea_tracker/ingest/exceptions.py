"""Ingest pipeline exceptions.

These are raised at the boundary between modules and caught by the Streamlit
layer, which surfaces them as user-facing banners (see `streamlit_app.py`).
"""

from __future__ import annotations


class IngestError(Exception):
    """Base class for any ingest failure. Always carries a human-readable message."""


class FetchError(IngestError):
    """The HTTP fetch failed (network error, 4xx, 5xx, etc.)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class CalibrationError(IngestError):
    """The image was fetched but its axes could not be re-calibrated.

    Causes: OCR found too few axis labels, the title timestamp regex didn't
    match, the plot bounding box couldn't be located, etc.
    """


class TraceError(IngestError):
    """The image was fetched and calibrated but the data curve couldn't be
    extracted (HSV mask was empty, the curve was too short, etc.).
    """
