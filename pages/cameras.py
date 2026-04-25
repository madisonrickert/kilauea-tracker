"""Cameras — full 8-camera live grid.

A thin page: just delegates to ``ui.cameras.show_grid()``. The strip
variant of the same module renders inside the Now page.
"""

from __future__ import annotations

from kilauea_tracker.ui import cameras

cameras.show_grid()
