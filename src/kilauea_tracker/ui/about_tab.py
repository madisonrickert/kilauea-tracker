"""About tab — how the model works + data sources + attribution.

Fully self-contained: no external data or computed inputs. The copy was
extracted from the ``ℹ️ How does this work?`` expander in the old
``streamlit_app.py``.
"""

from __future__ import annotations


_OVERVIEW_MARKDOWN = """
### The model in three sentences

Uēkahuna's tiltmeter rises approximately exponentially between fountain
events as magma re-pressurizes the summit reservoir, then drops sharply
when a fountain releases it. The app fits an exponential saturation curve
to the current rise and draws a linear trendline through recent peak
heights — the next fountain event is predicted at the intersection of
those two curves. A 200-sample Monte Carlo draw over the fit parameters
gives the 80% confidence band.

### Tunable parameters

- **Trendline window** — how many recent peaks feed the linear fit.
  Narrower windows react faster to recent changes; wider windows are
  smoother.
- **Peak sensitivity** (advanced) — the three sliders control `min
  prominence`, `min distance between peaks`, and `min height`, fed to
  `scipy.signal.find_peaks`.

### Data source

- **UWD tiltmeter**, azimuth 300°, Uēkahuna bluff, ~800 m NW of the
  Halemaʻumaʻu caldera rim.
- Operated by the **USGS Hawaiian Volcano Observatory**.
- Published only as PNGs; the app OCRs the axis labels and traces the
  plotted curves to recover numeric values. Five PNG sources (2-day,
  1-week, 1-month, 3-month, and Dec 2024 → now) plus a digital-only
  source (Jan–Jun 2025) are reconciled into a single 15-minute time series.

### Why azimuth 300°?

Kīlauea's summit tilt was historically recorded in *radial / tangential*
directions relative to the caldera. On 9 July 2025, USGS switched UWD's
published azimuth to **300°** — the direction that best isolates the
deflation signature of fountain events at the current vent. All data
shown here is on the 300° axis, including older archival data that has
been rotated to match.

### Attribution

Raw tilt and webcam imagery: **USGS Hawaiian Volcano Observatory**.
This app is not affiliated with USGS. Predictions are a model, not an
official forecast — see [USGS HVO](https://www.usgs.gov/observatories/hvo)
for official advisories.
"""


def show() -> None:
    """Render the About tab."""
    import streamlit as st

    st.markdown(_OVERVIEW_MARKDOWN)
