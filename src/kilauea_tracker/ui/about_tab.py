"""About tab — project story + how the model works + attribution.

Fully self-contained: no external data or computed inputs. The copy was
extracted from the ``ℹ️ How does this work?`` expander in the old
``streamlit_app.py`` and expanded with context about the project and author.
"""

from __future__ import annotations

from .. import __version__ as _KT_VERSION

_ABOUT_PROJECT_MARKDOWN = f"""
### About this project

Since late 2024 Kīlauea has entered a distinctive eruptive mode: discrete
**fountain events** separated by quiet weeks of inflation. Each fountain
pulses for hours, drains the summit reservoir a few µrad's worth, then
the cycle resets. The characteristic pre-eruption rise is visible on the
summit tiltmeter days before the next pulse, which makes the timing
modestly predictable.

This app turns that signal into an accessible forecast. The
[USGS Hawaiian Volcano Observatory](https://www.usgs.gov/observatories/hvo)
publishes tilt data only as PNG plots (no raw API), so the pipeline
OCRs the axis labels, traces the plotted curve back into numeric
samples, reconciles five overlapping time-windows into a single
15-minute series, then fits an exponential saturation curve to the
current rise plus a linear trendline through recent peak heights.
The next fountain event is predicted at the intersection of those two
curves. A 200-sample Monte Carlo draw over the fit parameters produces
the 80% confidence band you see on the chart.

It's not an official forecast — see
[USGS HVO](https://www.usgs.gov/observatories/hvo) for advisories and
the [volcano alert notification
system](https://volcanoes.usgs.gov/hans2/about/vans) — but it has
historically landed within about two days of each event since the
current eruptive phase began.

The source is [open on GitHub](https://github.com/madisonrickert/kilauea-tracker);
bug reports and corrections welcome. Running version **v{_KT_VERSION}**.
"""


_HOW_IT_WORKS_MARKDOWN = """
### The model in three sentences

Uēkahuna's tiltmeter rises approximately exponentially between fountain
events as magma re-pressurizes the summit reservoir, then drops sharply
when a fountain releases it. The app fits an exponential saturation curve
to the current rise and draws a linear trendline through recent peak
heights — the next fountain event is predicted at the intersection of
those two curves. A 200-sample Monte Carlo draw over the fit parameters
gives the 80% confidence band.

### Tunable parameters

- **Trendline window** (Chart tab) — how many recent peaks feed the
  linear fit. Narrower windows react faster to recent changes; wider
  windows are smoother.
- **Peak sensitivity** (Chart tab → *Advanced model tuning*) — the three
  sliders control `min prominence`, `min distance between peaks`, and
  `min height`, fed to `scipy.signal.find_peaks`.

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
"""


_AUTHOR_MARKDOWN = """
### About the author

Built by **Madison Rickert**. *Natively multimodal.* Senior software
engineer with twenty years across software architecture and fifteen
across media production, currently building at the intersection of
technology and creative work. This tracker is a side project: a place
to watch physical systems the way she usually watches creative ones,
with a model, a fingerprint, and a confidence band.

More work at [madisonrickert.com](https://madisonrickert.com) · code at
[github.com/madisonrickert](https://github.com/madisonrickert).

Feedback, bug reports, and pull requests are welcome at the
[project repository](https://github.com/madisonrickert/kilauea-tracker).
If you spot a transcription defect in the chart (a sample that looks
wildly off the trendline), the Pipeline tab has a PNG-overlay inspector
that lets you see exactly which source, which pixel, and which OCR pass
produced it. That's the best place to start a bug report.

### Attribution

Raw tilt data and webcam imagery: **USGS Hawaiian Volcano Observatory**.
Safety alerts: **USGS HANS** (aviation color codes) + **NWS Honolulu**
(tephra / air-quality advisories). This app is not affiliated with USGS
or NWS. Predictions are a model, not an official forecast.
"""


def show() -> None:
    """Render the About tab."""
    import streamlit as st

    st.markdown(_ABOUT_PROJECT_MARKDOWN)
    st.markdown("---")
    st.markdown(_HOW_IT_WORKS_MARKDOWN)
    st.markdown("---")
    st.markdown(_AUTHOR_MARKDOWN)
