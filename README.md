# 🌋 Kīlauea Fountain Event Tracker

[![Refresh tilt cache](https://github.com/madisonrickert/kilauea-tracker/actions/workflows/refresh-cache.yml/badge.svg)](https://github.com/madisonrickert/kilauea-tracker/actions/workflows/refresh-cache.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kilauea-tracker.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

**Live app:** <https://kilauea-tracker.streamlit.app/>

Predicts the next eruption pulse at Kīlauea by tracing tilt data straight from
USGS plot images and fitting an exponential saturation curve against the recent
peak trendline. Built as an interactive Streamlit app — non-technical viewers
just open a URL and see the current prediction.

This is **v2.0**, a full rewrite of the original Colab notebook prototype
(`legacy/eruption_projection.py`). v1.0 required manual digitization of USGS
graphs with PlotDigitizer plus hardcoded peak lists. v2.0 automates the entire
loop: fetch PNGs from USGS, OCR the axes, trace the curves, run the model,
render the chart.

## What's in v2.0

- **Automated image ingestion** — fetches the four USGS tilt PNGs (2-day,
  week, month, 3-month), re-calibrates the y-axis on every fetch via
  Tesseract OCR (USGS shifts the y-offset between captures), traces the blue
  Az 300° curve via OpenCV HSV masking, and merges new samples into a local
  history CSV with conflict detection.
- **Auto peak detection** — `scipy.signal.find_peaks` over a 1-hour
  resampled tilt series. Replaces v1.0's hardcoded 6-peak list. Tunable from
  the sidebar.
- **Same v1.0 model** — linear trendline through recent peaks × exponential
  saturation fit on the current episode, intersected via `scipy.optimize.brentq`.
  Bracketed root finding (more robust than v1.0's `fsolve`).
- **Monte Carlo confidence band** — 200 draws from the exp fit covariance
  give a 10th–90th percentile range on the predicted date.
- **Interactive Plotly chart** with hover, pan, zoom, and a shaded
  confidence band — no hardcoded axis limits.
- **Conditional fetching** with `If-Modified-Since` headers, so refresh
  doesn't hammer USGS.
- **Self-watering cache** — a daily GitHub Actions cron runs the ingest
  pipeline and commits the freshened `data/tilt_history.csv` back to `main`,
  triggering a Streamlit Cloud redeploy. First-time viewers always see
  recent data even if no one's clicked Refresh in a while.
- **42 tests** covering the model, peaks, cache, plotting, calibration,
  and trace modules.

## Run locally

You'll need [`uv`](https://docs.astral.sh/uv/) (`brew install uv`) and
Tesseract (`brew install tesseract`).

```bash
uv sync
uv run streamlit run streamlit_app.py
```

The app opens at <http://localhost:8501>. The first ingest takes ~5 seconds
(four PNG fetches + OCR); after that, the cache TTL is 15 minutes.

## Deploy to Streamlit Community Cloud

Streamlit Community Cloud is a free hosting service for Streamlit apps —
push your repo to GitHub and they build and serve the app for you. The
non-technical end users just visit a URL.

**Step by step (first time):**

1. Sign in to <https://share.streamlit.io> with your GitHub account and
   approve the OAuth app.
2. Push this repo to a **public** GitHub repository. Private repos require a
   paid Streamlit tier.
3. On <https://share.streamlit.io>, click **"New app"**.
4. Select your repo, branch (`main`), and main file path (`streamlit_app.py`).
5. Click **"Advanced settings"** → set **Python version** to `3.11`. No
   secrets needed (the USGS PNGs are public).
6. Click **Deploy**. The first build takes ~5 minutes while Streamlit Cloud
   installs Tesseract via `packages.txt` and pip installs the dependencies
   from `requirements.txt`.
7. Your app lives at a URL like
   `https://<your-username>-kilauea-tracker-streamlit-app-<hash>.streamlit.app`.
8. Every push to `main` auto-redeploys.

Streamlit Community Cloud puts inactive apps to sleep after a period of no
traffic. Cold-start takes ~30 seconds when the first viewer arrives. After
that, page loads are fast.

## Project layout

```
kilauea-tracker/
├── streamlit_app.py            # the Streamlit entrypoint
├── pyproject.toml              # uv-managed dep list
├── requirements.txt            # pinned, used by Streamlit Cloud
├── packages.txt                # apt packages for Streamlit Cloud (tesseract)
├── .python-version             # 3.11
├── .streamlit/config.toml      # dark theme + lava-orange accent
├── data/
│   └── tilt_history.csv        # the live cache (bootstraps from legacy/)
├── src/kilauea_tracker/
│   ├── config.py               # the 4 USGS URLs, paths, defaults
│   ├── ingest/
│   │   ├── fetch.py            # GET with If-Modified-Since
│   │   ├── calibrate.py        # OCR axes → pixel↔data transforms
│   │   ├── trace.py            # HSV mask + per-column curve extraction
│   │   ├── pipeline.py         # orchestration, fallbacks, error reporting
│   │   └── exceptions.py
│   ├── peaks.py                # find_peaks wrapper
│   ├── model.py                # predict() — curve fit + brentq intersection
│   ├── cache.py                # CSV history + dedupe + conflict detection
│   └── plotting.py             # Plotly figure builder
├── tests/                      # 41 tests
├── legacy/
│   ├── eruption_projection.py  # the v1.0 Colab script (frozen)
│   └── Tiltmeter Data - Sheet1.csv
└── README.md                   # this file
```

## Data source

Electronic tilt at Uēkahuna (UWD), azimuth 300°, published as auto-updating
PNGs by [USGS Hawaiian Volcano Observatory](https://www.usgs.gov/volcanoes/kilauea/science/monitoring-data-kilauea).
Specific image URLs:

- 2-day:   `https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-TILT-2day.png`
- 1-week:  `https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-POC-TILT-week.png`
- 1-month: `https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-POC-TILT-month.png`
- 3-month: `https://volcanoes.usgs.gov/vsc/captures/kilauea/UWD-TILT-3month.png`

USGS does not publish raw tilt data as CSV/JSON — only these auto-updating
plot images. v2.0 traces the curves directly out of the PNGs.

## Known limits

1. **Y-axis OCR fragility.** If USGS changes their plot fonts, colors, or
   tick spacing, the OCR can fail to recover enough labels. The committed
   PNG fixture in `tests/fixtures/` is dated, so test_calibrate.py will fail
   loudly the day this happens — that's our regression alarm.
2. **Calibration drift between captures.** Each USGS PNG has its own
   y-offset, so the same time bucket can produce slightly different tilts
   depending on which capture window you ingested. The cache surfaces these
   as soft "conflict" warnings.
3. **The model assumes the current episode is rising.** If tilt is in a
   post-eruption recovery / flat regime, the exp fit may not produce a
   meaningful intersection — the app correctly shows "—" rather than
   inventing a date.
4. **Streamlit Community Cloud cold starts** take ~30 seconds after periods
   of inactivity. Document this for your viewers if relevant.

## Tests

```bash
uv run pytest
```

Coverage: 42 tests across model, peaks, cache, plotting, calibration,
trace, and ingest pipeline modules.

## License

MIT — see [LICENSE](./LICENSE).
