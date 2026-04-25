# CLAUDE.md

Collaborator guide for this repo. The user-facing doc is `README.md`; this file is for whoever (human or agent) is editing the code.

## What this app is

A Streamlit app that predicts the next eruption pulse at Kīlauea by tracing tilt data out of USGS plot images. Live deploy: <https://kilauea-tracker.streamlit.app/>. Audience is non-technical — viewers open a URL and see a status banner + chart.

USGS publishes the UWD (Uēkahuna summit) tiltmeter at azimuth 300° as auto-updating PNGs only — no raw API. v2.0 fetches the PNGs, OCRs the axes, traces the blue Az-300 curve, and merges into a local CSV cache. v1.0 (`legacy/eruption_projection.py`) was a Colab notebook with hand-digitized peaks; v2.0 automates that loop.

## What the data looks like

Tilt is a roughly **sawtooth wave**: tilt rises slowly over days as magma accumulates (an "inflation" phase), then drops sharply over hours during an eruption pulse (a "deflation" phase). One full sawtooth = one **episode**, numbered sequentially. Episodes occur 1–2× per month and typically last 8–10 hours of active fountaining.

Together, all episodes since December 2024 comprise Kīlauea's current summit eruption.

### Recent episodes (reference for fixtures, peak detection, sanity checks)

| # | Date | Duration | Notes |
|---|---|---|---|
| 41 | 2026-01-24 | 8h 18m | |
| 42 | 2026-02-15 | ~10h | |
| 43 | 2026-03-10 | 9h | significant tephra fall |
| 44 | 2026-04-09 | 8.5h | |
| 45 | 2026-04-23 | 8.5h | |

The current regime peaks at 2–4 µrad with deflation drops of 7–30 µrad. Earlier in the eruption (2025) peaks were larger (8–12 µrad). Peak-detector defaults rely on **prominence**, not absolute height, so the detector adapts as the regime shifts — see `PeakDetectionDefaults` in `config.py`.

## Running the app

Always use `uv run` — never `source .venv/bin/activate && python …`.

```bash
uv sync                                          # install deps
uv run streamlit run streamlit_app.py            # localhost:8501
uv run pytest                                    # tests
uv run ruff check .                              # lint
uv run python -m kilauea_tracker.ingest.pipeline # ingest once (CLI)
```

Tesseract is a system dep: `brew install tesseract` locally; `packages.txt` installs it on Streamlit Cloud.

## Repo orientation

`streamlit_app.py` is the entrypoint and must stay at the repo root. Code lives in `src/kilauea_tracker/`. The non-obvious parts:

- `config.py` — every tuning knob, with a comment justifying the chosen value.
- `ingest/` — `fetch.py` (HTTP w/ `If-Modified-Since`), `calibrate.py` (OCR axes), `trace.py` (HSV mask + per-column extraction), `pipeline.py` (orchestration), `exceptions.py` (typed boundary errors).
- `model.py`, `reconcile.py`, `peaks.py`, `safety_alerts.py` — **pure** compute layers (no I/O).
- `cache.py`, `archive.py` — CSV I/O.
- `ui/` — Streamlit tab modules + palette/styles.
- `tests/fixtures/` — committed dated PNGs; intentional regression alarm for OCR drift.
- `.github/workflows/refresh-cache.yml` — daily cron (12:07 UTC) commits cache updates back to `main`.
- `legacy/eruption_projection.py` — frozen v1.0 reference; cite line numbers when porting math.
- `.claude/plans/foamy-yawning-horizon.md` — design rationale for the Phase-1 + Phase-2 alignment.

## How the ingest pipeline is structured

Two layers, decoupled on purpose:

1. **Raw layer** — `data/sources/<source>.csv`, one CSV per USGS PNG source. `ingest(source)` fetches + traces + appends with intra-source frame alignment. Sources never touch each other's files.
2. **Reconciled view** — `data/tilt_history.csv`, rebuilt by `reconcile.reconcile_sources()` after every ingest. Reads every per-source CSV plus the digital reference and the archive, solves per-source (a, b) corrections via pairwise OLS, merges by best effective resolution per 15-min bucket, and emits one merged history.

The decoupling is what makes the cache deterministic and order-independent. Reconciliation is a pure function of the raw inputs and can be re-run without re-fetching.

`archive.py` promotes reconciled rows into a frozen historical record (quorum-gated to avoid contamination from a single flaky source). Archive rows younger than `ARCHIVE_MAX_AGE_FOR_PRIORITY_DEMOTION_DAYS` are demoted in the merge so a higher-quality live source can override a recently-archived bad row.

## Data files

All under `data/`, **always committed** (the daily cron's `git add data/` sweeps them in):

- `tilt_history.csv` — merged view the model reads. `Date,Tilt (microradians)`, 15-min cadence.
- `archive.csv` — append-only frozen archive (modulo the youth-window override above). `archive_v1.csv` is the prior schema, kept for back-compat.
- `sources/*.csv` — per-source raw + quality CSVs (5 sources × 2 files). **Load-bearing**: frame alignment requires the previous fetch to be on disk.
- `last_modified.json` — per-source HTTP `Last-Modified` headers driving `If-Modified-Since`.
- `run_reports/*.json` — per-run diagnostics, bounded by `_prune_old_run_reports` to `MAX_RUN_REPORTS = 90`.
- `uwd_digital_az300.csv`, `y_slope_history.json` — processed digital reference and Phase-1a slope history.

**Gitignored** (regenerate; never commit): `data/raw/uwd_digital/UWD_*.csv` (regenerate via `scripts/import_digital_data.py`), `data/last_refresh.json` (per-container fcntl lock state), `NOTES.md`.

## Refresh model

- **Daily cron** runs the ingest pipeline and commits cache updates back to `main`. The push triggers a Streamlit Cloud redeploy. Viewers see at-most-24-hour-stale data even with no live traffic.
- **In-app refresh** is non-blocking: the page renders immediately from the committed cache, then a background thread refreshes and the app reruns when new data lands. Cooldown via `try_acquire_refresh_slot` against `data/last_refresh.json` (fcntl flock). Don't reintroduce a blocking refresh — page load must show fresh data.

## Coding conventions

Python 3.11+ is pinned (`pyproject.toml`). Use modern syntax everywhere; this isn't a stylistic preference, it's the bar.

**Types**

- `from __future__ import annotations` at the top of every module.
- `X | None`, never `Optional[X]`. `list[T]` / `dict[K, V]`, never `List` / `Dict`. Import `Callable`, `Iterator`, `Mapping`, etc. from `collections.abc`, not `typing`.
- Type-hint all public APIs. `dataclass(frozen=True)` for config-shaped objects and report payloads.

**Purity**

- Compute layers are pure: `model.py`, `reconcile.py`, `peaks.py`, `safety_alerts.py`. No I/O, no module-level mutable state, no `datetime.now()` / `time.time()` — pass time in if you need it. Same inputs → same outputs.
- I/O is isolated to `ingest/`, `cache.py`, `archive.py`, and `pipeline.py`. Don't leak it back into the compute layers.

**Errors**

- Raise typed exceptions at module boundaries. Subclass `IngestError` (`ingest/exceptions.py`) for ingest-domain failures; never return error sentinels (`None`, `-1`, `(value, ok)`).
- Catch the narrowest exception that handles the case. No bare `except:` and no `except Exception:` without naming and re-raising.
- The Streamlit layer is the only translation point from typed exceptions to user-facing banners.

**I/O & utilities**

- `pathlib.Path` only — never `os.path.join` or raw string paths.
- f-strings only — never `.format()` or `%`-formatting.
- `logger = logging.getLogger(__name__)` per module. No `print()` in `src/`.
- Timestamps are stored tz-naive UTC. Convert to `Pacific/Honolulu` (`DISPLAY_TIMEZONE` in `config.py`) only at the display boundary.

**Constants & magic numbers**

- Cross-module knobs live in `config.py`, each with a comment citing the incident, measurement, or rationale behind the value.
- Module-internal magic numbers go in `_SCREAMING_CASE` module-level constants with the same justification comment. See `model.py:_INTERSECTION_TILT_MIN` for the pattern — the comment cites the legacy line number, the regime change that broke the old bound, and what `brentq` already guarantees.

**Modules & comments**

- Module-level docstring explains *why* the module exists and what tradeoffs it makes — not just what it does. Inline comments call out hidden constraints (USGS quirks, OCR failure modes, past incidents) and cite line numbers in `legacy/eruption_projection.py` when porting math.
- Imports grouped: stdlib → third-party → local, blank line between groups. No `from x import *`. No mutable default arguments.

**Column conventions**

- Date columns are tz-naive UTC pandas `Timestamp`s, stored as ISO strings with `.000000000` nanosecond precision.
- Tilt values are µrad. The 15-min bucket grid is the canonical resampling cadence — used by alignment, reconcile, archive promotion, and the merged view.

**Tooling**

- `uv run ruff check .` before committing structural changes.
- Tests mirror module names (`test_<module>.py`). Pytest config in `pyproject.toml`: `addopts = "-q --strict-markers"`. Currently 63+ tests across model, peaks, cache, plotting, calibration, trace, ingest, reconcile, archive, safety alerts, and UI structure.
- The dated PNG fixtures in `tests/fixtures/` are an *intentional* OCR-drift alarm: when `test_calibrate.py` fails, the question is "did USGS change their plot rendering?" — not "should we update the fixture?".

## Things to avoid

- Never `source .venv/bin/activate && python …` — always `uv run python …`.
- Never `git add -A` or `git add .` from the repo root — `NOTES.md` and other personal scratch can sneak in. Stage files explicitly.
- Don't reintroduce user-initiated "Refresh" buttons as the primary path. Page load must always show the freshest available data; refresh is background.
- Don't add a `min_height` floor back to peak detection — the regime shifts; rely on prominence.
- Don't add I/O to `model.py` / `reconcile.py` / `peaks.py` / `safety_alerts.py`. They are pure on purpose.
