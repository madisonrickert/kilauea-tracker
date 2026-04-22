# Regression fixtures

`continuity_boundaries.csv` lists user-reported step-discontinuity boundaries
from the 2026-04 "whack-a-mole" period. Each row is a known-bad transition in
the merged `tilt_history.csv` output that should remain continuous (Δ ≤
`CONTINUITY_WARNING_THRESHOLD_MICRORAD`, 4 µrad by default) after the
v3 alignment rewrite is fully deployed.

**Columns:**
- `boundary_before_utc` — last good sample before the step (UTC, naive).
- `boundary_after_utc` — first sample after the step (UTC, naive). May be blank if only the day was captured in the report.
- `reported_delta_microrad` — measured Δ if the user measured it; blank otherwise.
- `source_phase` — which round of patches surfaced this report (`v1-patches-era`, `post-archive-clean`, etc.).
- `user_report_date` — when it was reported.
- `notes` — free-form.

**Lifecycle:** new regressions SHOULD append rows here rather than create separate files. Rows should never be deleted — even after a fix, the row serves as a continuity assertion.

**Usage in tests:** `tests/test_continuity_regression.py` will load this file, cross-reference each boundary against `data/tilt_history.csv` after a full reconcile, and assert `|Δtilt| < CONTINUITY_WARNING_THRESHOLD_MICRORAD` for every row with both timestamps filled in.

PNG fixtures for these captures are NOT checked in because USGS rolls off the source PNGs after their window. We can only regression-test at the merged-output level, not trace-by-trace.
