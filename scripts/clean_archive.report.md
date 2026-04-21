# Cache cleanup diff report

## Per-source CSVs — rows flagged by the rolling-median filter

Applied the `trace._filter_rolling_median_outliers` filter (threshold ±4.0 µrad) to each per-source CSV. Rows listed below were in the CSVs from before the filter existed. With `--clean-sources`, they are removed from the on-disk file.

| Source | Drops |
|---|---:|
| three_month | 0 |
| month | 0 |
| week | 0 |
| two_day | 0 |
| dec2024_to_now | 0 |

## Archive — rows flagged for removal

Scanned 16190 archive rows.
  • `truth` rows were compared to the re-reconciled live-sources value at the same bucket (threshold ±5.0 µrad).
  • `orphan` rows have no live-source coverage; they were compared to a ±6h rolling median of the archive itself (threshold ±4.0 µrad). Catches residual phantoms whose contributing source was already cleaned upstream.
Flagged 2 row(s) total.

| Date (UTC) | Archive tilt | Reference | Δ | Reason |
|---|---:|---:|---:|---|
| 2026-04-12 08:37:28 | -26.54 | -17.10 | -9.44 | orphan-isolated |
| 2026-04-12 09:18:22 | -26.47 | -17.10 | -9.37 | orphan-isolated |
