---
name: distilled-orange_line_eda
description: Curvature-flip of a 7.5min cubic fit leads slope-flip by ~20s median, causally — a candidate early exit signal, not yet turned into a strategy.
metadata: {type: distilled, topic: orange_line_eda, status: concluded}
---
## Verdict
Asked whether a causal early-exit signal exists ahead of the "orange line" turn
(a 7.5-min cubic OLS fit to 1s closes, evaluated at the trailing endpoint —
value/slope/curvature all computed causally from the trailing 7.5 min only).
Found: curvature zero-crossing (inflection) reliably precedes the slope
zero-crossing (the turn) by a positive lead time, across 10 days of 1s data.
Read-out is descriptive EDA only — no P&L, no entry/exit rule was backtested.

## Key numbers (with CIs where they exist)
- Window: 2024_03_01..2024_03_14 (10 trading days), 1s bars, orange fit = 7.5min
  cubic (ORANGE_W=450 samples, ORANGE_DEG=3).
- Swings (slope zero-cross to zero-cross): n=3776. Amplitude (pts): median
  3.31, p75 8.45, p90 16.70. Duration (min): median 3.1, p75 4.6.
- BIG swings (amp>=p75, n=944): median amp 14.41 pts, median dur 3.8 min.
- Curvature-flip-inside-swing: 2676/3776 swings (~70.9%).
- LEAD time (curvature-flip -> slope-flip): median 20s, p25 10s, p75 37s.
- No CIs reported in the source file (single-run EDA, not bootstrapped).

## Graveyard / never-retry (if any)
- None recorded — this is an early-stage EDA note, not a killed experiment.

## Reusable assets
- `research/orange_line_eda/tools/orange_line_eda.py` — computes causal
  cubic-fit value/slope/curvature via fixed convolution weights
  (`_cubic_weights`, `_roll`) from trailing 1s ATLAS data; produces the
  3-panel plot (price+curve, slope, curvature) and the swing/lead-time stats
  above. Run: `python research/orange_line_eda/tools/orange_line_eda.py [day]`.

## Data locations
- Reads `DATA/ATLAS/1s/<day>.parquet` (columns: timestamp, close).
- Produced `research/orange_line_eda/reports/orange_line_eda_2024_03_18.png`
  (example day plot).

## Open threads
- Whether curvature-flip-as-exit actually improves $/day vs the slope-flip
  (lagging) baseline is explicitly flagged as untested next step in the
  report ("the lever to test next") — no follow-up run found in this folder.
- README.md and project.md (DMAIC/PDCA) are both empty stubs — no Define/
  Measure/Analyze/Improve/Control content was ever filled in.

## Sources
- research/orange_line_eda/reports/orange_line_eda.md
- research/orange_line_eda/tools/orange_line_eda.py
- research/orange_line_eda/reports/orange_line_eda_2024_03_18.png
- research/orange_line_eda/README.md (stub)
- research/orange_line_eda/project.md (stub)

## Archive recommendation
ARCHIVE (single-run EDA note, no strategy/backtest built on it, no follow-up
in 4+ months since the report; the curvature-lead finding is worth keeping as
a citable data point but the folder itself is inactive — check for CLI-false-
orphan refs to `orange_line_eda.py` before moving).
