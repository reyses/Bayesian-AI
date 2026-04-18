---
name: Audit lookahead before trusting any baseline
description: 2026-04-16 discovery — higher-TF aggregation in build_dataset had 6-hour lookahead. $740/day was inflation. Audit any aggregation that uses searchsorted/index lookups.
type: feedback
---

# Always audit lookahead before trusting a baseline

**Rule:** Before reporting or using any IS/OOS baseline, trace the
feature aggregation for lookahead. Specifically check any
`np.searchsorted` or index-lookup that maps a low-TF timestamp to a
higher-TF bar — bars labeled at their START contain data from their
END. Subtract the TF period before lookup.

**Why:** 2026-04-17 found this exact bug in `training/build_dataset.py`.
Inner loop used:
```python
idx = np.searchsorted(tf_ts, target_ts, side='right') - 1
```
The matched TF bar started at or before `target_ts`, but its OHLCV
aggregated 5s bars forward up to its end. Every label at time T had
future data baked in for TFs ≥ 1m.

Fix:
```python
idx = np.searchsorted(tf_ts, target_ts - period, 'right') - 1
```

**Impact:** Baseline dropped from $740/day IS (lookahead) to -$164/day
(honest). Every analysis done between the bug introduction and the fix
is suspect. The 79D feature folder was renamed/moved during the fix:
`DATA/FEATURES_79D_1m/` → `DATA/ATLAS/FEATURES_5s/`.

**How to apply:**
- Any time a baseline seems too clean (low variance across OOS days,
  high $/day, IS/OOS nearly equal), suspect lookahead before celebrating.
- When building new feature aggregation code, test with a sentinel:
  inject a huge value into the last 5s bar of a TF window and verify
  that it doesn't appear in the label at the TF start.
- When merging feature code from other projects, re-audit — lookahead
  bugs often travel with good-looking results.
