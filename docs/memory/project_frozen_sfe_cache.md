---
name: Frozen SFE cache bug (fixed 2026-04-16)
description: LiveFeatureEngine SFE cache keyed on valid_idx alone collided after 5000-bar trim. Stale SFE state since mid-Feb. Every live session before fix traded on frozen features.
type: project
---

# Frozen SFE cache bug — fixed 2026-04-16

**What happened:** `training/live_feature_engine.py`'s `_compute_features`
cached SFE state keyed on `valid_idx` (count of valid bars up to current
timestamp). After the bar store hit its 5000-bar trim limit, every new
bar's `valid_idx` stayed at 5000 (newest always at end after trim), so
the cache returned stale SFE state from whichever bar first reached 5000.

**Observable symptom:** In the 2026-04-16 Phase 6 live session, z was
frozen at +3.95 for 1,346 bars (2 hours) during an MTF_BREAKOUT trade.
Only 2 unique z values across the entire trade (-2.07 and +3.95). Every
z-dependent exit (`fade_mean_reached`, `fade_z_expanding`, oscillation
tracking) never fired.

**Broken since mid-February.** Every live session from mid-Feb through
2026-04-16 was trading on frozen features. All live PnL from that window
is noise relative to the actual engine signal.

**Fix:**
```python
# BEFORE: cache key = valid_idx (collides after trim)
if cached and cached[0] == valid_idx:

# AFTER: cache key = (valid_idx, latest_bar_ts) — invalidates on new bar
if cached and cached[0] == (valid_idx, latest_bar_ts):
```

**Also fixed in same session:**
- `_find_today_start` used UTC midnight — fixed to match batch
  `get_day_start` file boundaries
- 5000-bar trim cap removed (root cause, not symptom)
- Mock mode `exclude_day` extended to `day_name >= exclude_day` so LFE
  store doesn't contain bars from the replay day or later
- Accumulators reset in mock mode to avoid stale carry-over

**First honest live session post-fix**: $900 peak PnL on first 2-hour
trade — frozen z had been hiding real edge. (Session was later stopped
when chain trades died on a bad MTF_BREAKOUT signal.)

**How to apply:** Any future caching in LFE or live components must key
on `(logical_state, latest_bar_ts)` — state identity alone isn't enough
if the state re-populates the same slot.
