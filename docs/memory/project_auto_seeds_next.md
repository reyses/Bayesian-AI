---
name: Auto seeds as template library (next major feature)
description: 31,605 auto seeds across 312 days with direction, MFE, MAE, duration. Replace HDBSCAN templates with seed matching.
type: project
---

Auto seeds are at `DATA/regime_seeds/auto_swing/auto_seeds_edited_20260313_212432.json`.

**Why:** HDBSCAN on full ATLAS produced only 3 templates. IS underperforms OOS ($0.97 vs $4.93/trade). Seeds are human-trained, pre-labeled with direction and expected outcomes. No clustering needed.

**How to apply:**
1. Augment seeds with peak detection + sensor data + 192D context
2. At runtime: peak fires -> extract 10-bar shape + context -> match nearest seed
3. Seed tells you: direction, expected MFE, MAE, duration
4. Then add quantum features as additional context dimensions
5. FibonacciPivots (S1-S3, R1-R3) as support/resistance context

Seed structure: `{trade_id, direction, entry_price, exit_price, mfe_ticks, mae_ticks, duration_mins, lookback_bars: 10, ...}`
