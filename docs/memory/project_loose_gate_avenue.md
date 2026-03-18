---
name: OOS compressed gate loosening avenue
description: Removing cascade/struct from OOS gate → 3x trades, $12K OOS (excl Feb 9 outlier) but lower $/trade
type: project
---

Removing `cascade_detected OR structure_confirmed` from OOS compressed signal gate:
- 44% of bars have pattern_type but no cascade/struct — previously blocked
- Loosening: 5,087 trades, $40K OOS ($12K excl Feb 9 outlier), $2.52/trade
- Tight (current): 1,691 trades, $6.7K OOS, $3.99/trade

**Why deferred:** Lower $/trade ($2.52 vs $6.16 previous best) means commissions
and slippage eat more of the edge in live. Need to filter low-quality pattern
trades before loosening.

**How to apply when ready:**
1. Loosen the gate (pattern_type alone)
2. Add quality filter: coherence > 0.60, or ADX > 20, or 5m DMI agrees
3. This should keep the extra trades but filter noise entries
4. Re-validate OOS excluding outlier days
