---
name: ADX chop filter on peak entries
description: 1m ADX < 15 blocks peak entries in choppy markets. Feb 9 went from -$8,869 to -$24. Biggest single improvement to OOS.
type: project
---

ADX chop filter added 2026-03-19 as Layer 3 of peak entry gate.

**Why:** Peak detection fires on every price wiggle in chop. 94% of entries in low-ADX markets are noise. The system was entering and exiting repeatedly in sideways conditions, bleeding money.

**How to apply:** `_1m_confirms_peak()` in `bar_processor.py` checks 1m ADX. Below 15 = block. This is the single biggest improvement: Feb 9 crash went from -$8,869 (222 trades) to -$24 (1 trade). OOS PnL jumped from $6,143 to $14,336.

Three-layer gate: (1) 1m sensor opposition, (2) fake peak (vol+fm), (3) ADX chop. All in `_1m_confirms_peak()`.
