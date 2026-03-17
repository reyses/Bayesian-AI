---
name: I-MR control chart for trade exit detection
description: Replace Brownian/fixed giveback with I-MR on per-trade price path — detects regime shifts without shape assumptions
type: project
---

Replace all giveback threshold logic with I-MR control chart on per-trade price path:
- Replay each trade's 15s bars from entry to exit
- Compute I-MR on the price series DURING the trade
- "In control" = hold (normal variation for THIS trade)
- "Out of control" = exit (regime shift detected)

**Why:** Current giveback assumes V-motion reversal. Real trades staircase, drift,
spike-fade, or cascade. Fixed/Brownian thresholds fire on consolidation (false alarm)
or miss slow drifts (late exit). I-MR adapts to the trade's actual behavior.

**Why I-MR specifically:**
- No shape assumption (works for V, staircase, drift, cascade)
- Control limits adapt to the trade's own volatility (MR = moving range)
- Detects the MOMENT behavior changes, not after a fixed % giveback
- Already validated in the system (I-MR regime segments tool)
- Six Sigma foundation — this is process monitoring applied to trades

**How to apply:**
1. Build `tools/imr_trade_replay.py` — replay IS trades with per-bar I-MR
2. For each trade: compute I-MR on close prices, detect first out-of-control bar
3. Compare: I-MR exit bar vs actual exit bar vs optimal exit bar (oracle MFE bar)
4. If I-MR catches reversals earlier → wire into exit engine as primary giveback
5. The MR (moving range) naturally handles staircase (high MR → wide limits)
   vs spike (low MR → tight limits)

**Prerequisite:** Need 15s bar data accessible during forward pass for per-trade
I-MR computation. Currently only have entry/exit/peak in trade log.
