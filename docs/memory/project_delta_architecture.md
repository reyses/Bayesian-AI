---
name: Delta vs Absolute Architecture
description: ALL engine computations are cumsum-based (absolute). This is why OOS != Live. Delta-based engine would fix parity permanently.
type: project
---

ALL MarketState fields depend on cumulative history (bar count from start):
- F_momentum = kp*error + ki*CUMSUM + kd*derror (ki*cumsum grows forever)
- DMI = Wilder smoothing (exponential cumsum)
- ADX = double smoothed cumsum
- P_at_center = regression window (history-dependent)
- velocity = derivative of cumsum
- sigma = regression band width (history window)

**Why:** PID controller integral term (ki * cumsum) accumulates over all bars.
OOS starts after 379K IS bars -> deep cumsum. Live starts from zero -> shallow.
Same bar, same price, different F_momentum (218 vs 12).

**How to apply:**
- Short-term: ATLAS warmup loads pre-computed states (deployed 2026-03-20)
- Long-term: refactor engine to output bar-to-bar DELTAS instead of absolute values
- Crow brain features: use deltas, not absolute MarketState values
- CNN input: trajectory of changes over 10 bars, not endpoint values
- Sensor gates: check "is decreasing?" not "is < threshold"
- This eliminates warmup, pre-computed states, and parity gap permanently
