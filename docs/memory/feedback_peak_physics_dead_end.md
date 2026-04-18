---
name: Peak-physics exits are a dead end
description: Features don't change at the peak. Don't build exits that try to detect the peak in real time. Baseline natural exit beats every physics rule.
type: feedback
---

# Peak-physics exits are a dead end

**Rule:** Do not propose or build exit logic that tries to detect the peak
of a trade from physics signals (velocity flip, acceleration reversal,
wick on other side, p_at_center hit, etc.). Every such rule lost to the
natural exit baseline on KILL_SHOT. The peak is a statistical maximum
over noise, not a detectable feature event.

**Why:** 2026-04-17 ran `tools/killshot_peak_physics.py` on 2,043
KILL_SHOT trades with peak > $3. Measured features AT peak vs ±3 bars:
- Velocity flips against trade: 3.3% fire rate
- Acceleration flips: 0.2%
- Wick on other side (>30% jump): 6.8%
- Largest Cohen-d across peak: 0.19 (1m_wick_ratio)

Back-test on that cohort:
| Rule | $/trade |
|---|---|
| Natural exit (baseline) | +$11.61 |
| Fixed $10 target | +$11.22 |
| 50% trail from peak | +$3.40 (worst — bails on every wiggle) |
| Velocity flip | +$6.83 |
| Every other physics rule | lost to baseline |

Full report: `reports/findings/2026-04-17_killshot_peak_physics.md`

**How to apply:**
- If a tier is losing and someone (incl. me) proposes "detect the peak
  and exit there," cite this finding and refuse.
- If the problem is that the tier gives back 50% of peak, the fix is
  either (a) a fixed target, (b) better entry filter to avoid the bad
  trades that never develop, or (c) tier rebuild from data. NOT
  physics peak detection.
- Exception: if a new tier is proposed with a fundamentally different
  structure (e.g., using order-flow or TF-level signals absent here),
  the finding doesn't automatically rule it out — but it must prove
  separability at the peak before we build exit logic around it.
