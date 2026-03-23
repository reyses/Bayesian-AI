---
name: PhysicsEngine price-awareness gap
description: PhysicsEngine K-NN has no price structure awareness — same physics at top vs bottom of range produces same match. Biggest edge improvement opportunity.
type: project
---

PhysicsEngine's 12 features are all statistical (fm, z, dmi, adx, vel, vol, hurst, P_center, coherence, sigma, pid). None encode WHERE in the price structure the bar sits.

**Why:** Two identical trajectories match the same seeds, but one at the top of a 1h range (no room) and one at the bottom (full room to run). This is why OOS is $264/day not $500+. The user identified this as the primary edge gap on 2026-03-22 during first live sim test.

**Features to add (research line):**
1. 1h z-score — position in hourly regression band
2. 1h F_momentum sign — with or against hourly trend
3. Distance to recent 1h pivots (ATR-normalized)
4. MTF alignment score (how many TFs agree on direction)

**Critical nuance (user insight 2026-03-22):**
Physics leads ENTRY correctly — momentum building, exhaustion forming. But EXIT (funnel flip)
is also purely physics. It doesn't know if the peak happened at a structural price level
(real reversal) or in the middle of nowhere (noise pause). Result: exits too early at noise
peaks, too late at real structural peaks.

**Implication:** Peak seeds and trend seeds MUST stay separate.
- Trends = physics events (PhysicsEngine handles these)
- Peaks = price events (need price structure: bands, pivots, Fibonacci)
- PhysicsEngine should NOT handle peak exits — it needs a price-aware exit layer

**How to apply:**
- Enrich seeds with higher-TF state at entry time (1h z, 1h fm, 15m z)
- Add to TRAJ_KEYS in physics_engine.py (expand from 12 to 15-16 features)
- For EXITS: add price-relative features (distance to regression bands, prior pivots)
- Re-run OOS to validate improvement
- Related: docs/Active/RESEARCH_MTF_POSITION.md
