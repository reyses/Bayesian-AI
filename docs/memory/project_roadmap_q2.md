---
name: Q2 2026 Roadmap — PhysicsEngine (bandaid) + AdvanceEngine (real build)
description: 3-month roadmap to NQ graduation. Two parallel tracks: PhysicsEngine runs now for data/funding, AdvanceEngine gets the full grounded rebuild.
type: project
---

## PhysicsEngine (bandaid — running NOW, funds research)
- [x] Deployed to sim (2026-03-22, commit 9bf32cb8)
- [ ] Fix ORPHAN_FLATTEN bug (-$572 in first session)
- [ ] Fix 0-bar SL entries (entering at bad price, instant reversal)
- [ ] IS leave-one-out seed weighting (slow learning, 1/100th per match)
- [ ] Rebalance IS/OOS: 6 months seeds / 8 months validation
- [ ] Distance threshold (reject garbage matches, don't trade "least bad")
- Keep running, keep collecting data

## AdvanceEngine (the real build)

### Goal 1 — Week 1-2: Ground the measurement system
- Replace 12 ungrounded features with 13 grounded ones in SFE
- Grounded derivatives (4): velocity, z_score, acceleration, dmi_diff
- Distribution (3): std_price, std_volume, variance_ratio
- Structural (3): fib_position, higher_tf_z, session_phase
- Cross signals (3): volume_delta, price×volume, dmi×volume_exhaustion
- Each feature answers ONE nameable question
- Reference: docs/reference/FIRST_PRINCIPLES_FRAMEWORK.md

### Goal 2 — Week 3-4: Rebuild the matching
- Drop templates/centroids — raw K-NN against seed library
- Templates were premature optimization for a constraint that doesn't exist
- IS leave-one-out seed weighting
- Distance threshold (reject garbage matches)
- Rebalance IS/OOS ratio (6 months IS / 8 months OOS)
- Validate weighted K-NN outperforms unweighted on expanded OOS

### Goal 3 — Month 2: Exhaustion detection
- Optimize for "is the move done?" not "which direction?"
- The flip timing IS the edge (blind flip = $7,967 OOS)
- Volume profile feature (historical liquidity clusters from ATLAS)
- 1s building the 1m bar in real-time (bottom-up estimation)
- Higher TF z-score (structural position / liquidity proximity)

### Goal 4 — Month 3: Merge + graduate
- AdvanceEngine replaces PhysicsEngine as single live engine
- One engine, grounded features, proven on expanded OOS
- 30+ days consistent profitability on MNQ sim
- Graduate to NQ (same system, 10x capital)
- Target: 2026-06-23

## Operational Costs & Break-Even
- Claude API: $100/month
- Power: ~$40/month
- Daily target: **$60/day** ($1,320/month) to cover costs + incentive
- On MNQ: 120 ticks/day profit, or ~4 clean trades at $15 average
- On NQ: 12 ticks/day profit — trivial if system works
- First session (2026-03-22): $1,495 gross, $264 without outlier
- Biggest leak: ORPHAN_FLATTEN (-$572) + bad SL entries (-$354) = -$926 lost to plumbing

## Key Principles (from 2026-03-22 session)
- Base measurements: Price, Time, Volume — everything else is derived
- DOE principle: measure the PROCESS (market) not the EQUIPMENT (system)
- Derivatives OK at any order if each step answers a nameable question
- The market is NOT Brownian — edge = deviations from random
- Variance ratio = the Brownian test (replaces Hurst + ADX)
- Liquidity is latent — measure effects (volume at price, rejection)
- Templates were premature optimization — raw K-NN at 38K is fast enough
- Optimize for exhaustion detection, not direction prediction
- Everything grounded in probability with sample size
