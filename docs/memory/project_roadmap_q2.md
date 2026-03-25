---
name: Q2 2026 Roadmap — PhysicsEngine (bandaid) + AdvanceEngine (real build)
description: 3-month roadmap to NQ graduation. Two parallel tracks: PhysicsEngine runs now for data/funding, AdvanceEngine gets the full grounded rebuild.
type: project
---

## DMI Flipper (running NOW, funds research — $208/day backtest)
- [x] Deployed to sim (2026-03-25, Sim101, bridge v6.8.1)
- [x] Cross mode: smoothed DMI zero-cross + TP=10t repeating + SL=40t
- [x] Safety lock: refuses non-sim accounts
- [x] Connection loss: bridge sends CONNECTION_LOST, Python flattens + stops
- [x] Bridge reduced to 3 TFs (1s/1m/1h) — 83% less memory
- [ ] Collect 30+ days of data for statistical validation
- Backtest: $208/day (14 months), $82/day OOS, 29.8% WR, 3:1 win/loss

## AdvanceEngine V2 (the real build — grounded templates)

### V2 — K-Means Templates (BUILT, ready to run)
- [x] 70D grounded features (7 features × 10 TFs): `core/grounded_features.py`
- [x] Template matcher with CNN-ready interface: `core/template_matcher.py`
- [x] Training pipeline: `training/advance_v2_trainer.py`
- [x] Full lookahead labeling (Phase 3)
- [x] OOS validation (Phase 5)
- [x] Per-template configs (SL/TP/direction/hold)
- [x] Trade marker logger for inspection
- [ ] Run `python -m training.advance_v2_trainer --phase all`
- [ ] Compare to DMI flipper baseline ($208/day)

### V3 — Brain + Templates (next)
- Load old brain (pre-lookahead-fix) OR train new brain on IS
- Brain provides direction per template state
- DMI provides timing (when to trade)
- Templates provide recognition (what state is this)

### V4 — CNN Direction Predictor (research)
- Simple 1D CNN: 70D input → LONG/SHORT
- Custom loss: maximize PnL not accuracy
- Train IS (6mo), validate OOS (8mo)
- Must beat K-means baseline

### V5 — TCN Multi-Resolution (research)
- Input: (10 TFs, 7 features) as 2D matrix — NOT flattened
- TCN convolves across TF axis with dilations
- Learns cross-TF relationships automatically
- Greedy layer training: add layers until no OOS gain
- See `memory/research_tcn_v5.md`
- Reference: `examples/Overview.md`

### Goal: Merge + Graduate (Month 3)
- Best version replaces DMI flipper as single live engine
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
