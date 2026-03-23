---
name: Live wiring — PhysicsEngine + AdvanceEngine rename
description: Next session priority. Wire PhysicsEngine into live, rename BarProcessor to AdvanceEngine.
type: project
---

**Priority 1: Rename BarProcessor → AdvanceEngine**
- 53 references across 14 files
- Rename class, file (bar_processor.py → advance_engine.py), all imports
- Mechanical but needs clean context to avoid typos

**Priority 2: Wire PhysicsEngine into live/live_engine.py**
- PhysicsEngine runs on 1m bars (not 15s)
- Needs: StatisticalFieldEngine for state computation
- Needs: enriched seed library loaded at startup (426MB)
- Flag to switch between AdvanceEngine and PhysicsEngine
- Both coexist — competing engines, user picks which one trades

**Priority 3: NT8 bridge integration**
- PhysicsEngine sends ENTER/EXIT/FLIP to NT8
- FLIP = exit + immediate re-enter (one bridge message or two?)
- SL still needed as capital protection (PhysicsEngine has no SL)

**PhysicsEngine config (proven $264/day OOS):**
- 12 features x 10-bar trajectory K-NN
- K=20, consensus > 0.65, coherence < 0.6, magnitude > p25
- FLIP exit: funnel direction change = exit + re-enter
- Max hold from matched seed median (3-20 bars)
- Seed library: DATA/regime_seeds/auto_seeds_all_20260322_154729.json

**How to apply:** Start next session with rename, then wire, then test with --dry-run.
