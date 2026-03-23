---
name: Live wiring — PhysicsEngine DEPLOYED in sim
description: PhysicsEngine wired into live and running on NT8 sim as of 2026-03-22. AdvanceEngine rename complete.
type: project
---

**COMPLETED 2026-03-22:**
1. Rename BarProcessor → AdvanceEngine (14 files, commit 00228fb5)
2. Deleted dead weight: history_replay.py, replay_bridge.py, atlas_loader.py
3. Wired PhysicsEngine into live_engine.py (commit 9bf32cb8)
4. Fixed: belief_network None guards, 1s bar routing, unrealized PnL, session report append
5. Added verbose decision logging, skip diamonds, deferred FLIP confirmation

**How to run:** `python -m live.launcher --physics`
- Auto-finds latest seed JSON in DATA/regime_seeds/
- Forces anchor_tf='1m'
- Skips TBN/brain/pattern library — just SFE + seeds
- SL: 40 ticks (10 points MNQ) at 1s resolution
- FLIP: deferred re-entry waits for close FILL (prevents 274-contract bug)

**Known gap: NOT PRICE AWARE**
- See memory/project_physics_price_aware.md
- 12 features are all physics, none encode position in higher TF structure
- Adding 1h z-score + 1h fm_sign would be biggest improvement

**How to apply:** Collect sim data, then enrich seeds with higher-TF features.
