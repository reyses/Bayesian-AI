---
name: MTF Two-Layer Counter-Proposal (Approved)
description: Approved architecture — 29D features, two-layer CNN (Direction + Duration), per-TF z-score norm
type: project
---

## Counter-Proposal: MTF Two-Layer Architecture

**Status**: APPROVED (2026-03-27). Spec: `docs/Active/COUNTER_PROPOSAL_MTF_TWO_LAYER.md`

**29D Features** (13D base + 16D MTF):
- 13D base: dmi_diff, dmi_gap, vol_rel, dir_vol, velocity, z_se, price_accel + 4D regime + 2D context
- 16D MTF: 4 TFs (1s, 5m, 15m, 1h) × 4 features each
  - 1s = pulse/noise floor (what's happening NOW inside forming bar)
  - 5m = swing structure
  - 15m = session trend
  - 1h = structural walls
- Per-TF z-score normalization (each TF independent before concatenation)

**Two Layers**:
- Layer 1: StatePredictor (direction) — 29D input, predicts feature state at t+10
- Layer 2: DurationPredictor (take/skip + hold commitment) — reduces trade fragmentation

**Build Order (4 phases)**:
- Phase A: 29D feature pipeline + MTF alignment validation + .npy cache
- Phase B: Train Layer 1 on 29D
- Phase C: Train Layer 2 DurationPredictor
- Phase D: Full two-layer simulation with 2s slippage

**Why:** Trade fragmentation (271-400 trades/day at 1m) costs 4 ticks round-trip. Layer 2 fixes this.
**How to apply:** Execute phases A→D sequentially. Baseline to beat: $1,609/day from 13D single-layer.
