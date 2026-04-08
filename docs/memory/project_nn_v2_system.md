---
name: nn_v2 3-CNN Trading System
description: Current active system — 79D features, NMP+blended engine, 3 CNNs, $613/day OOS on clean data
type: project
---

## nn_v2 3-CNN System (2026-04-07)

### Results
- IS: $620/day, 96% win days
- OOS: $613/day, 91% win days, $22/trade
- BASE_NMP improved from $0.30/trade to $7.90/trade with CNN flip (26x)

### Architecture
```
ticker (1s) → aggregator (all TFs) → SFE → 79D features → NMP → blended engine → 3 CNNs
```

**Blended Engine** (`nn_v2/nightmare_blended.py`):
- Cascade tier: aggressive z-based
- Kill shot tier: |z|>2 + vr<1 + wick rejection (96% WR, $42/day)  
- Base NMP tier: standard z_se>2, vr<1

**Three CNNs**:
1. **CNN Flip** (`cnn_flip.py`): 70.6% direction accuracy from 6×13 TF grid at entry
2. **CNN Hold** (`cnn_hold.py`): 94.8% accuracy, 98.9% HOLD, 69.6% EXIT
3. **CNN Risk** (`cnn_risk.py`): cuts losers (0% WR on cuts = correct)

### Key Learnings
- Trees exhausted at 55% direction — CNN sees cross-TF patterns trees can't
- Regret is the teacher, CNN is the student (skip trees/rules/books)
- Entry-only CNN > path CNN (path data adds noise)
- CNN distillation: primary split = 15m_wick_ratio (same as kill shot)
- Zero crossing pattern: odd = winner, even = loser

### Next Step: Stage 2
1. Run regret on CNN-flipped trades
2. Extract 79D at optimal entry points
3. Train Stage 2 CNN on new entry physics
4. Cluster → validate → expand ExNMP roster

**Why:** This is the first system with positive OOS on clean data ($613/day).
**How to apply:** All new work builds ON TOP of nn_v2. No changes to legacy core/.
