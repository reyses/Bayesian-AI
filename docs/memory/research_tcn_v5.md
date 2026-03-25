---
name: research_tcn_v5
description: TCN architecture for AdvanceEngine v5 — multi-TF learned pattern recognition replacing K-means
type: project
---

## TCN / Dilated 1D CNN — AdvanceEngine V5

**Why:** K-means (v2) manually flattens 70D features. TCN learns which TF relationships matter.

**Architecture:**
- Input: (10 TFs, 7 features) = (10, 7) matrix — NOT flattened
- Conv1D across TF axis with dilations (1, 2, 4) — sees adjacent to full TF range
- Flatten → Dense → sigmoid → LONG/SHORT
- Each dilation level discovers cross-TF patterns (1m+3m, 1m+1h, 1s+1W)

**Key design decisions:**
- Keep TFs separated as raw base features, let TCN figure out relationships
- Don't pre-compute 70D — that's manual feature engineering the TCN should learn
- Hurst is low (0.004), market isn't truly fractal — it's regime-switching with persistence
- TCN handles regime-switching naturally via dilated receptive fields
- If some TFs are noise (2m, 3m), TCN learns zero weights — self-pruning

**Layers vs Lookback distinction:**
- Layers = computation depth (how many feature combinations)
- Lookback = how many bars of history
- Higher TFs ENCODE lookback (1h DMI = 60 bars of context)
- So lookback = 1 bar of (10, 7) snapshot — TCN convolves across TF axis, not time axis
- This makes it closer to an MLP with structured input than a traditional temporal CNN

**Greedy layer training:**
- Start 1 layer, measure OOS, add layer, measure, stop when no gain
- Each layer adds a level of TF interaction discovery
- Feature tree says 3 levels max → expect 3 layers optimal

**Training:**
- PyTorch, train on IS (6 months), validate on OOS (8 months) — fixed ratio
- Custom loss: maximize PnL, not accuracy
- ONNX export for inference

**Hardware:** RTX 3060 12GB (training), GTX 1060 3GB (inference if needed)

**Prerequisites (must complete first):**
1. K-means v2 baseline (AdvanceEngine V2) — establishes the number to beat
2. 70D feature extraction validated on OOS
3. IS/OOS ratio fixed (6/8 months)

**Reference:** `examples/Overview.md` — external research on TCN vs LSTM vs Transformer for multi-scale signals
