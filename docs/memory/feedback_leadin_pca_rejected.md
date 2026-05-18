---
name: feedback-leadin-pca-rejected
description: Lead-in PCA signatures hurt direction classifier at all lookback lengths (60/240/720 bars); V2 entry features already encode macro setup
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

Tested whether lead-in PCA signatures (centroid + direction vector in 184-D V2 space, computed over the past K 5s-bars before entry) improve a binary LONG/SHORT direction classifier on the daisy-chain regret-oracle trades. Three lookbacks (60-bar/5min, 240-bar/20min, 720-bar/60min). **All hurt.**

| Variant | Test AUC | Test Brier | Train-test gap |
|---|---|---|---|
| Baseline (V2 entry only) | **0.864** | **0.142** | 0.000 |
| + 60-bar lead-in | 0.850 | 0.152 | 0.038 |
| + 240-bar lead-in | 0.842 | 0.156 | 0.047 |
| + 720-bar lead-in | 0.849 | 0.153 | 0.037 |

**Why:** Train AUC rises (~0.864 → ~0.888) while test AUC falls — textbook overfit. The 368 extra features (centroid 184-D + direction 184-D) carry noise the linear model latches onto.

**How to apply:**
- For direction prediction on regret-oracle-style trades, stop trying to add lead-in trajectory features as PCA-line summaries to a linear model.
- V2 entry features (L1-L3 × 5s/15s/1m/5m/15m/1h/4h/1D) at the entry bar already encode the multi-TF macro setup — the 4h/1D-layer features carry the regime info a lead-in PCA would extract.
- PCA signatures are lossy 2-vector summaries of a K×184 matrix. Unit-direction vectors going into a linear classifier are geometrically incoherent — the model can't use them without per-trade context.
- Next AUC lever is **non-linear** (GBM, CNN), not more features. The model class has more headroom than the feature set does.
- L3 clusters may still help for magnitude/risk/exit prediction, but NOT for direction routing.

**Doesn't mean lead-in is useless universally** — direct ridge regression on lead-in centroid (no entry features) gave R²=0.146 on signed_mfe (better than entry-feature → cluster route at R²=-0.05). The failure mode is concatenation with entry features into a linear classifier, where the entry features carry the same info more cleanly.

Connected: [[feedback-kway-r2-saturation]], [[feedback-regret-research-methodology]], [[project-bayesian-archetypes-pending]], [[project-regret-six-layer-architecture]].
