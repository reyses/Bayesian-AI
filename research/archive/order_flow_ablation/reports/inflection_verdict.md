# Stage 1 Exhaustion Verdict (Full Causal Gauntlet)

## 5-Fold Walk-Forward & Sub-Period Stability
- **L1 (Baseline):** 0.6156 avg AUC (In Bounds [0.55-0.75])
- **L2 (OHLCV Wicks):** 0.6179 avg AUC
- **L3 (True Delta Enriched):** 0.6214 avg AUC (Lift over L2: +0.0036)

### Fold-by-Fold Stability (L3 AUCs)
- **Fold 1:** 0.5819
- **Fold 2:** 0.6135
- **Fold 3:** 0.6338
- **Fold 4:** 0.6417
- **Fold 5:** 0.6363

## Fourier Null Gauntlet (Fold 5)
- **L3 Fold 5 AUC:** 0.6363
- **Fourier Null 95th pctile (N=20):** 0.6406
- **True Lift over Null:** -0.0043

## Final Verdict: BREAK
**Tick Data Purchase Decision:** NO (Stick to Free Wicks or Baseline)
