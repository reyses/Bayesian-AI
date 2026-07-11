# F-Space DOE Statistical Report: OHLC-01_Prior_Day

> **Status:** Pending Data Pipeline Execution

## 1. Stepwise Elimination Impact

### Pre-Stepwise (All 416 Features)
* **AIC:** [Pending]
* **BIC:** [Pending]
* **Pseudo R-Squared:** [Pending]
* **Baseline Win Rate:** [Pending]

### Post-Stepwise (Vital Few Features)
* **AIC:** [Pending]
* **BIC:** [Pending]
* **Pseudo R-Squared:** [Pending]
* **Augmented Predictive Win Rate:** [Pending]

## 2. Standardized Effects (Pareto)
*(Visuals pending script generation)*
* **Top Linear Effects:** [Pending]
* **Top Quadratic Effects:** [Pending]
* **Top Cubic Effects:** [Pending]

## 3. Interaction Plot Highlights
*(Visuals pending script generation)*
* **Interaction 1:** [Pending]
* **Interaction 2:** [Pending]

## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 551
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.1736
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_5s_Tminus2_L2_1m_price_sigma_30`
- `Ph1_15s_Tminus4_L4_15m_vr_exact`
- `Ph1_5s_Tminus1_L3_5s_reversion_prob_30`
- `Ph1_5m_Tminus1_L1_5m_vol_accel_1b`
- `Ph1_15s_Tminus4_L3_15s_reversion_prob_30`
- `Ph1_5m_Tminus1_L3_1m_hurst_30`
- `Ph1_5m_Tminus1_L1_15s_vol_accel_1b`
- `Ph1_15s_Tminus3_L1_4h_price_accel_1b`
- `Ph1_15s_Tminus3_L3_5m_reversion_prob_30`
- `Ph1_15s_Tminus2_L1_15s_price_accel_1b`
- `Ph1_1s_Tminus1_L3_5m_z_low_9`
- `Ph1_15s_Tminus4_L3_5s_reversion_prob_30`
- `Ph1_15s_Tminus3_L4_5s_lambda_t_12`
- `Ph1_15s_Tminus2_L5_15s_ldist_skew`
- `Ph1_5m_Tminus1_L3_15s_reversion_prob_30`