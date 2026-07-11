# Logistic Regression DOE: RSI-06_Divergence
**Total Events:** 485
**Base Rate:** 0.4804
**ROC AUC:** 0.5130
**Log Loss:** 0.7070

## Magnitude Weighted Evaluation
> Weights applied during fit based on MFE (wins) and MAE (losses).
> **OOS Guard:** Probabilities generated via Stratified 5-Fold Cross-Validation.

| Tier | N | Mean Post. | Actual WR | Base Delta | Mean MFE | Mean MAE |
|---|---|---|---|---|---|---|
| (0.271, 0.351] | 49 | 0.3228 | 0.3878 | -9.27 pp | 76.94 | -128.01 |
| (0.351, 0.386] | 48 | 0.3681 | 0.5000 | +1.96 pp | 82.83 | -75.41 |
| (0.386, 0.418] | 49 | 0.4015 | 0.5306 | +5.02 pp | 97.67 | -101.42 |
| (0.418, 0.447] | 48 | 0.4330 | 0.5208 | +4.04 pp | 79.32 | -100.19 |
| (0.447, 0.472] | 49 | 0.4624 | 0.3469 | -13.35 pp | 61.03 | -121.97 |
| (0.472, 0.492] | 48 | 0.4833 | 0.5208 | +4.04 pp | 87.48 | -99.18 |
| (0.492, 0.514] | 48 | 0.5028 | 0.5208 | +4.04 pp | 83.80 | -92.97 |
| (0.514, 0.548] | 49 | 0.5305 | 0.4694 | -1.10 pp | 110.80 | -99.34 |
| (0.548, 0.58] | 48 | 0.5623 | 0.5625 | +8.21 pp | 119.77 | -87.39 |
| (0.58, 0.709] | 49 | 0.6213 | 0.4490 | -3.14 pp | 113.18 | -100.05 |
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 483
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.1591
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_1s_Tminus2_L4_5s_lambda_hat_21`
- `Ph1_1s_Tminus1_L4_1h_lambda_se_12`
- `Ph1_15s_Tminus3_L5_15m_ldist_outlier_pct`
- `Ph1_5s_Tminus1_L3_5m_reversion_prob_30`
- `Ph1_1s_Tminus5_L1_15m_vol_accel_1b`
- `Ph1_5s_Tminus3_L1_15s_vol_velocity_1b`
- `Ph1_5m_Tminus1_L4_15s_lambda_hat_12`
- `Ph1_5m_Tminus1_L3_5m_hurst_30`
- `Ph1_15s_Tminus4_L5_15s_ldist_skew`
- `Ph1_1s_Tminus1_L4_5m_lambda_se_30`
- `Ph1_1s_Tminus3_L1_4h_price_velocity_1b`
- `Ph1_5s_Tminus3_L4_4h_lambda_t_21`
- `Ph1_5s_Tminus3_L4_1h_lambda_se_30`
- `Ph1_5s_Tminus3_L5_15s_ldist_outlier_pct`
- `Ph1_5m_Tminus1_L4_1m_lambda_hat_12`
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 483
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.1591
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_1s_Tminus1_L4_5s_lambda_hat_21`
- `Ph1_1s_Tminus1_L4_1h_lambda_se_12`
- `Ph1_15s_Tminus3_L5_15m_ldist_outlier_pct`
- `Ph1_5s_Tminus1_L3_5m_reversion_prob_30`
- `Ph1_1s_Tminus4_L1_15m_vol_accel_1b`
- `Ph1_5s_Tminus3_L1_15s_vol_velocity_1b`
- `Ph1_5m_Tminus1_L4_15s_lambda_hat_12`
- `Ph1_5m_Tminus1_L3_5m_hurst_30`
- `Ph1_15s_Tminus4_L5_15s_ldist_skew`
- `Ph1_1s_Tminus4_L4_5m_lambda_se_30`
- `Ph1_1s_Tminus2_L1_4h_price_velocity_1b`
- `Ph1_15s_Tminus2_L4_4h_lambda_t_21`
- `Ph1_5s_Tminus3_L4_1h_lambda_se_30`
- `Ph1_5s_Tminus3_L5_15s_ldist_outlier_pct`
- `Ph1_5m_Tminus1_L4_1m_lambda_hat_12`
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 483
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.1591
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_1s_Tminus2_L4_5s_lambda_hat_21`
- `Ph1_1s_Tminus1_L4_1h_lambda_se_12`
- `Ph1_15s_Tminus3_L5_15m_ldist_outlier_pct`
- `Ph1_5s_Tminus1_L3_5m_reversion_prob_30`
- `Ph1_1s_Tminus1_L1_15m_vol_accel_1b`
- `Ph1_5s_Tminus3_L1_15s_vol_velocity_1b`
- `Ph1_5m_Tminus1_L4_15s_lambda_hat_12`
- `Ph1_5m_Tminus1_L3_5m_hurst_30`
- `Ph1_15s_Tminus4_L5_15s_ldist_skew`
- `Ph1_1s_Tminus3_L4_5m_lambda_se_30`
- `Ph1_1s_Tminus2_L1_4h_price_velocity_1b`
- `Ph1_5s_Tminus1_L4_4h_lambda_t_21`
- `Ph1_15s_Tminus2_L4_1h_lambda_se_30`
- `Ph1_5s_Tminus3_L5_15s_ldist_outlier_pct`
- `Ph1_5m_Tminus1_L4_1m_lambda_hat_12`