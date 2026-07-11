# Logistic Regression DOE: SEASON-12_DayOfWeek
**Total Events:** 387
**Base Rate:** 0.5607
**ROC AUC:** 0.4737
**Log Loss:** 0.7066

## Magnitude Weighted Evaluation
> Weights applied during fit based on MFE (wins) and MAE (losses).
> **OOS Guard:** Probabilities generated via Stratified 5-Fold Cross-Validation.

| Tier | N | Mean Post. | Actual WR | Base Delta | Mean MFE | Mean MAE |
|---|---|---|---|---|---|---|
| (0.29, 0.496] | 39 | 0.4535 | 0.6410 | +8.03 pp | 95.23 | -51.33 |
| (0.496, 0.524] | 39 | 0.5107 | 0.6667 | +10.59 pp | 125.08 | -80.89 |
| (0.524, 0.549] | 38 | 0.5364 | 0.5263 | -3.44 pp | 113.05 | -78.09 |
| (0.549, 0.569] | 39 | 0.5587 | 0.5128 | -4.79 pp | 91.29 | -107.83 |
| (0.569, 0.584] | 39 | 0.5764 | 0.5897 | +2.90 pp | 121.42 | -90.33 |
| (0.584, 0.604] | 38 | 0.5928 | 0.5000 | -6.07 pp | 107.70 | -105.71 |
| (0.604, 0.623] | 39 | 0.6137 | 0.5128 | -4.79 pp | 72.81 | -62.93 |
| (0.623, 0.644] | 38 | 0.6323 | 0.3947 | -16.60 pp | 69.03 | -127.81 |
| (0.644, 0.681] | 39 | 0.6589 | 0.6410 | +8.03 pp | 144.35 | -80.24 |
| (0.681, 0.795] | 39 | 0.7215 | 0.6154 | +5.47 pp | 117.07 | -74.34 |
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 386
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.2150
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_5s_Tminus2_L4_4h_lambda_hat_12`
- `Ph1_1s_Tminus2_L5_15m_ldist_kurtosis`
- `Ph1_15s_Tminus2_L5_15s_ldist_outlier_pct`
- `Ph1_5m_Tminus1_L2_1m_price_sigma_30`
- `Ph1_1s_Tminus1_L1_1D_upper_wick`
- `Ph1_15s_Tminus4_L1_5s_price_velocity_1b`
- `Ph1_15s_Tminus3_L3_5s_z_close_vs_high_30`
- `Ph1_1s_Tminus1_L2_5m_vol_accel_9`
- `Ph1_5s_Tminus1_L4_4h_lambda_se_12`
- `Ph1_5m_Tminus1_L3_5m_reversion_prob_30`
- `Ph1_5s_Tminus1_L3_5m_reversion_prob_30`
- `Ph1_5s_Tminus1_L4_1h_vr_exact`
- `Ph1_1s_Tminus1_L4_5m_lambda_t_12`
- `Ph1_15s_Tminus3_L4_15s_lambda_t_30`
- `Ph1_15s_Tminus2_L5_15s_ldist_kurtosis`
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 386
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.2150
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_5s_Tminus2_L4_4h_lambda_hat_12`
- `Ph1_1s_Tminus1_L5_15m_ldist_kurtosis`
- `Ph1_15s_Tminus2_L5_15s_ldist_outlier_pct`
- `Ph1_5m_Tminus1_L2_1m_price_sigma_30`
- `Ph1_5s_Tminus2_L1_1D_upper_wick`
- `Ph1_15s_Tminus4_L1_5s_price_velocity_1b`
- `Ph1_15s_Tminus3_L3_5s_z_close_vs_high_30`
- `Ph1_1s_Tminus1_L2_5m_vol_accel_9`
- `Ph1_5s_Tminus1_L4_4h_lambda_se_12`
- `Ph1_5m_Tminus1_L3_5m_reversion_prob_30`
- `Ph1_15s_Tminus4_L3_5m_reversion_prob_30`
- `Ph1_5s_Tminus1_L4_1h_vr_exact`
- `Ph1_1s_Tminus1_L4_5m_lambda_t_12`
- `Ph1_15s_Tminus3_L4_15s_lambda_t_30`
- `Ph1_15s_Tminus2_L5_15s_ldist_kurtosis`