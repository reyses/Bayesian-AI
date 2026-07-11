# Logistic Regression DOE: VWMA-10_Divergence
**Total Events:** 485
**Base Rate:** 0.4536
**ROC AUC:** 0.4908
**Log Loss:** 0.7151

## Magnitude Weighted Evaluation
> Weights applied during fit based on MFE (wins) and MAE (losses).
> **OOS Guard:** Probabilities generated via Stratified 5-Fold Cross-Validation.

| Tier | N | Mean Post. | Actual WR | Base Delta | Mean MFE | Mean MAE |
|---|---|---|---|---|---|---|
| (0.301, 0.419] | 49 | 0.3879 | 0.5714 | +11.78 pp | 29.74 | -19.63 |
| (0.419, 0.463] | 48 | 0.4405 | 0.3542 | -9.94 pp | 25.46 | -25.93 |
| (0.463, 0.485] | 49 | 0.4758 | 0.4490 | -0.46 pp | 21.14 | -28.41 |
| (0.485, 0.509] | 48 | 0.4982 | 0.4583 | +0.47 pp | 25.79 | -19.66 |
| (0.509, 0.525] | 49 | 0.5166 | 0.4286 | -2.50 pp | 25.06 | -25.32 |
| (0.525, 0.54] | 48 | 0.5337 | 0.4583 | +0.47 pp | 25.79 | -14.98 |
| (0.54, 0.562] | 48 | 0.5511 | 0.3750 | -7.86 pp | 13.12 | -21.53 |
| (0.562, 0.584] | 49 | 0.5738 | 0.5918 | +13.82 pp | 26.83 | -14.16 |
| (0.584, 0.611] | 48 | 0.5991 | 0.4792 | +2.56 pp | 20.52 | -16.23 |
| (0.611, 0.705] | 49 | 0.6405 | 0.3673 | -8.63 pp | 13.89 | -15.78 |
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 479
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.1596
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_5s_Tminus3_L1_1m_vol_velocity_1b`
- `Ph1_1s_Tminus1_L2_5m_vol_accel_9`
- `Ph1_5s_Tminus3_L2_15s_vol_accel_30`
- `Ph1_1s_Tminus5_L3_5s_reversion_prob_9`
- `Ph1_5m_Tminus1_L5_15s_ldist_outlier_pct`
- `Ph1_1s_Tminus2_L1_4h_price_accel_1b`
- `Ph1_5s_Tminus3_L1_1m_lower_wick`
- `Ph1_5s_Tminus1_L2_5s_vol_velocity_30`
- `Ph1_15s_Tminus4_L3_1h_hurst_30`
- `Ph1_1s_Tminus5_L4_5s_lambda_t_30`
- `Ph1_5s_Tminus2_L4_5m_lambda_se_21`
- `Ph1_5s_Tminus1_L4_5s_lambda_hat_12`
- `Ph1_15s_Tminus4_L2_1m_vol_accel_30`
- `Ph1_1s_Tminus1_L4_5s_lambda_se_12`
- `Ph1_1s_Tminus1_L4_5m_lambda_hat_12`
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 479
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.1596
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_5s_Tminus3_L1_1m_vol_velocity_1b`
- `Ph1_1s_Tminus2_L2_5m_vol_accel_9`
- `Ph1_5s_Tminus3_L2_15s_vol_accel_30`
- `Ph1_1s_Tminus5_L3_5s_reversion_prob_9`
- `Ph1_5m_Tminus1_L5_15s_ldist_outlier_pct`
- `Ph1_1s_Tminus1_L1_4h_price_accel_1b`
- `Ph1_5s_Tminus3_L1_1m_lower_wick`
- `Ph1_5s_Tminus1_L2_5s_vol_velocity_30`
- `Ph1_15s_Tminus4_L3_1h_hurst_30`
- `Ph1_1s_Tminus5_L4_5s_lambda_t_30`
- `Ph1_5s_Tminus2_L4_5m_lambda_se_21`
- `Ph1_5s_Tminus1_L4_5s_lambda_hat_12`
- `Ph1_15s_Tminus4_L2_1m_vol_accel_30`
- `Ph1_1s_Tminus2_L4_5s_lambda_se_12`
- `Ph1_1s_Tminus5_L4_5m_lambda_hat_12`