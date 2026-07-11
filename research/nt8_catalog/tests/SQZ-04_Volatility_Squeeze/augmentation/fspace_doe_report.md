# Logistic Regression DOE: SQZ-04_Volatility_Squeeze
**Total Events:** 130
**Base Rate:** 0.5692
**ROC AUC:** 0.3745
**Log Loss:** 0.8346

## Magnitude Weighted Evaluation
> Weights applied during fit based on MFE (wins) and MAE (losses).
> **OOS Guard:** Probabilities generated via Stratified 5-Fold Cross-Validation.

| Tier | N | Mean Post. | Actual WR | Base Delta | Mean MFE | Mean MAE |
|---|---|---|---|---|---|---|
| (0.199, 0.455] | 13 | 0.3848 | 0.8462 | +27.69 pp | 108.15 | -40.33 |
| (0.455, 0.518] | 13 | 0.4864 | 0.6923 | +12.31 pp | 72.29 | -68.42 |
| (0.518, 0.569] | 13 | 0.5387 | 0.7692 | +20.00 pp | 34.63 | -11.48 |
| (0.569, 0.594] | 13 | 0.5785 | 0.3846 | -18.46 pp | 27.83 | -43.29 |
| (0.594, 0.625] | 13 | 0.6092 | 0.3846 | -18.46 pp | 41.75 | -23.02 |
| (0.625, 0.645] | 13 | 0.6344 | 0.7692 | +20.00 pp | 39.94 | -19.17 |
| (0.645, 0.686] | 13 | 0.6709 | 0.3846 | -18.46 pp | 30.19 | -19.27 |
| (0.686, 0.736] | 13 | 0.7065 | 0.4615 | -10.77 pp | 37.42 | -63.42 |
| (0.736, 0.83] | 13 | 0.7788 | 0.6923 | +12.31 pp | 49.87 | -50.15 |
| (0.83, 0.977] | 13 | 0.8803 | 0.3077 | -26.15 pp | 47.60 | -61.31 |
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 120
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.8728
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_5s_Tminus3_L4_5s_lambda_se_21`
- `Ph1_15s_Tminus3_L5_5s_ldist_skew`
- `Ph1_5m_Tminus1_L1_5m_lower_wick`
- `Ph1_1s_Tminus2_L4_4h_lambda_t_30`
- `Ph1_1s_Tminus2_L3_15m_z_low_12`
- `Ph1_5m_Tminus1_L2_1m_vol_sigma_30`
- `Ph1_1s_Tminus5_L3_5s_z_se_9`
- `Ph1_15s_Tminus4_L1_15s_vol_velocity_1b`
- `Ph1_15s_Tminus2_L1_15s_price_accel_1b`
- `Ph1_5s_Tminus3_L3_5s_z_se_30`
- `Ph1_5m_Tminus1_L4_5m_lambda_t_21`
- `Ph1_1s_Tminus1_L4_1D_lambda_hat_12`
- `Ph1_15s_Tminus4_L2_1m_vol_accel_30`
- `Ph1_1s_Tminus1_L4_15m_lambda_se_30`
- `Ph1_15s_Tminus3_L4_1D_lambda_t_12`
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 120
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.8726
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_5s_Tminus3_L4_5s_lambda_se_21`
- `Ph1_15s_Tminus3_L5_5s_ldist_skew`
- `Ph1_5m_Tminus1_L1_5m_lower_wick`
- `Ph1_1s_Tminus5_L4_4h_lambda_t_30`
- `Ph1_1s_Tminus1_L3_15m_z_low_12`
- `Ph1_5m_Tminus1_L2_1m_vol_sigma_30`
- `Ph1_1s_Tminus5_L3_5s_z_se_9`
- `Ph1_15s_Tminus4_L1_15s_vol_velocity_1b`
- `Ph1_15s_Tminus2_L1_15s_price_accel_1b`
- `Ph1_5s_Tminus3_L3_5s_z_se_30`
- `Ph1_5m_Tminus1_L4_5m_lambda_t_21`
- `Ph1_1s_Tminus3_L4_1D_lambda_hat_12`
- `Ph1_15s_Tminus4_L2_1m_vol_accel_30`
- `Ph1_1s_Tminus3_L4_15m_lambda_se_30`
- `Ph1_5s_Tminus3_L4_1D_lambda_t_12`