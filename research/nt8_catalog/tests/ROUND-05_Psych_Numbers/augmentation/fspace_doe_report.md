# Logistic Regression DOE: ROUND-05_Psych_Numbers
**Total Events:** 483
**Base Rate:** 0.3975
**ROC AUC:** 0.5107
**Log Loss:** 0.6903

## Magnitude Weighted Evaluation
> Weights applied during fit based on MFE (wins) and MAE (losses).
> **OOS Guard:** Probabilities generated via Stratified 5-Fold Cross-Validation.

| Tier | N | Mean Post. | Actual WR | Base Delta | Mean MFE | Mean MAE |
|---|---|---|---|---|---|---|
| (0.151, 0.322] | 49 | 0.2761 | 0.3061 | -9.14 pp | 9.87 | -8.97 |
| (0.322, 0.36] | 48 | 0.3443 | 0.4583 | +6.08 pp | 6.86 | -6.95 |
| (0.36, 0.391] | 48 | 0.3767 | 0.4375 | +4.00 pp | 8.23 | -8.11 |
| (0.391, 0.415] | 48 | 0.4027 | 0.3125 | -8.50 pp | 7.97 | -10.55 |
| (0.415, 0.442] | 49 | 0.4277 | 0.4694 | +7.19 pp | 6.51 | -8.12 |
| (0.442, 0.462] | 48 | 0.4525 | 0.3542 | -4.33 pp | 6.42 | -8.44 |
| (0.462, 0.483] | 48 | 0.4725 | 0.4375 | +4.00 pp | 13.54 | -10.46 |
| (0.483, 0.512] | 48 | 0.4965 | 0.3750 | -2.25 pp | 7.48 | -8.37 |
| (0.512, 0.55] | 48 | 0.5316 | 0.4583 | +6.08 pp | 15.04 | -7.55 |
| (0.55, 0.752] | 49 | 0.6057 | 0.3673 | -3.02 pp | 14.35 | -8.95 |
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 482
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.1523
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_15s_Tminus3_L5_15m_ldist_kurtosis`
- `Ph1_1s_Tminus5_L4_15s_z_21`
- `Ph1_15s_Tminus4_L4_15s_lambda_t_12`
- `Ph1_5s_Tminus3_L1_15s_price_velocity_1b`
- `Ph1_15s_Tminus3_L1_15s_body`
- `Ph1_5s_Tminus3_L1_5s_price_accel_1b`
- `Ph1_5m_Tminus1_L1_15m_upper_wick`
- `Ph1_5s_Tminus3_L1_5s_lower_wick`
- `Ph1_15s_Tminus4_L2_5s_vol_velocity_30`
- `Ph1_1s_Tminus1_L1_4h_lower_wick`
- `Ph1_5m_Tminus1_L1_5s_lower_wick`
- `Ph1_15s_Tminus3_L3_1h_SE_high_30`
- `Ph1_5m_Tminus1_L5_1m_ldist_skew`
- `Ph1_5m_Tminus1_L1_5s_vol_velocity_1b`
- `Ph1_5s_Tminus2_L5_5s_ldist_skew`
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 482
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.1523
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_15s_Tminus3_L5_15m_ldist_kurtosis`
- `Ph1_1s_Tminus5_L4_15s_z_21`
- `Ph1_15s_Tminus4_L4_15s_lambda_t_12`
- `Ph1_5s_Tminus3_L1_15s_price_velocity_1b`
- `Ph1_15s_Tminus3_L1_15s_body`
- `Ph1_5s_Tminus3_L1_5s_price_accel_1b`
- `Ph1_5m_Tminus1_L1_15m_upper_wick`
- `Ph1_5s_Tminus3_L1_5s_lower_wick`
- `Ph1_15s_Tminus4_L2_5s_vol_velocity_30`
- `Ph1_1s_Tminus3_L1_4h_lower_wick`
- `Ph1_5m_Tminus1_L1_5s_lower_wick`
- `Ph1_15s_Tminus3_L3_1h_SE_high_30`
- `Ph1_5m_Tminus1_L5_1m_ldist_skew`
- `Ph1_5m_Tminus1_L1_5s_vol_velocity_1b`
- `Ph1_5s_Tminus2_L5_5s_ldist_skew`
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 482
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.1523
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_15s_Tminus3_L5_15m_ldist_kurtosis`
- `Ph1_1s_Tminus5_L4_15s_z_21`
- `Ph1_15s_Tminus4_L4_15s_lambda_t_12`
- `Ph1_5s_Tminus3_L1_15s_price_velocity_1b`
- `Ph1_15s_Tminus3_L1_15s_body`
- `Ph1_5s_Tminus3_L1_5s_price_accel_1b`
- `Ph1_5m_Tminus1_L1_15m_upper_wick`
- `Ph1_5s_Tminus3_L1_5s_lower_wick`
- `Ph1_15s_Tminus4_L2_5s_vol_velocity_30`
- `Ph1_1s_Tminus1_L1_4h_lower_wick`
- `Ph1_5m_Tminus1_L1_5s_lower_wick`
- `Ph1_15s_Tminus3_L3_1h_SE_high_30`
- `Ph1_5m_Tminus1_L5_1m_ldist_skew`
- `Ph1_5m_Tminus1_L1_5s_vol_velocity_1b`
- `Ph1_5s_Tminus2_L5_5s_ldist_skew`