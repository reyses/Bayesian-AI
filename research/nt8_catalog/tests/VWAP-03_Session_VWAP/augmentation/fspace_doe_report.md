# Logistic Regression DOE: VWAP-03_Session_VWAP
**Total Events:** 485
**Base Rate:** 0.6907
**ROC AUC:** 0.4391
**Log Loss:** 0.7985

## Magnitude Weighted Evaluation
> Weights applied during fit based on MFE (wins) and MAE (losses).
> **OOS Guard:** Probabilities generated via Stratified 5-Fold Cross-Validation.

| Tier | N | Mean Post. | Actual WR | Base Delta | Mean MFE | Mean MAE |
|---|---|---|---|---|---|---|
| (0.21, 0.319] | 49 | 0.2844 | 0.7755 | +8.48 pp | 8.29 | -9.94 |
| (0.319, 0.354] | 48 | 0.3363 | 0.6667 | -2.41 pp | 7.24 | -14.52 |
| (0.354, 0.381] | 49 | 0.3689 | 0.7959 | +10.52 pp | 9.53 | -8.41 |
| (0.381, 0.402] | 48 | 0.3920 | 0.7292 | +3.84 pp | 7.71 | -6.57 |
| (0.402, 0.421] | 49 | 0.4115 | 0.6531 | -3.77 pp | 6.03 | -17.14 |
| (0.421, 0.442] | 48 | 0.4315 | 0.6875 | -0.32 pp | 5.67 | -12.78 |
| (0.442, 0.46] | 48 | 0.4514 | 0.6458 | -4.49 pp | 5.36 | -5.24 |
| (0.46, 0.487] | 49 | 0.4717 | 0.7143 | +2.36 pp | 6.56 | -9.35 |
| (0.487, 0.529] | 48 | 0.5063 | 0.6250 | -6.57 pp | 6.32 | -15.62 |
| (0.529, 0.693] | 49 | 0.5731 | 0.6122 | -7.85 pp | 5.18 | -19.24 |
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 482
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.1938
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_15s_Tminus3_L1_15s_price_accel_1b`
- `Ph1_5s_Tminus2_L4_1m_lambda_se_21`
- `Ph1_5s_Tminus1_L4_15s_lambda_t_21`
- `Ph1_5m_Tminus1_L4_1m_lambda_se_30`
- `Ph1_5s_Tminus3_L3_1D_hurst_30`
- `Ph1_5s_Tminus1_L4_5m_lambda_t_30`
- `Ph1_5m_Tminus1_L2_5m_price_accel_30`
- `Ph1_1s_Tminus1_L1_1m_vol_accel_1b`
- `Ph1_1s_Tminus5_L1_5s_lower_wick`
- `Ph1_1s_Tminus5_L4_5s_lambda_se_30`
- `Ph1_5s_Tminus2_L3_5m_z_close_vs_low_30`
- `Ph1_15s_Tminus4_L3_5m_band_pos_30`
- `Ph1_5s_Tminus3_L1_5m_body`
- `Ph1_1s_Tminus1_L1_1m_upper_wick`
- `Ph1_15s_Tminus4_L5_1m_ldist_skew`
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 482
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.1938
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_15s_Tminus3_L1_15s_price_accel_1b`
- `Ph1_5s_Tminus2_L4_1m_lambda_se_21`
- `Ph1_5s_Tminus1_L4_15s_lambda_t_21`
- `Ph1_5m_Tminus1_L4_1m_lambda_se_30`
- `Ph1_5s_Tminus1_L3_1D_hurst_30`
- `Ph1_5s_Tminus1_L4_5m_lambda_t_30`
- `Ph1_5m_Tminus1_L2_5m_price_accel_30`
- `Ph1_1s_Tminus1_L1_1m_vol_accel_1b`
- `Ph1_1s_Tminus5_L1_5s_lower_wick`
- `Ph1_1s_Tminus5_L4_5s_lambda_se_30`
- `Ph1_5s_Tminus2_L3_5m_z_close_vs_low_30`
- `Ph1_15s_Tminus4_L3_5m_band_pos_30`
- `Ph1_5s_Tminus3_L1_5m_body`
- `Ph1_1s_Tminus1_L1_1m_upper_wick`
- `Ph1_15s_Tminus4_L5_1m_ldist_skew`