# Logistic Regression DOE: SCALP-18_VWAP_EMA
**Total Events:** 43
**Base Rate:** 0.9535
**ROC AUC:** 0.7927
**Log Loss:** 0.3294

## Magnitude Weighted Evaluation
> Weights applied during fit based on MFE (wins) and MAE (losses).
> **OOS Guard:** Probabilities generated via Stratified 5-Fold Cross-Validation.

| Tier | N | Mean Post. | Actual WR | Base Delta | Mean MFE | Mean MAE |
|---|---|---|---|---|---|---|
| (0.010496, 0.62879] | 5 | 0.2311 | 0.8000 | -15.35 pp | -1.90 | -1.90 |
| (0.62879, 0.88814] | 4 | 0.7864 | 1.0000 | +4.65 pp | -5.12 | -5.12 |
| (0.88814, 0.95774] | 4 | 0.9376 | 1.0000 | +4.65 pp | -1.38 | -1.38 |
| (0.95774, 0.97942] | 4 | 0.9660 | 0.7500 | -20.35 pp | -22.56 | -22.56 |
| (0.97942, 0.98596] | 5 | 0.9833 | 1.0000 | +4.65 pp | 3.20 | 3.20 |
| (0.98596, 0.9926] | 4 | 0.9902 | 1.0000 | +4.65 pp | -3.94 | -3.94 |
| (0.9926, 0.9994] | 4 | 0.9963 | 1.0000 | +4.65 pp | -1.44 | -1.44 |
| (0.9994, 0.99982] | 4 | 0.9997 | 1.0000 | +4.65 pp | -1.50 | -1.50 |
| (0.99982, 0.99996] | 4 | 0.9998 | 1.0000 | +4.65 pp | 1.06 | 1.06 |
| (0.99996, 1.0] | 5 | 1.0000 | 1.0000 | +4.65 pp | -3.55 | -3.55 |
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 42
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 1.0000
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_5s_Tminus2_L5_15m_ldist_n`
- `Ph1_5s_Tminus2_L4_4h_lambda_t_21`
- `Ph1_1s_Tminus2_L4_5m_vr_exact`
- `Ph1_5s_Tminus3_L4_15s_lambda_t_21`
- `Ph1_1s_Tminus5_L5_1m_ldist_std`
- `Ph1_5s_Tminus3_L4_1h_lambda_hat_30`
- `Ph1_5s_Tminus1_L3_1D_reversion_prob_30`
- `Ph1_1s_Tminus3_L4_15s_lambda_hat_12`
- `Ph1_1s_Tminus1_L4_1h_lambda_se_30`
- `Ph1_1s_Tminus4_L2_4h_vol_sigma_18`
- `Ph1_15s_Tminus2_L3_1h_swing_noise_30`
- `Ph1_5s_Tminus1_L2_1h_vol_sigma_30`
- `Ph1_1s_Tminus4_L4_5m_lambda_hat_12`
- `Ph1_15s_Tminus2_L4_4h_lambda_se_30`
- `Ph1_5m_Tminus1_L4_1h_z_21`
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 42
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 1.0000
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_1s_Tminus2_L5_15m_ldist_n`
- `Ph1_15s_Tminus3_L4_4h_lambda_t_21`
- `Ph1_1s_Tminus4_L5_5s_ldist_std`
- `Ph1_1s_Tminus2_L3_1m_reversion_prob_15`
- `Ph1_15s_Tminus2_L1_1m_vol_accel_1b`
- `Ph1_5s_Tminus1_L4_5m_vr_exact`
- `Ph1_15s_Tminus3_L4_5m_lambda_se_21`
- `Ph1_1s_Tminus1_L3_5m_z_low_9`
- `Ph1_1s_Tminus3_L3_1D_hurst_5`
- `Ph1_1s_Tminus3_L1_15s_bar_range`
- `Ph1_1s_Tminus5_L2_5s_vol_sigma_9`
- `Ph1_1s_Tminus4_L2_1D_vol_mean_5`
- `Ph1_1s_Tminus2_L3_15m_hurst_12`
- `Ph1_5s_Tminus3_L2_15s_price_velocity_30`
- `Ph1_15s_Tminus3_L2_5m_price_sigma_30`
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 42
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 1.0000
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_1s_Tminus2_L5_15m_ldist_n`
- `Ph1_5m_Tminus1_L4_4h_lambda_t_21`
- `Ph1_15s_Tminus3_L4_5m_vr_exact`
- `Ph1_15s_Tminus4_L5_15m_ldist_skew`
- `Ph1_5s_Tminus2_L1_1m_vol_velocity_1b`
- `Ph1_5m_Tminus1_L1_1D_vol_velocity_1b`
- `Ph1_15s_Tminus2_L2_1m_vol_accel_30`
- `Ph1_1s_Tminus4_L1_1D_body`
- `Ph1_1s_Tminus3_L1_5m_price_velocity_1b`
- `Ph1_5s_Tminus3_L5_4h_ldist_q3`
- `Ph1_1s_Tminus5_L3_5m_SE_low_9`
- `Ph1_1s_Tminus2_L5_1h_ldist_q1`
- `Ph1_5m_Tminus1_L4_1m_lambda_t_30`
- `Ph1_5s_Tminus1_L3_1D_band_pos_30`
- `Ph1_1s_Tminus3_L2_4h_vol_velocity_18`