# Logistic Regression DOE: ATR-09_Statistical_Fade
**Total Events:** 151
**Base Rate:** 0.1258
**ROC AUC:** 0.6619
**Log Loss:** 0.5783

## Magnitude Weighted Evaluation
> Weights applied during fit based on MFE (wins) and MAE (losses).
> **OOS Guard:** Probabilities generated via Stratified 5-Fold Cross-Validation.

| Tier | N | Mean Post. | Actual WR | Base Delta | Mean MFE | Mean MAE |
|---|---|---|---|---|---|---|
| (0.014499999999999999, 0.0813] | 16 | 0.0522 | 0.0625 | -6.33 pp | 25.20 | -12.84 |
| (0.0813, 0.137] | 15 | 0.1109 | 0.0667 | -5.92 pp | 19.65 | -11.82 |
| (0.137, 0.193] | 15 | 0.1611 | 0.0667 | -5.92 pp | 8.33 | -11.33 |
| (0.193, 0.266] | 15 | 0.2190 | 0.0667 | -5.92 pp | 19.00 | -12.10 |
| (0.266, 0.344] | 15 | 0.2995 | 0.1333 | +0.75 pp | 17.13 | -11.30 |
| (0.344, 0.405] | 15 | 0.3761 | 0.0667 | -5.92 pp | 12.88 | -10.95 |
| (0.405, 0.465] | 15 | 0.4304 | 0.1333 | +0.75 pp | 18.32 | -11.60 |
| (0.465, 0.526] | 15 | 0.4954 | 0.2000 | +7.42 pp | 21.62 | -11.23 |
| (0.526, 0.699] | 15 | 0.6214 | 0.2667 | +14.08 pp | 24.55 | -10.03 |
| (0.699, 0.983] | 15 | 0.8319 | 0.2000 | +7.42 pp | 24.05 | -9.67 |
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 164
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 1.0000
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_5m_Tminus1_L2_5m_price_sigma_30`
- `Ph1_5m_Tminus1_L5_5m_ldist_kurtosis`
- `Ph1_15s_Tminus2_L3_5s_band_pos_30`
- `Ph1_1s_Tminus1_L3_1m_reversion_prob_15`
- `Ph1_15s_Tminus2_L4_5s_z_21`
- `Ph1_1s_Tminus2_L1_1h_lower_wick`
- `Ph1_15s_Tminus3_L1_15s_upper_wick`
- `Ph1_5s_Tminus2_L5_15s_ldist_outlier_pct`
- `Ph1_1s_Tminus3_L1_1D_vol_accel_1b`
- `Ph1_1s_Tminus2_L3_5s_z_se_9`
- `Ph1_15s_Tminus2_L5_5m_ldist_outlier_pct`
- `Ph1_15s_Tminus2_L3_1D_reversion_prob_30`
- `Ph1_15s_Tminus3_L3_1D_reversion_prob_30`
- `Ph1_5m_Tminus1_L4_15m_lambda_t_21`
- `Ph1_15s_Tminus3_L4_5m_vr_exact`