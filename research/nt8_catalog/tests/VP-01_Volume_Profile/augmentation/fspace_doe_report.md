# Logistic Regression DOE: VP-01_Volume_Profile
**Total Events:** 233
**Base Rate:** 0.2747
**ROC AUC:** 0.5364
**Log Loss:** 0.6260

## Magnitude Weighted Evaluation
> Weights applied during fit based on MFE (wins) and MAE (losses).
> **OOS Guard:** Probabilities generated via Stratified 5-Fold Cross-Validation.

| Tier | N | Mean Post. | Actual WR | Base Delta | Mean MFE | Mean MAE |
|---|---|---|---|---|---|---|
| (0.174, 0.283] | 24 | 0.2491 | 0.3333 | +5.87 pp | 11.76 | -8.52 |
| (0.283, 0.313] | 23 | 0.3009 | 0.3043 | +2.97 pp | 6.14 | -9.97 |
| (0.313, 0.346] | 23 | 0.3339 | 0.2174 | -5.73 pp | 7.10 | -9.91 |
| (0.346, 0.374] | 23 | 0.3592 | 0.2174 | -5.73 pp | 8.40 | -10.68 |
| (0.374, 0.395] | 24 | 0.3841 | 0.2083 | -6.63 pp | 7.83 | -9.32 |
| (0.395, 0.411] | 23 | 0.4023 | 0.2609 | -1.38 pp | 5.42 | -9.43 |
| (0.411, 0.439] | 23 | 0.4263 | 0.1739 | -10.08 pp | 3.66 | -11.40 |
| (0.439, 0.484] | 23 | 0.4636 | 0.1739 | -10.08 pp | 4.34 | -11.15 |
| (0.484, 0.526] | 23 | 0.5025 | 0.3913 | +11.66 pp | 15.21 | -8.15 |
| (0.526, 0.714] | 24 | 0.5804 | 0.4583 | +18.37 pp | 15.03 | -7.56 |
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 263
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.4349
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_5s_Tminus2_L1_5s_price_accel_1b`
- `Ph1_1s_Tminus1_L3_1h_z_high_12`
- `Ph1_5m_Tminus1_L4_15s_lambda_hat_30`
- `Ph1_1s_Tminus1_L2_1D_vol_velocity_5`
- `Ph1_5s_Tminus1_L3_5s_reversion_prob_30`
- `Ph1_5s_Tminus1_L4_1D_vr_exact`
- `Ph1_15s_Tminus2_L2_1m_vol_accel_30`
- `Ph1_5m_Tminus1_L3_15s_z_high_30`
- `Ph1_1s_Tminus1_L5_5m_ldist_outlier_pct`
- `Ph1_15s_Tminus3_L5_1m_ldist_kurtosis`
- `Ph1_1s_Tminus1_L1_15m_lower_wick`
- `Ph1_5s_Tminus1_L3_1D_hurst_30`
- `Ph1_15s_Tminus3_L4_15s_lambda_t_12`
- `Ph1_15s_Tminus3_L1_15s_body`
- `Ph1_15s_Tminus3_L3_15s_reversion_prob_30`