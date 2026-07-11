# Logistic Regression DOE: PIVOT-16_Floor_Levels
**Total Events:** 261
**Base Rate:** 0.1303
**ROC AUC:** 0.5601
**Log Loss:** 0.7060

## Magnitude Weighted Evaluation
> Weights applied during fit based on MFE (wins) and MAE (losses).
> **OOS Guard:** Probabilities generated via Stratified 5-Fold Cross-Validation.

| Tier | N | Mean Post. | Actual WR | Base Delta | Mean MFE | Mean MAE |
|---|---|---|---|---|---|---|
| (0.168, 0.294] | 27 | 0.2434 | 0.0741 | -5.62 pp | -2.66 | -2.66 |
| (0.294, 0.359] | 26 | 0.3301 | 0.1538 | +2.36 pp | 12.14 | 12.14 |
| (0.359, 0.395] | 26 | 0.3782 | 0.1538 | +2.36 pp | -2.88 | -2.88 |
| (0.395, 0.431] | 26 | 0.4091 | 0.0769 | -5.33 pp | -8.51 | -8.51 |
| (0.431, 0.488] | 26 | 0.4550 | 0.0769 | -5.33 pp | -2.84 | -2.84 |
| (0.488, 0.517] | 26 | 0.5014 | 0.1154 | -1.49 pp | 7.43 | 7.43 |
| (0.517, 0.555] | 26 | 0.5355 | 0.1154 | -1.49 pp | -6.05 | -6.05 |
| (0.555, 0.626] | 26 | 0.5862 | 0.1538 | +2.36 pp | 4.68 | 4.68 |
| (0.626, 0.705] | 26 | 0.6684 | 0.1923 | +6.20 pp | 7.93 | 7.93 |
| (0.705, 0.867] | 26 | 0.7603 | 0.1923 | +6.20 pp | 1.95 | 1.95 |
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 260
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.7955
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_5m_Tminus1_L3_1m_SE_low_30`
- `Ph1_1s_Tminus3_L5_5s_ldist_kurtosis`
- `Ph1_5m_Tminus1_L3_1h_z_close_vs_high_30`
- `Ph1_5m_Tminus1_L3_5s_z_close_vs_high_30`
- `Ph1_1s_Tminus1_L3_15s_reversion_prob_12`
- `Ph1_15s_Tminus4_L2_15s_price_accel_30`
- `Ph1_15s_Tminus4_L4_5m_lambda_se_12`
- `Ph1_1s_Tminus3_L4_1h_lambda_se_12`
- `Ph1_5s_Tminus3_L4_15s_lambda_hat_21`
- `Ph1_15s_Tminus4_L3_15m_hurst_30`
- `Ph1_15s_Tminus3_L3_1m_SE_high_30`
- `Ph1_15s_Tminus4_L1_1m_lower_wick`
- `Ph1_1s_Tminus1_L2_1D_vol_accel_5`
- `Ph1_5m_Tminus1_L5_1m_ldist_kurtosis`
- `Ph1_5s_Tminus3_L1_1D_upper_wick`