# Logistic Regression DOE: ADX-08_Trend_Gate
**Total Events:** 970
**Base Rate:** 0.5031
**ROC AUC:** 0.4808
**Log Loss:** 0.7074

## Magnitude Weighted Evaluation
> Weights applied during fit based on MFE (wins) and MAE (losses).
> **OOS Guard:** Probabilities generated via Stratified 5-Fold Cross-Validation.

| Tier | N | Mean Post. | Actual WR | Base Delta | Mean MFE | Mean MAE |
|---|---|---|---|---|---|---|
| (0.273, 0.416] | 97 | 0.3854 | 0.5876 | +8.45 pp | 104.38 | -102.77 |
| (0.416, 0.445] | 97 | 0.4299 | 0.5052 | +0.21 pp | 107.46 | -91.31 |
| (0.445, 0.468] | 97 | 0.4573 | 0.4639 | -3.92 pp | 97.83 | -93.41 |
| (0.468, 0.486] | 97 | 0.4770 | 0.5361 | +3.30 pp | 89.06 | -99.50 |
| (0.486, 0.503] | 97 | 0.4944 | 0.5361 | +3.30 pp | 104.97 | -76.03 |
| (0.503, 0.522] | 97 | 0.5132 | 0.3814 | -12.16 pp | 82.51 | -97.35 |
| (0.522, 0.539] | 97 | 0.5299 | 0.5464 | +4.33 pp | 97.24 | -91.26 |
| (0.539, 0.56] | 97 | 0.5485 | 0.5052 | +0.21 pp | 81.52 | -70.16 |
| (0.56, 0.59] | 97 | 0.5753 | 0.5361 | +3.30 pp | 82.47 | -104.14 |
| (0.59, 0.685] | 97 | 0.6197 | 0.4330 | -7.01 pp | 101.29 | -105.47 |
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 967
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.0467
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_1s_Tminus2_L2_1m_price_velocity_15`
- `Ph1_1s_Tminus5_L5_15s_ldist_skew`
- `Ph1_5m_Tminus1_L2_1m_vol_accel_30`
- `Ph1_5m_Tminus1_L1_1m_vol_velocity_1b`
- `Ph1_15s_Tminus2_L1_15s_price_accel_1b`
- `Ph1_15s_Tminus2_L1_5s_lower_wick`
- `Ph1_15s_Tminus2_L5_5s_ldist_kurtosis`
- `Ph1_5s_Tminus2_L4_5s_lambda_t_12`
- `Ph1_5s_Tminus3_L1_1m_upper_wick`
- `Ph1_15s_Tminus3_L1_5s_lower_wick`
- `Ph1_5s_Tminus1_L4_5s_lambda_hat_12`
- `Ph1_5m_Tminus1_L3_5s_reversion_prob_30`
- `Ph1_15s_Tminus4_L1_5s_vol_velocity_1b`
- `Ph1_5s_Tminus3_L5_5s_ldist_std`
- `Ph1_1s_Tminus1_L1_15s_bar_range`
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 967
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.0467
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_1s_Tminus1_L2_1m_price_velocity_15`
- `Ph1_1s_Tminus5_L5_15s_ldist_skew`
- `Ph1_5m_Tminus1_L2_1m_vol_accel_30`
- `Ph1_5m_Tminus1_L1_1m_vol_velocity_1b`
- `Ph1_15s_Tminus2_L1_15s_price_accel_1b`
- `Ph1_15s_Tminus2_L1_5s_lower_wick`
- `Ph1_15s_Tminus2_L5_5s_ldist_kurtosis`
- `Ph1_5s_Tminus2_L4_5s_lambda_t_12`
- `Ph1_5s_Tminus3_L1_1m_upper_wick`
- `Ph1_15s_Tminus3_L1_5s_lower_wick`
- `Ph1_5s_Tminus1_L4_5s_lambda_hat_12`
- `Ph1_5m_Tminus1_L3_5s_reversion_prob_30`
- `Ph1_15s_Tminus4_L1_5s_vol_velocity_1b`
- `Ph1_5s_Tminus3_L5_5s_ldist_std`
- `Ph1_1s_Tminus1_L1_15s_bar_range`
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 967
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.0467
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_1s_Tminus2_L2_1m_price_velocity_15`
- `Ph1_1s_Tminus5_L5_15s_ldist_skew`
- `Ph1_5m_Tminus1_L2_1m_vol_accel_30`
- `Ph1_5m_Tminus1_L1_1m_vol_velocity_1b`
- `Ph1_15s_Tminus2_L1_15s_price_accel_1b`
- `Ph1_15s_Tminus2_L1_5s_lower_wick`
- `Ph1_15s_Tminus2_L5_5s_ldist_kurtosis`
- `Ph1_5s_Tminus2_L4_5s_lambda_t_12`
- `Ph1_5s_Tminus3_L1_1m_upper_wick`
- `Ph1_15s_Tminus3_L1_5s_lower_wick`
- `Ph1_5s_Tminus1_L4_5s_lambda_hat_12`
- `Ph1_5m_Tminus1_L3_5s_reversion_prob_30`
- `Ph1_15s_Tminus4_L1_5s_vol_velocity_1b`
- `Ph1_5s_Tminus3_L5_5s_ldist_std`
- `Ph1_5s_Tminus1_L1_15s_bar_range`
## ML Feature Extraction & Selection
- **Target:** Binary 'Hit' (Win Rate)
- **Total Samples:** 967
- **Total Dimensionality Explored:** 4644 (Fractal Slice)
- **Pseudo R-Squared (McFadden):** 0.0467
- **Compute Engine:** PyTorch CUDA

### Top Selected Features (Stepwise Forward Elimination)
- `Ph1_1s_Tminus2_L2_1m_price_velocity_15`
- `Ph1_1s_Tminus5_L5_15s_ldist_skew`
- `Ph1_5m_Tminus1_L2_1m_vol_accel_30`
- `Ph1_5m_Tminus1_L1_1m_vol_velocity_1b`
- `Ph1_15s_Tminus2_L1_15s_price_accel_1b`
- `Ph1_15s_Tminus2_L1_5s_lower_wick`
- `Ph1_15s_Tminus2_L5_5s_ldist_kurtosis`
- `Ph1_5s_Tminus2_L4_5s_lambda_t_12`
- `Ph1_5s_Tminus3_L1_1m_upper_wick`
- `Ph1_15s_Tminus3_L1_5s_lower_wick`
- `Ph1_5s_Tminus1_L4_5s_lambda_hat_12`
- `Ph1_5m_Tminus1_L3_5s_reversion_prob_30`
- `Ph1_15s_Tminus4_L1_5s_vol_velocity_1b`
- `Ph1_5s_Tminus3_L5_5s_ldist_std`
- `Ph1_1s_Tminus1_L1_15s_bar_range`