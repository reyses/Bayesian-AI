# v1.0.4 Phase 2 EDA — feature-level regression

Generated: 2026-04-27 20:15

Trades: `examples/trades v1.0.4 playback.csv`

Features: `DATA/ATLAS_NT8/FEATURES_5s_v2`

window_bars: 5  (0 = entry only; >0 includes feature deltas at +1m..+Nm after entry)

Joined trades: 497 (winners: 187, losers: 310)

Features in design matrix: 906

Base WR: 37.6%


## Top 30 features by |Cohen-d| (winners vs losers at entry)

| Rank | Feature | Cohen-d | mean(W) | mean(L) | n(W) | n(L) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `L3_5m_reversion_prob_9__d2m` | -0.341 | -0.0043 | +0.0030 | 187 | 310 |
| 2 | `L3_5m_hurst_9__d5m` | +0.279 | +0.0120 | -0.0086 | 187 | 310 |
| 3 | `L3_5m_reversion_prob_9__d3m` | -0.257 | -0.0040 | +0.0027 | 187 | 310 |
| 4 | `L3_1m_swing_noise_15__d5m` | +0.246 | +40.4278 | +11.8968 | 187 | 310 |
| 5 | `L1_5s_price_velocity_1b__d4m` | +0.237 | +0.9131 | -0.9734 | 187 | 310 |
| 6 | `L3_5m_hurst_9__d4m` | +0.236 | +0.0102 | -0.0057 | 187 | 310 |
| 7 | `L2_1m_vol_sigma_15__d5m` | +0.232 | +254.7290 | +78.9122 | 187 | 310 |
| 8 | `L2_5s_price_accel_9__d3m` | +0.227 | +0.1139 | -0.1441 | 187 | 310 |
| 9 | `L2_15m_price_velocity_12` | -0.223 | +0.8189 | +4.0928 | 187 | 310 |
| 10 | `L2_5s_price_accel_9__d5m` | +0.222 | +0.1389 | -0.1184 | 187 | 310 |
| 11 | `L1_5s_body__d4m` | +0.222 | +0.7981 | -0.9710 | 187 | 310 |
| 12 | `L3_1m_swing_noise_15__d4m` | +0.220 | +36.2941 | +13.3968 | 187 | 310 |
| 13 | `L2_1m_price_accel_15__d2m` | -0.214 | -0.2749 | +0.2498 | 187 | 310 |
| 14 | `L3_5m_reversion_prob_9__d4m` | -0.211 | -0.0049 | +0.0019 | 187 | 310 |
| 15 | `L3_5m_reversion_prob_9__d5m` | -0.209 | -0.0073 | +0.0006 | 187 | 310 |
| 16 | `L2_1m_price_accel_15__d3m` | -0.206 | -0.3500 | +0.0501 | 187 | 310 |
| 17 | `L1_5s_price_velocity_1b__d3m` | +0.204 | +0.8610 | -0.7435 | 187 | 310 |
| 18 | `L1_5s_body__d3m` | +0.204 | +0.7741 | -0.8282 | 187 | 310 |
| 19 | `L3_5s_z_se_9__d4m` | +0.201 | +0.0758 | -0.1760 | 187 | 310 |
| 20 | `L2_1m_vol_accel_15` | +0.196 | +42.9807 | +11.0811 | 187 | 310 |
| 21 | `L2_15s_vol_sigma_12__d2m` | -0.195 | -22.1999 | +51.1399 | 187 | 310 |
| 22 | `L2_15m_vol_sigma_12` | -0.194 | +14137.8086 | +16771.8066 | 187 | 310 |
| 23 | `L3_5s_z_low_9__d4m` | +0.193 | +0.0907 | -0.1566 | 187 | 310 |
| 24 | `L3_15m_SE_low_12` | -0.192 | +33.6108 | +39.7052 | 187 | 310 |
| 25 | `L3_1m_swing_noise_15__d3m` | +0.192 | +32.2941 | +15.8097 | 187 | 310 |
| 26 | `L3_1h_SE_low_12__d2m` | +0.190 | +0.2228 | -0.0616 | 187 | 310 |
| 27 | `L1_5m_bar_range__d2m` | +0.189 | +6.7807 | -2.7782 | 187 | 310 |
| 28 | `L1_5m_bar_range__d3m` | +0.187 | +7.5214 | -2.1008 | 187 | 310 |
| 29 | `L1_1m_vol_velocity_1b__d2m` | -0.186 | -1076.0267 | -539.7774 | 187 | 310 |
| 30 | `L1_5s_price_accel_1b__d4m` | +0.186 | +1.6658 | -0.4081 | 187 | 310 |

## Lasso regression on PnL — feature selection

Best alpha (5-fold CV): 12.5979

Non-zero coefficients: 13

| Rank | Feature | Coefficient |
|---:|---|---:|
| 1 | `L3_1m_swing_noise_15__d5m` | +11.6186 |
| 2 | `L3_1m_z_se_15__d1m` | +6.0045 |
| 3 | `L2_5s_price_accel_9__d5m` | +5.8968 |
| 4 | `L3_5m_hurst_9__d5m` | +4.5523 |
| 5 | `L1_15s_price_accel_1b` | -4.4989 |
| 6 | `L2_15s_vol_sigma_12__d3m` | -3.9259 |
| 7 | `L3_5s_SE_low_9__d1m` | -3.2351 |
| 8 | `L3_5s_SE_low_9__d2m` | -3.2257 |
| 9 | `L1_5s_bar_range__d4m` | +3.0187 |
| 10 | `L1_15s_price_accel_1b__d5m` | +2.5751 |
| 11 | `L3_15s_SE_high_12__d4m` | -2.0161 |
| 12 | `L3_15s_z_high_12__d4m` | -0.9818 |
| 13 | `L3_1h_reversion_prob_12__d3m` | +0.6665 |

## Logistic regression — top 15 features (linear)

5-fold CV AUC: **0.640 ± 0.045**
(0.5 = no signal; >0.6 = weak; >0.7 = decent; >0.8 = strong)

Coefficients (refit on full data, standardized):

| Rank | Feature | Coefficient |
|---:|---|---:|
| 5 | `L1_5s_price_velocity_1b__d4m` | +0.8938 |
| 11 | `L1_5s_body__d4m` | -0.5278 |
| 4 | `L3_1m_swing_noise_15__d5m` | +0.4656 |
| 1 | `L3_5m_reversion_prob_9__d2m` | -0.3988 |
| 2 | `L3_5m_hurst_9__d5m` | +0.3513 |
| 13 | `L2_1m_price_accel_15__d2m` | -0.3257 |
| 9 | `L2_15m_price_velocity_12` | -0.2194 |
| 12 | `L3_1m_swing_noise_15__d4m` | -0.2106 |
| 7 | `L2_1m_vol_sigma_15__d5m` | +0.1374 |
| 8 | `L2_5s_price_accel_9__d3m` | +0.1368 |
| 15 | `L3_5m_reversion_prob_9__d5m` | -0.0807 |
| 3 | `L3_5m_reversion_prob_9__d3m` | +0.0687 |
| 6 | `L3_5m_hurst_9__d4m` | -0.0544 |
| 10 | `L2_5s_price_accel_9__d5m` | -0.0222 |
| 14 | `L3_5m_reversion_prob_9__d4m` | +0.0046 |

## Stepwise forward selection (win/loss prediction)

Candidate pool: top 50 features by |Cohen-d|.  Stop criterion: AUC improvement < 0.003.

| Step | Feature added | CV AUC | Δ AUC |
|---:|---|---:|---:|
| 1 | `L3_5m_hurst_9__d5m` | 0.581 | +0.0810 |
| 2 | `L3_5m_reversion_prob_9__d2m` | 0.614 | +0.0329 |
| 3 | `L3_5s_z_se_9__d4m` | 0.652 | +0.0384 |
| 4 | `L2_1m_price_accel_15__d2m` | 0.661 | +0.0085 |
| 5 | `L3_1m_swing_noise_15__d5m` | 0.669 | +0.0082 |
| 6 | `L2_15s_vol_sigma_12__d2m` | 0.682 | +0.0133 |
| 7 | `L1_5s_price_accel_1b__d4m` | 0.691 | +0.0082 |
| 8 | `L2_1m_vol_accel_15` | 0.695 | +0.0045 |

**Final selected: 8 features.** Final CV AUC: **0.695**

## Logistic regression — top 15 features × degree-2 polynomial

Total polynomial terms: 135
5-fold CV AUC: **0.532 ± 0.036**

Δ vs linear: **-0.108** — polynomial overfits.

Top 20 polynomial terms by |coefficient|:

| Rank | Term | Coefficient |
|---:|---|---:|
| 47 | `L3_5m_reversion_prob_9__d3m L1_5s_price_velocity_1b__d4m` | +0.9335 |
| 69 | `L3_1m_swing_noise_15__d5m L3_5m_reversion_prob_9__d5m` | -0.9275 |
| 99 | `L2_1m_vol_sigma_15__d5m L3_5m_reversion_prob_9__d5m` | +0.9208 |
| 5 | `L1_5s_price_velocity_1b__d4m` | +0.8865 |
| 100 | `L2_5s_price_accel_9__d3m^2` | -0.7985 |
| 19 | `L3_5m_reversion_prob_9__d2m L3_1m_swing_noise_15__d5m` | +0.7781 |
| 27 | `L3_5m_reversion_prob_9__d2m L3_1m_swing_noise_15__d4m` | -0.7696 |
| 98 | `L2_1m_vol_sigma_15__d5m L3_5m_reversion_prob_9__d4m` | -0.7347 |
| 88 | `L3_5m_hurst_9__d4m L2_1m_price_accel_15__d2m` | +0.6387 |
| 130 | `L2_1m_price_accel_15__d2m^2` | +0.5739 |
| 58 | `L3_1m_swing_noise_15__d5m^2` | +0.5557 |
| 84 | `L3_5m_hurst_9__d4m L2_15m_price_velocity_12` | +0.5525 |
| 13 | `L2_1m_price_accel_15__d2m` | -0.5518 |
| 62 | `L3_1m_swing_noise_15__d5m L2_5s_price_accel_9__d3m` | -0.5473 |
| 39 | `L3_5m_hurst_9__d5m L2_5s_price_accel_9__d5m` | -0.5215 |
| 111 | `L2_15m_price_velocity_12 L3_1m_swing_noise_15__d4m` | +0.5191 |
| 110 | `L2_15m_price_velocity_12 L1_5s_body__d4m` | -0.5154 |
| 112 | `L2_15m_price_velocity_12 L2_1m_price_accel_15__d2m` | +0.5149 |
| 20 | `L3_5m_reversion_prob_9__d2m L1_5s_price_velocity_1b__d4m` | +0.5078 |
| 8 | `L2_5s_price_accel_9__d3m` | +0.5011 |

## Honest summary

**MODEST SIGNAL.** AUC 0.60-0.70 = real but weak. Could support a weak filter.

Caveats:
- Only 506 trades in this Playback ledger. Underdetermined for 906-feature regression without regularization.
- Per playbook §9c: 'direction at entry is RANDOM on 91D'. We're predicting OUTCOME (win/loss) not direction — different question — but the small-N risk applies.
- Single 32-day window. Need OOS validation on a different window before trusting any selected feature.
- Polynomial expansion can overfit — compare CV AUC linear vs poly to detect.