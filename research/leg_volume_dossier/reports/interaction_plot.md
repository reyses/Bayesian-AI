# Detector interactions
detectors: ldist_std(lo), price_accel_1b(lo), vol_velocity_30(lo), lambda_se_21(hi), price_velocity_30(lo), swing_noise_30(hi)

## Anomaly-count curve (fwd 3-bar px, day-block 95% CI)
| active detectors | mean fwd px | CI | n |
|---|---|---|---|
| 0 | +5.05 | [+2.86, +7.38] | 3431 |
| 1 | +5.88 | [+1.83, +10.35] | 329 |
| 2 | -2.00 | [-7.42, +2.26] | 314 |
| 3+ | -4.47 | [-8.78, +0.81] | 423 |

## Pairwise synergy (both-on minus additive expectation, pts)
| pair | synergy |
|---|---|
| ldist_std × price_accel_1b | +11.15 |
| ldist_std × vol_velocity_30 | +2.01 |
| ldist_std × lambda_se_21 | +10.33 |
| ldist_std × price_velocity_30 | -3.20 |
| ldist_std × swing_noise_30 | +4.89 |
| price_accel_1b × vol_velocity_30 | +3.25 |
| price_accel_1b × lambda_se_21 | -7.04 |
| price_accel_1b × price_velocity_30 | +1.96 |
| price_accel_1b × swing_noise_30 | +5.42 |
| vol_velocity_30 × lambda_se_21 | -11.61 |
| vol_velocity_30 × price_velocity_30 | +6.02 |
| vol_velocity_30 × swing_noise_30 | -0.73 |
| lambda_se_21 × price_velocity_30 | +2.86 |
| lambda_se_21 × swing_noise_30 | -0.24 |
| price_velocity_30 × swing_noise_30 | +5.29 |
