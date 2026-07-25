# Wrong-direction detection — first 10 minutes (screening)
156 episodes, 25 days; base P(wrong)=37%.
BASELINE TO BEAT: early_neg (px<0 at minute 10).

| signal | n on | P(wrong|on) | lift | 95% CI | sig |
|---|---|---|---|---|---|
| early_neg | 33 | 67% | +30% | [+21%, +41%] | YES |
| d_lambda_se_21 | 28 | 46% | +10% | [-7%, +29%] |  |
| d_vol_velocity_30 | 29 | 45% | +8% | [-6%, +25%] |  |
| deep_adverse | 63 | 43% | +6% | [-1%, +14%] |  |
| faded | 56 | 39% | +3% | [-6%, +16%] |  |
| sick2 | 54 | 39% | +2% | [-9%, +15%] |  |
| conv_neg | 82 | 38% | +1% | [-7%, +11%] |  |
| d_price_velocity_30 | 38 | 37% | +0% | [-16%, +19%] |  |
| d_swing_noise_30 | 49 | 35% | -2% | [-14%, +10%] |  |
| d_price_accel_1b | 31 | 32% | -4% | [-17%, +10%] |  |
| d_ldist_std | 15 | 13% | -23% | [-40%, -5%] |  |
