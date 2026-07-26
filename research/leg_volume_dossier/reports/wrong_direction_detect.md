# Wrong-direction detection — first 10 minutes (screening)
156 episodes, 25 days; base P(wrong)=37%.
BASELINE TO BEAT: early_neg (px<0 at minute 10).

| signal | n on | P(wrong|on) | lift | 95% CI | sig |
|---|---|---|---|---|---|
| no_recovery | 15 | 73% | +37% | [+15%, +59%] | YES |
| mostly_under | 19 | 68% | +32% | [+16%, +46%] | YES |
| early_neg | 33 | 67% | +30% | [+21%, +41%] | YES |
| low_ER | 53 | 53% | +16% | [+9%, +25%] | YES |
| d_lambda_se_21 | 28 | 46% | +10% | [-6%, +29%] |  |
| d_vol_velocity_30 | 29 | 45% | +8% | [-6%, +25%] |  |
| deep_adverse | 63 | 43% | +6% | [-1%, +14%] |  |
| faded | 56 | 39% | +3% | [-7%, +16%] |  |
| sick2 | 54 | 39% | +2% | [-9%, +16%] |  |
| conv_neg | 82 | 38% | +1% | [-7%, +11%] |  |
| d_price_velocity_30 | 38 | 37% | +0% | [-16%, +18%] |  |
| fires_against | 156 | 37% | +0% | [+0%, +0%] |  |
| fires_with | 155 | 36% | -0% | [-1%, +0%] |  |
| d_swing_noise_30 | 49 | 35% | -2% | [-14%, +10%] |  |
| d_price_accel_1b | 31 | 32% | -4% | [-18%, +10%] |  |
| d_ldist_std | 15 | 13% | -23% | [-39%, -5%] |  |

COMPOSITE tune/holdout: threshold>=2 (tuned on 6 days) -> HOLDOUT P(wrong|on)=73% vs base 43%, lift +31%, 95% CI [+16%, +47%] over 15 fired / 103 eps -> GENERALIZES
