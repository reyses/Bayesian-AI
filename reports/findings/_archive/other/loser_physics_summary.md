# Loser Physics — Summary

Each row is a candidate "loser regime" — a feature-threshold condition where losses concentrate in the current NMP engine. `net $ if flipped` estimates what the tier would add if we flipped direction specifically on the flippable losers in this regime.

| Mode | Regime | N | flippable losers | flip $/trade | net $ if flipped |
|---|---|---:|---:|---:|---:|
| NMP_FADE | 1m_variance_ratio > 0.4004 | 4,684 | 2,351 (100%) | $+84.70 | $+282,880 |
| NMP_FADE | 1m_z_high > 0.9452 | 4,684 | 2,324 (100%) | $+81.56 | $+271,434 |
| NMP_FADE | 1D_wick_ratio > 0.5360 | 4,646 | 2,286 (100%) | $+82.12 | $+267,926 |
| NMP_FADE | 1m_z_se > -2.0350 | 4,684 | 2,326 (100%) | $+80.26 | $+267,444 |
| NMP_FADE | 1m_z_low > -2.2125 | 4,684 | 2,324 (100%) | $+79.70 | $+264,830 |
| NMP_FADE | 15s_z_se > -0.5063 | 4,684 | 2,314 (100%) | $+79.30 | $+262,485 |
| NMP_FADE | 15s_dmi_diff > -6.2414 | 4,684 | 2,315 (100%) | $+79.17 | $+262,317 |
| NMP_FADE | 15s_z_low > -1.0282 | 4,684 | 2,330 (100%) | $+78.39 | $+261,288 |
| NMP_FADE | 15s_dir_vol > 0.0000 | 4,297 | 2,108 (100%) | $+83.26 | $+250,968 |
| NMP_FADE | 1h_vol_rel < 0.4459 | 4,684 | 2,302 (100%) | $+54.04 | $+176,106 |
| NMP_RIDE | 5m_variance_ratio > 0.2541 | 256 | 129 (100%) | $+84.68 | $+14,640 |
| NMP_RIDE | 15s_z_low < -0.9940 | 256 | 130 (100%) | $+85.58 | $+14,560 |
| NMP_RIDE | 15s_z_high < 0.5822 | 256 | 130 (100%) | $+86.13 | $+14,542 |
| NMP_RIDE | 15s_z_se < -0.4176 | 256 | 134 (100%) | $+82.70 | $+14,490 |
| NMP_RIDE | 1m_hurst < 0.6977 | 256 | 126 (100%) | $+78.60 | $+13,379 |
| NMP_RIDE | 15m_z_low < -0.7993 | 256 | 131 (100%) | $+76.86 | $+13,299 |
| NMP_RIDE | 15s_dmi_gap < 33.0783 | 256 | 127 (100%) | $+77.80 | $+12,894 |
| NMP_RIDE | 15m_dir_vol > 0.0694 | 256 | 131 (100%) | $+74.20 | $+12,853 |
| NMP_RIDE | 1m_wick_ratio < 0.2195 | 256 | 132 (100%) | $+75.31 | $+12,836 |
| NMP_RIDE | 1m_p_at_center < 0.0459 | 256 | 131 (100%) | $+69.43 | $+11,968 |