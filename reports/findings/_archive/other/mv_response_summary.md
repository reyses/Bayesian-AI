# Multivariate Response Surface — Summary

| Tier | N | WR | Avg $ | 2D cells | Tree rules |
|---|---:|---:|---:|---:|---:|
| FADE_CALM | 24,039 | 48.9% | $-0.23 | 8 | 2 |
| RIDE_AGAINST | 39,721 | 48.0% | $-0.10 | 0 | 3 |
| KILL_SHOT | 4,411 | 51.5% | $-0.14 | 45 | 5 |
| CASCADE | 1,270 | 51.1% | $+1.34 | 141 | 4 |
| FADE_AGAINST | 4,532 | 49.4% | $+0.32 | 44 | 5 |
| MTF_BREAKOUT | 5,961 | 45.1% | $+0.23 | 41 | 3 |
| MTF_EXHAUSTION | 233 | 48.9% | $+5.90 | 1 | 2 |

## Strongest candidate gate rules across tiers

Pulled from the top interaction cells and tree rules. ΔWR is measured against the tier's own baseline.

| Tier | Rule | N | WR | ΔWR | avg $ |
|---|---|---:|---:|---:|---:|
| KILL_SHOT | 1m_dir_vol <= -0.92 AND 5m_dmi_gap > 9.36 AND 1h_variance_ratio <= 0.28 | 97 | 80.4% | +28.9 | — |
| FADE_AGAINST | 15s_dmi_gap <= 0.89 AND 15m_bar_range <= 145.50 AND 1h_dmi_diff <= -4.23 | 52 | 17.3% | -32.1 | — |
| MTF_BREAKOUT | 5m_reversion_prob > 0.95 AND 5m_z_high <= 1.39 | 84 | 71.4% | +26.3 | — |
| KILL_SHOT | 1m_dir_vol > -0.92 AND 15s_z_low <= -1.11 AND 1m_p_at_center > 0.79 | 94 | 28.7% | -22.8 | — |
| FADE_AGAINST | 15s_dmi_gap > 0.89 AND 15s_variance_ratio > 1.20 | 59 | 74.6% | +25.1 | — |
| KILL_SHOT | 1m_dir_vol > -0.92 AND 15s_z_low > -1.11 AND 5m_p_at_center > 0.79 | 64 | 73.4% | +21.9 | — |
| RIDE_AGAINST | 1m_reversion_prob <= 0.99 AND 1m_dmi_gap <= 8.19 AND 1m_z_high <= -0.04 | 309 | 32.4% | -15.7 | — |
| MTF_EXHAUSTION | 15m_hurst <= 0.70 AND 1h_z_se <= 0.03 | 57 | 70.2% | +21.2 | — |
| CASCADE | 1h_hurst > 0.58 AND 1m_bar_range > 56.50 AND 1m_z_high > 0.43 | 176 | 67.6% | +16.5 | — |
| MTF_EXHAUSTION | 15m_hurst > 0.70 AND 1D_dmi_diff <= -2.76 | 65 | 29.2% | -19.7 | — |
| CASCADE | 1h_hurst > 0.58 AND 1m_bar_range <= 56.50 AND 5m_dmi_gap > 20.43 | 80 | 32.5% | -18.6 | — |
| MTF_BREAKOUT | 5m_reversion_prob <= 0.95 AND 1h_z_high <= 0.64 AND 1h_reversion_prob > 1.00 | 172 | 60.5% | +15.4 | — |
| CASCADE | 1h_hurst <= 0.58 | 77 | 33.8% | -17.3 | — |
| CASCADE | 5m_dmi_gap in Q4 AND 15m_p_at_center in Q3 | 79 | 34.2% | -16.9 | $-7.73 |
| KILL_SHOT | 1m_dir_vol <= -0.92 AND 5m_dmi_gap <= 9.36 AND 1h_reversion_prob <= 0.95 | 62 | 69.4% | +17.8 | — |
| FADE_AGAINST | 15s_dmi_gap <= 0.89 AND 15m_bar_range > 145.50 AND 1h_variance_ratio > 0.57 | 53 | 67.9% | +18.5 | — |
| CASCADE | 1h_variance_ratio in Q4 AND 15m_p_at_center in Q3 | 76 | 34.2% | -16.9 | $-12.63 |
| CASCADE | 1h_dmi_gap in Q1 AND 1m_z_high in Q3 | 80 | 67.5% | +16.4 | $+10.71 |
| CASCADE | 1D_vol_rel in Q2 AND 1h_variance_ratio in Q3 | 71 | 67.6% | +16.5 | $+3.58 |
| CASCADE | 5m_dmi_gap in Q1 AND 1m_z_high in Q2 | 81 | 66.7% | +15.6 | $+10.15 |
| CASCADE | 1D_vol_rel in Q1 AND 1m_z_high in Q4 | 88 | 36.4% | -14.7 | $-4.28 |
| CASCADE | 1h_variance_ratio in Q4 AND 1m_z_high in Q1 | 65 | 35.4% | -15.7 | $+1.90 |
| FADE_AGAINST | 15s_dmi_gap <= 0.89 AND 15m_bar_range <= 145.50 AND 1h_dmi_diff > -4.23 | 113 | 36.3% | -13.1 | — |
| CASCADE | 1h_dmi_gap in Q1 AND 15s_hurst in Q3 | 75 | 65.3% | +14.2 | $+5.83 |
| CASCADE | 1m_z_high in Q3 AND 15m_p_at_center in Q1 | 80 | 65.0% | +13.9 | $+7.38 |
| CASCADE | 1D_vol_rel in Q2 AND 15s_hurst in Q3 | 92 | 64.1% | +13.0 | $+9.01 |
| CASCADE | 1h_dmi_gap in Q1 AND 15m_p_at_center in Q2 | 97 | 63.9% | +12.8 | $+8.82 |
| CASCADE | 1h_dmi_gap in Q3 AND 1h_variance_ratio in Q4 | 77 | 37.7% | -13.4 | $-0.02 |
| KILL_SHOT | 5m_dmi_gap in Q3 AND 5m_reversion_prob in Q4 | 249 | 41.0% | -10.5 | $-3.38 |
| CASCADE | 1m_z_high in Q4 AND 15m_p_at_center in Q2 | 73 | 64.4% | +13.3 | $+11.73 |
| RIDE_AGAINST | 1m_reversion_prob > 0.99 AND 1m_dmi_gap > 15.12 AND 5m_dmi_gap > 15.31 | 3,315 | 55.1% | +7.0 | — |
| CASCADE | 1D_vol_rel in Q4 AND 1h_variance_ratio in Q2 | 75 | 64.0% | +12.9 | $+13.19 |
| CASCADE | 5m_dmi_gap in Q4 AND 1m_z_high in Q1 | 66 | 37.9% | -13.2 | $-5.55 |
| FADE_AGAINST | 15s_dmi_gap <= 0.89 AND 15m_bar_range > 145.50 AND 1h_variance_ratio <= 0.57 | 94 | 37.2% | -12.2 | — |
| CASCADE | 15s_hurst in Q3 AND 1m_z_high in Q4 | 82 | 63.4% | +12.3 | $+14.11 |
| RIDE_AGAINST | 1m_reversion_prob > 0.99 AND 1m_dmi_gap <= 15.12 AND 1h_dmi_diff <= -30.48 | 614 | 56.4% | +8.3 | — |
| CASCADE | 15s_hurst in Q3 AND 15m_p_at_center in Q2 | 89 | 62.9% | +11.8 | $+12.09 |
| FADE_CALM | 1m_z_se > 0.09 AND 5m_dir_vol > -0.71 AND 1m_z_high <= 0.86 | 2,998 | 42.3% | -6.6 | — |
| CASCADE | 1D_vol_rel in Q1 AND 1h_variance_ratio in Q4 | 91 | 39.6% | -11.5 | $-2.36 |
| MTF_BREAKOUT | 5m_hurst in Q1 AND 1h_dmi_diff in Q1 | 373 | 53.9% | +8.8 | $+4.73 |