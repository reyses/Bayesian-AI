# Multivariate Response Surface — FADE_AGAINST

**Trades:** 4,532  **WR:** 49.4%  **Avg $/trade:** $+0.32

## Top univariate features (Cohen d)

| Feature | Cohen d | p | Win mean | Loss mean |
|---|---:|---:|---:|---:|
| 1m_p_at_center | -0.066 | 0.0265 | 0.6400 | 0.6517 |
| 1m_reversion_prob | -0.065 | 0.0279 | 0.9734 | 0.9770 |
| 15m_reversion_prob | +0.065 | 0.0275 | 0.9419 | 0.9334 |
| 1m_wick_ratio | -0.065 | 0.0300 | 0.5328 | 0.5494 |
| 5m_velocity | -0.060 | 0.0441 | 0.0477 | 0.3507 |
| 15s_vol_rel | +0.058 | 0.0499 | 0.9194 | 0.8725 |
| 15m_p_at_center | +0.055 | 0.0631 | 0.5138 | 0.5006 |
| 15m_z_low | +0.051 | 0.0839 | -0.6780 | -0.7382 |

## Top GBM importance

| Feature | Importance |
|---|---:|
| 15s_variance_ratio | 0.0428 |
| 15s_hurst | 0.0346 |
| 15m_z_high | 0.0237 |
| 15s_dmi_gap | 0.0227 |
| 15m_hurst | 0.0226 |
| 1m_z_high | 0.0218 |
| 5m_dmi_gap | 0.0216 |
| 15s_z_low | 0.0212 |

## 2D interaction cells (|ΔWR| ≥ 5pp, N ≥ 30)

Baseline WR = 49.4%. Cells listed with higher WR first; negative diverge_pp = cell is WORSE than baseline.

| feat_a | qa | feat_b | qb | N | WR | Δpp | avg $ |
|---|---|---|---|---:|---:|---:|---:|
| 15m_reversion_prob | Q4 | 15m_z_low | Q1 | 90 | 58.9% | +9.5 | $+7.99 |
| 15m_p_at_center | Q4 | 15m_z_low | Q1 | 90 | 58.9% | +9.5 | $+7.99 |
| 5m_velocity | Q1 | 15m_z_low | Q3 | 274 | 58.4% | +9.0 | $+4.36 |
| 1m_wick_ratio | Q1 | 15s_vol_rel | Q1 | 204 | 57.8% | +8.4 | $+1.20 |
| 15m_reversion_prob | Q3 | 5m_velocity | Q1 | 266 | 57.5% | +8.1 | $+3.74 |
| 5m_velocity | Q1 | 15m_p_at_center | Q3 | 266 | 57.5% | +8.1 | $+3.74 |
| 1m_p_at_center | Q1 | 15s_vol_rel | Q2 | 260 | 57.3% | +7.9 | $+2.67 |
| 1m_reversion_prob | Q1 | 15s_vol_rel | Q2 | 260 | 57.3% | +7.9 | $+2.67 |
| 15m_reversion_prob | Q3 | 15m_z_low | Q3 | 294 | 57.1% | +7.7 | $+5.57 |
| 15m_p_at_center | Q3 | 15m_z_low | Q3 | 294 | 57.1% | +7.7 | $+5.57 |
| 5m_velocity | Q1 | 15s_vol_rel | Q4 | 251 | 56.6% | +7.1 | $+4.31 |
| 1m_wick_ratio | Q4 | 5m_velocity | Q2 | 319 | 42.3% | -7.1 | $-2.77 |
| 1m_p_at_center | Q2 | 15m_reversion_prob | Q3 | 303 | 56.4% | +7.0 | $+7.91 |
| 1m_p_at_center | Q2 | 15m_p_at_center | Q3 | 303 | 56.4% | +7.0 | $+7.91 |
| 1m_reversion_prob | Q2 | 15m_reversion_prob | Q3 | 303 | 56.4% | +7.0 | $+7.91 |
| 1m_reversion_prob | Q2 | 15m_p_at_center | Q3 | 303 | 56.4% | +7.0 | $+7.91 |
| 5m_velocity | Q1 | 15s_vol_rel | Q2 | 297 | 56.2% | +6.8 | $+9.11 |
| 1m_p_at_center | Q3 | 15m_z_low | Q2 | 332 | 42.8% | -6.7 | $-6.29 |
| 1m_reversion_prob | Q3 | 15m_z_low | Q2 | 332 | 42.8% | -6.7 | $-6.29 |
| 15s_vol_rel | Q4 | 15m_z_low | Q2 | 282 | 42.9% | -6.5 | $-5.27 |
| 1m_p_at_center | Q2 | 15m_reversion_prob | Q1 | 268 | 42.9% | -6.5 | $-1.85 |
| 1m_p_at_center | Q2 | 15m_p_at_center | Q1 | 268 | 42.9% | -6.5 | $-1.85 |
| 1m_reversion_prob | Q2 | 15m_reversion_prob | Q1 | 268 | 42.9% | -6.5 | $-1.85 |
| 1m_reversion_prob | Q2 | 15m_p_at_center | Q1 | 268 | 42.9% | -6.5 | $-1.85 |
| 1m_p_at_center | Q3 | 15m_z_low | Q3 | 269 | 55.8% | +6.3 | $+3.30 |
| 1m_reversion_prob | Q3 | 15m_z_low | Q3 | 269 | 55.8% | +6.3 | $+3.30 |
| 15s_vol_rel | Q2 | 15m_z_low | Q3 | 287 | 55.7% | +6.3 | $+0.30 |
| 1m_wick_ratio | Q3 | 5m_velocity | Q4 | 294 | 43.2% | -6.2 | $-5.13 |
| 1m_wick_ratio | Q4 | 15m_z_low | Q2 | 280 | 43.2% | -6.2 | $-2.76 |
| 1m_wick_ratio | Q1 | 5m_velocity | Q1 | 297 | 55.6% | +6.1 | $+1.03 |
| 15s_vol_rel | Q4 | 15m_z_low | Q4 | 305 | 55.4% | +6.0 | $+5.26 |
| 1m_wick_ratio | Q1 | 15m_z_low | Q1 | 269 | 55.4% | +6.0 | $+3.48 |
| 15m_reversion_prob | Q1 | 1m_wick_ratio | Q4 | 283 | 43.5% | -6.0 | $-8.65 |
| 1m_wick_ratio | Q4 | 15m_p_at_center | Q1 | 283 | 43.5% | -6.0 | $-8.65 |
| 1m_p_at_center | Q4 | 15m_z_low | Q1 | 291 | 43.6% | -5.8 | $-6.81 |
| 1m_reversion_prob | Q4 | 15m_z_low | Q1 | 291 | 43.6% | -5.8 | $-6.81 |
| 15m_reversion_prob | Q3 | 1m_wick_ratio | Q1 | 273 | 54.9% | +5.5 | $+4.46 |
| 1m_wick_ratio | Q1 | 15m_p_at_center | Q3 | 273 | 54.9% | +5.5 | $+4.46 |
| 1m_p_at_center | Q4 | 5m_velocity | Q2 | 296 | 43.9% | -5.5 | $-3.67 |
| 1m_reversion_prob | Q4 | 5m_velocity | Q2 | 296 | 43.9% | -5.5 | $-3.67 |
| 1m_p_at_center | Q1 | 1m_wick_ratio | Q1 | 472 | 54.9% | +5.4 | $+2.38 |
| 1m_reversion_prob | Q1 | 1m_wick_ratio | Q1 | 472 | 54.9% | +5.4 | $+2.38 |
| 1m_p_at_center | Q3 | 5m_velocity | Q1 | 278 | 54.7% | +5.2 | $+5.82 |
| 1m_reversion_prob | Q3 | 5m_velocity | Q1 | 278 | 54.7% | +5.2 | $+5.82 |

## Shallow decision-tree rules (max_depth=3)

Top rules by |ΔWR| × log(N). Baseline WR = 49.4%.

| Rule | N | WR | ΔWR |
|---|---:|---:|---:|
| 15s_dmi_gap <= 0.886 AND 15m_bar_range <= 145.500 AND 1h_dmi_diff <= -4.230 | 52 | 17.3% | -32.1 |
| 15s_dmi_gap > 0.886 AND 15s_variance_ratio > 1.200 | 59 | 74.6% | +25.1 |
| 15s_dmi_gap <= 0.886 AND 15m_bar_range > 145.500 AND 1h_variance_ratio > 0.574 | 53 | 67.9% | +18.5 |
| 15s_dmi_gap <= 0.886 AND 15m_bar_range <= 145.500 AND 1h_dmi_diff > -4.230 | 113 | 36.3% | -13.1 |
| 15s_dmi_gap <= 0.886 AND 15m_bar_range > 145.500 AND 1h_variance_ratio <= 0.574 | 94 | 37.2% | -12.2 |
| 15s_dmi_gap > 0.886 AND 15s_variance_ratio <= 1.200 AND 5m_velocity <= -4.125 | 951 | 54.4% | +4.9 |
| 15s_dmi_gap > 0.886 AND 15s_variance_ratio <= 1.200 AND 5m_velocity > -4.125 | 3,210 | 48.5% | -0.9 |
