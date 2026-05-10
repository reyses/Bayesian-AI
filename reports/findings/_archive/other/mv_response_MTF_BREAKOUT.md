# Multivariate Response Surface — MTF_BREAKOUT

**Trades:** 5,961  **WR:** 45.1%  **Avg $/trade:** $+0.23

## Top univariate features (Cohen d)

| Feature | Cohen d | p | Win mean | Loss mean |
|---|---:|---:|---:|---:|
| 15m_z_high | +0.068 | 0.0088 | 0.9058 | 0.8047 |
| 5m_hurst | -0.067 | 0.0101 | 0.7181 | 0.7256 |
| 1h_z_high | -0.066 | 0.0112 | 0.5744 | 0.6472 |
| 1m_dmi_gap | -0.060 | 0.0219 | 14.1516 | 14.7814 |
| 5m_z_low | +0.048 | 0.0676 | -1.0775 | -1.1478 |
| 15s_bar_range | +0.046 | 0.0795 | 23.0216 | 21.8866 |
| 1h_dmi_diff | -0.046 | 0.0797 | -1.0805 | -0.3897 |
| 15m_z_se | +0.045 | 0.0860 | -0.1770 | -0.2625 |

## Top GBM importance

| Feature | Importance |
|---|---:|
| 1m_dmi_gap | 0.0435 |
| 5m_hurst | 0.0289 |
| 15s_hurst | 0.0250 |
| 15s_z_high | 0.0222 |
| 5m_dmi_gap | 0.0209 |
| 1m_variance_ratio | 0.0206 |
| 15s_variance_ratio | 0.0201 |
| 1h_dir_vol | 0.0191 |

## 2D interaction cells (|ΔWR| ≥ 5pp, N ≥ 30)

Baseline WR = 45.1%. Cells listed with higher WR first; negative diverge_pp = cell is WORSE than baseline.

| feat_a | qa | feat_b | qb | N | WR | Δpp | avg $ |
|---|---|---|---|---:|---:|---:|---:|
| 15m_z_high | Q4 | 15m_z_se | Q1 | 46 | 56.5% | +11.4 | $+12.40 |
| 15m_z_high | Q2 | 5m_z_low | Q4 | 47 | 55.3% | +10.2 | $+7.43 |
| 5m_hurst | Q1 | 1h_dmi_diff | Q1 | 373 | 53.9% | +8.8 | $+4.73 |
| 15m_z_high | Q3 | 15m_z_se | Q2 | 78 | 53.8% | +8.7 | $+3.97 |
| 5m_hurst | Q1 | 1h_z_high | Q2 | 366 | 53.3% | +8.2 | $+2.97 |
| 5m_hurst | Q4 | 1m_dmi_gap | Q4 | 450 | 37.1% | -8.0 | $-3.73 |
| 15m_z_high | Q1 | 5m_hurst | Q2 | 352 | 37.2% | -7.9 | $-2.23 |
| 5m_z_low | Q3 | 15m_z_se | Q1 | 231 | 52.4% | +7.3 | $+3.46 |
| 15m_z_high | Q4 | 1h_dmi_diff | Q3 | 371 | 52.3% | +7.2 | $+3.42 |
| 5m_hurst | Q1 | 15m_z_se | Q1 | 300 | 52.0% | +6.9 | $+4.85 |
| 15m_z_high | Q3 | 1m_dmi_gap | Q1 | 353 | 51.8% | +6.7 | $+2.93 |
| 15m_z_high | Q3 | 1h_dmi_diff | Q1 | 480 | 51.7% | +6.6 | $+6.11 |
| 15m_z_high | Q1 | 5m_z_low | Q1 | 293 | 38.6% | -6.5 | $+0.55 |
| 5m_hurst | Q1 | 1m_dmi_gap | Q1 | 347 | 51.6% | +6.5 | $+3.11 |
| 5m_hurst | Q4 | 15s_bar_range | Q2 | 315 | 38.7% | -6.4 | $-2.36 |
| 1h_z_high | Q2 | 1m_dmi_gap | Q1 | 377 | 51.5% | +6.3 | $-0.14 |
| 15m_z_high | Q1 | 1m_dmi_gap | Q4 | 276 | 38.8% | -6.3 | $+0.60 |
| 1h_dmi_diff | Q1 | 15m_z_se | Q3 | 411 | 51.3% | +6.2 | $+4.94 |
| 1m_dmi_gap | Q1 | 15s_bar_range | Q1 | 458 | 51.3% | +6.2 | $+0.92 |
| 5m_hurst | Q1 | 15s_bar_range | Q3 | 349 | 51.3% | +6.2 | $+3.25 |
| 15m_z_high | Q1 | 1h_dmi_diff | Q1 | 302 | 39.1% | -6.0 | $-5.34 |
| 1m_dmi_gap | Q1 | 15m_z_se | Q4 | 350 | 51.1% | +6.0 | $+1.23 |
| 1h_z_high | Q3 | 1h_dmi_diff | Q4 | 266 | 39.1% | -6.0 | $-3.71 |
| 1h_z_high | Q3 | 5m_z_low | Q1 | 386 | 39.1% | -6.0 | $-0.43 |
| 1m_dmi_gap | Q2 | 5m_z_low | Q2 | 354 | 39.3% | -5.8 | $-3.34 |
| 1h_z_high | Q4 | 15s_bar_range | Q3 | 414 | 39.4% | -5.7 | $-2.74 |
| 5m_z_low | Q1 | 15s_bar_range | Q2 | 304 | 39.5% | -5.6 | $-3.07 |
| 5m_hurst | Q1 | 1m_dmi_gap | Q2 | 422 | 50.7% | +5.6 | $+4.66 |
| 5m_z_low | Q3 | 1h_dmi_diff | Q3 | 369 | 50.7% | +5.6 | $+4.48 |
| 5m_hurst | Q4 | 1h_z_high | Q3 | 361 | 39.6% | -5.5 | $-1.41 |
| 1h_z_high | Q3 | 15s_bar_range | Q2 | 353 | 39.7% | -5.4 | $-1.43 |
| 15m_z_high | Q1 | 15m_z_se | Q2 | 726 | 39.7% | -5.4 | $-3.79 |
| 1h_z_high | Q2 | 15m_z_se | Q4 | 386 | 50.5% | +5.4 | $-1.20 |
| 1h_z_high | Q4 | 5m_z_low | Q2 | 473 | 39.7% | -5.4 | $-3.35 |
| 5m_z_low | Q1 | 15m_z_se | Q4 | 121 | 50.4% | +5.3 | $-2.45 |
| 5m_hurst | Q2 | 5m_z_low | Q2 | 359 | 39.8% | -5.3 | $-0.28 |
| 1h_z_high | Q3 | 1m_dmi_gap | Q4 | 399 | 39.8% | -5.3 | $-0.20 |
| 5m_hurst | Q2 | 1h_z_high | Q3 | 326 | 39.9% | -5.2 | $-0.68 |
| 1h_z_high | Q1 | 5m_z_low | Q2 | 253 | 39.9% | -5.2 | $-2.56 |
| 15m_z_high | Q1 | 1h_z_high | Q4 | 521 | 39.9% | -5.2 | $-0.38 |
| 15m_z_high | Q1 | 15s_bar_range | Q3 | 362 | 40.1% | -5.1 | $-1.99 |

## Shallow decision-tree rules (max_depth=3)

Top rules by |ΔWR| × log(N). Baseline WR = 45.1%.

| Rule | N | WR | ΔWR |
|---|---:|---:|---:|
| 5m_reversion_prob > 0.948 AND 5m_z_high <= 1.387 | 84 | 71.4% | +26.3 |
| 5m_reversion_prob <= 0.948 AND 1h_z_high <= 0.635 AND 1h_reversion_prob > 0.998 | 172 | 60.5% | +15.4 |
| 5m_reversion_prob <= 0.948 AND 1h_z_high > 0.635 AND 1h_variance_ratio > 0.253 | 1,933 | 39.8% | -5.3 |
| 5m_reversion_prob <= 0.948 AND 1h_z_high > 0.635 AND 1h_variance_ratio <= 0.253 | 955 | 47.4% | +2.3 |
| 5m_reversion_prob > 0.948 AND 5m_z_high > 1.387 | 78 | 48.7% | +3.6 |
| 5m_reversion_prob <= 0.948 AND 1h_z_high <= 0.635 AND 1h_reversion_prob <= 0.998 | 2,739 | 46.1% | +1.0 |
