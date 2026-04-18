# Multivariate Response Surface — CASCADE

**Trades:** 1,270  **WR:** 51.1%  **Avg $/trade:** $+1.34

## Top univariate features (Cohen d)

| Feature | Cohen d | p | Win mean | Loss mean |
|---|---:|---:|---:|---:|
| 5m_dmi_gap | -0.121 | 0.0310 | 9.7668 | 10.7157 |
| 1h_dmi_gap | -0.110 | 0.0511 | 12.5392 | 13.6217 |
| 1h_acceleration | +0.103 | 0.0659 | 1.8486 | 0.2323 |
| 1D_vol_rel | +0.100 | 0.0741 | 0.9689 | 0.9071 |
| 15s_hurst | +0.095 | 0.0924 | 0.6859 | 0.6752 |
| 1h_variance_ratio | -0.088 | 0.1164 | 0.4789 | 0.5120 |
| 1m_z_high | +0.086 | 0.1224 | 0.8312 | 0.7403 |
| 15m_p_at_center | -0.086 | 0.1248 | 0.5495 | 0.5668 |

## Top GBM importance

| Feature | Importance |
|---|---:|
| 15s_hurst | 0.0351 |
| 1h_hurst | 0.0325 |
| 1m_z_high | 0.0257 |
| 1h_bar_range | 0.0240 |
| 1h_dmi_gap | 0.0215 |
| 15s_dmi_gap | 0.0209 |
| 1m_bar_range | 0.0207 |
| 1m_z_se | 0.0198 |

## 2D interaction cells (|ΔWR| ≥ 5pp, N ≥ 30)

Baseline WR = 51.1%. Cells listed with higher WR first; negative diverge_pp = cell is WORSE than baseline.

| feat_a | qa | feat_b | qb | N | WR | Δpp | avg $ |
|---|---|---|---|---:|---:|---:|---:|
| 5m_dmi_gap | Q4 | 15m_p_at_center | Q3 | 79 | 34.2% | -16.9 | $-7.73 |
| 1h_variance_ratio | Q4 | 15m_p_at_center | Q3 | 76 | 34.2% | -16.9 | $-12.63 |
| 1D_vol_rel | Q2 | 1h_variance_ratio | Q3 | 71 | 67.6% | +16.5 | $+3.58 |
| 1h_dmi_gap | Q1 | 1m_z_high | Q3 | 80 | 67.5% | +16.4 | $+10.71 |
| 1h_variance_ratio | Q4 | 1m_z_high | Q1 | 65 | 35.4% | -15.7 | $+1.90 |
| 5m_dmi_gap | Q1 | 1m_z_high | Q2 | 81 | 66.7% | +15.6 | $+10.15 |
| 1D_vol_rel | Q1 | 1m_z_high | Q4 | 88 | 36.4% | -14.7 | $-4.28 |
| 1h_dmi_gap | Q1 | 15s_hurst | Q3 | 75 | 65.3% | +14.2 | $+5.83 |
| 1m_z_high | Q3 | 15m_p_at_center | Q1 | 80 | 65.0% | +13.9 | $+7.38 |
| 1h_dmi_gap | Q3 | 1h_variance_ratio | Q4 | 77 | 37.7% | -13.4 | $-0.02 |
| 1m_z_high | Q4 | 15m_p_at_center | Q2 | 73 | 64.4% | +13.3 | $+11.73 |
| 5m_dmi_gap | Q4 | 1m_z_high | Q1 | 66 | 37.9% | -13.2 | $-5.55 |
| 1D_vol_rel | Q2 | 15s_hurst | Q3 | 92 | 64.1% | +13.0 | $+9.01 |
| 1D_vol_rel | Q4 | 1h_variance_ratio | Q2 | 75 | 64.0% | +12.9 | $+13.19 |
| 1h_dmi_gap | Q1 | 15m_p_at_center | Q2 | 97 | 63.9% | +12.8 | $+8.82 |
| 15s_hurst | Q3 | 1m_z_high | Q4 | 82 | 63.4% | +12.3 | $+14.11 |
| 15s_hurst | Q3 | 15m_p_at_center | Q2 | 89 | 62.9% | +11.8 | $+12.09 |
| 5m_dmi_gap | Q4 | 1h_dmi_gap | Q2 | 61 | 39.3% | -11.8 | $+0.89 |
| 15s_hurst | Q4 | 1m_z_high | Q1 | 71 | 39.4% | -11.7 | $-10.72 |
| 5m_dmi_gap | Q4 | 15s_hurst | Q1 | 81 | 39.5% | -11.6 | $-7.57 |
| 1D_vol_rel | Q1 | 1h_variance_ratio | Q4 | 91 | 39.6% | -11.5 | $-2.36 |
| 5m_dmi_gap | Q2 | 1D_vol_rel | Q1 | 75 | 40.0% | -11.1 | $-13.25 |
| 1m_z_high | Q1 | 15m_p_at_center | Q3 | 72 | 40.3% | -10.8 | $-2.23 |
| 5m_dmi_gap | Q3 | 1m_z_high | Q3 | 76 | 61.8% | +10.7 | $+2.78 |
| 5m_dmi_gap | Q3 | 1m_z_high | Q1 | 91 | 40.7% | -10.4 | $-1.64 |
| 5m_dmi_gap | Q1 | 15s_hurst | Q4 | 88 | 61.4% | +10.3 | $+6.18 |
| 1h_dmi_gap | Q1 | 15s_hurst | Q2 | 88 | 40.9% | -10.2 | $-3.92 |
| 1D_vol_rel | Q4 | 15m_p_at_center | Q2 | 62 | 61.3% | +10.2 | $+9.90 |
| 1h_variance_ratio | Q3 | 1m_z_high | Q3 | 85 | 61.2% | +10.1 | $-1.91 |
| 15s_hurst | Q3 | 1h_variance_ratio | Q1 | 82 | 61.0% | +9.9 | $+5.96 |
| 1h_dmi_gap | Q1 | 1D_vol_rel | Q4 | 64 | 60.9% | +9.8 | $+8.41 |
| 1D_vol_rel | Q1 | 1m_z_high | Q1 | 75 | 41.3% | -9.8 | $+0.09 |
| 1h_dmi_gap | Q2 | 1D_vol_rel | Q2 | 69 | 60.9% | +9.8 | $+1.64 |
| 1m_z_high | Q1 | 15m_p_at_center | Q1 | 84 | 41.7% | -9.4 | $+1.01 |
| 1D_vol_rel | Q4 | 15s_hurst | Q2 | 76 | 60.5% | +9.4 | $+15.24 |
| 1h_dmi_gap | Q4 | 15m_p_at_center | Q1 | 91 | 60.4% | +9.3 | $+9.27 |
| 1h_dmi_gap | Q1 | 1h_variance_ratio | Q3 | 96 | 60.4% | +9.3 | $+8.96 |
| 5m_dmi_gap | Q2 | 1h_dmi_gap | Q4 | 67 | 41.8% | -9.3 | $-8.34 |
| 1h_dmi_gap | Q4 | 1m_z_high | Q1 | 86 | 41.9% | -9.2 | $-2.50 |
| 1D_vol_rel | Q1 | 1m_z_high | Q3 | 78 | 60.3% | +9.2 | $+1.34 |
| 1h_dmi_gap | Q3 | 15m_p_at_center | Q3 | 81 | 42.0% | -9.1 | $-3.62 |
| 15s_hurst | Q1 | 1m_z_high | Q2 | 81 | 42.0% | -9.1 | $-9.20 |
| 1D_vol_rel | Q3 | 1m_z_high | Q1 | 100 | 42.0% | -9.1 | $-5.01 |
| 1h_dmi_gap | Q4 | 15m_p_at_center | Q3 | 83 | 42.2% | -8.9 | $-4.81 |
| 15s_hurst | Q2 | 1h_variance_ratio | Q1 | 83 | 42.2% | -8.9 | $-4.02 |
| 1h_dmi_gap | Q4 | 1h_variance_ratio | Q1 | 50 | 60.0% | +8.9 | $-0.21 |
| 1D_vol_rel | Q3 | 1m_z_high | Q4 | 75 | 60.0% | +8.9 | $+11.77 |
| 5m_dmi_gap | Q3 | 15s_hurst | Q1 | 78 | 42.3% | -8.8 | $-1.98 |
| 1D_vol_rel | Q4 | 15s_hurst | Q4 | 82 | 59.8% | +8.7 | $+9.32 |
| 5m_dmi_gap | Q4 | 1h_variance_ratio | Q4 | 106 | 42.5% | -8.6 | $+0.97 |

## Shallow decision-tree rules (max_depth=3)

Top rules by |ΔWR| × log(N). Baseline WR = 51.1%.

| Rule | N | WR | ΔWR |
|---|---:|---:|---:|
| 1h_hurst > 0.580 AND 1m_bar_range > 56.500 AND 1m_z_high > 0.431 | 176 | 67.6% | +16.5 |
| 1h_hurst > 0.580 AND 1m_bar_range <= 56.500 AND 5m_dmi_gap > 20.429 | 80 | 32.5% | -18.6 |
| 1h_hurst <= 0.580 | 77 | 33.8% | -17.3 |
| 1h_hurst > 0.580 AND 1m_bar_range > 56.500 AND 1m_z_high <= 0.431 | 54 | 44.4% | -6.7 |
| 1h_hurst > 0.580 AND 1m_bar_range <= 56.500 AND 5m_dmi_gap <= 20.429 | 883 | 51.4% | +0.3 |
