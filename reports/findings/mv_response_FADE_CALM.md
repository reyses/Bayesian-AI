# Multivariate Response Surface — FADE_CALM

**Trades:** 24,039  **WR:** 48.9%  **Avg $/trade:** $-0.23

## Top univariate features (Cohen d)

| Feature | Cohen d | p | Win mean | Loss mean |
|---|---:|---:|---:|---:|
| 1m_dir_vol | -0.049 | 0.0001 | -0.0391 | 0.0169 |
| 1m_z_se | -0.046 | 0.0004 | -0.0504 | -0.0156 |
| 1h_dir_vol | +0.044 | 0.0007 | 0.0225 | -0.0438 |
| 15s_dmi_diff | -0.039 | 0.0027 | -0.4066 | 0.0823 |
| 15s_dmi_gap | -0.036 | 0.0047 | 9.8635 | 10.1455 |
| 1h_vol_rel | +0.034 | 0.0094 | 0.9988 | 0.9599 |
| 15m_dmi_diff | +0.032 | 0.0119 | -0.7921 | -1.2490 |
| 15s_z_low | -0.030 | 0.0193 | -0.5946 | -0.5616 |

## Top GBM importance

| Feature | Importance |
|---|---:|
| 15s_variance_ratio | 0.0283 |
| 1m_z_se | 0.0276 |
| 1m_vol_rel | 0.0264 |
| 15m_hurst | 0.0245 |
| time_of_day | 0.0245 |
| 1h_dmi_diff | 0.0232 |
| 1m_dir_vol | 0.0230 |
| 5m_dir_vol | 0.0226 |

## 2D interaction cells (|ΔWR| ≥ 5pp, N ≥ 30)

Baseline WR = 48.9%. Cells listed with higher WR first; negative diverge_pp = cell is WORSE than baseline.

| feat_a | qa | feat_b | qb | N | WR | Δpp | avg $ |
|---|---|---|---|---:|---:|---:|---:|
| 1h_dir_vol | Q2 | 1h_vol_rel | Q3 | 62 | 58.1% | +9.2 | $+10.56 |
| 1m_z_se | Q2 | 1h_dir_vol | Q1 | 113 | 57.5% | +8.6 | $+2.33 |
| 1m_dir_vol | Q4 | 15s_z_low | Q1 | 311 | 43.4% | -5.5 | $-2.65 |
| 1m_z_se | Q3 | 15s_dmi_diff | Q3 | 1,686 | 43.5% | -5.4 | $-2.29 |
| 1h_dir_vol | Q2 | 15s_dmi_diff | Q3 | 1,489 | 43.5% | -5.4 | $-2.29 |
| 1h_dir_vol | Q4 | 1h_vol_rel | Q2 | 100 | 54.0% | +5.1 | $-5.80 |
| 1m_z_se | Q4 | 15s_z_low | Q1 | 445 | 43.8% | -5.1 | $-4.68 |
| 1m_z_se | Q4 | 1h_dir_vol | Q3 | 171 | 43.9% | -5.0 | $-3.11 |

## Shallow decision-tree rules (max_depth=3)

Top rules by |ΔWR| × log(N). Baseline WR = 48.9%.

| Rule | N | WR | ΔWR |
|---|---:|---:|---:|
| 1m_z_se > 0.093 AND 5m_dir_vol > -0.706 AND 1m_z_high <= 0.859 | 2,998 | 42.3% | -6.6 |
| 1m_z_se <= 0.093 AND 5m_z_se <= -1.178 AND 1D_z_low > 0.138 | 715 | 41.7% | -7.2 |
| 1m_z_se > 0.093 AND 5m_dir_vol <= -0.706 AND 1m_z_high > 0.882 | 1,795 | 53.9% | +5.0 |
| 1m_z_se <= 0.093 AND 5m_z_se > -1.178 AND 15s_vol_rel > 0.450 | 8,013 | 52.4% | +3.5 |
| 1m_z_se > 0.093 AND 5m_dir_vol > -0.706 AND 1m_z_high > 0.859 | 3,618 | 46.8% | -2.1 |
| 1m_z_se > 0.093 AND 5m_dir_vol <= -0.706 AND 1m_z_high <= 0.882 | 1,680 | 47.6% | -1.3 |
| 1m_z_se <= 0.093 AND 5m_z_se > -1.178 AND 15s_vol_rel <= 0.450 | 3,044 | 48.5% | -0.4 |
| 1m_z_se <= 0.093 AND 5m_z_se <= -1.178 AND 1D_z_low <= 0.138 | 2,176 | 48.4% | -0.5 |
