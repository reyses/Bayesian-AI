# Multivariate Response Surface — MTF_EXHAUSTION

**Trades:** 233  **WR:** 48.9%  **Avg $/trade:** $+5.90

## Top univariate features (Cohen d)

| Feature | Cohen d | p | Win mean | Loss mean |
|---|---:|---:|---:|---:|
| 15m_hurst | -0.258 | 0.0516 | 0.6848 | 0.7123 |
| 1h_hurst | -0.235 | 0.0750 | 0.6928 | 0.7185 |
| 1h_reversion_prob | -0.222 | 0.0969 | 0.9226 | 0.9505 |
| 5m_hurst | +0.207 | 0.1178 | 0.7147 | 0.6943 |
| 15m_velocity | -0.186 | 0.1602 | -27.1645 | -19.7731 |
| 15m_variance_ratio | +0.183 | 0.1667 | 0.5965 | 0.5260 |
| 15m_acceleration | -0.177 | 0.1818 | -18.8487 | -12.6218 |
| 15m_bar_range | +0.173 | 0.1925 | 235.4912 | 209.7899 |

## Top GBM importance

| Feature | Importance |
|---|---:|
| 5m_z_high | 0.0631 |
| 15m_dmi_gap | 0.0551 |
| 15m_vol_rel | 0.0462 |
| 15m_hurst | 0.0447 |
| 5m_z_low | 0.0419 |
| 15m_variance_ratio | 0.0311 |
| 5m_variance_ratio | 0.0295 |
| 15s_hurst | 0.0284 |

## 2D interaction cells (|ΔWR| ≥ 5pp, N ≥ 30)

Baseline WR = 48.9%. Cells listed with higher WR first; negative diverge_pp = cell is WORSE than baseline.

| feat_a | qa | feat_b | qb | N | WR | Δpp | avg $ |
|---|---|---|---|---:|---:|---:|---:|
| 15m_velocity | Q3 | 15m_bar_range | Q1 | 36 | 38.9% | -10.0 | $-4.31 |

## Shallow decision-tree rules (max_depth=3)

Top rules by |ΔWR| × log(N). Baseline WR = 48.9%.

| Rule | N | WR | ΔWR |
|---|---:|---:|---:|
| 15m_hurst <= 0.698 AND 1h_z_se <= 0.029 | 57 | 70.2% | +21.2 |
| 15m_hurst > 0.698 AND 1D_dmi_diff <= -2.760 | 65 | 29.2% | -19.7 |
| 15m_hurst > 0.698 AND 1D_dmi_diff > -2.760 | 61 | 50.8% | +1.9 |
| 15m_hurst <= 0.698 AND 1h_z_se > 0.029 | 50 | 48.0% | -0.9 |
