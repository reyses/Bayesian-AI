# Multivariate Response Surface — RIDE_AGAINST

**Trades:** 39,721  **WR:** 48.0%  **Avg $/trade:** $-0.10

## Top univariate features (Cohen d)

| Feature | Cohen d | p | Win mean | Loss mean |
|---|---:|---:|---:|---:|
| 1m_dmi_gap | +0.049 | 0.0000 | 10.4305 | 10.0492 |
| 1m_p_at_center | +0.047 | 0.0000 | 0.6924 | 0.6849 |
| 5m_dmi_gap | +0.040 | 0.0001 | 10.7256 | 10.3976 |
| 5m_bar_range | +0.036 | 0.0003 | 93.1831 | 90.1321 |
| 1m_reversion_prob | +0.034 | 0.0008 | 0.9806 | 0.9785 |
| 1m_variance_ratio | +0.033 | 0.0009 | 0.4435 | 0.4349 |
| 5m_vol_rel | +0.031 | 0.0020 | 1.0993 | 1.0702 |
| 1h_dmi_gap | +0.029 | 0.0039 | 12.1993 | 11.9389 |

## Top GBM importance

| Feature | Importance |
|---|---:|
| 5m_z_high | 0.0367 |
| 1m_z_se | 0.0303 |
| 1m_reversion_prob | 0.0297 |
| 5m_dmi_gap | 0.0293 |
| 1h_dir_vol | 0.0286 |
| time_of_day | 0.0284 |
| 1m_dmi_gap | 0.0247 |
| 1m_vol_rel | 0.0240 |

## 2D interaction cells (|ΔWR| ≥ 5pp, N ≥ 30)

Baseline WR = 48.0%. Cells listed with higher WR first; negative diverge_pp = cell is WORSE than baseline.

| feat_a | qa | feat_b | qb | N | WR | Δpp | avg $ |
|---|---|---|---|---:|---:|---:|---:|

## Shallow decision-tree rules (max_depth=3)

Top rules by |ΔWR| × log(N). Baseline WR = 48.0%.

| Rule | N | WR | ΔWR |
|---|---:|---:|---:|
| 1m_reversion_prob <= 0.985 AND 1m_dmi_gap <= 8.188 AND 1m_z_high <= -0.042 | 309 | 32.4% | -15.7 |
| 1m_reversion_prob > 0.985 AND 1m_dmi_gap > 15.116 AND 5m_dmi_gap > 15.313 | 3,315 | 55.1% | +7.0 |
| 1m_reversion_prob > 0.985 AND 1m_dmi_gap <= 15.116 AND 1h_dmi_diff <= -30.477 | 614 | 56.4% | +8.3 |
| 1m_reversion_prob <= 0.985 AND 1m_dmi_gap <= 8.188 AND 1m_z_high > -0.042 | 4,461 | 44.1% | -3.9 |
| 1m_reversion_prob <= 0.985 AND 1m_dmi_gap > 8.188 AND 15m_z_low <= -0.857 | 2,542 | 44.7% | -3.3 |
| 1m_reversion_prob <= 0.985 AND 1m_dmi_gap > 8.188 AND 15m_z_low > -0.857 | 2,696 | 50.1% | +2.1 |
| 1m_reversion_prob > 0.985 AND 1m_dmi_gap > 15.116 AND 5m_dmi_gap <= 15.313 | 3,697 | 48.5% | +0.5 |
| 1m_reversion_prob > 0.985 AND 1m_dmi_gap <= 15.116 AND 1h_dmi_diff > -30.477 | 22,087 | 47.8% | -0.2 |
