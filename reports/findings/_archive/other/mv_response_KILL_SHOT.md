# Multivariate Response Surface — KILL_SHOT

**Trades:** 4,411  **WR:** 51.5%  **Avg $/trade:** $-0.14

## Top univariate features (Cohen d)

| Feature | Cohen d | p | Win mean | Loss mean |
|---|---:|---:|---:|---:|
| 1m_reversion_prob | -0.093 | 0.0020 | 0.9772 | 0.9813 |
| 1m_p_at_center | -0.092 | 0.0021 | 0.6572 | 0.6713 |
| 1D_z_high | +0.065 | 0.0319 | 0.6455 | 0.5800 |
| 1m_z_low | -0.060 | 0.0471 | -0.7389 | -0.6952 |
| 5m_dmi_gap | +0.059 | 0.0492 | 9.5542 | 9.1192 |
| 5m_reversion_prob | -0.054 | 0.0703 | 0.9733 | 0.9753 |
| 1D_wick_ratio | -0.054 | 0.0714 | 0.5147 | 0.5299 |
| 1h_dir_vol | +0.050 | 0.1001 | 0.0636 | -0.0076 |

## Top GBM importance

| Feature | Importance |
|---|---:|
| 1m_z_high | 0.0277 |
| 15s_dmi_gap | 0.0272 |
| 1m_z_low | 0.0263 |
| 1m_vol_rel | 0.0243 |
| 15s_hurst | 0.0227 |
| 15s_vol_rel | 0.0223 |
| 1m_dir_vol | 0.0209 |
| 15m_dmi_gap | 0.0209 |

## 2D interaction cells (|ΔWR| ≥ 5pp, N ≥ 30)

Baseline WR = 51.5%. Cells listed with higher WR first; negative diverge_pp = cell is WORSE than baseline.

| feat_a | qa | feat_b | qb | N | WR | Δpp | avg $ |
|---|---|---|---|---:|---:|---:|---:|
| 5m_dmi_gap | Q3 | 5m_reversion_prob | Q4 | 249 | 41.0% | -10.5 | $-3.38 |
| 5m_reversion_prob | Q2 | 1h_dir_vol | Q3 | 248 | 60.9% | +9.4 | $+1.95 |
| 5m_dmi_gap | Q4 | 1h_dir_vol | Q3 | 230 | 59.1% | +7.6 | $+2.00 |
| 5m_reversion_prob | Q1 | 1D_wick_ratio | Q4 | 274 | 44.2% | -7.3 | $-3.85 |
| 1D_z_high | Q4 | 1D_wick_ratio | Q4 | 205 | 44.4% | -7.1 | $-6.43 |
| 1m_reversion_prob | Q3 | 1D_wick_ratio | Q4 | 252 | 44.4% | -7.1 | $-3.24 |
| 1m_p_at_center | Q3 | 1D_wick_ratio | Q4 | 252 | 44.4% | -7.1 | $-3.24 |
| 1m_reversion_prob | Q1 | 5m_reversion_prob | Q1 | 246 | 58.5% | +7.0 | $-1.81 |
| 1m_p_at_center | Q1 | 5m_reversion_prob | Q1 | 246 | 58.5% | +7.0 | $-1.81 |
| 5m_reversion_prob | Q4 | 1h_dir_vol | Q2 | 254 | 44.9% | -6.6 | $-1.77 |
| 1m_reversion_prob | Q4 | 5m_dmi_gap | Q3 | 260 | 45.0% | -6.5 | $-1.82 |
| 1m_p_at_center | Q4 | 5m_dmi_gap | Q3 | 260 | 45.0% | -6.5 | $-1.82 |
| 1m_reversion_prob | Q2 | 5m_reversion_prob | Q1 | 276 | 45.3% | -6.2 | $+0.24 |
| 1m_p_at_center | Q2 | 5m_reversion_prob | Q1 | 276 | 45.3% | -6.2 | $+0.24 |
| 1D_z_high | Q4 | 5m_reversion_prob | Q2 | 274 | 57.7% | +6.2 | $+2.09 |
| 1m_z_low | Q1 | 5m_dmi_gap | Q4 | 269 | 57.6% | +6.1 | $+2.97 |
| 5m_dmi_gap | Q4 | 1D_wick_ratio | Q2 | 245 | 57.6% | +6.0 | $+2.33 |
| 1m_reversion_prob | Q1 | 1D_wick_ratio | Q1 | 280 | 57.5% | +6.0 | $+0.86 |
| 1m_p_at_center | Q1 | 1D_wick_ratio | Q1 | 280 | 57.5% | +6.0 | $+0.86 |
| 1m_reversion_prob | Q1 | 1h_dir_vol | Q3 | 282 | 57.4% | +5.9 | $-0.13 |
| 1m_p_at_center | Q1 | 1h_dir_vol | Q3 | 282 | 57.4% | +5.9 | $-0.13 |
| 1D_z_high | Q2 | 5m_dmi_gap | Q3 | 250 | 45.6% | -5.9 | $-1.58 |
| 1D_z_high | Q3 | 1D_wick_ratio | Q3 | 305 | 57.4% | +5.9 | $+5.53 |
| 1m_z_low | Q4 | 1D_wick_ratio | Q4 | 245 | 45.7% | -5.8 | $+0.00 |
| 5m_dmi_gap | Q1 | 5m_reversion_prob | Q1 | 234 | 45.7% | -5.8 | $-1.90 |
| 1D_z_high | Q3 | 1m_z_low | Q1 | 283 | 57.2% | +5.7 | $+6.62 |
| 1D_wick_ratio | Q3 | 1h_dir_vol | Q3 | 278 | 57.2% | +5.7 | $+0.79 |
| 1m_reversion_prob | Q1 | 5m_dmi_gap | Q4 | 301 | 57.1% | +5.6 | $-0.91 |
| 1m_p_at_center | Q1 | 5m_dmi_gap | Q4 | 301 | 57.1% | +5.6 | $-0.91 |
| 5m_reversion_prob | Q1 | 1D_wick_ratio | Q1 | 307 | 57.0% | +5.5 | $+2.01 |
| 1D_z_high | Q4 | 1h_dir_vol | Q3 | 293 | 57.0% | +5.5 | $-2.54 |
| 5m_dmi_gap | Q3 | 1h_dir_vol | Q2 | 252 | 46.0% | -5.5 | $-1.66 |
| 1m_z_low | Q1 | 5m_reversion_prob | Q1 | 271 | 56.8% | +5.3 | $+0.57 |
| 1D_z_high | Q3 | 5m_dmi_gap | Q4 | 294 | 56.8% | +5.3 | $+3.04 |
| 1m_reversion_prob | Q3 | 1m_z_low | Q4 | 253 | 46.2% | -5.3 | $+0.26 |
| 1m_p_at_center | Q3 | 1m_z_low | Q4 | 253 | 46.2% | -5.3 | $+0.26 |
| 5m_dmi_gap | Q4 | 5m_reversion_prob | Q1 | 319 | 56.7% | +5.2 | $+0.55 |
| 1m_z_low | Q2 | 1h_dir_vol | Q1 | 229 | 46.3% | -5.2 | $-2.69 |
| 1D_z_high | Q1 | 5m_dmi_gap | Q3 | 270 | 46.3% | -5.2 | $-2.46 |
| 1m_z_low | Q2 | 5m_reversion_prob | Q3 | 280 | 46.4% | -5.1 | $-2.28 |
| 1D_wick_ratio | Q4 | 1h_dir_vol | Q2 | 267 | 46.4% | -5.1 | $-0.64 |
| 1D_z_high | Q3 | 1m_z_low | Q3 | 274 | 56.6% | +5.1 | $-0.44 |
| 5m_reversion_prob | Q2 | 1D_wick_ratio | Q1 | 260 | 56.5% | +5.0 | $+2.60 |
| 1D_z_high | Q2 | 1m_z_low | Q3 | 284 | 46.5% | -5.0 | $-0.32 |
| 5m_dmi_gap | Q1 | 5m_reversion_prob | Q3 | 276 | 56.5% | +5.0 | $+3.01 |

## Shallow decision-tree rules (max_depth=3)

Top rules by |ΔWR| × log(N). Baseline WR = 51.5%.

| Rule | N | WR | ΔWR |
|---|---:|---:|---:|
| 1m_dir_vol <= -0.919 AND 5m_dmi_gap > 9.361 AND 1h_variance_ratio <= 0.277 | 97 | 80.4% | +28.9 |
| 1m_dir_vol > -0.919 AND 15s_z_low <= -1.113 AND 1m_p_at_center > 0.785 | 94 | 28.7% | -22.8 |
| 1m_dir_vol > -0.919 AND 15s_z_low > -1.113 AND 5m_p_at_center > 0.787 | 64 | 73.4% | +21.9 |
| 1m_dir_vol <= -0.919 AND 5m_dmi_gap <= 9.361 AND 1h_reversion_prob <= 0.946 | 62 | 69.4% | +17.8 |
| 1m_dir_vol <= -0.919 AND 5m_dmi_gap > 9.361 AND 1h_variance_ratio > 0.277 | 138 | 58.0% | +6.5 |
| 1m_dir_vol > -0.919 AND 15s_z_low <= -1.113 AND 1m_p_at_center <= 0.785 | 854 | 47.5% | -4.0 |
| 1m_dir_vol <= -0.919 AND 5m_dmi_gap <= 9.361 AND 1h_reversion_prob > 0.946 | 253 | 47.8% | -3.7 |
| 1m_dir_vol > -0.919 AND 15s_z_low > -1.113 AND 5m_p_at_center <= 0.787 | 2,849 | 51.6% | +0.1 |
