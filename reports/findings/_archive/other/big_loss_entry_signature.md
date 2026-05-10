# BIG_LOSS Entry Signature — physics-based filter candidate

BIG_LOSS (pnl < $-50.0): **2,224 trades** across all tiers. Can features at ENTRY predict them?

Methodology: Cohen d between BIG_LOSS cohort and Other (pnl >= $-50.0) cohort on all 91 entry features. `**` |d| ≥ 0.5 (strong, ship as filter). `*` |d| ≥ 0.3 (moderate).

## Cross-tier separators

| feature | Cohen d | BL mean | Other mean |
|---|---:|---:|---:|
| 15m_bar_range | +0.419  * | 200.394 | 135.055 |
| 5m_bar_range | +0.395  * | 122.356 | 82.464 |
| 1h_bar_range | +0.353  * | 397.123 | 289.316 |
| 1m_bar_range | +0.323  * | 90.586 | 62.733 |
| 15s_bar_range | +0.302  * | 31.273 | 21.914 |
| 15m_vol_rel | +0.298 | 1.562 | 1.110 |
| 15m_variance_ratio | +0.281 | 0.548 | 0.438 |
| 1h_vol_rel | +0.279 | 1.222 | 0.892 |
| 5m_vol_rel | +0.238 | 1.416 | 1.143 |
| 5m_variance_ratio | +0.198 | 0.509 | 0.441 |
| 1D_dmi_diff | -0.180 | -3.159 | -0.586 |
| time_of_day | +0.180 | 0.516 | 0.467 |
| 1h_p_at_center | -0.174 | 0.455 | 0.498 |
| 1D_bar_range | +0.168 | 1594.046 | 1369.479 |
| 1h_dmi_diff | -0.156 | -3.418 | -1.066 |

_No strong cross-tier separator. Top: `15m_bar_range` d=+0.42 — moderate. Cross-tier entry filter unlikely to help; try per-tier._

## Per-tier separators (top 5 each)

### NMP_FADE  (BL=1296, Other=6878)

| feature | d | BL mean | Other mean |
|---|---:|---:|---:|
| 5m_bar_range | +0.522 ** | 115.240 | 72.574 |
| 15m_bar_range | +0.516 ** | 212.549 | 131.088 |
| 1m_bar_range | +0.459  * | 83.404 | 53.211 |
| 1h_bar_range | +0.422  * | 417.823 | 282.151 |
| 15s_bar_range | +0.403  * | 28.782 | 18.578 |

**Strong separator**: `5m_bar_range` d=+0.52 — tier-specific entry filter candidate.

### RIDE_AGAINST  (BL=337, Other=3867)

| feature | d | BL mean | Other mean |
|---|---:|---:|---:|
| 5m_bar_range | +0.566 ** | 133.139 | 83.327 |
| 1h_bar_range | +0.494  * | 446.332 | 302.822 |
| 15m_bar_range | +0.491  * | 211.347 | 137.734 |
| 1m_bar_range | +0.476  * | 101.964 | 63.203 |
| 15s_bar_range | +0.419  * | 35.380 | 22.259 |

**Strong separator**: `5m_bar_range` d=+0.57 — tier-specific entry filter candidate.

### FADE_AGAINST  (BL=135, Other=191)

| feature | d | BL mean | Other mean |
|---|---:|---:|---:|
| 1D_vol_rel | +0.337  * | 1.022 | 0.814 |
| time_of_day | -0.309  * | 0.434 | 0.525 |
| 1m_variance_ratio | +0.262 | 0.434 | 0.364 |
| 5m_z_high | +0.238 | 0.634 | 0.381 |
| 1h_wick_ratio | +0.226 | 0.457 | 0.403 |

### MTF_BREAKOUT  (BL=124, Other=588)

| feature | d | BL mean | Other mean |
|---|---:|---:|---:|
| 15m_bar_range | +0.470  * | 267.944 | 175.993 |
| 5m_bar_range | +0.455  * | 185.718 | 119.240 |
| 15s_bar_range | +0.413  * | 43.468 | 27.946 |
| 1m_bar_range | +0.386  * | 114.169 | 80.022 |
| 1h_vol_rel | +0.366  * | 1.401 | 0.919 |

### NMP_RIDE  (BL=107, Other=192)

| feature | d | BL mean | Other mean |
|---|---:|---:|---:|
| time_of_day | -0.355  * | 0.265 | 0.364 |
| 1D_variance_ratio | -0.306  * | 0.467 | 0.559 |
| 1m_wick_ratio | -0.289 | 0.268 | 0.328 |
| 1h_vol_rel | -0.275 | 0.315 | 0.445 |
| 1D_wick_ratio | +0.264 | 0.490 | 0.406 |

### TREND_FOLLOWER  (BL=105, Other=675)

| feature | d | BL mean | Other mean |
|---|---:|---:|---:|
| 15s_bar_range | +0.279 | 62.676 | 46.101 |
| 1m_bar_range | +0.258 | 183.648 | 132.030 |
| 15m_bar_range | +0.180 | 185.305 | 157.129 |
| 1h_z_se | +0.176 | 0.196 | -0.004 |
| 1h_z_high | +0.173 | 0.813 | 0.629 |

### MTF_EXHAUSTION  (BL=57, Other=68)

| feature | d | BL mean | Other mean |
|---|---:|---:|---:|
| 15s_dir_vol | +0.461  * | -0.879 | -2.347 |
| 15s_vol_rel | -0.440  * | 1.990 | 3.112 |
| 1h_z_high | -0.413  * | 0.365 | 0.768 |
| 15s_velocity | +0.392  * | -1.614 | -4.452 |
| 1h_dmi_diff | -0.383  * | -3.379 | 1.846 |

### CASCADE  (BL=36, Other=77)

| feature | d | BL mean | Other mean |
|---|---:|---:|---:|
| 15m_velocity | -0.536 ** | -2.312 | 0.211 |
| 1D_vol_rel | -0.468  * | 0.786 | 1.061 |
| 15s_reversion_prob | -0.430  * | 0.774 | 0.871 |
| 1h_p_at_center | +0.364  * | 0.286 | 0.231 |
| 15m_dir_vol | -0.339  * | -0.369 | 0.138 |

**Strong separator**: `15m_velocity` d=-0.54 — tier-specific entry filter candidate.

### KILL_SHOT_CALM  (BL=20, Other=402)

| feature | d | BL mean | Other mean |
|---|---:|---:|---:|
| 1h_bar_range | +0.651 ** | 328.900 | 195.152 |
| 5m_bar_range | +0.629 ** | 86.000 | 44.709 |
| 1h_dmi_diff | -0.554 ** | -8.404 | 1.094 |
| 1D_dmi_diff | -0.505 ** | -7.218 | 0.871 |
| 1h_variance_ratio | +0.489  * | 0.589 | 0.407 |

**Strong separator**: `1h_bar_range` d=+0.65 — tier-specific entry filter candidate.
