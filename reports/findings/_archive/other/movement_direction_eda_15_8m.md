# Movement direction EDA — target $15, 8 min

## Phase 1 — single feature Cohen d

**Binary question**: at a bar with a $15 hit, does any single feature discriminate LONG-first from SHORT-first?

IS cohort sizes: LONG=1,423,322, SHORT=1,445,015, NEITHER=413,754
OOS cohort sizes: LONG=395,651, SHORT=406,024, NEITHER=59,172

### Top 30 features — LONG first vs SHORT first (IS)

Positive d = feature is HIGHER on LONG-first bars.

| Rank | Feature | d_IS | d_OOS | Walk-forward |
|---:|---|---:|---:|---|
| 1 | `1h_variance_ratio` | -0.010 | +0.008 | — |
| 2 | `15m_hurst` | -0.010 | +0.009 | — |
| 3 | `5m_z_se` | +0.009 | -0.019 | — |
| 4 | `5m_dmi_diff` | +0.008 | +0.008 | — |
| 5 | `5m_wick_ratio` | -0.008 | -0.005 | — |
| 6 | `15s_dmi_diff` | +0.007 | -0.011 | — |
| 7 | `15m_velocity` | -0.007 | -0.006 | — |
| 8 | `15s_dmi_gap` | +0.007 | +0.018 | — |
| 9 | `1m_variance_ratio` | +0.007 | +0.013 | — |
| 10 | `1D_dmi_diff` | +0.007 | +0.000 | — |
| 11 | `15s_z_se` | -0.007 | +0.001 | — |
| 12 | `1m_vol_rel` | +0.007 | +0.014 | — |
| 13 | `1D_p_at_center` | -0.006 | +0.013 | — |
| 14 | `1D_hurst` | -0.006 | -0.008 | — |
| 15 | `1m_hurst` | -0.006 | +0.008 | — |
| 16 | `1m_dmi_diff` | +0.006 | -0.013 | — |
| 17 | `15s_z_low` | -0.006 | +0.004 | — |
| 18 | `15s_hurst` | -0.006 | +0.008 | — |
| 19 | `5m_bar_range` | +0.006 | +0.002 | — |
| 20 | `5m_p_at_center` | -0.006 | -0.000 | — |
| 21 | `1D_dir_vol` | +0.006 | +0.018 | — |
| 22 | `1h_vol_rel` | +0.005 | +0.002 | — |
| 23 | `1m_z_se` | +0.005 | +0.003 | — |
| 24 | `5m_reversion_prob` | -0.005 | -0.010 | — |
| 25 | `15s_vol_rel` | +0.005 | +0.005 | — |
| 26 | `1m_wick_ratio` | -0.005 | +0.002 | — |
| 27 | `1h_dmi_diff` | +0.005 | +0.014 | — |
| 28 | `1h_z_se` | -0.005 | -0.008 | — |
| 29 | `1h_dmi_gap` | -0.005 | -0.003 | — |
| 30 | `1m_bar_range` | +0.004 | +0.002 | — |

### Walk-forward stable LONG-vs-SHORT features (0 total)

_No features clear |d| >= 0.05 on both sides. Pure direction prediction from single features is near-zero._

### Top 20 features — ANY hit vs NEITHER (dead-zone signal)

Positive d = feature is HIGHER on "opportunity" bars than dead zones.

| Rank | Feature | d_IS | d_OOS | Walk-forward |
|---:|---|---:|---:|---|
| 1 | `5m_bar_range` | +0.692 | +0.819 | ✓ |
| 2 | `1m_bar_range` | +0.687 | +0.777 | ✓ |
| 3 | `15m_bar_range` | +0.673 | +0.801 | ✓ |
| 4 | `1h_bar_range` | +0.665 | +0.758 | ✓ |
| 5 | `15s_bar_range` | +0.656 | +0.743 | ✓ |
| 6 | `1D_dmi_diff` | -0.554 | -0.290 | ✓ |
| 7 | `1h_vol_rel` | +0.538 | +0.429 | ✓ |
| 8 | `15m_variance_ratio` | +0.519 | +0.545 | ✓ |
| 9 | `15m_vol_rel` | +0.475 | +0.492 | ✓ |
| 10 | `1D_bar_range` | +0.416 | +0.410 | ✓ |
| 11 | `time_of_day` | +0.380 | +0.344 | ✓ |
| 12 | `1h_dmi_diff` | -0.373 | -0.528 | ✓ |
| 13 | `1D_z_se` | -0.328 | -0.489 | ✓ |
| 14 | `15m_dmi_diff` | -0.302 | -0.410 | ✓ |
| 15 | `5m_vol_rel` | +0.302 | +0.349 | ✓ |
| 16 | `15s_wick_ratio` | +0.269 | +0.196 | ✓ |
| 17 | `5m_dmi_diff` | -0.249 | -0.369 | ✓ |
| 18 | `5m_variance_ratio` | +0.247 | +0.310 | ✓ |
| 19 | `1h_reversion_prob` | -0.231 | -0.218 | ✓ |
| 20 | `15m_dmi_gap` | +0.206 | +0.107 | ✓ |

## Phase 2 — polynomial feature expansion

Expanding 6 single features: `1h_variance_ratio`, `15m_hurst`, `5m_z_se`, `5m_dmi_diff`, `5m_wick_ratio`, `15s_dmi_diff`

Expanded to 27 polynomial features (6 linear + 6 squared + 15 pair products).

IS: LONG=1,423,322, SHORT=1,445,015
OOS: LONG=395,651, SHORT=406,024

### Top 30 polynomial features — LONG vs SHORT

| Rank | Feature | d_IS | d_OOS | Walk-forward |
|---:|---|---:|---:|---|
| 1 | `5m_z_se * 15s_dmi_diff` | +0.013 | +0.002 | — |
| 2 | `1h_variance_ratio * 15m_hurst` | -0.012 | +0.011 | — |
| 3 | `1h_variance_ratio * 5m_wick_ratio` | -0.011 | +0.001 | — |
| 4 | `15m_hurst * 5m_wick_ratio` | -0.011 | -0.001 | — |
| 5 | `1h_variance_ratio` | -0.010 | +0.008 | — |
| 6 | `15m_hurst` | -0.010 | +0.009 | — |
| 7 | `5m_z_se` | +0.009 | -0.019 | — |
| 8 | `15m_hurst^2` | -0.009 | +0.007 | — |
| 9 | `1h_variance_ratio * 15s_dmi_diff` | +0.009 | -0.014 | — |
| 10 | `1h_variance_ratio^2` | -0.008 | +0.006 | — |
| 11 | `5m_dmi_diff * 5m_wick_ratio` | +0.008 | +0.001 | — |
| 12 | `15m_hurst * 15s_dmi_diff` | +0.008 | -0.013 | — |
| 13 | `15m_hurst * 5m_z_se` | +0.008 | -0.021 | — |
| 14 | `5m_dmi_diff` | +0.008 | +0.008 | — |
| 15 | `5m_wick_ratio` | -0.008 | -0.005 | — |
| 16 | `1h_variance_ratio * 5m_dmi_diff` | +0.008 | -0.005 | — |
| 17 | `15s_dmi_diff^2` | +0.007 | +0.018 | — |
| 18 | `15s_dmi_diff` | +0.007 | -0.011 | — |
| 19 | `15m_hurst * 5m_dmi_diff` | +0.006 | +0.007 | — |
| 20 | `5m_z_se^2` | +0.006 | +0.005 | — |
| 21 | `5m_z_se * 5m_wick_ratio` | +0.006 | -0.020 | — |
| 22 | `5m_wick_ratio^2` | -0.005 | -0.008 | — |
| 23 | `1h_variance_ratio * 5m_z_se` | +0.005 | -0.022 | — |
| 24 | `5m_z_se * 5m_dmi_diff` | -0.004 | +0.006 | — |
| 25 | `5m_wick_ratio * 15s_dmi_diff` | +0.003 | -0.011 | — |
| 26 | `5m_dmi_diff * 15s_dmi_diff` | -0.002 | -0.012 | — |
| 27 | `5m_dmi_diff^2` | -0.001 | +0.005 | — |

Walk-forward stable polynomial features: **0** of 27
