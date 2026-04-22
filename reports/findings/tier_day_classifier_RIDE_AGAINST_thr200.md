# Tier day classifier — RIDE_AGAINST

Threshold: BLEED = tier PnL <= -$200.0, HARVEST = tier PnL >= +$200.0.

## 1. Cohort stats

| Dataset | Class | Days | Total $ | Mean $ | Median $ | Trades |
|---|---|---:|---:|---:|---:|---:|
| IS | bleed | 92 | $-68,134 | $-741 | $-632 | 17,409 |
| IS | harvest | 76 | $+51,756 | $+681 | $+528 | 15,214 |
| IS | neutral | 105 | $-91 | $-1 | $-4 | 13,752 |
| OOS | bleed | 15 | $-11,366 | $-758 | $-558 | 3,228 |
| OOS | harvest | 25 | $+18,337 | $+733 | $+703 | 5,481 |
| OOS | neutral | 26 | $-495 | $-19 | $-33 | 3,324 |

_The bleed/harvest $ totals are what's at stake if we can separate them with a day-level filter._

## 2. Top-20 day features — IS

(N bleed=92, N harvest=76) Positive d => feature is HIGHER on bleed days.

| Rank | Feature | d | bleed mean | harvest mean |
|---:|---|---:|---:|---:|
| 1 | `day_mean_trade_pnl` | -1.25 | -3.697 | +3.182 |
| 2 | `mean_5m_variance_ratio` | +0.75 | +0.477 | +0.432 |
| 3 | `mean_15m_variance_ratio` | +0.56 | +0.511 | +0.441 |
| 4 | `mean_1h_z_range` | +0.42 | +1.485 | +1.329 |
| 5 | `day_entry_range` | -0.39 | +356.788 | +473.388 |
| 6 | `day_first_hour_frac` | -0.36 | +0.599 | +0.612 |
| 7 | `mean_1D_acceleration` | -0.30 | -0.764 | +0.584 |
| 8 | `mean_1h_variance_ratio` | -0.30 | +0.427 | +0.491 |
| 9 | `mean_1D_velocity` | -0.29 | -58.043 | +41.148 |
| 10 | `mean_1h_acceleration` | +0.29 | +0.114 | -0.098 |
| 11 | `mean_1h_z_low` | -0.29 | -0.852 | -0.700 |
| 12 | `mean_5m_p_at_center` | -0.29 | +0.492 | +0.501 |
| 13 | `mean_1h_dir_vol` | -0.28 | -0.087 | +0.033 |
| 14 | `mean_15s_velocity` | +0.24 | +0.076 | -0.016 |
| 15 | `mean_1h_hurst` | -0.24 | +0.707 | +0.727 |
| 16 | `mean_5m_velocity` | -0.22 | -1.236 | +0.265 |
| 17 | `day_target_tier_share` | -0.22 | +0.527 | +0.537 |
| 18 | `mean_1D_dmi_diff` | -0.22 | -4.239 | -1.025 |
| 19 | `day_target_tier_n` | -0.22 | +189.228 | +200.184 |
| 20 | `mean_1m_wick_ratio` | -0.21 | +0.527 | +0.533 |

## 3. Top-20 day features — OOS

(N bleed=15, N harvest=25)

| Rank | Feature | d | bleed mean | harvest mean |
|---:|---|---:|---:|---:|
| 1 | `day_mean_trade_pnl` | -1.96 | -2.066 | +2.624 |
| 2 | `mean_15m_dmi_gap` | -1.35 | +8.796 | +12.506 |
| 3 | `mean_1h_reversion_prob` | +1.09 | +0.952 | +0.918 |
| 4 | `mean_1h_variance_ratio` | -0.81 | +0.394 | +0.535 |
| 5 | `mean_5m_dmi_gap` | -0.76 | +9.918 | +11.082 |
| 6 | `mean_1D_variance_ratio` | +0.74 | +0.710 | +0.545 |
| 7 | `mean_5m_variance_ratio` | +0.69 | +0.460 | +0.418 |
| 8 | `mean_5m_vol_rel` | -0.66 | +1.007 | +1.058 |
| 9 | `mean_1m_z_low` | +0.63 | -0.758 | -0.800 |
| 10 | `mean_1h_velocity` | +0.61 | +5.448 | -4.895 |
| 11 | `mean_15m_hurst` | -0.61 | +0.696 | +0.724 |
| 12 | `mean_1D_acceleration` | +0.59 | +0.336 | -0.225 |
| 13 | `mean_1D_bar_range` | +0.58 | +2020.600 | +1511.480 |
| 14 | `mean_1m_z_se` | +0.56 | +0.026 | -0.016 |
| 15 | `mean_1h_z_range` | +0.56 | +1.440 | +1.281 |
| 16 | `day_entry_range` | -0.54 | +393.700 | +502.600 |
| 17 | `mean_15m_reversion_prob` | +0.54 | +0.943 | +0.935 |
| 18 | `mean_1D_hurst` | -0.53 | +0.641 | +0.704 |
| 19 | `mean_1D_z_low` | -0.52 | -1.224 | -0.638 |
| 20 | `mean_15m_dmi_diff` | +0.50 | -0.703 | -3.605 |

## 4. Walk-forward stable shortlist

Features where sign(d_IS) matches sign(d_OOS) AND min(|d_IS|, |d_OOS|) >= 0.30. Sorted by min |d| descending.

| Feature | d_IS | d_OOS | min |d| | IS bleed mean | IS harv mean | OOS bleed mean | OOS harv mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `day_mean_trade_pnl` | -1.25 | -1.96 | 1.25 | -3.697 | +3.182 | -2.066 | +2.624 |
| `mean_5m_variance_ratio` | +0.75 | +0.69 | 0.69 | +0.477 | +0.432 | +0.460 | +0.418 |
| `mean_1h_z_range` | +0.42 | +0.56 | 0.42 | +1.485 | +1.329 | +1.440 | +1.281 |
| `day_entry_range` | -0.39 | -0.54 | 0.39 | +356.788 | +473.388 | +393.700 | +502.600 |
| `mean_1h_variance_ratio` | -0.30 | -0.81 | 0.30 | +0.427 | +0.491 | +0.394 | +0.535 |

## 5. How to read this

- **Non-empty shortlist** = there is a real day-level signal to classify on. Next step: build a decision rule from the top 2-3 features (e.g. quantile split of d_IS>0.5 features) and backtest it on OOS.
- **Empty shortlist** = day outcomes are not predictable from aggregated market state alone. The bleed/harvest split comes from path-dependent noise within the day, not entry-time regime. If so, day-level classification is the wrong frame — move to path-based signals (e.g. drawdown in first N trades of the day as an intraday kill switch).
