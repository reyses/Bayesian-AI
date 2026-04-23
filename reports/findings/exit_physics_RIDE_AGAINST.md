# Tier Exit Physics — RIDE_AGAINST

## 1. Cohort summary

- **N:** 4,182  WR **65.0%**  (winners 2,719 / losers 1,463)
- **Avg PnL:** winner $+20.89, loser $-36.80  (asymmetry 1.76×)
- **Median PnL:** winner $+15.00, loser $-22.00
- **Avg peak:** winner $+23.37, loser $+4.42
- **Median hold:** winner 4m, loser 15m
- **Total PnL:** winners $+56,788, losers $-53,844, net $+2,944

## 2. Peak timing — when does max pnl occur?

| Cohort | n | mean | p25 | p50 | p75 | p90 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| winners | 2719 | 5.8 | 2 | 4 | 8 | 13 | 15 |
| losers | 1463 | 2.3 | 0 | 1 | 3 | 8 | 11 |

_Bar at which the trade reached its maximum PnL. Low p50 = winners peak fast (capture early). High p50 = winners develop slowly (hold longer)._

## 3. Bar-N trajectory (median PnL / MFE / MAE by cohort)

| bar | n_w | n_l | pnl_w | pnl_l | mfe_w | mfe_l | mae_w | mae_l |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2711 | 1463 | $+2.0 | $-4.0 | $+2.0 | $+0.0 | $+0.0 | $-4.0 |
| 2 | 2244 | 1451 | $+2.0 | $-7.0 | $+4.0 | $+0.0 | $-2.0 | $-8.5 |
| 3 | 1834 | 1434 | $+2.0 | $-9.5 | $+5.0 | $+0.0 | $-4.5 | $-12.0 |
| 5 | 1346 | 1390 | $+1.0 | $-14.0 | $+5.5 | $+0.0 | $-7.5 | $-18.5 |
| 7 | 1019 | 1350 | $+1.5 | $-17.5 | $+6.5 | $+0.0 | $-9.5 | $-23.5 |
| 10 | 723 | 1280 | $+4.0 | $-21.0 | $+7.5 | $+0.0 | $-11.5 | $-29.2 |
| 15 | 382 | 1185 | $+8.0 | $-26.2 | $+9.5 | $+0.0 | $-14.5 | $-38.0 |
| 20 | 21 | 12 | $+25.5 | $-24.5 | $+6.5 | $+0.0 | $-3.5 | $-7.5 |
| 25 | 16 | 11 | $+21.5 | $-9.0 | $+5.8 | $+0.0 | $-4.5 | $-9.0 |
| 30 | 15 | 10 | — | — | $+5.5 | $+0.0 | $-3.5 | $-7.5 |
| 40 | 15 | 10 | — | — | $+5.5 | $+0.0 | $-3.5 | $-7.5 |
| 60 | 15 | 10 | — | — | $+5.5 | $+0.0 | $-3.5 | $-7.5 |

**Fork bar (winner − loser ≥ $5):** bar 1  (winner $+2.0 vs loser $-4.0, spread $6.0)

## 4. Give-back from peak

| Cohort | n_with_peak | median retrace % | mean retrace % | round-trip losers |
|---|---:|---:|---:|---:|
| winners | 2719 | 0% | 9% | 0 |
| losers | 777 | 353% | 975% | 465 |

_Retrace % = (peak − final) / peak. High retrace = winners give back gains before exit → trail stop helps. Round-trip losers = trades that had a peak ≥ $5 but exited negative → those were catchable with a tighter trail or trailing peak-rule gate._

## 5. Regression-mean slope β (1m horizon, 12 × 5s)

|β| is the magnitude of price drift over the last 60s. "% decayed" = fraction of trades where |β| < 0.05 (diminishing-returns threshold).

| Cohort | checkpoint | n | mean |β| | median |β| | % decayed |
|---|---|---:|---:|---:|---:|
| winners | entry | 2719 | 0.483 | 0.283 | 12% |
| winners | peak | 2719 | 0.502 | 0.304 | 10% |
| winners | exit | 2719 | 0.503 | 0.308 | 10% |
| losers | entry | 1463 | 0.455 | 0.247 | 12% |
| losers | peak | 1463 | 0.446 | 0.261 | 13% |
| losers | exit | 1463 | 0.434 | 0.269 | 12% |

_Winners entry→peak → |β| growth = riding acceleration. Winners peak→exit → |β| decay = natural slow-down (use as trailing exit signal). Losers entry β tells us whether the setup fires in a stagnant or trending regime._

## 6. Cut-rule scan — (bar × peak threshold) loser%−winner% delta

Each cell = (winner% < thr) / (loser% < thr) / Δ. `**` = Δ ≥ 20pp (strong cut candidate). `*` = Δ ≥ 15pp (moderate).

| bar | <$1 | <$3 | <$5 | <$10 | <$15 | <$20 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44/75/+30 ** | 53/83/+30 ** | 62/89/+27 ** | 75/97/+21 ** | 84/98/+14 | 89/99/+9 |
| 2 | 35/67/+32 ** | 44/76/+32 ** | 54/83/+29 ** | 72/95/+23 ** | 82/97/+15 | 88/98/+10 |
| 3 | 30/62/+32 ** | 40/71/+31 ** | 50/80/+30 ** | 71/94/+23 ** | 82/96/+15 | 87/98/+10 |
| 5 | 27/59/+32 ** | 37/68/+31 ** | 47/78/+31 ** | 70/95/+25 ** | 78/97/+18  * | 84/98/+14 |
| 7 | 22/57/+35 ** | 32/66/+35 ** | 43/77/+34 ** | 70/97/+27 ** | 78/97/+19  * | 83/98/+16  * |
| 10 | 17/55/+38 ** | 25/65/+40 ** | 34/75/+40 ** | 63/97/+35 ** | 71/98/+27 ** | 79/99/+20  * |
| 15 | 2/54/+51 ** | 7/64/+57 ** | 15/74/+59 ** | 51/98/+47 ** | 66/99/+33 ** | 72/99/+27 ** |
| 20 | 24/75/+51 ** | 24/75/+51 ** | 33/75/+42 ** | 62/92/+30 ** | 76/92/+15  * | 76/100/+24 ** |
| 25 | 31/73/+41 ** | 31/73/+41 ** | 44/73/+29 ** | 75/82/+7 | 88/91/+3 | 88/100/+12 |
| 30 | 33/80/+47 ** | 33/80/+47 ** | 47/80/+33 ** | 80/90/+10 | 93/90/-3 | 93/100/+7 |
| 40 | 33/80/+47 ** | 33/80/+47 ** | 47/80/+33 ** | 80/90/+10 | 93/90/-3 | 93/100/+7 |
| 60 | 33/80/+47 ** | 33/80/+47 ** | 47/80/+33 ** | 80/90/+10 | 93/90/-3 | 93/100/+7 |

## 7. Entry-time discrimination (Cohen d: winners vs losers at entry)

| feature | d | W mean | L mean |
|---|---:|---:|---:|
| 1m_dmi_diff | -0.127 | -2.320 | -0.122 |
| 1h_bar_range | +0.123 | 326.110 | 294.422 |
| 15m_dmi_diff | -0.122 | -1.982 | -0.267 |
| 5m_dmi_diff | -0.121 | -1.898 | -0.340 |
| 1h_dmi_diff | -0.120 | -2.488 | -0.679 |
| 15m_bar_range | +0.117 | 149.249 | 134.189 |
| 1D_z_se | -0.114 | -0.091 | 0.036 |
| 1h_vol_rel | +0.110 | 1.024 | 0.898 |
| 1m_dir_vol | -0.107 | -0.180 | 0.099 |
| 15s_dmi_diff | -0.106 | -2.555 | 0.170 |
| 1D_z_low | -0.105 | -0.819 | -0.700 |
| 5m_bar_range | +0.105 | 90.337 | 82.153 |
| 15s_bar_range | +0.103 | 24.370 | 21.493 |
| 1m_bar_range | +0.102 | 68.894 | 61.848 |
| 15m_p_at_center | -0.099 | 0.507 | 0.530 |

_No entry discrimination. Winners and losers identical at entry → the fix is an exit rule, not an entry filter._

## 8. Peak-signature features (entry→peak delta, Cohen d/σ)

Top features where value shifts most from entry to peak.  `d/σ > 2` is a strong signal; `> 5` is dominant.

| feature | d/σ | mean Δ | std Δ | n |
|---|---:|---:|---:|---:|
| 1m_p_at_center | +1.45 | +0.386 | 0.266 | 4204 |
| time_of_day | +0.92 | +0.004 | 0.004 | 4204 |
| 1m_reversion_prob | +0.89 | +0.219 | 0.247 | 4204 |
| 15s_dmi_gap | -0.87 | -11.358 | 13.089 | 4204 |
| 1m_wick_ratio | +0.75 | +0.229 | 0.306 | 4204 |
| 15s_variance_ratio | -0.71 | -0.306 | 0.429 | 4204 |
| 1m_vol_rel | -0.60 | -0.880 | 1.468 | 4204 |
| 15s_vol_rel | -0.45 | -0.785 | 1.730 | 4204 |
| 1m_dmi_gap | -0.44 | -3.714 | 8.451 | 4204 |
| 1m_bar_range | -0.42 | -20.496 | 49.093 | 4204 |
| 15s_p_at_center | +0.34 | +0.104 | 0.309 | 4204 |
| 5m_dmi_gap | +0.28 | +1.404 | 5.006 | 4204 |
| 15s_reversion_prob | +0.27 | +0.051 | 0.188 | 4204 |
| 1m_z_low | +0.25 | +0.488 | 1.922 | 4204 |
| 15s_bar_range | -0.25 | -5.319 | 21.264 | 4204 |

## 9. Rule candidates (synthesized)

- **Cut candidate:** `bars_held >= 15 AND peak_pnl < $5`  (Δ=+59pp, cuts 74% of losers / 15% of winners)
- **Winner peak p50/p90:** 4m / 13m — timeout candidate at p90 + buffer ≈ 18m

---
_Generated by `tools/tier_exit_physics.py --tier RIDE_AGAINST`_