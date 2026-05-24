**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**

# Tier Exit Physics — KILL_SHOT

## 1. Cohort summary

- **N:** 297  WR **61.6%**  (winners 183 / losers 114)
- **Avg PnL:** winner $+18.52, loser $-32.75  (asymmetry 1.77×)
- **Median PnL:** winner $+14.00, loser $-16.75
- **Avg peak:** winner $+20.78, loser $+5.21
- **Median hold:** winner 4m, loser 30m
- **Total PnL:** winners $+3,388, losers $-3,734, net $-345

## 2. Peak timing — when does max pnl occur?

| Cohort | n | mean | p25 | p50 | p75 | p90 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| winners | 183 | 6.5 | 1 | 3 | 9 | 17 | 23 |
| losers | 114 | 4.7 | 0 | 1 | 6 | 18 | 23 |

_Bar at which the trade reached its maximum PnL. Low p50 = winners peak fast (capture early). High p50 = winners develop slowly (hold longer)._

## 3. Bar-N trajectory (median PnL / MFE / MAE by cohort)

| bar | n_w | n_l | pnl_w | pnl_l | mfe_w | mfe_l | mae_w | mae_l |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 181 | 114 | $+1.0 | $-3.0 | $+1.5 | $+0.0 | $+0.0 | $-3.5 |
| 2 | 138 | 97 | $+0.5 | $-6.0 | $+2.5 | $+0.0 | $-2.5 | $-6.5 |
| 3 | 112 | 96 | $+1.0 | $-7.0 | $+3.5 | $+0.0 | $-6.0 | $-11.0 |
| 5 | 86 | 95 | $-1.5 | $-11.5 | $+3.0 | $+0.0 | $-8.8 | $-16.0 |
| 7 | 70 | 87 | $-1.0 | $-10.8 | $+4.5 | $+0.0 | $-11.2 | $-22.0 |
| 10 | 53 | 80 | $-1.5 | $-21.0 | $+5.5 | $+0.0 | $-13.0 | $-26.5 |
| 15 | 30 | 74 | $+2.0 | $-25.0 | $+5.0 | $+1.5 | $-18.0 | $-35.2 |
| 20 | 21 | 67 | $-0.5 | $-25.8 | $+5.0 | $+1.5 | $-22.5 | $-40.0 |
| 25 | 12 | 63 | $-2.0 | $-26.5 | $+5.5 | $+1.5 | $-21.2 | $-48.0 |
| 30 | 9 | 60 | $+12.0 | $-28.5 | $+12.0 | $+1.8 | $-21.0 | $-51.0 |
| 40 | 0 | 1 | — | — | — | $+5.5 | — | $-16.0 |
| 60 | 0 | 1 | — | — | — | $+5.5 | — | $-16.0 |

**Fork bar (winner − loser ≥ $5):** bar 2  (winner $+0.5 vs loser $-6.0, spread $6.5)

## 4. Give-back from peak

| Cohort | n_with_peak | median retrace % | mean retrace % | round-trip losers |
|---|---:|---:|---:|---:|
| winners | 183 | 0% | 8% | 0 |
| losers | 65 | 393% | 740% | 39 |

_Retrace % = (peak − final) / peak. High retrace = winners give back gains before exit → trail stop helps. Round-trip losers = trades that had a peak ≥ $5 but exited negative → those were catchable with a tighter trail or trailing peak-rule gate._

## 5. Regression-mean slope β (1m horizon, 12 × 5s)

|β| is the magnitude of price drift over the last 60s. "% decayed" = fraction of trades where |β| < 0.05 (diminishing-returns threshold).

| Cohort | checkpoint | n | mean |β| | median |β| | % decayed |
|---|---|---:|---:|---:|---:|
| winners | entry | 183 | 0.435 | 0.271 | 10% |
| winners | peak | 183 | 0.484 | 0.332 | 7% |
| winners | exit | 183 | 0.448 | 0.299 | 9% |
| losers | entry | 114 | 0.374 | 0.182 | 22% |
| losers | peak | 114 | 0.407 | 0.185 | 15% |
| losers | exit | 114 | 0.413 | 0.261 | 12% |

_Winners entry→peak → |β| growth = riding acceleration. Winners peak→exit → |β| decay = natural slow-down (use as trailing exit signal). Losers entry β tells us whether the setup fires in a stagnant or trending regime._

## 6. Cut-rule scan — (bar × peak threshold) loser%−winner% delta

Each cell = (winner% < thr) / (loser% < thr) / Δ. `**` = Δ ≥ 20pp (strong cut candidate). `*` = Δ ≥ 15pp (moderate).

| bar | <$1 | <$3 | <$5 | <$10 | <$15 | <$20 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 46/79/+33 ** | 56/86/+30 ** | 63/94/+31 ** | 77/97/+21 ** | 86/100/+14 | 91/100/+9 |
| 2 | 43/67/+24 ** | 51/77/+26 ** | 61/88/+27 ** | 77/96/+19  * | 86/98/+12 | 88/99/+11 |
| 3 | 37/62/+26 ** | 46/76/+31 ** | 56/83/+27 ** | 79/96/+17  * | 88/98/+10 | 94/99/+5 |
| 5 | 35/55/+20  * | 45/69/+24 ** | 58/81/+23 ** | 79/95/+16  * | 88/96/+7 | 92/97/+5 |
| 7 | 26/56/+31 ** | 33/68/+35 ** | 51/78/+27 ** | 76/97/+21 ** | 81/98/+16  * | 90/98/+8 |
| 10 | 25/54/+29 ** | 32/66/+34 ** | 45/75/+30 ** | 70/96/+26 ** | 77/98/+20 ** | 89/98/+9 |
| 15 | 23/43/+20  * | 37/59/+23 ** | 47/72/+25 ** | 67/97/+31 ** | 73/99/+25 ** | 80/99/+19  * |
| 20 | 19/43/+24 ** | 29/54/+25 ** | 48/70/+23 ** | 71/97/+26 ** | 76/99/+22 ** | 76/99/+22 ** |
| 25 | 25/44/+19  * | 33/56/+22 ** | 50/68/+18  * | 67/97/+30 ** | 75/98/+23 ** | 83/98/+15  * |
| 30 | 11/43/+32 ** | 11/55/+44 ** | 22/65/+43 ** | 44/98/+54 ** | 67/98/+32 ** | 78/98/+21 ** |
| 40 | — | — | — | — | — | — |
| 60 | — | — | — | — | — | — |

## 7. Entry-time discrimination (Cohen d: winners vs losers at entry)

| feature | d | W mean | L mean |
|---|---:|---:|---:|
| 1D_wick_ratio | -0.450  * | 0.456 | 0.583 |
| 1D_variance_ratio | +0.332  * | 0.497 | 0.413 |
| 1D_dmi_diff | -0.300  * | -1.454 | 2.616 |
| 1m_acceleration | -0.230 | -1.059 | 2.816 |
| 15m_acceleration | -0.230 | -0.128 | 0.156 |
| 15s_p_at_center | -0.226 | 0.333 | 0.388 |
| 1m_velocity | -0.222 | -1.321 | 2.724 |
| 1D_z_low | -0.197 | -0.722 | -0.505 |
| 15m_wick_ratio | +0.197 | 0.888 | 0.876 |
| 15m_hurst | -0.184 | 0.684 | 0.703 |
| 15s_wick_ratio | +0.174 | 0.450 | 0.398 |
| 1D_z_se | -0.173 | 0.004 | 0.194 |
| 1h_bar_range | +0.173 | 281.842 | 242.096 |
| 1D_hurst | +0.168 | 0.678 | 0.663 |
| 1m_hurst | -0.165 | 0.685 | 0.702 |

_Moderate separator. Entry gate marginal; exit rules preferred._

## 8. Peak-signature features (entry→peak delta, Cohen d/σ)

_Peak signature unavailable — path features were stripped from pickle (run single-tier for features-intact pickle).  Re-run with `--tier KILL_SHOT` to populate this section._

## 9. Rule candidates (synthesized)

- **Cut candidate:** `bars_held >= 30 AND peak_pnl < $10`  (Δ=+54pp, cuts 98% of losers / 44% of winners)
- **Winner peak p50/p90:** 3m / 17m — timeout candidate at p90 + buffer ≈ 22m

---
_Generated by `tools/tier_exit_physics.py --tier KILL_SHOT`_