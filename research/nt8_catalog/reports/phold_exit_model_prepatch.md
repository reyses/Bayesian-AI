# P_hold(tau) — during-trade confidence on the full V2 F-space

Binary logistic run DURING the open trade on the full V2 vector, no fixed horizon. P_hold = P(active AI label still agrees with entry direction). The confidence should decay as the entry-direction move turns over.

## Population
- entry-P p90 threshold (frozen on train 2024) = **0.76023**
- train: 31260 fires >= thr -> **13292 engagements** after 60s/day/dir de-dup
- test : 64661 fires >= thr -> **24083 engagements** after de-dup
- during-trade rows saved: 990294 (train+test); test rows 642634 across 24078 engagements
- full V2 F-space = **409** feature columns (41 families) + 4 context + nan_count

## Readout 1 — OOS AUC (test): FULL vs BASELINE (context-only bar)
- **overall FULL 0.6472  vs  BASELINE 0.6894  (delta -0.0422)**; day-block 95% CI on delta [-0.0538, -0.0309]

| tau bucket | N | FULL AUC | BASE AUC | delta |
|---|---|---|---|---|
| 1-5 | 120347 | 0.6273 | 0.7416 | -0.1143 |
| 6-10 | 120186 | 0.6646 | 0.7650 | -0.1004 |
| 11-20 | 180544 | 0.6590 | 0.7114 | -0.0524 |
| 21-40 | 166025 | 0.6430 | 0.6479 | -0.0049 |
| 41-60 | 55532 | 0.6138 | 0.5971 | +0.0167 |

**KILL-POINT A**: overall delta -0.0422 — **BELOW** the 0.05 house bar; the F-space adds ~nothing over trivial during-trade state (elapsed/drift/entry_P/trail_vol).

## Readout 2 — Calibration of FULL P_hold (deciles)
| decile | N | pred mean | obs mean | day-block 95% CI |
|---|---|---|---|---|
| 0 | 64264 | 0.147 | 0.359 | [0.312,0.404] |
| 1 | 64263 | 0.368 | 0.385 | [0.358,0.413] |
| 2 | 64263 | 0.495 | 0.462 | [0.437,0.486] |
| 3 | 64264 | 0.584 | 0.517 | [0.495,0.540] |
| 4 | 64263 | 0.654 | 0.558 | [0.538,0.578] |
| 5 | 64263 | 0.713 | 0.593 | [0.572,0.612] |
| 6 | 64264 | 0.766 | 0.626 | [0.607,0.645] |
| 7 | 64263 | 0.816 | 0.666 | [0.648,0.683] |
| 8 | 64263 | 0.869 | 0.708 | [0.689,0.728] |
| 9 | 64264 | 0.937 | 0.753 | [0.731,0.774] |

## Readout 3 — Decay curves: mean P_hold vs tau
(a) engagements not-yet-flipped at tau vs (b) already-flipped-so-far

| tau | N not-flipped | mean P (a) | N flipped | mean P (b) |
|---|---|---|---|---|
| 1 | 24078 | 0.674 | 0 | nan |
| 2 | 16660 | 0.699 | 7416 | 0.608 |
| 3 | 15691 | 0.705 | 8379 | 0.598 |
| 5 | 14031 | 0.711 | 10026 | 0.582 |
| 10 | 10597 | 0.726 | 13426 | 0.562 |
| 15 | 8096 | 0.739 | 10326 | 0.557 |
| 20 | 6216 | 0.747 | 7738 | 0.543 |
| 30 | 3628 | 0.758 | 4511 | 0.509 |
| 40 | 2062 | 0.757 | 2653 | 0.491 |
| 50 | 1165 | 0.752 | 1569 | 0.453 |
| 60 | 677 | 0.739 | 845 | 0.418 |

## Readout 4 — Flip lead-time (P_hold<0.5 sustained 2m − label-flip minute)
- N flipped engagements w/ sustained P<0.5 cross = 11367
- **mode +1.5 min | median +3.0 | p25 -2.0 | p75 +6.0 | mean +0.72** (negative = early warning)
- share with early warning (lead<=0): 0.271

## Readout 5 — Exit-policy captured displacement (points; mode-first)
| policy | N eng | mode | median | mean | day-block 95% CI | capture-ratio median |
|---|---|---|---|---|---|---|
| fixed 5-min hold (ref) | 24078 | +1.0 | +1.75 | +2.23 | [+1.61,+2.82] | +0.014 |
| P_hold<0.6 sust 2m | 24078 | -5.0 | -3.00 | +3.76 | [+2.24,+5.31] | -0.021 |
| P_hold<0.5 sust 2m | 24078 | -3.0 | -2.75 | +4.66 | [+2.95,+6.44] | -0.022 |
| ORACLE (label end) | 24078 | +1.0 | +27.50 | +28.49 | [+25.32,+31.65] | +0.230 |

**KILL-POINT B**: fixed-5m median +1.75 pts vs P<0.6 -3.00 / P<0.5 -2.75 — neither P_hold policy beats the fixed-5m median capture: the open-ended exit is **NOT yet earned**.

## Top-30 |coef| FULL features (F-space dims carrying the exit signal)
| coef | feature |
|---|---|
| +3.9472 | L5_4h_ldist_median |
| -3.4062 | L5_4h_ldist_level |
| +1.6812 | L2_1h_vwap_30 |
| +1.6751 | L5_1h_ldist_median |
| -1.3528 | L5_1h_ldist_level |
| +1.2257 | L5_15m_ldist_median |
| -1.1455 | L5_4h_ldist_q3 |
| -0.9963 | L5_1h_ldist_min |
| +0.9776 | L5_5m_ldist_max |
| +0.9544 | L5_15m_ldist_level |
| -0.9304 | L4_1D_lambda_hat_30 |
| +0.8927 | L4_1D_lambda_t_30 |
| -0.8731 | L2_15m_vwap_30 |
| -0.7915 | L2_5m_vwap_30 |
| -0.7882 | L5_1h_ldist_max |
| -0.7766 | L5_4h_ldist_min |
| +0.7642 | L5_4h_ldist_mean |
| +0.7343 | L2_1D_price_mean_30 |
| +0.7077 | L5_5m_ldist_min |
| -0.6940 | L2_5m_price_mean_30 |
| +0.6930 | L2_1m_vwap_30 |
| +0.6883 | L2_15s_vwap_30 |
| +0.6677 | drift_so_far |
| -0.6503 | L2_1h_price_mean_30 |
| -0.6502 | L5_1h_ldist_q1 |
| +0.6474 | L5_15m_ldist_min |
| +0.6053 | L5_5m_ldist_level |
| +0.6044 | L3_1D_z_se_30 |
| -0.6017 | L3_4h_z_se_30 |
| -0.5932 | L5_4h_ldist_max |