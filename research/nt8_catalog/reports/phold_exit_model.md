# P_hold(tau) — during-trade confidence on the full V2 F-space

Binary logistic run DURING the open trade on the full V2 vector, no fixed horizon. P_hold = P(active AI label still agrees with entry direction). The confidence should decay as the entry-direction move turns over.

## Population
- entry-P p90 threshold (frozen on train 2024) = **0.76023**
- train: 31260 fires >= thr -> **13292 engagements** after 60s/day/dir de-dup
- test : 64661 fires >= thr -> **24083 engagements** after de-dup
- during-trade rows saved: 990294 (train+test); test rows 642634 across 24078 engagements
- full V2 F-space = **409** feature columns (41 families) + 4 context + nan_count

## Readout 1 — OOS AUC (test): FULL vs BASELINE (context-only bar)
- **overall FULL 0.6381  vs  BASELINE 0.6846  (delta -0.0465)**; day-block 95% CI on delta [-0.0582, -0.0350]

| tau bucket | N | FULL AUC | BASE AUC | delta |
|---|---|---|---|---|
| 1-5 | 120347 | 0.6170 | 0.7316 | -0.1146 |
| 6-10 | 120186 | 0.6537 | 0.7587 | -0.1050 |
| 11-20 | 180544 | 0.6486 | 0.7064 | -0.0579 |
| 21-40 | 166025 | 0.6363 | 0.6450 | -0.0087 |
| 41-60 | 55532 | 0.6075 | 0.5950 | +0.0125 |

**KILL-POINT A**: overall delta -0.0465 — **BELOW** the 0.05 house bar; the F-space adds ~nothing over trivial during-trade state (elapsed/drift/entry_P/trail_vol).

## Readout 2 — Calibration of FULL P_hold (deciles)
| decile | N | pred mean | obs mean | day-block 95% CI |
|---|---|---|---|---|
| 0 | 64264 | 0.145 | 0.365 | [0.316,0.411] |
| 1 | 64263 | 0.367 | 0.400 | [0.372,0.428] |
| 2 | 64263 | 0.493 | 0.471 | [0.445,0.497] |
| 3 | 64264 | 0.581 | 0.519 | [0.496,0.540] |
| 4 | 64263 | 0.650 | 0.564 | [0.544,0.583] |
| 5 | 64263 | 0.708 | 0.591 | [0.572,0.611] |
| 6 | 64264 | 0.760 | 0.622 | [0.604,0.641] |
| 7 | 64263 | 0.810 | 0.657 | [0.639,0.675] |
| 8 | 64263 | 0.862 | 0.700 | [0.682,0.719] |
| 9 | 64264 | 0.932 | 0.739 | [0.718,0.761] |

## Readout 3 — Decay curves: mean P_hold vs tau
(a) engagements not-yet-flipped at tau vs (b) already-flipped-so-far

| tau | N not-flipped | mean P (a) | N flipped | mean P (b) |
|---|---|---|---|---|
| 1 | 24078 | 0.669 | 0 | nan |
| 2 | 16660 | 0.692 | 7416 | 0.609 |
| 3 | 15691 | 0.698 | 8379 | 0.599 |
| 5 | 14031 | 0.704 | 10026 | 0.581 |
| 10 | 10597 | 0.717 | 13426 | 0.563 |
| 15 | 8096 | 0.729 | 10326 | 0.558 |
| 20 | 6216 | 0.737 | 7738 | 0.543 |
| 30 | 3628 | 0.748 | 4511 | 0.509 |
| 40 | 2062 | 0.747 | 2653 | 0.492 |
| 50 | 1165 | 0.741 | 1569 | 0.455 |
| 60 | 677 | 0.726 | 845 | 0.419 |

## Readout 4 — Flip lead-time (P_hold<0.5 sustained 2m − label-flip minute)
- N flipped engagements w/ sustained P<0.5 cross = 11324
- **mode +1.5 min | median +3.0 | p25 -3.0 | p75 +6.0 | mean +0.44** (negative = early warning)
- share with early warning (lead<=0): 0.287

## Readout 5 — Exit-policy captured displacement (points; mode-first)
| policy | N eng | mode | median | mean | day-block 95% CI | capture-ratio median |
|---|---|---|---|---|---|---|
| fixed 5-min hold (ref) | 24078 | -1.0 | +1.75 | +2.19 | [+1.58,+2.78] | +0.014 |
| P_hold<0.6 sust 2m | 24078 | -3.0 | -2.75 | +3.55 | [+2.09,+5.00] | -0.020 |
| P_hold<0.5 sust 2m | 24078 | -3.0 | -2.75 | +4.53 | [+2.85,+6.26] | -0.020 |
| ORACLE (label end) | 24078 | +1.0 | +27.50 | +28.49 | [+25.32,+31.65] | +0.230 |

**KILL-POINT B**: fixed-5m median +1.75 pts vs P<0.6 -2.75 / P<0.5 -2.75 — neither P_hold policy beats the fixed-5m median capture: the open-ended exit is **NOT yet earned**.

## Top-30 |coef| FULL features (F-space dims carrying the exit signal)
| coef | feature |
|---|---|
| +4.1041 | L5_4h_ldist_median |
| -3.4455 | L5_4h_ldist_level |
| +1.7974 | L5_1h_ldist_median |
| +1.7654 | L2_1h_vwap_30 |
| -1.3123 | L5_1h_ldist_level |
| +1.2689 | L5_15m_ldist_median |
| -1.2328 | L5_4h_ldist_q3 |
| -0.9972 | L2_15m_vwap_30 |
| +0.9925 | L5_15m_ldist_level |
| -0.9541 | L5_1h_ldist_min |
| -0.8712 | L4_1D_lambda_hat_30 |
| +0.8610 | L5_5m_ldist_max |
| +0.8417 | L4_1D_lambda_t_30 |
| -0.8263 | L5_4h_ldist_min |
| -0.8245 | L2_5m_vwap_30 |
| +0.7980 | L2_15s_vwap_30 |
| -0.7880 | L2_5m_price_mean_30 |
| +0.7818 | L5_4h_ldist_mean |
| -0.7557 | L2_1h_price_mean_30 |
| -0.7523 | L5_1h_ldist_max |
| +0.7449 | L5_15m_ldist_min |
| +0.7401 | L2_1D_price_mean_30 |
| +0.7313 | L3_1D_z_se_30 |
| +0.7089 | L2_1m_vwap_30 |
| +0.7002 | L5_5m_ldist_level |
| -0.6855 | L3_4h_z_se_30 |
| -0.6671 | L5_1h_ldist_q1 |
| +0.6528 | drift_so_far |
| +0.6392 | L5_15m_ldist_mean |
| -0.6264 | L5_4h_ldist_max |