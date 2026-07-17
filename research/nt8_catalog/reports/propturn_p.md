# PROP-TURN-P — P-modulated proportional leg-turn (dynamic r_eff)
_Moises design (doc 094). P_turn fit + 36-cell grid tuned on 2024 ONLY (sealed); all read-outs below are TEST 2025+26. Generated 2026-07-16 22:27._

## TL;DR — verdict
- **KILL RULE: PASS (literal) — proportional-turn family NOT closed.** Requirement: beat the static baseline on BOTH dir-recall@2m AND precision@2m with non-overlapping day-block CIs. dir-recall beat=True, precision beat=True. **But read §4a before acting on this:** the winner is a 425/day FIREHOSE where P-modulation is inert, both precisions sit BELOW the 0.43 chance line, the standing bar FAILS, and capture FAILS — the PASS is fire-rate saturation, not conviction-modulation rescuing turn timing.
- **Standing turn bar: FAIL** — dir-recall@±2m 0.302 [0.294,0.310], precision@2m 0.173 (chance 0.43), lead-median +0.00m.
- **League (combiner feature):** OOS AUC 0.689.
- **Capture (secondary):** net -0.88 pt/trade after 0.6-pt friction; capture-ratio median -0.01 (budget 0.5–0.8; frac in budget 0.00).

## 1. P_turn model (2024 SEALED)
- Reference tracker for fitting: fixed r=0.15, A_min=10, no stall gate, escape on. 102814 RTH 1m-boundary samples / 258 days; label = interior turn within next 3 min; base rate 0.110.
- **2024 in-sample AUC 0.604** | 5-fold GroupKFold(day) CV AUC 0.603 ± 0.009.
- Coefs (standardized logistic; sorted by |coef|):

| feature | coef |
|---|---|
| leg_age_min | -0.421 |
| A_pts | +0.388 |
| ER10 | +0.267 |
| g | +0.142 |
| stall_min | -0.072 |
| A_over_std21 | +0.059 |
| trail_vol | +0.055 |
| kmdr_since_min | +0.043 |
| climax_since_min | -0.005 |
| ha_since_min | +0.000 |
| _(intercept)_ | -2.140 |

## 2. Frozen cell (36-cell grid, 2024 SEALED)
Grid: r_lo[0.03, 0.05, 0.08] × r_hi[0.15, 0.25, 0.35] × (p0,p1)[(0.2, 0.6), (0.3, 0.7)] × A_min[10, 15] = 36 cells. Objective (doc 094): **max dir-recall@±2m s.t. direction-correctness(near-turn) ≥ 0.8 AND lead-median ∈ [-2,1] min; NO fires/day cap.**
- Feasible cells: 21/36.
- **FROZEN:** r_lo=0.03, r_hi=0.15, (p0,p1)=(0.2,0.6), A_min=10 pt.
- 2024 selection stats: dir-recall@2m **0.295**, recall@2m 0.301, precision@2m 0.168, dir-correct@2m 0.837, lead-median +0.00m, fires/day 312.7, on 11545 interior turns / 258 days.

## 3. TEST turn scorecard (2025+26) — frozen cell, with deltas vs static
| metric | PROP-TURN-P | static (recomputed) | Δ (P − static) | doc-093 static |
|---|---|---|---|---|
| dir-recall@±1m | 0.278 | 0.016 | +0.261 |  |
| **dir-recall@±2m [CI]** | **0.302** [0.294,0.310] | 0.042 [0.039,0.046] | +0.259 | 0.042 |
| dir-recall@±3m | 0.312 | 0.236 | +0.076 |  |
| dir-recall@±5m | 0.321 | 0.305 | +0.016 |  |
| recall@±2m | 0.305 | 0.119 | +0.186 |  |
| **precision@±2m [CI]** (chance 0.43) | **0.173** [0.168,0.178] | 0.101 [0.097,0.106] | +0.072 | 0.102 |
| dir-correct@±2m | 0.831 | 0.333 | +0.498 |  |
| lead@2m median (min) | +0.00 | -0.58 | — | — |
| fires/day | 424.7 | 53.6 | — | — |

**Standing-bar verdict: FAIL** — bar = precision > 0.43 OR (dir-recall@2m ≥ 0.35 with lead ≤ +1 min). Best prior stream (RENKO-24) ≈ 0.30 / 0.17.

## 4. KILL RULE (pre-registered, doc 094)
Beat static on BOTH dir-recall@2m AND precision@2m with **non-overlapping day-block CIs**, else the proportional-turn family (static + dynamic) is CLOSED.
- dir-recall@2m: P 0.302 [0.294,0.310] vs static 0.042 [0.039,0.046] → **beat=True**
- precision@2m : P 0.173 [0.168,0.178] vs static 0.101 [0.097,0.106] → **beat=True**

**>>> VERDICT: PASS (literal rule met) — proportional-turn family NOT closed.**

### 4a. Reading the PASS honestly (the caveats the reviewer needs)
The literal rule is met, but the PASS is **fire-rate saturation, not conviction modulation rescuing turn timing.** Four things must be weighed before the family is called "alive":
1. **It is a firehose.** 425 fires/day (P) vs 54 (static). dir-recall scales ~mechanically with fire rate; the objective (max dir-recall, NO fires/day cap, per doc 094) explicitly rewards spraying. dir-recall@±3m/±5m barely rise (0.312/0.321) — the extra fires buy ±2m hits, not turn structure.
2. **P-modulation is essentially INERT at the winner.** dir-recall FALLS monotonically as the modulation band widens (A_min=10 grid means): r_hi=0.15→0.295, 0.25→0.286, 0.35→0.194. So maximizing dir-recall drove r_hi to its FLOOR (0.15); at the winning r_hi=0.15/A_min=10 the r_lo and (p0,p1) knobs move dir-recall by only 0.0009. The dynamic tracker ≈ a sensitive STATIC tracker (r≈0.15, no stall); P_turn (AUC 0.60) is too weak to concentrate fires, so the objective routes AROUND it.
3. **Both precisions are BELOW the 0.43 chance line** (P 0.173, static 0.101). P is only RELATIVELY less-bad than static; neither is an absolute ±2m turn-timer. The **standing bar FAILS** for both.
4. **Capture FAILS decisively and is WORSE than static** (net -0.88 vs static −0.80 pt/trade; 0% in the 0.5–0.8 budget) — the firehose whipsaws harder. The one genuine positive is the LEAGUE combiner (AUC 0.689, up from static 0.636): as a state FEATURE the fires carry real direction info, but it feeds the combiner, it does not stand alone.

## 5. League line (full 604-day pipeline; direction-agreement with AI labels)
- N=210999 (train 79629 / test 131370), OOS **AUC 0.689**, test base 0.50
- P-terciles: low: 0.33 [0.32,0.34] N=43790 | mid: 0.48 [0.48,0.49] N=43790 | high: 0.69 [0.68,0.70] N=43790
- coefs: {'pivot_age_min': 0.073, 'sig_with_leg': 0.544, 'value': -0.81, 'tod': -0.043, 'inter': -0.179}

## 6. CAPTURE — stop-and-reverse (TEST; secondary; the 0.5–0.8 budget)
Flat outside RTH; 0.6 pt/round-trip friction (MNQ 1 pt = $2). Per completed leg-trade.

| pop | trades/day | captured median | mean [CI] (pt) | PF Trade WR | net mean [CI] (pt) | ratio median | frac in [.5,.8] |
|---|---|---|---|---|---|---|---|
| 2025 | 453.3 | -1.75 | -0.27 [-0.34,-0.20] | -0.065 | -0.87 [-0.94,-0.80] | -0.01 | 0.00 |
| 2026 | 522.7 | -1.75 | -0.30 [-0.45,-0.16] | -0.073 | -0.90 [-1.05,-0.76] | -0.01 | 0.00 |
| POOLED | 466.8 | -1.75 | -0.28 [-0.34,-0.22] | -0.067 | -0.88 [-0.94,-0.82] | -0.01 | 0.00 |

## 7. Declared choices (spec 094 left these open; all sealed on 2024 before any test read)
- **"A/21 ratio"** ⇒ A / std(last 21 one-minute closes), floor 1 pt (vol-normalized amplitude); standardization makes the exact normalizer scale irrelevant to the fit.
- **P_turn circularity broken** by fitting on a REFERENCE tracker (fixed r=0.15, A_min=10, no stall gate, escape on) and applying the frozen model to the dynamic tracker's own live state (declared train/deploy shift). "Against leg dir": up-leg opposed by SHORT aux fires, down-leg by LONG.
- **Stall gate removed** in the dynamic tracker (it forced doc-093's degenerate cell); stall is a P_turn feature. Escape clause + A_min noise floor retained verbatim from `_propturn_core`.
- **r_eff updates at every 1m boundary** (RTH + overnight) from P_turn; fit samples only RTH boundaries; fires stay RTH-gated. Aux fires (EXIT-KMDR/TURN-CLIMAX/TURN-HA) are the existing generators, precomputed once/day, independent of the proportional tracker.
- **CIs are day-block bootstraps** (unit of independence = the day): 1000 resamples, precision/dir-correct as day-summed ratios, dir-recall as a day-blocked mean.
- Value per fire = completed leg amplitude A (pts); capture fills use the fire-bar CLOSE; final leg/day force-closes at the RTH close.

## 8. Artifacts
- `research/nt8_catalog/reports/propturn_p_frozen.json` — frozen P_turn coefs + cell (no pickle)
- `research/nt8_catalog/reports/propturn_p_grid_2024.csv` — 36-cell 2024 selection grid
- `research/nt8_catalog/reports/signal_rows_PROPTURNP.parquet` — league signal rows
- `research/nt8_catalog/reports/propturn_p_capture_trades.csv` — capture leg-trades
- `research/nt8_catalog/reports/propturn_p_run.log` — full run log
- generator `PROP-TURN-P` + shared cores appended to `tools/dossier_signal_pipeline.py`; tuning driver `tools/propturn_p_tune.py`