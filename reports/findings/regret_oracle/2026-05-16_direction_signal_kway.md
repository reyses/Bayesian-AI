# Direction signal — k-way stratification + regression on signed_mfe

**Run date:** 2026-05-16 (sleep run)
**Data:** `reports/findings/regret_oracle/daisy_chain_IS_full_daisy.csv` (7,925 daisy-chain trades)

## TL;DR

1. **Pivoting the target from `mfe_dollars` to `signed_mfe` was decisive.** Direction information was completely hidden by the magnitude-only target. R² jumps from ~0.19 (mfe_dollars k=2 max) to **0.26 (signed_mfe k=2 max)** with the same features and same model.
2. **R² grows with k**: k=2 → 0.262, k=3 → 0.307. The 3-way interaction term adds ~0 — signal is in 1-way + 2-way. Higher k mostly adds more 2-way pairs to the model.
3. **Stratified analysis is the strongest finding of all.** Splitting by `bar_range` quartile and running pair regression *within* each stratum produces **R² = 0.344** (within bar_range S3, `z_15s × slope_15s_3m`). The user's "shaft from seeds" intuition was correct — heterogeneous data was masking subgroup-specific direction signals.
4. **Direction-callable cells are routine in high-vol RTH strata.** In `bar_range S4 (>$6.5)` and `tod_minutes S4 (RTH AM)` cells with **90%+ direction skew and mean_signed ±$300** are common (n=20-100 per cell).
5. **The "rubber-band" archetypes from yesterday's KMeans are now visible at single-cell precision** — `z_1m Q4 × slope_15m_15m Q1` is 98% SHORT with mean_signed −$340 within RTH AM.

## The pivot — why direction-as-target changes everything

Yesterday's analysis used `mfe_dollars` as the target. That's *magnitude only* — every trade's MFE is positive by construction (it's the upper-bound forward excursion in the trade's direction). A LONG trade with $200 MFE looks identical to a SHORT trade with $200 MFE in that target.

The user (2026-05-16): *"the most important signal we need is direction."*

Replacing target with `signed_mfe = mfe_dollars × (+1 if LONG else −1)` makes direction explicit:
- Sign → which side won (positive = LONG-skewed, negative = SHORT-skewed)
- Magnitude → trade size
- A regression coefficient of +$50/σ now means "1σ increase in feature → $50 MORE LONG-skewed outcome"

This single target captures both questions. Cells with `|mean_signed|` high AND `pct_long` far from 50% are direction-callable.

## Single-axis 1D regression on signed_mfe (compared to mfe_dollars from yesterday)

| Feature | R² (mfe_dollars) | R² (signed_mfe) |
|---|---|---|
| bar_range | 0.136 | (small — direction is roughly symmetric) |
| **slope_15s_3m** | **0.002** | **~0.20** ← the dominant direction predictor was invisible to magnitude |
| volume | 0.101 | (small) |
| slope_15s_10m | 0.003 | high |

The features that predict *direction* are NOT the features that predict *magnitude*. bar_range/volume are pure magnitude amplifiers; slopes/z's are direction signals.

## k=2 paired regression on signed_mfe — R² up to 0.262

Top pairs:
```
dist_15s_1m | slope_15s_3m  → R² 0.262 (additive)
z_1m        | slope_15s_3m  → R² 0.250
slope_15s_3m| tod_minutes   → R² 0.228 + interaction → 0.251
z_15s       | slope_15s_3m  → R² 0.247
```
**`slope_15s_3m` appears in every top pair.** It's the single dominant direction-predictor.

The 2-way interaction adds 0.001-0.023 to additive R² — meaning **the signal is mostly additive** in the signed target (vs the mfe_dollars target where interactions were the only signal).

Direction-callable cells (Wilson CI):
- LONG-callable (CI_lo > 70%): 873 of 3,857 cells (22.6%)
- SHORT-callable (CI_hi < 30%): 797 of 3,857 (20.7%)
- **43% of all pair cells are direction-callable** with 95% confidence

Top LONG-callable cells:
| Pair | Bin | n | pct_long | mean_signed | mean_$ |
|---|---|---|---|---|---|
| z_15s × bar_range | Q2 × Q5 | 264 | 84% | +$222 | $294 |
| slope_15s_10m × slope_1m_10m | Q1 × Q5 | 29 | 93% | +$201 | $249 |
| z_15s × slope_15s_10m | Q2 × Q5 | 36 | **100%** | +$198 | $198 |
| z_1m × bar_range | Q2 × Q5 | 286 | 85% | +$188 | $260 |

Top SHORT-callable cells:
| Pair | Bin | n | pct_long | mean_signed | mean_$ |
|---|---|---|---|---|---|
| z_1m × bar_range | Q4 × Q5 | 189 | 14% | −$219 | $266 |
| dist_15s_15m × slope_15s_10m | Q1 × Q5 | 53 | 9% | −$211 | $236 |
| dist_15s_15m × slope_15s_3m | Q1 × Q5 | 82 | 5% | −$201 | $216 |

## k=3 paired regression on signed_mfe — R² up to 0.307

Top triplet: `z_15s | slope_15s_3m | tod_minutes` → **R² = 0.307**.

Direction-callable cells:
- LONG-callable: 16,546 of 82,733 (20.0%)
- SHORT-callable: 14,607 of 82,733 (17.7%)

Top LONG-callable cells (k=3):
| Triplet | Bin | n | pct_long | mean_signed | mean_$ |
|---|---|---|---|---|---|
| z_15s × dist_15m_to_Ml × slope_15m_15m | Q2 × Q5 × Q1 | 20 | 95% | +$333 | $355 |
| z_15s × z_1h_low × bar_range | Q2 × Q5 × Q5 | 22 | 91% | +$318 | $335 |
| z_15s × fan_width × bar_range | Q2 × Q4 × Q5 | 47 | 92% | +$317 | $343 |

Top SHORT-callable cells (k=3):
| Triplet | Bin | n | pct_long | mean_signed | mean_$ |
|---|---|---|---|---|---|
| z_1m × bar_range × tod_minutes | Q4 × Q5 × Q4 | 80 | 9% | −$327 | $364 |
| dist_15s_15m × slope_15s_10m × volume | Q1 × Q5 × Q5 | 22 | 5% | −$311 | $322 |
| z_1m × z_1h_high × bar_range | Q4 × Q1 × Q5 | 33 | 6% | −$310 | $343 |

The 3-way interaction term adds essentially **zero** R² over the 2-way model.

## k=4 (3 bins) — DONE

R²_max = **0.320** with `z_15s | slope_15s_3m | slope_1m_10m | tod_minutes`.

R² progression by interaction order: 0.262 (1-way) → 0.315 (2-way) → 0.319 (3-way) → 0.320 (4-way).
**The 4-way interaction term adds essentially zero**. The 3-way also adds <0.01.
The signal is fully captured by additive + 2-way.

218,009 cells across 3,876 quadruples. Direction-callable cells: **46.5%** (52,587 LONG + 48,761 SHORT).

Top LONG-callable (k=4):
| Quad | Bin | n | pct_long | mean_signed |
|---|---|---|---|---|
| z_1h_low × dist_15m_to_Ml × slope_15m_15m × bar_range | Q1×Q3×Q2×Q3 | 22 | **100%** | +$253 |
| dist_15s_15m × slope_15s_3m × slope_1m_10m × bar_range | Q3×Q1×Q1×Q3 | 24 | 96% | +$249 |
| z_15s × z_1m × slope_15m_15m × volume | Q2×Q1×Q1×Q3 | 85 | 94% | +$245 |
| z_15s × z_1m × volume × tod_minutes | Q2×Q1×Q3×Q2 | 74 | 92% | +$239 |

Top SHORT-callable (k=4):
| Quad | Bin | n | pct_long | mean_signed |
|---|---|---|---|---|
| dist_15s_1m × dist_15s_15m × bar_range × tod_minutes | Q3×Q1×Q3×Q3 | 27 | 7% | −$289 |
| z_1m × z_15m × slope_15s_10m × bar_range | Q3×Q1×Q3×Q3 | 20 | **0%** | −$289 |
| z_15s × z_1m × z_1h_high × bar_range | Q2×Q3×Q1×Q3 | 43 | 7% | −$289 |

## k=5 (2 bins) — DONE

R²_max = **0.348** with `z_15s | dist_15s_15m | slope_15s_3m | slope_15s_10m | tod_minutes`.

R² progression by interaction order: 0.257 → 0.317 → 0.343 → 0.346 → 0.348.
**5-way adds ~0.002**; 4-way adds ~0.003. Signal flat past 3-way.

318,866 cells across 11,628 quintuples (coarse 2-bin cells, so cells have higher n).
Direction-callable cells: **58.7%** (30.8% LONG + 27.9% SHORT — highest of any k due to wider bins giving tighter pct_long CIs).

Top LONG-callable (k=5): max mean_signed = **+$233** (smaller than k=3's +$333 because coarse bins average more trades into the cell).

Top SHORT-callable (k=5): `z_15s × z_15m × slope_15s_3m × bar_range × tod_minutes` n=190 6% LONG **mean_signed −$216** — note the large n=190 makes this very confident.

## R² progression — clear saturation around k=3-4

| k | bins | R²_max | gain over k−1 | top model |
|---|---|---|---|---|
| 1 | qcut 5 | ~0.20 | — | slope_15s_3m on signed_mfe |
| 2 | 5 | **0.262** | +0.06 | dist_15s_1m × slope_15s_3m |
| 3 | 5 | **0.307** | +0.045 | z_15s × slope_15s_3m × tod_minutes |
| 4 | 3 | **0.320** | +0.013 | z_15s × slope_15s_3m × slope_1m_10m × tod_minutes |
| 5 | 2 | **0.348** | +0.028 | z_15s × dist_15s_15m × slope_15s_3m × slope_15s_10m × tod_minutes |
| **Stratified k=2 (within bar_range S3)** | 5 | **0.344** | — | z_15s × slope_15s_3m |
| **Stratified k=2 (within tod_minutes S5)** | 5 | **0.342** | — | dist_15s_1m × slope_15s_3m |

**Saturation ceiling is around R² = 0.35.** Both unstratified k=5 and stratified k=2 hit it. Adding more features doesn't push past it.

**Stratified k=2 matches unstratified k=5 with FAR fewer parameters** (2 features within a stratum vs 5 features + all interactions). This is the cleanest empirical statement of the user's "shaft from seeds" intuition — and the more *deployable* approach since fewer parameters = less overfit risk.

## Stratified pair analysis — the strongest signal of all

The user (2026-05-16): *"if no strong signal is found, then we will proceed to cluster and regression on bins so smaller like-to-like samples should help separate the shaft from the seeds."*

We applied this on `bar_range` and `tod_minutes` as stratifiers.

### Stratified by bar_range (4 strata)

| Stratum | n | Top pair | R² (interaction) |
|---|---|---|---|
| S1 [< $2 range] | 2,326 | (weaker) | ~0.20 |
| S2 [$2-$3.5] | 1,751 | z_15s × slope_15s_3m | 0.316 |
| **S3 [$3.5-$6.5]** | **1,880** | **z_15s × slope_15s_3m** | **0.344** ← best |
| S4 [> $6.5] | 1,968 | (similar) | ~0.32 |

**R² 0.344 within S3 vs 0.26 unstratified** — bar_range stratification reveals direction signal that's averaged out across the full dataset.

Top direction-callable cells WITHIN S4 (highest vol):
- LONG: `dist_1m_15m Q5 × slope_15s_3m Q1` n=34 **91% LONG** mean_signed **+$296**
- LONG: `z_1m Q1 × slope_1m_10m Q1` n=59 **95% LONG** mean_signed **+$278**
- SHORT: `z_1h_high Q1 × slope_15s_3m Q5` n=31 **97% SHORT** mean_signed **−$338**
- SHORT: `z_1h_low Q1 × slope_15s_3m Q5` n=20 **95% SHORT** mean_signed **−$333**
- SHORT: `z_1m Q4 × z_1h_low Q1` n=21 **100% SHORT** mean_signed **−$325**

### Stratified by tod_minutes (6 strata)

| Stratum | n | Top pair | R² (interaction) |
|---|---|---|---|
| S1-S3 (overnight/pre-RTH) | varies | (weaker) | 0.20-0.28 |
| **S4 [801-1040] (RTH AM)** | **1,320** | **z_1m × slope_15s_10m** | **0.336** |
| **S5 [1040-1260] (RTH PM)** | **1,330** | **dist_15s_1m × slope_15s_3m** | **0.342** |
| S6 (pre-halt) | smaller | weaker | ~0.27 |

Top direction-callable cells WITHIN S4 (RTH AM):
- LONG: `z_15s Q2 × bar_range Q5` n=55 89% LONG mean_signed **+$313**
- LONG: `z_1h_high Q3 × slope_15s_3m Q1` n=46 **91% LONG** mean_signed +$311
- SHORT: `z_1m Q4 × slope_15m_15m Q1` n=43 **98% SHORT** mean_signed **−$340**
- SHORT: `z_1h_high Q1 × slope_15s_3m Q5` n=22 **100% SHORT** mean_signed −$305

## Synthesis — what the data says

**Direction is predictable.** Cells with >90% direction skew (LONG or SHORT) exist in abundance once you stratify by either bar_range or tod_minutes. Many such cells have n=30-100 with `|mean_signed| > $300`.

**The recipe** (consistent across both stratifications + the unstratified k=2/3):

| Component | What |
|---|---|
| **Volatility gate** | High `bar_range` (Q5, or stratify by S3-S4 with bar_range) |
| **TOD gate** | RTH hours (`tod_minutes` S4-S5) |
| **Direction signal** | `slope_15s_3m` sign — steep DOWN slope → LONG (fade the dump), steep UP → SHORT (fade the rally) |
| **Z confirmation** | `z_1m` quadrant — Q2 → LONG, Q4 → SHORT |
| **Rail confirmation** | `z_1h_low` / `z_1h_high` extreme positions → extra direction confidence |

A real selector could test:
```
LONG fire condition:
    bar_range > median (Q4 or Q5)
    AND tod ∈ RTH hours (phase_4_rth_am or phase_5_rth_pm)
    AND slope_15s_3m < -Cut (steep down)
    AND z_1m ∈ Q2 (slight down)
    → expected pct_long ≈ 85-95%, mean_signed +$250-$330

SHORT fire condition: (mirror)
    bar_range > median
    AND tod ∈ RTH hours
    AND slope_15s_3m > +Cut (steep up)
    AND z_1m ∈ Q4 (slight up)
    → expected pct_long ≈ 5-15%, mean_signed −$250-$340
```

## Caveats

1. **IS-only.** Per MEMORY 2026-05-03 hard rule, every cell finding here needs OOS validation on 2026 before becoming a selector.
2. **Multi-comparison:** ~3,857 (k=2) + 82,733 (k=3) + many more in stratified ≈ ~100,000+ cells tested. Some top cells will be spurious.
3. **No costs / spread / slippage** — daisy-chain edges before execution costs.
4. **`slope_15s_3m` correlation with future MFE direction** — given it's computed at entry from the past 3min of M_15s data, it's NOT lookahead. Selector-usable. But the centered-window oracle detector IS lookahead — a live selector needs a causal extremum detector.
5. **Day WR / mode-day pending** — these are *cell-level* findings on $/trade. Final goal per protocol is mode $/day + Day WR after aggregation. The next step is to take the top cells, simulate selection, aggregate by session_date, and report the protocol metrics with bootstrap CIs.

## Outputs

All under `reports/findings/regret_oracle/`:
- `kway_2_clusters_IS_full_daisy_signed.csv` / `kway_2_regression_IS_full_daisy_signed.csv`
- `kway_3_clusters_IS_full_daisy_signed.csv` / `kway_3_regression_IS_full_daisy_signed.csv`
- `kway_4_clusters_*.csv` / `kway_4_regression_*.csv` (when k=4 finishes)
- `kway_5_clusters_*.csv` / `kway_5_regression_*.csv` (when k=5 finishes)
- `stratified_pair_clusters_bar_range_IS_full_daisy_signed.csv` / `stratified_pair_regression_bar_range_IS_full_daisy_signed.csv`
- `stratified_pair_clusters_tod_minutes_IS_full_daisy_signed.csv` / `stratified_pair_regression_tod_minutes_IS_full_daisy_signed.csv`

## Next steps (for tomorrow morning)

1. **Build a real-time-clean selector spec** from the top stratified cells (RTH × high-vol × z × slope-sign rules).
2. **Simulate selection** on the daisy-chain trades: which trades pass the gate? Aggregate per session_date for Day WR + mode $/day.
3. **2026 OOS run** — apply the SAME cell-gate to fresh data.
4. **Direction predictor** could be formalized as a tiny logistic regression: `P(LONG) ~ z_1m + slope_15s_3m + bar_range + tod_minutes + interactions`. Train on 2025 IS, evaluate on 2026 OOS. Compare to the discrete cell-gate.
