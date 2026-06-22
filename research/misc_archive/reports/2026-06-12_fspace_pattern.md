# Does the F-space follow a measurable selection pattern? (post-hoc / lookahead)

- valid segments: 80,717  (PRISTINE/RECOVERED). Lookahead acknowledged: whole-day scaler + completed window.
- feature names mapped from FEATURES_5s_v2 (day 2025_01_01); first 177 non-(timestamp/price_mean/vwap) columns.

## 1. Selection profile
- surviving terms/segment: median 15, p90 15, max 15
- active features/segment (p_local): median 25, p90 42
- term kinds: linear 61,160 (5.2%), quadratic 1,103,974 (94.8%), oob 0
  (linear-dominant => individual indicators explain price; quad-dominant => joint/interaction structure)

## 2. Concentration (is selection structured or uniform?)
- Gini of per-feature selection frequency: 0.602  (0 = uniform/no structure, 1 = all in one feature)
- top-10 features account for 30.0% of all selections; top-20 -> 47.9%
- features NEVER selected: 15/177

## 3. Null model (uniform selection from each segment's own poly space)
- null Gini (mean of 5): 0.545  +/- 0.000
- REAL Gini 0.602 vs NULL 0.545  ->  MEASURABLE PATTERN (real concentration exceeds chance)
  Note: even the null is concentrated if active_grid_cells (stage-1 availability) is itself non-uniform;
  the REAL>NULL gap isolates the elastic-net's preference ON TOP of stage-1 availability.

## 4. Indicators worth seeing (top 25 by selection frequency)
| rank | feature | selected in % of segments | via-linear % | beta sign stability |
|---|---|---|---|---|
| 1 | L3_5s_hurst_9 | 43.4% | 5% | 0.51 (+) |
| 2 | L3_5s_reversion_prob_9 | 43.1% | 2% | 0.51 (+) |
| 3 | L3_5s_z_high_9 | 38.2% | 3% | 0.74 (+) |
| 4 | L3_5s_z_low_9 | 37.9% | 3% | 0.71 (+) |
| 5 | L3_15s_z_high_12 | 36.9% | 8% | 0.67 (+) |
| 6 | L3_15s_z_low_12 | 36.5% | 8% | 0.67 (+) |
| 7 | L2_5s_price_velocity_9 | 36.3% | 6% | 0.76 (+) |
| 8 | L3_15s_z_se_12 | 32.4% | 9% | 0.68 (+) |
| 9 | L3_15s_hurst_12 | 31.5% | 8% | 0.50 (-) |
| 10 | L3_15s_reversion_prob_12 | 30.2% | 3% | 0.52 (+) |
| 11 | L3_5s_z_se_9 | 29.6% | 3% | 0.72 (+) |
| 12 | L2_5s_price_accel_9 | 25.3% | 2% | 0.72 (+) |
| 13 | L2_15s_price_accel_12 | 22.7% | 3% | 0.65 (+) |
| 14 | L1_5s_price_accel_1b | 21.1% | 1% | 0.66 (+) |
| 15 | L1_5s_upper_wick | 20.8% | 2% | 0.51 (+) |
| 16 | L1_5s_lower_wick | 20.6% | 2% | 0.58 (-) |
| 17 | L3_1m_z_se_15 | 19.6% | 10% | 0.61 (+) |
| 18 | L1_15s_upper_wick | 19.6% | 2% | 0.52 (-) |
| 19 | L2_15s_price_velocity_12 | 19.3% | 5% | 0.64 (+) |
| 20 | L1_15s_lower_wick | 18.9% | 2% | 0.53 (+) |
| 21 | L1_5s_body | 18.3% | 2% | 0.73 (+) |
| 22 | L1_15s_price_accel_1b | 16.3% | 2% | 0.51 (+) |
| 23 | L3_1m_z_high_15 | 15.9% | 10% | 0.57 (+) |
| 24 | L3_1m_z_low_15 | 15.5% | 9% | 0.55 (+) |
| 25 | L1_5s_price_velocity_1b | 15.4% | 2% | 0.74 (+) |

## 5. Stratify by volatility_tier (is structure within-vol, or just re-deriving vol regimes?)
Top-10 feature overlap between vol tiers (Jaccard); if ~1.0 the selected set is vol-invariant (real structure), if low it tracks vol.
- vol_tier 2: 32,947 segs; top-5 = ['L3_5s_hurst_9', 'L3_5s_reversion_prob_9', 'L3_5s_z_high_9', 'L3_5s_z_low_9', 'L3_15s_z_high_12']
- vol_tier 1: 27,367 segs; top-5 = ['L3_5s_reversion_prob_9', 'L3_5s_hurst_9', 'L3_5s_z_high_9', 'L3_5s_z_low_9', 'L2_5s_price_velocity_9']
- vol_tier 5: 3,889 segs; top-5 = ['L3_5s_hurst_9', 'L3_5s_reversion_prob_9', 'L3_15s_z_high_12', 'L3_15s_z_low_12', 'L3_5s_z_high_9']
- vol_tier 6: 3,757 segs; top-5 = ['L3_5s_hurst_9', 'L3_5s_reversion_prob_9', 'L3_15s_z_low_12', 'L3_15s_z_high_12', 'L2_5s_price_velocity_9']
- vol_tier 4: 3,598 segs; top-5 = ['L3_5s_hurst_9', 'L3_5s_reversion_prob_9', 'L3_15s_z_low_12', 'L3_15s_z_high_12', 'L3_5s_z_low_9']
- vol_tier 7: 3,420 segs; top-5 = ['L3_5s_hurst_9', 'L3_15s_z_high_12', 'L3_15s_z_low_12', 'L3_5s_reversion_prob_9', 'L2_5s_price_velocity_9']
- vol_tier 8: 2,976 segs; top-5 = ['L3_5s_hurst_9', 'L3_15s_z_high_12', 'L2_5s_price_velocity_9', 'L3_15s_z_low_12', 'L3_5s_reversion_prob_9']
- vol_tier 3: 2,763 segs; top-5 = ['L3_5s_hurst_9', 'L3_5s_reversion_prob_9', 'L3_15s_z_high_12', 'L3_5s_z_high_9', 'L3_5s_z_low_9']
- Jaccard(top10 vol 2, vol 1) = 0.82
- Jaccard(top10 vol 2, vol 5) = 1.00
- Jaccard(top10 vol 2, vol 6) = 1.00
- Jaccard(top10 vol 2, vol 4) = 1.00
- Jaccard(top10 vol 2, vol 7) = 1.00
- Jaccard(top10 vol 2, vol 8) = 1.00
- Jaccard(top10 vol 2, vol 3) = 0.82
- Jaccard(top10 vol 1, vol 5) = 0.82
- Jaccard(top10 vol 1, vol 6) = 0.82
- Jaccard(top10 vol 1, vol 4) = 0.82
- Jaccard(top10 vol 1, vol 7) = 0.82
- Jaccard(top10 vol 1, vol 8) = 0.82
- Jaccard(top10 vol 1, vol 3) = 0.82
- Jaccard(top10 vol 5, vol 6) = 1.00
- Jaccard(top10 vol 5, vol 4) = 1.00
- Jaccard(top10 vol 5, vol 7) = 1.00
- Jaccard(top10 vol 5, vol 8) = 1.00
- Jaccard(top10 vol 5, vol 3) = 0.82
- Jaccard(top10 vol 6, vol 4) = 1.00
- Jaccard(top10 vol 6, vol 7) = 1.00
- Jaccard(top10 vol 6, vol 8) = 1.00
- Jaccard(top10 vol 6, vol 3) = 0.82
- Jaccard(top10 vol 4, vol 7) = 1.00
- Jaccard(top10 vol 4, vol 8) = 1.00
- Jaccard(top10 vol 4, vol 3) = 0.82
- Jaccard(top10 vol 7, vol 8) = 1.00
- Jaccard(top10 vol 7, vol 3) = 0.82
- Jaccard(top10 vol 8, vol 3) = 0.82


## VERDICT — honest read (do not oversell the "measurable pattern" flag)

**A stable, volatility-invariant VOCABULARY exists — that part is real and useful.**
- The same features get reached for regardless of volatility_tier: Jaccard(top-10) = 0.82-1.00
  across all 8 tiers. So this is NOT just re-deriving vol regimes.
- The vocabulary is the FAST-TF REVERSION/EXTENSION family: `L3_5s/15s_{hurst,reversion_prob,
  z_high,z_low,z_se}`, `L2_5s/15s_price_{velocity,accel}`, `L1_5s/15s_{wick,body}`.
- CONVERGENCE WORTH NOTING: this is the NMP feature family. The segment regressions
  independently reach for z_se / z_high / z_low / reversion_prob — the same quantities the
  NMP trigger and lambda-completion (lambda_hat = slope of log|z_se|) are built on. A
  post-hoc curve-fit and the a-priori NMP equation point at the SAME indicators. That is
  the most valuable thing this run produced.

**But three caveats gut the strong interpretation:**
1. **Concentration barely beats chance.** Real Gini 0.602 vs NULL 0.545 = +0.057 only. The
   NULL already sits at 0.545 because stage-1 `active_grid_cells` is itself non-uniform, so
   most of the apparent concentration is stage-1's correlation pre-screen, NOT an elastic-net
   discovery. Given what stage-1 makes available, WHICH terms survive is near-random.
   => the top-feature RANKING is essentially the stage-1 AVAILABILITY ranking, lightly tilted.
2. **94.8% of surviving terms are QUADRATIC (interactions); only 5.2% linear.** The top
   features are credited almost entirely via feature x feature products (via-linear column
   = 1-10%). 15 terms chosen from a ~350-term poly space to fit a 32-bar window is an
   overfit basis; reaching for cross-terms is the overfit signature, not joint structure
   you can trust.
3. **Beta signs are not stable.** Ranks 1-2 (hurst, reversion_prob) sign-stability ~0.51 =
   coin flip. Best are z_high/z_low/velocity at 0.67-0.76 -- moderate, not monotone. So even
   the selected features have no simple, consistent direction->price mapping.

**Bottom line for the stated goal ("which indicators worth seeing"):**
- WHICH: real and actionable -> the fast reversion/extension (NMP) family, vol-invariant.
  Worth computing/watching causally; everything else (the 15 never-selected + the long tail)
  is prunable.
- HOW: NOT a simple per-indicator signal. The relationship is interaction-heavy and
  sign-unstable -> consistent with the 2026-05-03 finding that the composite must be
  CONDITIONAL (modifier-quantile), not additive. A linear "watch indicator X" rule won't work.
- The "F-space follows a measurable pattern" claim is TRUE only in the weak sense (which
  features are reached for is stable); it is FALSE in the strong sense (a stable, simple,
  recurring functional form). Do not build on the strong reading.

**Caveat on the caveat (lookahead):** all of this is post-hoc (whole-day scaler, completed
window). It legitimately prunes the indicator set for causal work; it does NOT establish any
of these features carry CAUSAL signal -- that needs a causal nowcast + OOS test (the KT2 gate).
