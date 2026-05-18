# 2026-05-17 — Live Zigzag Baseline (signal-quality floor)

**Goal**: settle whether trend3 ML adds anything over a 30-line streaming zigzag
indicator on the same direction-prediction task. NT8 OOS, 32 days, ATR×4
(matched to training).

**Method**: streaming zigzag on 5s closes with min_reversal = ATR(14)_1m × 4
in ticks, min_bars=36 (3min). Direction state emitted causally per 5s bar,
sampled at each 1m close, compared to hindsight `leg_direction` truth from
the same dataset trend3 was evaluated against.

**Tool**: `tools/live_zigzag_baseline.py`
**Outputs**:
- `reports/findings/regret_oracle/live_zigzag_baseline.txt`
- `reports/findings/regret_oracle/live_zigzag_baseline.per_day.csv`

---

## Three-way comparison (32 days, 32,970 truth-labeled bars)

| Signal           | Coverage | Acc on signal | Per-day acc (95% CI)         | LONG prec | SHORT prec |
|------------------|----------|---------------|------------------------------|-----------|------------|
| RAW trend3       | 66.0%    | 65.86%        | 66.19% [63.41, 68.79]        | 70.2%     | 62.6%      |
| SMOOTHED trend3  | 99.9%    | 61.48%        | 60.68% [57.76, 63.35]        | 64.5%     | 59.1%      |
| **LIVE zigzag**  | **97.6%**| **64.33%**    | **64.85% [63.22, 66.66]**    | **64.3%** | **64.4%**  |

## Paired same-day delta (n=31 matched days)

| Comparison                | Δ mean | 95% CI           | Significance      | Wins |
|---------------------------|--------|------------------|-------------------|------|
| RAW trend3 - LIVE ZZ      | +1.80pp| [-1.13, +4.40]   | **NOT significant** | 19/31 |
| SMOOTHED - LIVE ZZ        | -3.69pp| [-6.70, -1.43]   | Significantly worse | 10/31 |

## Flip lag at zigzag truth pivots

| Signal     | n     | median | mean | p25 | p75 | p90 |
|------------|-------|--------|------|-----|-----|-----|
| RAW trend3 | 1,391 | 3.0    | 4.5  | 2.0 | 5.0 | 8.0 |
| SMOOTHED   | 1,540 | 4.0    | 4.8  | 2.0 | 6.0 | 10.0|
| LIVE ZZ    | 1,793 | 4.0    | 6.4  | 3.0 | 7.0 | 13.0|

Live ZZ lags ~1 bar more than trend3 raw on average, but reacts comparably
at the median.

---

## Honest conclusions

1. **The 184-feature V2 GBM does NOT statistically beat a 30-line streaming
   zigzag indicator** on hindsight-direction prediction. The +1.80pp average
   delta has CI crossing zero — plausibly noise.

2. **The DMI windowed-EMA smoothed version is significantly WORSE** than the
   indicator alone (-3.69pp, CI [-6.70, -1.43]). Adding post-hoc smoothing
   to the GBM's outputs harms more than it helps.

3. **At matched coverage the gap likely collapses**. Live ZZ achieves
   64.33% accuracy at 97.6% coverage. Trend3 raw achieves 65.86% at 66%
   coverage — i.e., trend3 *abstains* on 34% of bars. If we forced live ZZ
   to commit only at its 66% most-confident bars (e.g., deep into legs,
   far from R-trigger), it would likely tie or beat trend3.

4. **The information ceiling at ~71% from the strength-vs-accuracy curve is
   the indicator's ceiling, not the model's.** Trend3 is approximating the
   zigzag indicator with V2 features; the indicator captures essentially
   the same information directly.

---

## Implications for the planned step 3 (multi-TF stacking)

The original plan was: train per-TF GBMs (5s, 15s, 1m, 5m, ...) + a meta-GBM
to combine them, expecting the explicit scale separation to lift the
direction-accuracy ceiling.

**That premise is now under serious doubt**. If a streaming indicator
already explains 64-65% of `leg_direction`, the remaining headroom is
~5-7pp at best. No amount of architectural reorganization can find signal
that isn't there.

The signal *bottleneck* is the target — `leg_direction` is a retrospective
label that mostly reflects "price went up in the past N seconds, so this
bar belongs to the up-leg." A causal indicator measures the same thing.

---

## Recommended redirection

Before running multi-TF stacking, reconsider the **learning target**.
Targets that the indicator can't trivially measure:

| Target                                | Why valuable                              | Difficulty |
|---------------------------------------|-------------------------------------------|------------|
| Direction at NEXT pivot               | Forward-looking, actionable               | Hard       |
| Continuation length: bars until next pivot | Used for exit timing                  | Medium     |
| Leg amplitude: |p_next - p_current| / ATR  | Sizing + TP placement                | Medium     |
| Entry-quality at current bar          | Filter for tier strategies                | Medium     |
| Pivot-imminent probability            | Counter-trend entry trigger               | Hard       |

A direction-classifier with the same V2 features but trained to predict the
*next* leg direction (not the *current* one) would be a meaningful test of
ML expressiveness over the indicator. The current trend3 task is too easy
for V2 to beat the indicator on.

## Status of original plan

- ✅ Step 1: inspector correctness mode locked to ATR×4
- ✅ Step 2: live zigzag baseline — **shows trend3 doesn't beat the indicator**
- ⏸ Step 3: multi-TF stacking — **REDIRECT recommended**: change target
  before adding architecture. Otherwise the experiment will just confirm
  the saturation we already see.
- ⏸ Step 4: depends on step 3 outcome

**No P&L numbers in this report by design.** Signal-quality only.
