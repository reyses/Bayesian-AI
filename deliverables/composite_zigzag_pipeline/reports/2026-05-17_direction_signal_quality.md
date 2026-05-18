# 2026-05-17 — Direction Signal Quality (raw vs DMI-smoothed)

**Goal**: measure how well the trend3 classifier's predicted direction matches
the zigzag leg_direction (truth) bar-by-bar. NOT trading. Pure signal-quality.

**Truth source**: `tools/build_zigzag_pivot_dataset.py --root DATA/ATLAS_NT8
--target oos --atr-mult 4.0` → `zigzag_pivot_dataset_NT8_OOS_atr4.parquet`
(32 days, 34,844 1m bars, 130 pivots/day median, leg_direction derived from
pivot-to-pivot legs).

**Predicted signals compared**:
- **RAW**     : `argmax(p_long, p_short, p_neutral)` from `trend3_cache_OOS_NT8.parquet`
- **SMOOTHED**: `regime_dir` from windowed-EMA + DMI state machine
  (span=5, ADX span=10, window=180 bars, margin=0.05, ADX floor=15)

**Evaluator**: `tools/direction_signal_accuracy.py`
**Outputs**:
- `reports/findings/regret_oracle/direction_signal_accuracy.txt`
- `reports/findings/regret_oracle/direction_signal_accuracy.per_day.csv`

---

## Headline (32 days, 32,970 truth-labeled bars)

| Signal    | Coverage (non-NEUTRAL) | Accuracy on signal bars | Per-day mean acc (95% CI)    |
|-----------|------------------------|-------------------------|------------------------------|
| RAW       | 66.0%   (21,771)       | **65.86%**              | 66.19%   [63.41%, 68.79%]    |
| SMOOTHED  | 99.9%   (32,940)       | 61.48%                  | 60.68%   [57.76%, 63.35%]    |

**Delta (smoothed - raw) on matched per-day comparison: -5.51pp,
95% CI [-7.72pp, -3.75pp] — SIGNIFICANTLY WORSE.**

Smoothed beats raw on 3 / 32 days.

---

## Per-class precision/recall

| Signal       | LONG prec | LONG rec | SHORT prec | SHORT rec |
|--------------|-----------|----------|------------|-----------|
| RAW          | 70.2%     | 38.7%    | 62.6%      | 48.5%     |
| SMOOTHED     | 64.5%     | 55.7%    | 59.1%      | 67.5%     |

Smoothed lifts recall (catches more legs) at the cost of precision (more wrong
calls). Raw is the more conservative, higher-precision option.

---

## Flip lag at zigzag pivots (truth direction flips)

| Signal    | n     | median | mean | p25 | p75 | p90 |
|-----------|-------|--------|------|-----|-----|-----|
| RAW       | 1,391 | 3.0    | 4.5  | 2.0 | 5.0 | 8.0 |
| SMOOTHED  | 1,540 | 4.0    | 4.8  | 2.0 | 6.0 | 10.0 |

Both lag 3-5 bars (3-5 min) catching up to a new leg. Smoothing adds ~1 bar
of additional latency — expected, the EMA inertia.

The smoothed signal catches up MORE truth-flips (1,540 vs 1,391) because raw
sometimes never commits a new direction (sticks in NEUTRAL).

---

## Accuracy vs directional strength (P_dir - P_neutral)

**RAW**:
- strength 0.00-0.05:  n= 1,695  acc=64.66%
- strength 0.30-0.50:  n= 6,797  acc=64.23%
- **strength 0.50+    :  n= 4,147  acc=71.69%**  ← clear high-confidence lift

**SMOOTHED**:
- strength < 0.00:     n=12,580  acc=57.77%  ← sticky-state holdouts hurt here
- strength 0.30-0.50:  n= 5,979  acc=63.56%
- **strength 0.50+    :  n= 3,052  acc=71.10%**  ← same ceiling

Both signals show a confidence-accuracy relationship at the top end (~71% at
P_dir - P_neut > 0.5). That's the realistic ceiling for V2 entry features on
this 3-class task.

---

## Interpretation

1. **Raw classifier is the better direction signal.** When it commits (66% of
   bars), it's 5.5pp more accurate than the smoothed sticky-state version, with
   significance.

2. **Smoothing trades accuracy for coverage.** It pushes coverage to 99.9% by
   sticking with a direction when momentum is unclear — those sticky bars
   pull average accuracy down. If the downstream consumer can use NEUTRAL as
   "no information" (i.e. an abstain), raw is strictly better.

3. **The windowed-EMA's 3-hour cutoff did NOT prevent the smoothing penalty.**
   The smoothing inertia itself is the issue, not stale infinite-history bias.

4. **Hard ceiling ≈ 71% at high confidence.** Same for both signals.
   This is the V2-feature information limit for direction classification on
   zigzag legs. More smoothing on top of the same features can't raise it.

5. **Flip-lag is comparable to inspector observation** (3-5 bars). Smoothing
   adds ~1 bar of inertia.

---

## Implications for next iteration

- **Don't apply DMI-style smoothing post-hoc.** Per the 2026-05-17 process
  rule, this confirms the user's instinct that smoothing should be INTEGRATED
  into the ML (LSTM with learned transition probabilities), not bolted on as
  a state machine over independent per-bar predictions.

- **The 71% ceiling at high confidence is the budget.** An LSTM that learns
  prior bars' state transitions might raise this — if it can't, no smoothing
  scheme on the V2-feature classifier will.

- **Raw signal at strength > 0.5 (n=4,147 bars = ~130/day) is the high-confidence
  set worth instrumenting visually.** Inspector already shows this via
  alpha-coded dots.

## Next

- Visualize: overlay smoothed +DI/-DI (p_long_ema, p_short_ema) as lines on
  the inspector for direct comparison alongside the existing raw-classifier
  ribbon. NOT a decision tool, a diagnostic.
- LSTM-integrated direction (queued per process rule).

**No P&L numbers in this report by design.** This is signal-quality only.
