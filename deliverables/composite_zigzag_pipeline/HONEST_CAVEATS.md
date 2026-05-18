# Honest Caveats

What's leaky, optimistic, or untested in this pipeline. Read this BEFORE
believing the headline numbers.

## Performance numbers — context matrix

| Variant | $/day mean | What's wrong with it |
|---------|------------|---------------------|
| Oracle entries + GBM sizing | $5,781 | Pivot-entry timing peek (can't enter at actual pivot live) |
| Hardened entries + GBM sizing | $927 | B7 trained on pivot-bar features, applied at R-trigger bar (small distribution shift) |
| Hardened + B8 hour gate | $1,505 | Same as above + B8 also has pivot-bar feature peek |

**Use the hardened $927/day number** as the credible OOS estimate. The
$5,781 is fully cooked. The $1,505 is partially cooked.

## Known peeks (in order of severity)

### 1. B7 trained on pivot-bar features, applied at R-trigger bar
**Severity: moderate.** B7 was trained on V2 features computed at the
zigzag pivot bar (the actual extreme). At inference, we apply it to
legs entering at R-trigger fire (which is bar-after-pivot, with the
extreme features now decayed). This is distribution shift, not
mathematical leakage, but it slightly inflates results.

**Fix**: retrain B7 on V2 features at the R-trigger bar (the entry
moment). Build a new IS dataset by detecting R-trigger fires on IS
days and grabbing V2 features there.

**Expected impact**: another 10-30% reduction in hardened forward
pass. Realistic floor: $600-700/day.

### 2. B8 hour-risk uses pivot-aware leg P&L for labels
**Severity: moderate.** B8 was trained to predict "forward 60-min total
leg P&L" where the leg P&L was computed as `leg_amplitude - 2R - friction`
— the theoretical R-trigger-based formula. In practice, R-trigger
entry/exit prices may differ from `pivot ± R` due to bar slippage.

**Fix**: rebuild B8 labels using actual R-trigger detection (same as
the hardened forward pass). The label distribution would shift slightly.

**Expected impact**: small. The label is already approximating
R-trigger reality; the actual numbers are close.

### 3. R-trigger detection on 5s closes (slight optimism)
**Severity: low.** Our R-trigger fires when the 5s CLOSE crosses the
threshold. In live NinjaTrader execution, R-trigger fires when 5s OHLC
INTRA-BAR crosses — so live detection is *faster* than our sim by
0-5 seconds.

But the order fill is at MARKET (or limit), with slippage. Net:
roughly cancels. Could be ±$1-2/leg in either direction.

### 4. Zigzag detection uses retrospective pivot location
**Severity: low.** Pivots are detected by `detect_swings()` which walks
forward through 5s closes confirming reversals. The pivot timestamp
is the actual extreme bar, which is only knowable retrospectively.

For TRAINING labels, this is correct. For LIVE detection, the pivot
location is determined when R-trigger fires (i.e., the running extreme
becomes the confirmed pivot at R-trigger time).

Our forward pass uses LIVE R-trigger detection — so this is honest.
But the truth dataset uses retrospective pivots, which is fine for
label generation.

### 5. No regime gating
**Severity: moderate, deliberate.** We did NOT use the project's
existing `DATA/ATLAS/regime_labels_2d.csv` for filtering, because that
file is computed end-of-day (lookahead). We considered training a
forward-pass-honest regime detector but didn't have time. Some days
that fall into "DOWN_CHOPPY" or similar bad regimes are not filtered;
this is the source of the 5 net-negative OOS days.

**Fix**: build a session-start regime classifier using overnight gap,
prior-day stats, calendar events.

### 6. Friction estimate ($6/leg)
**Severity: low.** Commission $4 + slippage $2 = $6 per leg round-trip.
Real-world MNQ retail commissions can be $1.20-$2.40/round-trip
(IB, TD, etc.). Slippage on stop-entries during fast moves could be
3-5 ticks ($1.50-$2.50 each side). So realistic friction is $2-$7/leg,
centered on our $6.

**Sensitivity**: at +$0/leg friction, $/day +$340. At $12/leg friction,
$/day -$340. Real friction within ±$3 of our estimate.

## What's NOT tested

### No live deployment data
The entire pipeline is backtest only. Live deployment may differ in:
- Order fill quality (slippage on entries/exits)
- Connectivity latency (5s feed delays)
- Discretionary intervention (the user babysitting reduces edge variance)
- Regime persistence (live regime ≠ NT8 OOS regime)

### Per-day CIs are autocorrelated
Day-to-day P&L is correlated (regime persists). The 31 OOS days aren't
truly IID. Bootstrap CIs are mildly optimistic. Need >50-100 OOS days
for reliable inference.

### No partial-exit testing
Strategy is full-position entry + full-position exit. No scale-in,
no partial-target-fill, no Kelly-style add. These are unexplored
levers that could improve risk-adjusted return.

### No correlation analysis between B-models
B1-B8 likely have correlated errors (all use V2 features). Stacking
them in the composite assumes some independence; reality is noisier.

### No outright failure mode analysis
On the 5 negative days, we don't have a granular per-leg post-mortem.
What specifically went wrong? Choppy chop? News spikes? Need to log
+ analyze.

## What's robust

### Direction accuracy
Zigzag construction guarantees ~97% direction correctness for any leg
that has a pivot at both ends. This part is structural and not at
risk of model degradation.

### B7 monotonic calibration
Predicted-amp_R buckets reliably rank actual outcomes:
- pred 1.5-2.0 → actual median 1.78
- pred 3.0-4.0 → actual median 2.76

The ranking holds across IS and OOS. For SIZING use, this is what
matters; absolute prediction error is less important.

### Daily distribution shape
Hardened OOS shows 84% days positive, 77% over $200, median per-day
$627. This shape is consistent across the 31-day OOS sample.

## The "too good to be true" check

User flagged "$1,613/day OOS sounds too good." After full hardening:
- $475/day flat (naked indicator) — modest but positive
- $927/day with B7 sizing — meaningful lift (+95%)
- $1,505/day with B8 gate — meaningful lift (+62% over $927)

The progression makes physical sense:
1. R-trigger indicator alone gives ~$500/day on MNQ — consistent with industry knowledge of zigzag-style strategies
2. Sizing by amplitude prediction adds ~$450/day — proportional to the variance in leg amplitudes
3. Hour-level risk gating adds ~$580/day — by de-risking in predicted-bad hours

Each component is incremental and explainable. Not "too good" once
peeled.

## Recommended sanity checks for outside reviewer

1. **Reproduce the hardened forward pass** from the provided caches.
   Run `python tools/composite_forward_pass_hardened.py` from the
   standalone folder and verify the output matches reports/.

2. **Cross-check B7 calibration on OOS**. Bucket OOS predictions and
   verify monotonic relationship with actual outcomes. See
   `caches/b7_leg_sizer_OOS.parquet`.

3. **Run sensitivity analysis on friction**: rerun forward pass with
   $0, $4, $6, $10 per leg to bracket the realistic range.

4. **Inspect the 5 losing days**. Look at their hourly leg sequence.
   Were they news-driven? Choppy? Was B8 predicting low hour P&L
   correctly?

5. **Check for leakage by feature**. Compare feature distribution at
   pivot bar vs R-trigger bar. If they differ substantially, B7 peek
   matters more than estimated.

## What we'd build if continuing

In order of expected payoff:

1. Day-level regime classifier (overnight + calendar + intermarket features) — addresses bad-day filtering, the main remaining problem
2. B7 retraining on R-trigger-bar features — removes the last clear peek
3. Run through `training_iso_v2/` streaming engine — full end-to-end validation
4. Partial-exit logic — take some off at +1R, ride rest to R-trigger
5. Live deployment + data logging — actual ground truth

What we'd NOT build:

- Transformer/LSTM on same V2 features — won't break the Pearson 0.22 ceiling
- RL policy on same V2 features — same ceiling, far higher cost
- More B-models on same data — diminishing returns
