# Research: Multi-TF Position in Structure

## Hypothesis

A 1m trade's profitability depends on WHERE it sits in the higher TF structure.
A 1m LONG at 20% of a 1h SHORT trend has room to run (pullback within fresh trend).
A 1m LONG at 90% of a 1h SHORT trend has nowhere to go (1h about to reverse).

**The higher TF position IS the conviction gate — not features, not gates, not the brain.**

## What to Prove

1. Do 1m trades near the END of higher TF seeds (>80% duration/MFE) perform worse?
2. Do 1m trades at the START of higher TF seeds (<20%) perform better?
3. Does TF alignment (1m direction = 5m direction = 1h direction) predict PnL?
4. Does TF disagreement at exhaustion predict the reversal?

## Data Required

- Auto seeds at multiple TFs: 1m, 5m, 15m, 1h
  - Currently only 1m exists (31K seeds)
  - Need to run `auto_swing_marker.py` with different params for 5m, 15m, 1h
  - OR: resample existing 1m seeds into higher TF swings
- Peak seeds (258K) with their 1m confirmation state
- Cross-reference: for each 1m seed, find which 5m/15m/1h seed it lives inside
  and compute its position (0-100%) within that higher TF seed

## Method

### Step 1: Generate Higher TF Seeds
Run `auto_swing_marker.py` at 5m, 15m, 1h scale:
```
python tools/auto_swing_marker.py --all --min-reversal 100 --min-bars 5 --max-bars 15
# 5m scale: 100 ticks min reversal, 5-15 bars = 25-75 min swings

python tools/auto_swing_marker.py --all --min-reversal 200 --min-bars 4 --max-bars 12
# 15m scale: 200 ticks, 4-12 bars = 1-3 hour swings

python tools/auto_swing_marker.py --all --min-reversal 400 --min-bars 3 --max-bars 8
# 1h scale: 400 ticks, 3-8 bars = 3-8 hour session swings
```

### Step 2: Nest Seeds
For each 1m seed, find:
- Which 5m seed contains it (by timestamp overlap)
- Which 15m seed contains it
- Which 1h seed contains it
- Position within each: `(seed_ts - parent_ts_start) / parent_duration`

### Step 3: Analyze Position vs Outcome
For each 1m seed, group by position in parent TF:
- 0-20%: early in parent swing
- 20-50%: middle
- 50-80%: late
- 80-100%: near parent exhaustion

Measure: avg PnL, WR, MFE, MAE per bucket.
If early = profitable and late = losing, hypothesis confirmed.

### Step 4: TF Alignment
Count how many TFs agree on direction at each 1m seed entry:
- 1 TF (1m only): no context
- 2 TFs (1m + 5m): partial
- 3 TFs (1m + 5m + 15m): strong
- 4 TFs (all agree): maximum conviction

Measure PnL per alignment level.

### Step 5: Cross-Reference with Peak Seeds
For the 258K peak pivots, overlay the same higher TF position.
Do peak fakeouts cluster at higher TF exhaustion points?
Does the 80% fakeout-is-counter-trend finding align with higher TF position?

## Expected Output

- `reports/findings/mtf_position_analysis.txt`
- `reports/findings/mtf_position_by_tf.csv`
- Answer: "1m trades at >X% of 1h duration lose Y% more" or "no effect"

## Connection to System

If confirmed, the two-worker architecture gets a third dimension:
- **Trend worker**: 1m direction + hold (from trend seeds)
- **Peak worker**: entry timing + fakeout filter (from peak seeds)
- **Structure worker**: position in 5m/15m/1h hierarchy (from nested seeds)

The structure worker answers: "does the higher TF have room for this trade?"
That's the missing conviction gate.

## Dependencies

- `tools/auto_swing_marker.py` — needs to run at multiple TF scales
- ATLAS 1m data (exists)
- Pivot seeds CSV (exists)
- Nested seed cross-reference tool (to build)

## Priority

HIGH — this may explain why the blind flip fails on full sample but works on
individual days. Some days the 1h structure supports oscillation, others it
trends all day. The position in structure tells you which day you're on.
