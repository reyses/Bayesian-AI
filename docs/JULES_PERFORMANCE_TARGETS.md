# Jules Task: Performance Improvement — 3 Targets

## Targets
| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Win Rate | 37.5% | 80% | +42.5pp |
| Optimal Exit (capture rate ≥80% of MFE) | 3.0% of trades | 80% of trades | +77pp |
| Wrong Direction After Entry | 46.1% | ~23% (50% reduction) | -23pp |

## Reference Files (READ FIRST)

- `docs/checkpoint_reference/SCHEMAS.md` — All data structures
- `docs/checkpoint_reference/run_snapshot.json` — Latest metrics
- `reports/is/oracle_trade_log.csv` — 3,754 trades with full diagnostics
- `reports/phase4_report.txt` — Latest forward pass report

Key source files:
- `training/orchestrator.py` — Gate logic, forward pass, exit mechanics
- `training/wave_rider.py` — Position management, trail stops, exits
- `training/timeframe_belief_network.py` — Direction signals, belief network, decay cascade

## Current Baseline (IS, 10 months, 3,754 trades)

```
Win Rate:           37.5%
Total PnL:          $5,834
Avg PnL/trade:      $1.55
Direction correct:  45.2%
Direction wrong:    46.1%
Capture rate:       0.1% mean (median: -1.6%)
MFE utilization:    -17.1%

Exit breakdown:
  profit_target:      1,135 trades  100.0% WR   $18.58 avg   capture: 37.8%
  trail_stop:         2,221 trades    0.9% WR   -$9.02 avg   capture: -21.7%
  belief_flip:          283 trades   58.0% WR   $10.08 avg   capture:  9.5%
  structural_break:     115 trades   78.3% WR   $16.70 avg   capture: 23.8%

CRITICAL: Even correct-direction trades only win 35.7% of the time.
          99.9% of trail stops NEVER reach activation threshold.
```

## Root Cause Analysis

### Problem 1: Trail Stop = Broken Stop-Loss (59% of trades, 0.9% WR)

**This is the #1 problem.** 2,221 trades exit via trail_stop with $-9.02 avg PnL.

Facts:
- Average trail activation threshold: $64.66
- 99.9% of trail stops never reach activation → trail never arms
- Average hold: 2.9 bars (44 seconds) — killed almost immediately
- Trail distance: 481 ticks avg, but SL is 270 ticks — SL hits first
- Even correct-direction trail stops lose: 1,038 correct-dir trades → $-9.04 avg

**Diagnosis**: The "trail stop" is really just the initial SL firing. Price hasn't
moved enough to activate the trail, so the trailing mechanism never engages. The
system enters a trade, sits 2-3 bars, gets noise-stopped out, logs it as "trail_stop."

**Files**: `training/wave_rider.py` — `update_trail()`, `check_exit()`

### Problem 2: Direction Model Worse Than Random (46.1% wrong)

Facts:
- SHORT trades: 2,321 (62%) with only 35.1% correct direction
- LONG trades: 1,433 (38%) with 61.5% correct direction
- Heavy SHORT bias — system favors SHORT but gets it wrong 65% of the time
- Logistic regression `_logistic_prob()` is the primary direction source
- Per-cluster `dir_coeff` trained on small samples → overfit/noisy

**Files**: `training/timeframe_belief_network.py` — `_logistic_prob()`
          `training/orchestrator.py` — direction gating hierarchy

### Problem 3: Capture Rate Near Zero (0.1% mean)

Facts:
- Avg oracle MFE: $93.93 (huge potential)
- Avg actual PnL: $1.55 (captures almost nothing)
- 83.3% of trades have oracle MFE > TP target — the moves are there
- Only 3% of trades achieve ≥80% capture
- Profit target trades capture 37.8%, structural breaks capture 23.8%
- Trail stops capture -21.7% (negative = losing money)

**Root cause**: Combination of (1) wrong direction → guaranteed loss, and
(2) trail stop kills correct-direction trades before they mature.

## Implementation Strategy

**The three targets are deeply interconnected.** Fixing the trail stop alone would
improve all three metrics. Fixing direction alone would improve all three. The
order matters:

### Phase A: Fix the Trail Stop System (Highest Impact)

The trail stop produces 2,221 losing trades. Fixing it directly improves WR and capture.

#### A1. Separate SL from Trail Stop

Currently trail_stop fires as the initial SL before trail activates. These are
different mechanisms and should be tracked separately.

In `wave_rider.py`:
- If trail has NOT activated yet and price hits SL → exit_reason = `"stop_loss"`
- If trail HAS activated and price retraces through trail → exit_reason = `"trail_stop"`
- This alone changes nothing about behavior, but gives us clean data to work with

#### A2. Widen Initial Stop-Loss for High-Conviction Entries

Current SL = `mean_mae * 1.1` (template-based). For most trades this is too tight
at 15-second resolution — 2-3 bars of noise triggers it.

Proposed fix in `wave_rider.py` or wherever SL is computed:
```python
# Scale SL by entry conviction — high conviction = wider leash
conviction_multiplier = 1.0 + 0.5 * max(0, belief_conviction - 0.5)
# E.g. conviction=0.7 → 1.1x, conviction=0.9 → 1.2x
adjusted_sl = base_sl * conviction_multiplier

# Minimum SL floor: never less than 2x ATR(15s) to survive noise
min_sl = 2.0 * atr_15s
final_sl = max(adjusted_sl, min_sl)
```

#### A3. Lower Trail Activation Threshold

Current: avg $64.66 activation (essentially unreachable in 15s moves).

Proposed: Activation = `max(3 ticks, TP * 0.15)` instead of current formula.
The trail should activate once we have ANY meaningful unrealized profit, not after
the trade is halfway to TP.

#### A4. Adaptive Trail Distance

Once trail activates, use physics to set distance:
```python
if wave_maturity < 0.3:
    trail_dist = base_trail * 1.5  # wide early — let trade develop
elif wave_maturity < 0.7:
    trail_dist = base_trail * 1.0  # standard
else:
    trail_dist = base_trail * 0.5  # tight late — protect gains
```

### Phase B: Fix Direction Model (Second Highest Impact)

#### B1. Add Direction Confidence Gate

Don't trade when direction model is uncertain:
```python
p_long = _logistic_prob(feat_s, lib)
dir_confidence = abs(p_long - 0.5)  # 0.0 = no idea, 0.5 = certain

if dir_confidence < 0.15:
    skip  # direction model is guessing — don't trade
```

This filters out the ~30-40% of trades where the model has no directional edge.

#### B2. Fix SHORT Bias

The system takes 62% SHORT trades but only gets direction right 35% of the time
on shorts. Something in the direction hierarchy is biased toward SHORT.

**Diagnostic first** — log `direction_source` on every trade:
```python
# Which level of the hierarchy decided direction?
# 1=oracle, 2=logistic, 3=template_bias, 4=dmi, 5=velocity
```

Then identify: is the logistic model SHORT-biased? Is DMI fallback the culprit?
If logistic `dir_coeff` is poorly trained, consider:
- Requiring minimum 30 samples per cluster for logistic fit (vs current minimum)
- Falling back to belief network direction consensus instead of DMI

#### B3. Multi-Timeframe Direction Consensus

Before entering, check if ≥2 of the top-3 weighted workers agree on direction:
```python
worker_dirs = [(w.tf, w.belief.direction) for w in active_workers[:3]]
long_votes = sum(1 for _, d in worker_dirs if d == 'LONG')
consensus_dir = 'LONG' if long_votes >= 2 else 'SHORT'
consensus_strength = max(long_votes, 3 - long_votes) / 3  # 0.67 or 1.0
```

Only trade if `consensus_strength >= 0.67` (at least 2 of 3 agree).

### Phase C: Improve Capture Rate (Exit Optimization)

#### C1. Dynamic Profit Target Based on Oracle Stats

Currently TP is fixed from template stats. But 83.3% of trades have MFE > TP,
meaning we're leaving the majority of the move on the table.

```python
# If trade reaches TP and conviction is still high:
if unrealized_pnl >= tp_target and belief_conviction >= 0.6:
    # Move to "runner mode" — raise TP to 1.5x, tighten trail
    new_tp = tp_target * 1.5
    trail_dist = trail_dist * 0.6  # protect the base gain
```

#### C2. Use Decay Cascade for Smart Exits

The decay cascade system was just cherry-picked from unified-cluster. Wire it in:
```python
cascade = belief_network.get_decay_cascade()
if cascade['should_exit']:
    exit(reason='decay_cascade', detail=f"score={cascade['cascade_score']:.2f}")
```

This gives physics-informed exits that respond to z-score drift from the expected
trajectory, rather than blind trailing.

#### C3. Structural Break as Primary Exit

Structural breaks have 78.3% WR and 23.8% capture — the best exit signal by far.
Make structural integrity checks more frequent and responsive:
- Check every bar instead of every N bars
- Lower the structural death threshold slightly (currently too conservative?)

## Measurement & Verification

After EACH phase (A, B, C), run:
```bash
python tools/run_benchmark.py --tag "phase_A" --is-only
```

**Phase A success** (trail fix alone):
- Trail stop WR should rise from 0.9% to ≥30%
- Overall WR should rise from 37.5% to ≥55%
- Capture rate should rise from 0.1% to ≥20%

**Phase B success** (+ direction fix):
- Wrong direction should drop from 46.1% to ≤30%
- Overall WR should rise to ≥65%
- Trade count may drop (filtering uncertain trades is OK)

**Phase C success** (+ exit optimization):
- Capture rate ≥80% of MFE should reach ≥50% of trades
- Overall WR should reach ≥75%
- PnL per trade should rise significantly

**Combined target**: 80% WR, 80% optimal exits, 50% wrong direction reduction.
Trade count reduction is acceptable — quality over quantity.

## Implementation Order

```
Phase A (trail fix)  → benchmark → commit
Phase B (direction)  → benchmark → commit
Phase C (exits)      → benchmark → commit
```

Each phase is independently valuable. Do NOT skip benchmarking between phases.

## Files to Modify

| File | Phase | Changes |
|------|-------|---------|
| `training/wave_rider.py` | A | SL/trail separation, activation threshold, adaptive trail |
| `training/orchestrator.py` | A,B | Conviction-scaled SL, direction confidence gate, consensus check |
| `training/timeframe_belief_network.py` | B,C | Direction consensus method, decay cascade wiring |

## What NOT to Change

- Do NOT change the clustering algorithm or template library
- Do NOT change the feature vector (16D)
- Do NOT change the data pipeline or ATLAS structure
- Do NOT add new dependencies
- Do NOT modify the 1s inner loop bar aggregation
- Do NOT change the benchmark tooling (`tools/run_benchmark.py`)

## Priority

**CRITICAL** — This is the core performance improvement task. All three metrics
are interconnected through the trail stop bug. Phase A alone should move the needle
significantly.
