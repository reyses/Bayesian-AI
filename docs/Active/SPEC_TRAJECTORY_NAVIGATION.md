# SPEC: Trajectory Navigation System

**Status:** Approved architecture — build next session
**Date:** 2026-03-28
**Foundation:** Three DualHeadPredictor models all passing 95% gate

---

## Executive Summary

The system navigates price like a GPS navigates terrain. Three CNN models at
different zoom levels (1h, 1m, 1s) each predict P(D) — the probability that
the next N bars continue in direction D. The decay of P(D) across horizons
measures the model's "sight distance." Entries, exits, and hold decisions
emerge from the trajectory shape and the interaction between timeframes.

No fixed SL. No fixed TP. No fixed hold duration. No L2 duration model. No L3
retreat model. The physics of uncertainty and directional conviction replace
all fixed rules.

---

## Core Concept: P(D) Trajectory

Each model outputs P(D) at multiple forward horizons:

```
P(D) at n+1 = 0.95  → "I can see this clearly"
P(D) at n+2 = 0.88  → "I can still see"
P(D) at n+3 = 0.71  → "getting foggy"
P(D) at n+4 = 0.55  → "can't see past here"
```

Where:
- D = the direction of the current trend (not always LONG)
- P(D) = calibrated probability (via binomial logistic regression) that bar n+k
  continues in direction D
- The decay is not about the market — it's about the MODEL'S SIGHT DISTANCE
- Recalculated from scratch every bar with fresh data

---

## Three Regimes (from P(D) lookback)

### 1. TRENDING
```
N-4=0.90, N-3=0.91, N-2=0.89, N-1=0.92, N=0.90
```
P(D) has been consistently above the chop zone. Model can see clearly.
Trade in direction D. Sight distance is long.

### 2. INFLECTION (peak/trough)
```
N-4=0.92, N-3=0.85, N-2=0.71, N-1=0.58, N=0.48
```
P(D) was high, now rapidly crossing through 50%. The model was confident,
now it's changing its mind. This is the oscillation peak — EXIT.
Not uncertainty — a measured directional change.

### 3. CHOP (no signal)
```
N-4=0.51, N-3=0.49, N-2=0.52, N-1=0.50, N=0.48
```
P(D) has been near 50% for multiple bars. The model genuinely can't see
a direction. No trade at this TF. Go up one level.

The distinction between INFLECTION and CHOP is critical:
- P(D) = 0.48 after being 0.92 → INFLECTION (exit/flip)
- P(D) = 0.48 after being 0.51 → CHOP (no trade)

---

## Three Timeframes, Three Jobs

| TF | Question | Sight Distance | Forward |
|----|----------|---------------|---------|
| 1h | "Where are we going?" | Hours | t+1 (1 hour) |
| 1m | "Where in the wave?" | ~4 bars (~4 min) | t+1 to t+4 |
| 1s | "When to step in?" | ~5 bars (~5 sec) | t+1 to t+5 |

```
1h: ────────────/───────────\────────   (structural trend, long sight)
1m: ──/\──/\──/\──/\──/\──/\──/\────   (oscillation waves, medium sight)
1s: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   (tick precision, short sight)
```

1h doesn't see the oscillation — that's correct. You don't want 1h to see
the waves. The waves are 1m's job.

---

## Trade Logic

### Entry
1. **1h says D=LONG** (P(D) above chop zone, long sight) → only look for longs
2. **1m in CHOP** → wave is pulling back, wait
3. **1m exits chop zone in 1h direction** (P(D=LONG) rises above chop) → new wave starting
4. **1s confirms** (P(D) high at n+1) → step in NOW

### Hold
5. **Every bar**: recalculate 1m trajectory [n+1, n+2, n+3, n+4]
6. **Trajectory still above chop zone** → HOLD (wave still going)
7. **Trajectory extending** (stronger than last bar) → HOLD (trend strengthening)

### Exit
8. **1m trajectory contracting** → wave approaching peak, prepare
9. **1m P(D) entering chop zone** → wave peaked → EXIT
10. **1m P(D) crossing through 50% toward other side** → INFLECTION → EXIT immediately

### No Trade
11. **1m in chop AND 1h in chop** → no direction at any TF, stay flat

### Re-entry
12. **After exit, wait for next 1m chop period** → wave resetting
13. **1m exits chop in 1h direction** → next wave → enter again

---

## Calibration (Binomial Logistic Regression)

Each model's raw P(D) output is calibrated against actual outcomes:

```python
# Fit on validation data
logistic.fit(model_P_long, actual_went_long)

# Produces calibrated probability:
# When model says 80% → actual is 79.9% (1m) or 78.6% (1h)
```

The chop zone boundaries come from this calibration:
- **1h chop zone**: P(D) = 50-62% (CI includes 50%)
- **1m chop zone**: P(D) ~ 42-50% (very narrow — well calibrated)

These are MEASURED from the data, not chosen.

---

## Trajectory Decay = Sight Distance

The trajectory [P(D) at n+1, n+2, n+3, n+4] changes shape based on
market conditions:

```
Strong trend:  [0.95, 0.93, 0.91, 0.88]  → long sight, hold through
Normal wave:   [0.95, 0.85, 0.70, 0.55]  → 3 bars of runway, then exit
Near peak:     [0.80, 0.55, 0.40, 0.35]  → peak is NOW, exit
Volatile:      [0.92, 0.60, 0.52, 0.50]  → can only see 1 bar, be quick
```

The decay shape tells you:
- **Flat**: strong trend, hold
- **Steep**: approaching inflection, prepare to exit
- **Already below 50% at n+2**: the flip is imminent

---

## What This Replaces

| Old System | New System |
|------------|------------|
| Hard SL (40 ticks) | P(D) trajectory entering chop zone |
| Breakeven protection | Not needed — exits before SL hit |
| Trailing stop | Not needed — trajectory contraction IS the trail |
| L2 DurationPredictor | Not needed — sight distance IS the duration |
| L3 RetreatPredictor | Not needed — inflection detection IS the retreat |
| 13-layer exit cascade | Single mechanism: P(D) trajectory shape |
| Fixed confidence threshold | Calibrated chop zone from binomial logistic regression |

---

## Foundation (proven)

| Model | Forward | Dir Acc | Top 5% | P-value | Gate |
|-------|---------|---------|--------|---------|------|
| 1h | t+1 | 78.7% | 95.4% | 0.00 | PASS |
| 1m | t+4 | 75.1% | 96.3% | 0.00 | PASS |
| 1s | t+5 | 72.7% | 96.6% | 0.00 | PASS |

All statistically significant. CI does not include 50% for any model.
Calibration verified via binomial logistic regression.

---

## Build Order

### Phase 1: Multi-Horizon Output Heads
- Modify DualHeadPredictor: one P(D) head per horizon (4 heads for 1m)
- Same backbone, shared features, separate direction heads
- Train with same walk-forward, MSE on state + BCE on each horizon

### Phase 2: Calibration Layer
- Fit binomial logistic regression per horizon per TF on validation data
- Store calibration coefficients alongside model checkpoints
- Raw P(D) → calibrated P(D) → measured chop zone boundaries

### Phase 3: Trajectory Logic
- Per-bar: compute trajectory [P(D) at n+1..n+4]
- Per-bar: compare to previous trajectory (contracting/extending)
- Regime detection from P(D) lookback (trending/inflection/chop)

### Phase 4: Trade Orchestration
- 1h provides D (structural direction)
- 1m provides wave entry/exit (trajectory shape)
- 1s provides timing (step-in confirmation)
- No trade when both 1m and 1h in chop

### Phase 5: Validation
- Run on IS daily, compare to $1,609/day baseline
- Run on OOS
- Run on forward dates (Mar 25-26) that failed the old system

---

## Files

| File | Purpose |
|------|---------|
| `core/direction_cnn.py` | DualHeadPredictor + multi-horizon heads |
| `core/trajectory_engine.py` | NEW: trajectory computation + regime detection |
| `core/calibration.py` | NEW: binomial logistic regression per horizon |
| `training/train_direction.py` | Multi-horizon training pipeline |
| `live/cnn3_layer.py` | Replace with trajectory navigation module |
