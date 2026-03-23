# Research: Fibonacci Price Structure for PhysicsEngine

**Status:** Open
**Created:** 2026-03-22
**Priority:** High — biggest known edge gap in PhysicsEngine

## Problem

PhysicsEngine's 12 features are all physics (fm, z, dmi, adx, vel, vol, hurst, P_center,
coherence, sigma, pid). None encode WHERE in the price structure the bar sits. Two identical
trajectories produce the same match, but one at a structural high (no room) and one at a
structural low (full room to run).

Physics leads ENTRY correctly. But EXIT (funnel flip) is also purely physics — it doesn't
know if the peak happened at a meaningful price level or noise.

## Hypothesis

5-day rolling high/low defines the weekly trading range. Fibonacci retracement levels
(0.236, 0.382, 0.5, 0.618, 0.786) mark structural price levels where reversals cluster.
Adding price position relative to these levels will:
1. Improve entry quality (skip entries near structural resistance in trade direction)
2. Improve exit timing (hold through noise peaks, exit at structural peaks)
3. Separate real peaks from noise peaks (peaks AT Fibonacci = real, peaks between = noise)

## Features to Compute

1. **fib_position** — where current price sits in 5-day range (0.0=low, 1.0=high)
2. **dist_nearest_fib** — distance to nearest Fibonacci level (ticks)
3. **fib_level_id** — which Fibonacci zone (0-0.236, 0.236-0.382, etc.)
4. **range_width** — 5-day high-low in ticks (context for how wide the structure is)

## Research Steps

### Step 1: Validate Fibonacci clustering in existing seeds
- Load enriched auto seeds (38K)
- For each seed entry, compute 5-day high/low from ATLAS 1m data
- Compute fib_position at entry
- Plot: PnL distribution by fib_position bucket
- Question: do seeds entered near Fibonacci levels have different PnL?

### Step 2: Validate for peak seeds separately
- Load pivot seeds (258K peaks)
- Same analysis: fib_position at peak time
- Plot: real vs fake peak rate by Fibonacci zone
- Question: are real reversals clustered at Fibonacci levels?

### Step 3: Add to PhysicsEngine trajectory
- Expand TRAJ_KEYS from 12 to 14-16 features
- Re-enrich seeds with fib_position and dist_nearest_fib per bar
- Re-run OOS with expanded trajectory
- Compare: 12-feat vs 14-feat vs 16-feat performance

### Step 4: Price-aware exit filter
- On funnel flip: check if price is near a Fibonacci level
- If yes: real structural peak, exit immediately
- If no: noise peak, hold (or tighten SL instead of hard exit)

## Data Requirements

- ATLAS 1m data (already available)
- 5-day lookback per bar (rolling window from aggregator)
- Enriched seeds re-generated with Fibonacci features

## Expected Impact

Current: $264/day OOS (physics-only, no price awareness)
Target: $400+/day (physics + Fibonacci structure)
Basis: price-aware exits should capture more of each move by holding through
noise peaks and exiting at structural peaks.
