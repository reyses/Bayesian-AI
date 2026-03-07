# Liquidation Levels = Three-Body Anchors

## Core Insight (2026-03-04)

The three bodies in the quantum field engine are not abstract statistical constructs —
they are **algorithmic liquidation pools**. Each body is a cluster of stop orders at a
price level. Price orbits between them, gets pulled toward them, bounces off them,
and occasionally breaks through (which reorganizes the entire field).

## How It Works

### 1. Levels (macro, daily TF)
- Visible as horizontal lines through clustered swing peaks on the daily chart
- A day's high can be another day's low at the same level — it's the PRICE that matters
- These are where the big money's stops sit (liquidation zones)
- Identified visually or via peak clustering (DBSCAN on swing H/L prices)

### 2. Physics Confirmation (matrix)
- The three-body states (z-score, barrier height, tunnel probability) confirm
  whether a level is currently active
- "Both sides pulling or pushing" = real force field at that price
- Without confirmation, the level is just a line on a chart

### 3. Pattern (micro, 15m/15s TF)
- The local shape (ramp, V-reversal, step, etc.) tells you what's happening
  at a confirmed level
- Same pattern means different things depending on level context:
  - V-reversal AT a liquidation zone = bounce off the pool
  - V-reversal in empty space = random noise
  - Step THROUGH a liquidation zone = cascade/breakout

### 4. Fractal Property
- Liquidation pools exist at EVERY timeframe and they nest
- Daily level at 22,000 has micro-pools on 1h at 21,950 and 22,080
- Price bounces between micro-pools first, but they orbit the macro attractor
- Eventually collapses to the daily level
- The DNA tree (parent/child TF) already models this nesting
- `depth` and `parent_z` in the 16D vector capture fractal position

## Mapping to Existing System

| Concept | Current Implementation | Missing Piece |
|---------|----------------------|---------------|
| Three-body orbits | z-score from statistical mean | Anchor to real liquidation levels |
| Barrier height | OU potential V(z) = 0z^2/2 | Barriers should be AT the level prices |
| Tunnel probability | erfi-based analytical P(break) | P(break through liquidation zone) |
| Fractal depth | DNA tree parent/child | Parent equilibrium = parent's liquidation level |
| Shape primitives | 20 seed functions | Shape tells you HOW price interacts with the level |
| Pattern context | 16D feature vector | Add: distance to nearest level above/below |

## Seed Primitive Reinterpretation

The 20 shapes describe HOW price moves between liquidation pools:
- **Ramp** = smooth drift toward the attractor
- **V-reversal** = hit the pool and bounced
- **Step** = broke through one pool, snapped to the next
- **Sigmoid** = slow approach to a pool (deceleration)
- **Oscillation** = trapped between two pools
- **Damped oscillator** = settling into a pool (energy dissipation)

## Actionable Signal

All three layers together = trade signal:
1. Where are the liquidation pools? (levels from daily chart)
2. Is the physics confirming activity at this level? (three-body states)
3. What pattern is forming here? (CNN / seed primitive match)

Any one layer alone is incomplete. The combination is the edge.

## Implementation Path

1. **Manual levels first**: User identifies 5-10 active levels from daily chart,
   stores in `checkpoints/price_levels.json`
2. **Distance features**: For each bar, compute dist_to_nearest_level_above/below
3. **Physics anchoring**: Replace statistical mean with nearest level as equilibrium
4. **CNN context**: Feed distance-to-level as dual-path scalar alongside OHLCV shape
5. **Validate**: Does adding level context improve direction accuracy and WR?
6. **Auto-detect later**: Once proven, build algorithmic level detection (peak clustering)
