# Session Context Features — Live Intraday Metrics
> Date: 2026-04-05
> Status: PARKED — design captured

## Problem
1D features in the 79D are dead (zero variance). They compute daily aggregates
that don't change bar to bar. But intraday session context IS useful — it just
needs to be computed differently.

## Proposed Features (5-6, added to feature vector)

### Knowable at open:
- `prev_day_return`: previous day close vs open (red/green, magnitude)
- `prev_day_range`: previous day high-low in ticks (volatile vs quiet day)

### Grows with each bar (no lookahead):
- `session_range_so_far`: (day_high - day_low) / tick — how volatile is today so far
- `position_in_range`: (price - day_low) / (day_high - day_low) — near high (1.0) or low (0.0)
- `distance_from_open`: (price - open) / tick — trending away or oscillating
- `bars_since_extreme`: bars since last daily high or low was set — momentum fading?

### Why these matter for ExNMP:
- Near daily high + long reversion = risky (hitting ceiling)
- Wide session range + z extreme = stronger reversion (more room to move)
- Red previous day + z extreme down = continuation risk (not reversion)
- Near open + narrow range = early session, patterns differ from mid-day

### Implementation:
- Computed in aggregator or a new session tracker
- Added to feature vector (79D → 85D)
- Requires: tracking day_open, day_high, day_low, prev_day stats
- The aggregator already has bar history — session stats are trivial to compute

### Connection to 1D features:
- Replace the 14 dead 1D features (all zero) with these 6 live session features
- 79D becomes 71D (drop 14 dead 1D, add 6 session context)
- Or keep as 85D and let the tree decide what matters

### Prerequisite:
- Validate that these features have discriminative power (run derive_physics with them)
- Need access to previous day's data at session open
