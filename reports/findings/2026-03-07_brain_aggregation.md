# How the Brain Aggregates Signals Across Time
> Date: 2026-03-07 | Source: core/timeframe_belief_network.py

## Signal Flow: From Raw Market State to Trade Direction

### Stage 1: Quantum Engine -> Per-TF State
Each of 11 timeframes (4h, 1h, 30m, 15m, 5m, 3m, 1m, 30s, 15s, 5s, 1s) has
pre-computed `ThreeBodyQuantumState` objects from the quantum field engine.

Key state fields used by workers:
- `z_score` — deviation from regression center (SE bands)
- `velocity` — dp/dt (particle velocity, signed)
- `net_force` — d2p/dt2 (acceleration, signed)
- `pattern_maturity` — how developed the current move is (0-1)
- `reversion_probability` — P(revert to center)
- `regression_sigma`, `regression_center` — SE band parameters

### Stage 2: Worker Tick (event-driven, NOT continuous)
Each TF worker updates ONLY when its TF bar closes:
- 4h worker: updates every 960 base bars (4 hours at 15s resolution)
- 1h worker: updates every 240 base bars
- 5m worker: updates every 20 base bars
- 1m worker: updates every 4 base bars
- 15s worker: updates every bar
- Sub-resolution (5s, 1s): mapped from base bars

**Between bar closes, the worker's belief is FROZEN.**
A 4h worker literally holds the same belief for up to 4 hours.

File: `timeframe_belief_network.py:170-201` (tick/tick_at methods)

### Stage 3: Per-Worker Direction (template bias + physics blend)
When a worker ticks, `_analyze()` computes:

1. **Feature extraction**: 16D vector from quantum state
2. **Cluster matching**: find nearest template(s) in pattern library
3. **Template direction**: `_logistic_prob(features, template)` -> P(LONG)
   - If fitted logistic model: `sigmoid(dot(features, coefficients))`
   - If NOT fitted: `long_bias / (long_bias + short_bias)`
   - Example: template #21 has long_bias=0.03 -> P(LONG) = 0.03 (97% SHORT)

4. **Physics direction**: momentum-aware (NOT mean-reverting z-score)
   ```
   momentum = velocity + 0.5 * acceleration
   phys_dir = sigmoid(momentum * sensitivity)
   sensitivity = 0.5 + 0.5 * log(bars_per_update) / log(240)  [0.5 to 1.0]
   ```
   - Higher TFs get higher sensitivity (more bars aggregated = stronger signal)
   - Positive momentum -> P(LONG) high, negative -> P(SHORT) high

5. **Blend**: `dir_prob = 0.5 * physics + 0.5 * template` (if fitted)
   - If no fitted model: `dir_prob = physics` (pure physics fallback)

**PROBLEM**: The 50/50 blend is the same for all TFs. A 4h worker gets the
same template weight as a 1s worker. But at 4h scale, trends dominate and
physics should have MORE weight. At 1s scale, noise dominates and template
patterns may be more relevant.

File: `timeframe_belief_network.py:219-315` (_analyze method)

### Stage 4: Band Context (per worker)
Each worker also computes SE band position from its quantum state:
- `z_score` -> which sigma band (-3 to +3)
- `at_support` = z <= -1.0
- `at_resistance` = z >= +1.0
- `band_position` = continuous [-1, +1]

This is a per-TF snapshot. The 1m worker's z_score oscillates rapidly
(hitting +1 resistance on every micro-rally), while the 4h z_score is stable.

File: `timeframe_belief_network.py:287-303`

### Stage 5: Network Aggregation (get_belief)
All active worker beliefs are combined into one BeliefState:

**Direction**: Weighted geometric mean of all worker dir_probs
```python
weights = [5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.25, 0.1]
#           4h   1h   30m  15m   5m   3m   1m  30s  15s    5s    1s
w_normalized = weights / sum(weights)
path_long = exp(dot(w_normalized, log(dir_probs)))
path_short = exp(dot(w_normalized, log(1 - dir_probs)))
```

Geometric mean means: if ANY high-weighted worker says strongly SHORT, the
product gets pulled toward SHORT. With 11 workers and most templates biased
SHORT, the geometric mean amplifies this — even if physics on 2-3 slow TFs
says LONG, the 8 other workers pulling SHORT dominates.

**Band confluence override** (NEW, 2026-03-07):
After geometric mean, if multi-TF bands agree on direction with strength >0.3:
```python
bc_weight = strength * 0.4  # max 40% influence
path_long = path_long * (1 - bc_weight) + bc_weight * 0.75  # blend toward LONG
```
This provides a structural trend signal that bypasses per-worker template bias.

**Final direction**: whichever path (long/short) has higher probability wins.

File: `timeframe_belief_network.py:559-637`

### Stage 6: Band Confluence Aggregation
`get_band_confluence()` aggregates band positions across all workers:
- Each worker's band z_score weighted by TF_WEIGHTS
- Support score: sum of (weight * |z|) for workers at_support
- Resistance score: sum of (weight * z) for workers at_resistance
- Direction: LONG if support > 2*resistance AND support > 0.5
- Direction: SHORT if resistance > 2*support AND resistance > 0.5
- Otherwise: MIXED (no signal)

**Used in two places:**
1. Entry: blended into path probabilities in get_belief() (40% max)
2. Exit: tighten/widen/urgent signals in get_exit_signal()

File: `timeframe_belief_network.py:770-835`

### Stage 7: Exit Signal (band + time + conviction)
`get_exit_signal()` for position management:
- `band_tighten`: price approaching resistance (LONG) or support (SHORT)
- `band_widen`: price at support (LONG) or resistance (SHORT) AND aligned
- `band_urgent`: band direction flips against trade with strength >0.5
- `time_tighten` / `time_urgent`: from avg_mfe_bar timescale exhaustion
- `wave_mature`: from wave_maturity threshold

**EXIT PROBLEM**: Fast TFs (1m, 5m) hit local resistance bands constantly
during trends. Their at_resistance=True triggers band_tighten, even though
4h/1h bands still say "trending, at support." The exit logic checks
per-worker bands individually, not the aggregated confluence direction.

File: `timeframe_belief_network.py:940-994`

## Three Problems Identified

### Problem 1: Template Bias is TF-Agnostic
All TF workers use the same pattern library with the same template biases.
Template #21: long_bias=0.03 (97% SHORT) applied identically at 4h and 1s.

**Impact**: At 4h scale, trend should dominate (physics). But the 50/50
blend means template #21 forces P(LONG)=0.39 even when physics says 0.75.
Multiply this across 8+ workers with similar bias → geometric mean is
heavily SHORT even during LONG trends.

**Evidence**: 574 counter-trend scalps, workers agree with oracle 57.4%
but dir_prob stays SHORT. Templates #21, #54, #55 produce most scalps.

### Problem 2: Workers Update Only on Bar Close (Stale Signals)
The 4h worker freezes for 4 hours. During that time:
- Price may rally 50+ ticks
- 1m worker updates 240 times with noisy micro-signals
- The aggregated belief shifts because fast workers update while slow
  workers are frozen at their last bar-close state

**Impact**: Entry direction may be correct at bar close, but between bar
closes the fast TF noise gradually shifts the aggregate. By the time the
next 4h bar closes, the trade may already be exited.

### Problem 3: Exit Side Ignores Trend Agreement
Band confluence exit logic (get_exit_signal) checks each worker's band
individually. A 1m worker at z=+1.2 (resistance) triggers band_tighten
for a LONG trade. But the 4h worker at z=-0.5 (still trending up from
support) doesn't suppress this tighten.

The exit side uses `band_confluence['direction']` only for URGENT exits
(full direction flip). For tighten/widen, it looks at aggregate
support_score/resistance_score, which are dominated by fast TF bands
that hit boundaries more frequently.

**Impact**: Correct-direction trades exit too early (207 trades captured
<20% of oracle move) because fast TF bands keep tightening the trail.

## Proposed Fixes (priority order)

### Fix 1: TF-Scaled Physics Weight (small change, high impact)
```python
# Current: fixed 50/50
dir_prob = 0.5 * phys_dir + 0.5 * template_prob

# Proposed: TF-scaled
phys_weight = 0.3 + 0.5 * (tf_seconds / 14400)  # 0.3 at 1s, 0.8 at 4h
dir_prob = phys_weight * phys_dir + (1 - phys_weight) * template_prob
```
At 4h: 80% physics, 20% template (trend dominates)
At 1m: 30% physics, 70% template (pattern dominates)
At 1s: 30% physics, 70% template

Scope: 2 lines in _analyze(). Can test immediately.

### Fix 2: Exit Trend Guard (medium change, targets too-early exits)
When band_confluence['direction'] matches trade side AND strength > 0.5,
suppress fast-TF band_tighten signals:
```python
if bc_direction == side and bc_strength > 0.5:
    _band_tighten = False  # don't tighten, trend is with us
```

Scope: 3-5 lines in get_exit_signal(). Can test immediately.

### Fix 3: Partial Bar Aggregation (large change, comprehensive fix)
Blend completed bar state with forming bar state, weighted by bar maturity.
Addresses both stale-signal and noise problems simultaneously.

Scope: Jules-sized. Worker tick loop, quantum engine partial states, band
context interpolation. See `docs/ROADMAP.md` item #2.

## Data References
- Analysis tools: `tools/analyze_scalps.py`, `analyze_scalp_timing.py`,
  `analyze_scalp_vs_early_exit.py`, `analyze_wrong_dir.py`
- Related findings: `2026-03-07_scalp_timescale.md`
- Code: `core/timeframe_belief_network.py` (lines referenced above)
