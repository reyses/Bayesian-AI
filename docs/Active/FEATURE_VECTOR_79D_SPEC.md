# 79D Unified Feature Vector Spec

## Core Insight
The atomic unit is 5s — the Planck constant of MNQ. Every higher TF is
an aggregation of 5s bars. The 1m bar IS 12 x 5s bars smoothed.
The 1h bar IS 720 x 5s bars smoothed. Same data, different noise levels.

The entire trading problem reduces to: **how many 5s bars of noise
should I hold through before the signal resolves?**

- Hold too short → exit in noise, random result
- Hold too long → signal expired, gave back profit
- Hold just right → captured the move at its natural duration

The 79D state across 6 TFs tells the NN the signal-to-noise ratio at
every smoothing level simultaneously. The NN learns the **half-life
of each signal** — when the edge decays back to noise.

## Design Principle
One set of features measured at every timeframe. No "structural" vs "grounded" split.
The fractal state IS the feature vector — same measurements at 15s, 1m, 5m, 15m, 1h, 1D.
Each TF is just a different aggregation window on the same 5s atomic data.

## Timeframes (6)
| TF | Role |
|----|------|
| 15s | Execution (entry timing, tick-level exits) |
| 1m | Anchor (decision timeframe) |
| 5m | Short structure |
| 15m | Intraday trend |
| 1h | Session trend |
| 1D | Daily bias |

## Core Features (10 per TF = 60D)
Cluster at full weight. These are the primary signals.

| # | Name | What it measures | Proven? |
|---|------|-----------------|---------|
| 0 | `z_se` | Position — where in the regression band (SE units) | Yes (Roche limit) |
| 1 | `dmi_diff` | Direction — DI+ minus DI- (raw) | Yes |
| 2 | `variance_ratio` | Regime — short/long vol ratio. <1 = mean-revert, >1 = trending | Yes (lambda) |
| 3 | `velocity` | Momentum — price rate of change | Yes |
| 4 | `acceleration` | Chop/buildup — velocity change. Oscillating = chop, sustained = power | Yes |
| 5 | `vol_rel` | Conviction — volume vs 30-bar SMA. High = climax or conviction | Yes (exhaustion fade) |
| 6 | `bar_range` | Risk — (high-low)/tick per bar | Yes |
| 7 | `hurst` | Persistence — Hurst exponent. >0.5 = trending, <0.5 = mean-revert | Yes (top template differentiator) |
| 8 | `reversion_prob` | P(revert) — from OU first-passage error function | Computed, untested standalone |
| 9 | `p_at_center` | P(at mean) — 3-class probability distribution | Computed, untested standalone |

## Helper Features (3 per TF + 1 global = 19D)
Cluster at reduced weight (0.5x after StandardScaler). Derived from core features
but carry information that helps separate patterns.

| # | Name | Derived from | Why keep |
|---|------|-------------|----------|
| 0 | `dmi_gap` | abs(dmi_diff) | Strength regardless of direction |
| 1 | `dir_vol` | sign(velocity) * vol_rel | Volume WITH vs AGAINST the move |
| 2 | `wick_ratio` | 1 - abs(close-open)/range | Rejection/acceptance signal per candle |
| + | `time_of_day` | timestamp % 86400 / 86400 | Session context (1 value, global) |

## Total: 60D core + 19D helper = 79D

## Layout (column order)
```
# 15s core (0-9)
15s_z_se, 15s_dmi_diff, 15s_variance_ratio, 15s_velocity, 15s_acceleration,
15s_vol_rel, 15s_bar_range, 15s_hurst, 15s_reversion_prob, 15s_p_at_center,

# 1m core (10-19)
1m_z_se, 1m_dmi_diff, 1m_variance_ratio, 1m_velocity, 1m_acceleration,
1m_vol_rel, 1m_bar_range, 1m_hurst, 1m_reversion_prob, 1m_p_at_center,

# 5m core (20-29)
5m_z_se, ...

# 15m core (30-39)
15m_z_se, ...

# 1h core (40-49)
1h_z_se, ...

# 1D core (50-59)
1d_z_se, ...

# 15s helpers (60-62)
15s_dmi_gap, 15s_dir_vol, 15s_wick_ratio,

# 1m helpers (63-65)
1m_dmi_gap, 1m_dir_vol, 1m_wick_ratio,

# 5m helpers (66-68), 15m (69-71), 1h (72-74), 1D (75-77)
...

# Global (78)
time_of_day
```

## What was dropped and why
| Feature | Reason |
|---------|--------|
| `adx` | Smoothed abs(dmi_diff). Redundant — dmi_diff across TFs IS trend strength |
| `momentum` (abs) | Velocity of velocity = acceleration. Already kept |
| `price_accel` | Same as acceleration. Kept under better name |
| `std_price` | Captured by bar_range + variance_ratio |
| `vwap_distance` | z_se already measures distance from regression mean |
| `entropy_normalized` | 0.07 sigma separation. Dead signal |
| `pid_signal` | 0.04 sigma separation. Dead signal |
| `oscillation_tightness` | Captured by variance_ratio + hurst |
| `breakout_prob` | 1 - reversion_prob. Literally redundant |
| `p_near_upper/lower` | Derivable from p_at_center + sign(z_se) |
| `reversion_potential` | Formula of z_se (0.025*(9-z^2)). Redundant |
| `regime_stability` | Captured by variance_ratio + hurst |
| `parent_extreme` | Binary. Replaced by actual z_se at each TF |
| `parent_z` | Replaced by z_se at each TF |
| `parent_dmi` | Replaced by dmi_diff at each TF |
| `root_extreme` | Binary. Replaced by z_se at root TF |
| `tf_alignment` | Derivable from dmi_diff signs across TFs |
| `tf_scale` | TFs are fixed columns, not a variable |
| `fractal_depth` | Discovery artifact, irrelevant with per-TF columns |
| `path_slope` | = velocity. Already kept |
| `path_curvature` | = acceleration. Already kept |
| `path_efficiency` | = hurst. Same concept |
| `path_range` | = bar_range at higher TF |
| `path_end_position` | = z_se. Same concept |
| `path_monotonicity` | = hurst. Same concept |
| `lookback_0..5` | Replaced by path geometry which was replaced by core features at each TF |

## How to read the fractal state
Example: "I'm at 3 sigma on 1m, bouncing off daily support"
```
1m_z_se = 3.0       # extreme on anchor
1m_variance_ratio = 0.4  # mean-revert regime
1m_reversion_prob = 0.85 # high probability of revert
1h_z_se = 0.3       # near mean at session level
1h_dmi_diff = 5.2   # session trending up
1d_z_se = -1.8      # at daily lower band (support)
1d_dmi_diff = -2.1  # daily bearish but at support
```
This IS the nightmare protocol in numbers.

## Implementation
1. SFE computes MarketState per TF (already does this)
2. New function: `extract_79d(states_by_tf: dict) -> np.ndarray`
3. Input: `{'15s': MarketState, '1m': MarketState, '5m': MarketState, ...}`
4. Output: 79-float array in the layout above
5. Missing TFs get zeros (warmup period)

## Performance Ceiling (cord length analysis, 2026-04-03)
The bar-to-bar |delta| = theoretical max PnL per trade at that hold duration.
Sum of deltas per day = ceiling if every trade was perfectly timed.

| TF | Median Move | Daily Cord (ceiling) | At 55% WR (10% net capture) |
|----|-------------|---------------------|----------------------------|
| 1s | $1.00 | $95,134 | $9,513/day — but $0.50 cost = needs 75% WR |
| 5s | $2.00 | $47,040 | $4,704/day — viable at 55% WR |
| 15s | $2.50 | $27,277 | $2,728/day |
| 1m | $4.50 | $13,384 | $1,338/day |
| 5m | $10.50 | ~$5,000 | $500/day |

Move scales as sqrt(time) — random walk diffusion (~30% per TF step).
Cost is fixed ($0.50/RT = 1 tick). Higher frequency = more opportunities but cost drag grows.

Sweet spot: **1m decision, 5s execution**
- 1m gives $4.50 median per directional call (73% of bars > $2)
- 5s gives 12 entries per minute to optimize timing
- 5% net capture at 5s = $2,350/day target

## Weighting for clustering
After StandardScaler:
- Core features (dims 0-59): weight 1.0
- Helper features (dims 60-78): weight 0.5
Applied by multiplying helper columns by 0.5 before K-Means distance calc.

---

## Target Architecture: Strategy Router NN

### Concept
Every bar is tradable. The NN doesn't predict price — it selects which strategy
to deploy given the current 79D fractal state. Some bars the answer is "no_trade".

### Two-layer decision
```
Layer 1 — DECISION (1m anchor):
  Every 1m bar: compute 79D state across all 6 TFs
  NN classifies → strategy_id (or no_trade)
  Output: strategy + exit params

Layer 2 — EXECUTION (5s cord length):
  Drop to 5s resolution within the 1m bar
  Strategy rules handle entry timing + exits
  5s captures the actual price path, not the 1m summary
  A "flat" 1m bar may have +20 pt swings at 5s
```

### Strategy classes (initial set, expand from data)
| ID | Strategy | Entry | Exit | When |
|----|----------|-------|------|------|
| 0 | `no_trade` | — | — | Chop, unclear state, or risk too high |
| 1 | `reversion_tight` | Fade z_se extreme | Mean reached or 8-bar max hold | z_se extreme + vr < 1 + stable regime |
| 2 | `reversion_wide` | Fade z_se extreme | Wider target, trail stop | z_se extreme + multi-TF confirms |
| 3 | `trend_ride` | Enter with trend | Trail stop, trend validity check | Strong trend + dmi aligned across TFs |
| 4 | `exhaustion_fade` | Fade climax spike | Quick exit, tight stop | High vol_rel + z extreme + acceleration spike |
| 5 | `range_scalp` | Buy low / sell high | Mean target | Low bar_range + high p_at_center + hurst < 0.5 |

### NN Output: Direction + Duration + Expected PnL

The NN doesn't just pick a strategy — it picks **how long to hold**.

```
79D state → {
  direction: LONG | SHORT | NO_TRADE,
  duration:  5s | 15s | 30s | 1m | 5m | 15m,
  expected_pnl: $X.XX
}
```

A state with z_se=3 + 1h_dmi positive + vr=0.4 might return:
- $0.50 at 5s  (noise)
- $2.00 at 30s (emerging)
- $4.50 at 1m  (visible)
- $18.00 at 15m (full reversion plays out)
→ NN labels this "15m LONG, expected $18"

A state with high vol_rel + z extreme + acceleration spike might return:
- $8.00 at 5s  (quick snapback)
- $4.00 at 30s (fading)
- $1.00 at 5m  (gone)
→ NN labels this "5s SHORT, expected $8" (exhaustion scalp)

### Training pipeline
1. Replay every 1m bar across 311 days of clean ATLAS data
2. At EACH bar compute 79D state
3. Look forward at MULTIPLE durations: 5s, 15s, 30s, 1m, 5m, 15m
4. Record actual PnL for LONG and SHORT at each duration
5. Label = (best_direction, best_duration, pnl) — the hold period
   that maximizes risk-adjusted return from this state
6. ~700 bars/day x 311 days = ~218,000 labeled samples (every bar tradeable)

### Why duration matters more than direction
The nightmare ticker's problem: hardcoded MAX_HOLD=20 bars for everything.
A 30s reversion held for 20 minutes = gave back profit and then some.
A 5m trend exited after 4 bars = left $40 on the table.

The cord length table IS the ground truth:
| Duration | Median Move | What it teaches the NN |
|----------|-------------|----------------------|
| 5s  | $2.00  | Quick scalps, exhaustion fades |
| 15s | $2.50  | Fast reversions |
| 30s | $3.50  | Standard reversions |
| 1m  | $4.50  | Confirmed reversions |
| 5m  | $10.50 | Trend rides |
| 15m | $18.00 | Full trend / session moves |

The NN learns: "this state pays if you wait 5 minutes, don't panic
at the 30s noise." Or: "this state pays NOW, take the $8 and run."

### Exit = Half-Life Decay (already built)
The exit engine (`core/exits/envelope.py`) already implements exponential
half-life decay: `decay = exp(-ln2 * bars_held / half_life)`

The envelope starts at the TP target and decays toward a floor.
It already modulates half-life based on giveback, band exhaustion, ADX slope.

The connection: **the NN predicts the half-life, the envelope executes it.**
- NN says "5m half-life" → effective_hl = 60 bars at 5s
- Envelope decays from peak toward floor over those bars
- If price keeps running, peak resets, envelope extends
- If price stalls, decay catches up, exit triggers

No hardcoded exit strategies. One decay function. The strategy IS the half-life.
A "quick scalp" = half-life of 5 bars. A "trend ride" = half-life of 180 bars.

### Duration = natural TF of the trade
If NN says "5m hold" → that's a 5m bar trade. Enter, hold one 5m bar, exit.
The 5m delta ($10.50 median) is the expected payoff.
Execution still happens at 5s for precise timing within that 5m window.

### Key insight
The NN is NOT the trading system. The strategies ARE the trading system.
The NN picks direction + duration. The execution layer handles entry timing
at 5s resolution and exit management. If the NN is wrong, the strategy's
exit rules (stop loss = cord length of the duration TF) limit the damage.

### Why this works with our data size
- Multi-label classification: direction (3 classes) x duration (6 classes) = 18 classes
- 218,000 samples at 79D with 18 classes = ~12,000 per class
- Shallow network (2-3 hidden layers, 128-64-32) is sufficient
- XGBoost as baseline / interpretable alternative
- Can also frame as regression: predict PnL at each duration directly

### Execution at 5s (cord length)
The 1m bar is a straight line from open to close. The 5s path within that
minute is the actual cord — pullbacks, spikes, the real price action.
At 5s we can:
- Time entry to get a better fill (wait for pullback in trend direction)
- Exit tighter (5s trailing stop instead of waiting for next 1m close)
- Capture intra-bar moves (a "flat" 1m bar can have +20 pt swing at 5s)
- Detect early failure (trade goes wrong in first 10 seconds = cut immediately)
