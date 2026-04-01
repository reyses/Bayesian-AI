# Probabilistic Forward Pass + 4-Brain Architecture

## Why

The deterministic system (templates → gates → binary direction → hard exits) has
structural limits: binary decisions where reality is probabilistic, hard gates that
block good trades, a brain that counts wins/losses instead of calibrating probabilities,
and exits driven by timers instead of probability collapse.

The probabilistic system replaces binary decisions with continuous probability. Templates
shift from entry gatekeepers to exit calibration lookups. The Bayesian brain shifts from
direction correction to probability calibration across four frozen/live layers.

## Architecture Overview

```
                   CNN (frozen)
                       |
              P(long) at 10 horizons
                       |
            +----------+----------+
            |                     |
     IS Brain (frozen)    Template Library
            |                (exit params)
     OOS Brain (frozen)          |
            |                    |
     Live Brain (learns)         |
            |                    |
       Calibrated P(long)   MFE/MAE/SL/TP
            |                    |
            +-------- TRADE -----+
```

## Layer 1: ProbabilisticTrajectory CNN (frozen)

**Already built**: `training/train_probabilistic.py`

- Input: 10-bar lookback x 22D features (13D base + 4D wave + 2D prob + 3D level)
- Output: 10 horizons x (22D predicted features + P(long)) = 230D
- Per-feature-group encoders: base(13D), wave(4D), prob(2D), level(3D)
- Each horizon head: 64D merged → 48D → 23D (22 features + 1 P(long))

**What it provides per bar:**
- P(long) at n+1, n+2, n+3, n+4, n+5, n+6, n+7, n+8, n+9, n+10
- 22D predicted feature state at each horizon
- The SHAPE of the P(long) curve is the signal, not individual values

**Entry signal**: P(long) > entry_threshold at n+1 through n+3 (immediate conviction)
**Exit signal**: P(long) crosses 0.5 (direction flip) or collapses toward 0.5 (uncertainty)

## Layer 2: Template Library (exit calibration)

**Already built**: `core/fractal_clustering.py`, `core/shape_primitives.py`

Templates shift from entry gatekeeping to exit management:

| Old Role (deterministic) | New Role (probabilistic) |
|--------------------------|--------------------------|
| Gate: distance < threshold | Not used for entry |
| Direction: template long_bias | Not used (CNN decides) |
| Confidence: distance to centroid | Not used |
| Exit params: MFE, MAE, giveback | **KEPT — primary role** |
| Shape calibration: peak bar, envelope | **KEPT — primary role** |

**How templates are matched:**
Once CNN triggers an entry, the current feature vector is matched to the nearest
template centroid. This doesn't gate the trade — it provides exit parameters:
- `p75_mfe_ticks` → TP target
- `p25_mae_ticks` → SL distance
- `expected_peak_bar` → when to tighten trail
- `giveback_pct` → retracement tolerance
- `envelope_halflife_mult` → decay rate

If no template matches within a loose distance, use default exit params.

## Layer 3: 4-Brain Cascade (probability calibration)

### Brain Architecture

Each brain is an instance of `BayesianBrain` — a probability table indexed by
(template_id, direction, feature_bucket). It stores:
- Observed win rate for this (template, direction) combo
- Running average PnL
- Sample count (confidence grows with N)
- Calibration offset: `actual_win_rate - cnn_predicted_probability`

### The Four Brains

#### Brain 0: CNN (frozen, offline)
- Raw P(long) output from ProbabilisticTrajectory CNN
- Never changes after training
- Checkpoint: `checkpoints/probabilistic/best_model.pt`

#### Brain 1: IS Brain (frozen, post-IS)
- Built during IS forward pass
- For each trade: CNN predicted P(long) = X, actual outcome = W/L
- Accumulates calibration: "when CNN says 0.72, IS reality was 0.68"
- Frozen after IS completes
- Checkpoint: `checkpoints/brains/is_brain.pkl`

#### Brain 2: OOS Brain (frozen, post-OOS)
- Starts as copy of IS Brain
- Updated during OOS forward pass with OOS trade outcomes
- Provides validation-calibrated probabilities
- Frozen after OOS completes
- Checkpoint: `checkpoints/brains/oos_brain.pkl`

#### Brain 3: Live Brain (mutable, production)
- Starts as copy of OOS Brain
- Updated after every live trade
- Continuously recalibrates as market conditions evolve
- Checkpoint: `checkpoints/brains/live_brain.pkl` (saved after every trade)
- Rollback: reset to OOS Brain at any time

### Calibration Math

```python
# CNN predicts:
p_cnn = 0.72  # raw P(long) at n+2

# IS Brain observed: 50 trades where CNN said ~0.72, 34 won
p_is = is_brain.calibrate(template_id, 'LONG', p_cnn)  # → 0.68

# OOS Brain observed: 15 more trades, 9 won
p_oos = oos_brain.calibrate(template_id, 'LONG', p_is)  # → 0.65

# Live Brain observed: 8 trades this week, 5 won
p_live = live_brain.calibrate(template_id, 'LONG', p_oos)  # → 0.63

# Final decision uses p_live
if p_live > entry_threshold:
    enter(direction='LONG', confidence=p_live)
```

Each brain applies a Bayesian update:
```
P_calibrated = (N_observed * observed_rate + prior_weight * P_input) / (N_observed + prior_weight)
```
Where `prior_weight` decays as N_observed grows (more data = less reliance on prior).

### Rollback Chain

```
Live brain bad?  → reset to OOS brain checkpoint
OOS brain stale? → reset to IS brain checkpoint
IS brain stale?  → retrain CNN, rebuild all four
```

## Trading Loop (Probabilistic Forward Pass)

```python
for each bar:
    # 1. CNN prediction (frozen)
    trajectory = cnn.predict(last_10_bars)  # 10 x P(long)
    p_raw = trajectory[0:3].mean()  # avg P(long) at n+1 through n+3

    # 2. Brain calibration cascade
    p_is = is_brain.calibrate(nearest_template, direction, p_raw)
    p_oos = oos_brain.calibrate(nearest_template, direction, p_is)
    p_live = live_brain.calibrate(nearest_template, direction, p_oos)

    # 3. Entry decision
    if not in_position:
        if p_live > ENTRY_THRESHOLD:  # e.g., 0.65
            direction = 'LONG' if p_live > 0.5 else 'SHORT'
            template = match_nearest_template(features)
            exit_params = template.get_exit_params()
            enter(direction, exit_params)

    # 4. Position management (every bar while in trade)
    if in_position:
        # Recalculate trajectory
        new_trajectory = cnn.predict(last_10_bars)
        p_now = brain_cascade(new_trajectory, template)

        # Exit conditions (checked in order):
        # a. Hard SL from template MAE
        if unrealized_loss > exit_params.sl_ticks:
            exit('stop_loss')

        # b. Probability collapse: P(direction) dropped below hold threshold
        if p_now < HOLD_THRESHOLD:  # e.g., 0.45
            exit('probability_collapse')

        # c. Trajectory inflection: P(direction) was rising, now falling
        if trajectory_slope < 0 and bars_held > MIN_HOLD:
            tighten_trail(exit_params.giveback_pct)

        # d. Template peak bar reached: expected MFE timing hit
        if bars_held >= exit_params.expected_peak_bar:
            activate_trail(exit_params.trail_ticks)

        # e. Trail stop hit
        if trail_triggered:
            exit('trail_stop')

    # 5. Post-trade learning (live brain only)
    if trade_just_closed:
        live_brain.update(template_id, direction, p_at_entry, outcome)
```

## Key Constants (TradingConfig)

```python
# === Probabilistic System ===
prob_entry_threshold: float = 0.65     # min calibrated P(direction) to enter
prob_hold_threshold: float = 0.45      # P below this = probability collapse exit
prob_horizon_entry: list = [0, 1, 2]   # horizons to average for entry (n+1 to n+3)
prob_min_hold_bars: int = 2            # minimum bars before trail/collapse exits
prob_brain_prior_weight: float = 10.0  # prior weight in Bayesian calibration
prob_template_match_dist: float = 10.0 # loose match for exit params (not gating)
```

## Files to Build

| File | Purpose |
|------|---------|
| `core/probabilistic_engine.py` | **NEW** — trading loop: entry/exit/position management |
| `core/brain_cascade.py` | **NEW** — 4-brain calibration chain |
| `training/train_probabilistic_forward.py` | **NEW** — IS/OOS forward pass building IS+OOS brains |
| `live/prob_launcher.py` | **NEW** — live launcher for probabilistic system |

## Files to Reuse (no changes)

| File | What it provides |
|------|-----------------|
| `training/train_probabilistic.py` | CNN training (already done) |
| `core/trade_cnn.py` / `core/direction_cnn.py` | Model architectures |
| `core/fractal_clustering.py` | Template building (exit params) |
| `core/shape_primitives.py` | Geometry features |
| `core/exit_engine.py` | SL/trail/envelope mechanics |
| `core/bayesian_brain.py` | Base brain class (instantiated 3x) |

## Build Phases

```
Phase A: core/brain_cascade.py
         - BrainCascade class wrapping 4 brain instances
         - calibrate(template, direction, p_raw) → p_final
         - update(template, direction, p_entry, outcome)
         - freeze() / rollback(checkpoint)

Phase B: core/probabilistic_engine.py
         - ProbabilisticEngine class
         - process_bar() → entry/exit/hold decisions
         - Uses CNN + BrainCascade + template exit params
         - No gates, no deterministic direction

Phase C: training/train_probabilistic_forward.py
         - IS pass: run ProbabilisticEngine on IS data
         - Build IS Brain from outcomes → freeze
         - OOS pass: continue with IS Brain → build OOS Brain → freeze
         - Compare to deterministic baseline

Phase D: live/prob_launcher.py
         - Load frozen CNN + OOS Brain → copy to Live Brain
         - Run ProbabilisticEngine on live data
         - Live Brain updates every trade
```

## Verification

1. Train CNN: `python training/train_probabilistic.py --tf 1h` (existing)
2. IS/OOS pass: `python training/train_probabilistic_forward.py --fresh`
3. Compare: IS/OOS PnL vs deterministic baseline ($1,609/day)
4. Live sim: `python -m live.prob_launcher --dry-run` (one session, watched)
5. Check brain divergence: IS brain vs OOS brain calibration offsets

## Success Criteria

- OOS PnL >= $800/day (half of deterministic baseline = acceptable for v1)
- Fewer trades with higher avg PnL (quality over quantity)
- Brain calibration converges (offset shrinks over time, not grows)
- Probability collapse exits capture more MFE than regime_decay
- Live brain doesn't diverge > 0.1 from OOS brain in first week
