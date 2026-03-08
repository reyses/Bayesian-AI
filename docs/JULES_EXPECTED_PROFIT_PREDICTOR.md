# Jules Spec: Expected Profit Predictor (E[PnL])

## Core Thesis

The entire system answers two questions:
1. **Should I trade?** → E[PnL] > 0
2. **How much will I make?** → E[PnL] = predicted_ticks × tick_value

Everything else — gates, direction, exits, re-entry — derives from the predicted outcome.
Without E[PnL], every decision is blind.

## Current State

- Workers output: direction (P(long)), conviction, wave_maturity
- Workers do NOT output: expected ticks, predicted hold time
- Exit sizing: generic ATR-based TP/SL, self-tuned halflife from past outcomes
- The waveform research (`tools/standalone_research.py` analysis J) achieved **92% R²**
  on price prediction from physics features
- Seed library (`docs/JULES_WAVEFORM_SEED_INTEGRATION.md`) has 20 mathematical shapes
  that classify price trajectories with fitted parameters

## Architecture

### Phase 1: Worker-Level Prediction

**Each TF worker produces `predicted_ticks` alongside existing outputs.**

File: `core/timeframe_belief_network.py`

Each worker already has access to its TF's physics state (16D feature vector):
- abs(z), log1p(v), log1p(m), coherence, tf_scale, depth, parent_ctx,
  self_adx, self_hurst, self_dmi_diff, parent_z, parent_dmi_diff,
  root_is_roche, tf_alignment, self_pid, osc_coh

The predictor uses these features to output signed expected ticks:
- Positive = expected move up (favor long)
- Negative = expected move down (favor short)
- Magnitude = expected size of move

**Option A: Seed Library Match (preferred — no training needed)**

```python
# In worker update():
# 1. Observe last N bars of price delta from current position
# 2. Match to seed library (closest Pearson correlation)
# 3. Seed shape has known continuation: e.g., V_UP predicts +X more ticks
# 4. Scale by ATR and physics confidence

shape, corr = seed_library.classify(recent_deltas)
predicted_continuation = seed_library.predict_continuation(shape, corr, atr)
predicted_ticks = predicted_continuation * sign_from_physics
```

**Option B: Linear Regression on 16D features → signed MFE**

The signed MFE regression already exists per-template (`signed_mfe_coeff` in pattern
library). Extend it to per-worker using the worker's own TF features.

### Phase 2: Belief Aggregation

File: `core/timeframe_belief_network.py` — `get_belief()` method

Current aggregation produces conviction-weighted direction. Add:

```python
# Weighted average of worker predicted_ticks
# Weight by: conviction × coherence × (1 / tf_scale)  — closer TFs matter more
predicted_ticks = sum(w.predicted_ticks * w.weight for w in active_workers)
predicted_ticks /= sum(w.weight for w in active_workers)

# Also aggregate predicted hold time
predicted_hold_bars = sum(w.predicted_hold * w.weight for w in active_workers)
```

Output: `BeliefState.predicted_ticks`, `BeliefState.predicted_hold_bars`

### Phase 3: Entry Gate

File: `core/execution_engine.py` — `_finalize_entry()`

After direction cascade, before sizing:

```python
# E[PnL] gate: skip if predicted outcome is negative
e_pnl_ticks = belief.predicted_ticks
if e_pnl_ticks is not None and abs(e_pnl_ticks) > 0:
    # Flip direction check: predicted direction must agree with chosen side
    if (side == 'long' and e_pnl_ticks < 0) or (side == 'short' and e_pnl_ticks > 0):
        return HOLD  # predicted direction disagrees → skip

    # Minimum expected profit gate
    if abs(e_pnl_ticks) < min_expected_ticks:  # tunable, e.g., 3 ticks
        return HOLD  # not worth the spread + slippage
```

### Phase 4: Exit Sizing from Prediction

File: `core/exit_engine.py` — `make_position()`

Currently: TP/SL from generic ATR multipliers.
New: TP/SL derived from predicted ticks.

```python
# TP = predicted_ticks (that's literally what we expect)
tp_ticks = max(4, int(predicted_ticks))  # floor at 4 for spread protection

# SL = fraction of predicted ticks (risk/reward ratio)
sl_ticks = max(4, int(predicted_ticks * sl_ratio))  # e.g., sl_ratio = 0.5

# Envelope halflife = predicted hold time
# If workers say "this move takes 40 bars", halflife = 40
envelope_halflife = max(8, int(predicted_hold_bars))
```

### Phase 5: Live Exit Comparison

File: `core/exit_engine.py` — exit cascade

The exit decision becomes: **am I on track relative to the prediction?**

```python
# Current progress vs prediction
progress_ratio = current_ticks / predicted_ticks  # 0.0 = just entered, 1.0 = hit target
time_ratio = bars_held / predicted_hold_bars       # 0.0 = just entered, 1.0 = expected duration

# Case 1: Ahead of schedule (progress > time) → patient, let it run
# Case 2: Behind schedule (progress < time * 0.5) → prediction failing, tighten
# Case 3: At target → bank it, check re-entry

if progress_ratio >= 1.0:
    # Hit predicted target → take profit
    # Check if workers still predict continuation → re-enter
    exit_reason = 'predicted_target'

elif time_ratio > 1.5 and progress_ratio < 0.3:
    # Way past expected time, barely moved → prediction wrong, abort
    exit_reason = 'prediction_timeout'

elif progress_ratio < 0 and time_ratio > 0.3:
    # Going backwards past 30% of expected time → abort
    exit_reason = 'prediction_invalidated'
```

### Phase 6: Re-Entry Decision

File: `live/live_engine.py` — auto-TP re-entry (already exists at line 832)

Current: re-enter if belief direction agrees.
New: re-enter if **workers predict another +N ticks** from current price.

```python
# After taking profit:
new_prediction = belief_network.get_belief()
if new_prediction.predicted_ticks > min_reentry_ticks:
    # Workers see more room → re-enter with fresh prediction
    reenter(side, predicted_ticks=new_prediction.predicted_ticks)
```

## Files to Modify

| File | Changes |
|------|---------|
| `training/seed_library.py` | NEW — extract from standalone_research.py (Part 1 of existing spec) |
| `core/timeframe_belief_network.py` | Add `predicted_ticks` to worker output, aggregate in `get_belief()` |
| `core/execution_engine.py` | E[PnL] gate in `_finalize_entry()`, record prediction on TradeAction |
| `core/exit_engine.py` | Prediction-based TP/SL/halflife in `make_position()`, progress comparison in exit cascade |
| `live/live_engine.py` | Prediction-aware re-entry decision |
| `training/trainer.py` | Record `predicted_ticks` in oracle trade log, add calibration report section |

## Tuning Parameters (checkpoints/tuning.json)

```json
{
    "min_expected_ticks": 3,
    "sl_to_prediction_ratio": 0.5,
    "prediction_timeout_multiplier": 1.5,
    "prediction_invalidation_threshold": 0.3,
    "min_reentry_ticks": 5
}
```

## Validation

### IS Report: Prediction Calibration
- Predicted vs actual PnL scatter plot data (R², slope, intercept)
- Binned accuracy: predicted 5-10 ticks → actual avg X ticks
- Direction accuracy: did predicted sign match actual sign?
- Calibration: is the system overconfident (predicted 20, actual 5) or underconfident?

### OOS: Prediction Stability
- Same calibration metrics on unseen data
- If IS R² = 0.80 and OOS R² = 0.30 → overfit, need regularization

## Dependencies

- Seed library extraction (Part 1 of JULES_WAVEFORM_SEED_INTEGRATION.md) — needed for Option A
- Waveform research results (`tools/standalone_research.py` analysis J) — 92% R² model

## Implementation Order

1. **seed_library.py** — extract from standalone_research (already spec'd)
2. **Worker predicted_ticks** — each worker outputs signed expected ticks
3. **get_belief() aggregation** — consensus E[PnL]
4. **Entry gate** — E[PnL] > min_threshold
5. **Exit sizing** — TP/SL/halflife from prediction
6. **Live exit comparison** — progress vs prediction
7. **Re-entry** — prediction-aware re-entry after TP
8. **Calibration report** — predicted vs actual correlation
