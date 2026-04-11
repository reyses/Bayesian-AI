# CNN Physics Classifier — Design Spec

## Problem
Current CNNs learn from regret (hindsight-optimal actions). They degrade the $599/day physics baseline because they solve "what should've happened" not "what will happen." The physics engine already knows WHEN to enter (z extreme + vr < 1). What it doesn't know is: given THIS specific physics state, what direction works, how long does it last, and what signals the end?

## Core Concept
The CNN is a **physics pattern classifier**. It looks at the 91D state at entry and classifies the trade into a cluster of historically similar setups. Each cluster has known statistics: win rate, median duration, median PnL, typical exit physics. The CNN doesn't predict — it RECOGNIZES.

## Architecture: 3 Classification Tasks

### Task 1: Direction Classifier
**Question**: Given this 91D state at entry, should the trade be long or short?

**Labels**: NOT from regret. From actual realized outcomes:
- Take all historical entries where physics triggered (|z| > 2, vr < 1)
- For each entry: what direction actually made money?
- Label = 1 (long wins) if going long from this state produced positive PnL
- Label = 0 (short wins) if going short produced positive PnL
- Exclude trades where BOTH directions lose (bad entry, not a direction problem)

**Key difference from current**: Current CNN flip uses regret's SAME/COUNTER which is relative to the physics default (fade z). The new label is absolute: long or short, based on what the market actually did after this physics state.

**Input**: 91D entry features reshaped to 6x15 grid (6 TFs x 15 features per TF, padded from 12 core + 3 helper)

**Why this works**: The physics engine defaults to "fade z" (z > 0 → short, z < 0 → long). But some z-extreme states reverse immediately (fade works), while others are momentum breakouts (ride works). The CNN learns to distinguish these from the multi-TF physics pattern.

### Task 2: Duration Classifier  
**Question**: Given this 91D state at entry, how long will a winning trade last?

**Labels**: From realized winner durations, binned:
- SHORT (0): winner peaks within 10 bars (< 10 min). Quick scalp.
- MEDIUM (1): winner peaks in 10-40 bars (10-40 min). Standard trade.
- LONG (2): winner peaks after 40 bars (> 40 min). Extended hold.

**How labels are computed**:
- Take all winning trades (PnL > 0)
- `peak_bar` = bar index where trade reached maximum PnL
- Bin into SHORT/MEDIUM/LONG based on peak_bar
- Losing trades get the label of their nearest winning neighbor in feature space (KNN with k=5) — this teaches the CNN what duration SHOULD have been, not what happened

**Why this matters**: The physics engine uses fixed exit timing (p_center confirmation after N bars). But optimal hold duration varies wildly by entry state. A high-velocity z-extreme should be held longer than a low-velocity one. The CNN learns this mapping.

**Output**: 3-class softmax. The ENGINE uses the predicted duration to set exit patience:
- SHORT → tight exit, small target, 3-bar p_center confirmation
- MEDIUM → standard exit, normal target, 5-bar confirmation  
- LONG → patient exit, wide target, 10-bar confirmation

### Task 3: Exit Signal Classifier
**Question**: Given the current 91D state DURING a trade, is the trade done?

**Labels**: From realized exit physics:
- HOLD (1): trade has not yet reached its peak PnL
- EXIT (0): trade has passed its peak and is giving back profits

**Key difference from current CNN hold**: Current hold uses regret's optimal exit bar. New exit classifier uses a CAUSAL rule:
- For each bar in a winning trade: if PnL at this bar > 80% of eventual peak → still HOLD
- For each bar after the peak where PnL drops below 50% of peak → EXIT
- This is computable from realized data and represents a reasonable "don't give back too much" rule

**Why this is better**: The label is based on a simple, robust definition of "the trade is done" rather than the hindsight-exact optimal bar. A trade that peaks at $100 and is now at $45 is EXIT regardless of what happens next.

**Input**: 91D current features + trade context (bars_held, current_pnl, peak_pnl, entry_z)

## Training Data

### Source
All trades from the physics-only forward pass (no CNN influence):
```bash
python training/run.py blended --from 1 --to 1   # Phase 1: physics IS
```
This produces `training/output/trades/blended_is.pkl` — trades with:
- `entry_79d`: 91D features at entry
- `path`: list of {timestamp, price, features_79d} per bar during trade
- `pnl`: final PnL
- `peak`: maximum PnL during trade
- `held`: bars held
- `dir`: long/short
- `entry_tier`: CASCADE/KILL_SHOT/FADE_CALM/etc.

### Label Generation (new module: `training/physics_labels.py`)
```python
def generate_direction_labels(trades) -> DataFrame:
    """For each trade: did the physics direction win or should it have been opposite?"""

def generate_duration_labels(trades) -> DataFrame:
    """For each winning trade: SHORT/MEDIUM/LONG duration bin."""

def generate_exit_labels(trades) -> DataFrame:
    """For each bar in each trade: HOLD or EXIT based on peak proximity."""
```

Labels are deterministic — no randomness, no regret counterfactuals.

## How the Pipeline Changes

### Current Pipeline (regret-based)
```
Phase 1: Physics IS → trades
Phase 2: Regret analysis → counterfactual labels
Phase 3: Train CNN flip on regret labels
Phase 4: Forward pass + flip → new trades
Phase 5: Regret on flipped trades → hold/risk labels  
Phase 6: Train CNN hold + risk
Phase 7: Forward pass with all CNNs
```

### New Pipeline (physics-based)
```
Phase 1:  Physics IS → trades (unchanged)
Phase 1b: Physics OOS baseline (unchanged)
Phase 1c: Physics OOS-NT8 baseline (unchanged)
Phase 2:  Generate physics labels from IS trades (direction, duration, exit)
Phase 3:  Train direction CNN
Phase 4:  Train duration CNN
Phase 5:  Train exit CNN
Phase 6:  Forward pass IS + OOS + OOS-NT8 with all 3 CNNs
```

Key differences:
- NO regret analysis step
- ALL labels come from realized trade outcomes
- CNNs trained on physics IS trades, validated on OOS + OOS-NT8
- Pipeline is shorter (6 phases vs 7+)
- Each CNN trains independently (no cascading dependency)

## How BlendedEngine Uses the CNNs

### At Entry (every 5s bar with NMP trigger):
1. Physics detects entry condition (|z| > 2, vr < 1)
2. Physics classifies tier (CASCADE/KILL_SHOT/FADE_CALM/etc.)
3. **Direction CNN**: predicts long/short from 91D. If disagrees with physics AND confidence > threshold → override direction
4. **Duration CNN**: predicts SHORT/MEDIUM/LONG. Sets exit patience parameters for this trade

### During Trade (every 5s bar):
5. **Exit CNN**: predicts HOLD/EXIT from current 91D + trade context. If EXIT AND confirmed for N bars → close trade
6. Physics exits still active as safety net (hard stop, daily loss limit, z reversal)

### Thresholds (configurable, not hardcoded):
- Direction override: CNN confidence > 75% (not 50%)
- Exit confirmation: CNN says EXIT for 3 consecutive bars (not single bar)
- Duration bins: SHORT < 10 bars, MEDIUM < 40 bars, LONG >= 40 bars (tunable)

## Expected Outcome
- Physics baseline: $599/day (deterministic floor)
- +Direction CNN: should IMPROVE on FADE_CALM/RIDE tiers where direction is ambiguous
- +Duration CNN: should IMPROVE exit timing by adapting patience to entry state
- +Exit CNN: should REDUCE giveback by detecting peak proximity
- Combined: should be >= $599/day with lower variance than current CNNs

If any CNN degrades performance below physics, it gets disabled for that tier. The physics baseline is the FLOOR — CNNs can only add, never subtract.

## Guard Rails
1. **Per-tier CNN enable/disable**: if CASCADE works best without CNN, disable CNN for CASCADE
2. **Confidence thresholds**: never act on < 75% confidence
3. **Confirmation bars**: never exit on a single CNN prediction
4. **A/B tracking**: every trade logs whether CNN agreed or disagreed with physics, and what the outcome was. This creates the data to tune thresholds post-deployment.
5. **Determinism**: fixed seeds, no dropout at inference, ensemble voting with odd number of models

## Files to Create/Modify
- `training/physics_labels.py` — NEW: label generation from realized trades
- `training/cnn_direction.py` — NEW: replaces cnn_flip.py
- `training/cnn_duration.py` — NEW: replaces cnn_hold.py  
- `training/cnn_exit.py` — MODIFY: retrain with causal exit labels
- `training/nightmare_blended.py` — MODIFY: new CNN integration points
- `training/run.py` — MODIFY: new pipeline phases
