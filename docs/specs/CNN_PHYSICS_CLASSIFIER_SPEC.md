# CNN Physics Classifier — Design Spec v3

## Philosophy
The regret oracle shows the PERFECT trades — optimal entry, direction, duration, exit.
The CNN's job: learn which 91D physics patterns correspond to which oracle outcomes.
The physics engine is the teacher's curriculum. The oracle is the answer key.
The CNN is the student trying to recognize the patterns that lead to A+ answers.

## Architecture: 5 Jobs × N Tiers = Specialist Models

Each tier gets its own set of 5 specialist CNNs. Each CNN masters ONE decision
using only trades from its tier. The tier determines the training data.
The job determines the architecture.

```
training/output/nn/
  CASCADE/
    entry_cnn.pt        — "Is this a good CASCADE entry?"
    direction_cnn.pt    — "Long or short for CASCADE?"
    duration_cnn.pt     — "How long does a CASCADE winner last?"
    exit_cnn.pt         — "Is this CASCADE trade done?"
    loser_id_cnn.pt     — "Is this CASCADE trade failing?"
  KILL_SHOT/
    entry_cnn.pt
    direction_cnn.pt
    ...
  FADE_CALM/
    ...
```

Tiers with < 100 IS trades skip CNN entirely — physics only for those.

## Why Per-Tier
- CASCADE: 96% WR, wick rejection, 1h aligned. Direction is almost always "fade z."
- FADE_CALM: ~50% WR, no wick, low velocity. Direction is ambiguous — CNN's real job.
- One model trained on both learns the average of 96% and 50% = useless.
- Per-tier models see a tight, learnable distribution specific to their physics.

## The 5 Jobs

### Job 1: Entry Gate
**Question**: Is this a good entry for this tier, or should we skip it?
- Input: 91D at entry
- Label: 1 if trade was a winner (PnL > 0), 0 if loser — from oracle outcomes
- Output: P(good_entry) — continuous probability
- Engine: if P(good_entry) < ENTRY_THRESHOLD → skip this entry entirely
- **Value**: filters out the 40-50% of entries that physics triggers but oracle says lose

### Job 2: Direction
**Question**: Long or short for this tier's entry?
- Input: 91D at entry
- Label: oracle's optimal direction (from regret best_action)
- Output: P(long), P(short) — continuous probability
- Engine: if P(oracle_direction) > DIRECTION_THRESHOLD → override physics default
- **Value**: catches the ~35% of FADE trades where riding was actually better

### Job 3: Duration
**Question**: How long should a winner in this tier be held?
- Input: 91D at entry
- Label: oracle's optimal exit bar, binned SHORT(<10)/MEDIUM(10-40)/LONG(>40)
- Output: 3-class softmax probability
- Engine: sets exit patience parameters (confirmation bars, giveback tolerance)
- **Value**: adapts hold time to entry conditions instead of fixed-for-all

### Job 4: Exit
**Question**: Is this trade done? (runs every 5s bar while in position)
- Input: 91D current state + trade context (bars_held, pnl, peak_pnl, entry_z)
- Label: HOLD(1) if bar < oracle's optimal exit bar, EXIT(0) if past it
- Output: P(EXIT) — continuous probability
- Engine: if P(EXIT) > EXIT_THRESHOLD for N consecutive bars → close
- **Value**: detects peak exhaustion from multi-TF physics, not just p_center

### Job 5: Loser ID
**Question**: This trade is underwater — will it recover or is it dead?
- Input: 91D current state + trade context (bars_held, pnl, peak_pnl, entry_z)
- Label: RECOVER(1) if trade eventually ended positive, DEAD(0) if not
- Fires ONLY when pnl < 0 (not every bar)
- Output: P(DEAD) — continuous probability
- Engine: if P(DEAD) > LOSER_THRESHOLD for N consecutive bars → cut losses
- **Value**: kills losers early without waiting for hard stop

## Confidence Gating

| | Current | New |
|---|---|---|
| Entry gate | None (all triggers trade) | P(good) > 60% or skip |
| Direction | 50% majority vote | P(direction) > 75% or use physics default |
| Duration | Fixed per tier | Predicted class sets patience params |
| Exit | Single bar argmax | P(EXIT) > 70% for 3+ consecutive bars |
| Loser cut | Binary DEAD/RECOVER | P(DEAD) > 80% for 3+ consecutive bars |
| Per-tier | One model | Separate models per tier |
| Fallback | CNN overrides physics | Physics default, CNN boosts when confident |

**The rule**: below threshold → physics runs unmodified. CNN can only help.

## Pipeline

```
Phase 1:   Physics forward pass IS → trades with entry_91d + trade paths
Phase 1b:  Physics OOS baseline (deterministic floor)
Phase 1c:  Physics OOS-NT8 baseline (live parity floor)
Phase 2:   Regret analysis → oracle labels per trade
Phase 3:   Split IS trades by tier
Phase 4:   Per tier (where count >= 100):
             4a. Train entry gate CNN
             4b. Train direction CNN
             4c. Train duration CNN
             4d. Train exit CNN
             4e. Train loser ID CNN
Phase 5:   Forward pass IS with all CNNs (confidence-gated)
Phase 6:   Forward pass OOS + OOS-NT8 with all CNNs
Phase 7:   Per-tier report: physics vs +CNN, per OOS dataset
           Disable CNNs for tiers where they hurt OOS
```

### Phase 7: Per-Tier Report
```
TIER REPORT (OOS):
  CASCADE:       Physics $42/day → +CNN $48/day (+$6)  ✓ KEEP
  KILL_SHOT:     Physics $35/day → +CNN $33/day (-$2)  ✗ DISABLE
  FADE_CALM:     Physics $10/day → +CNN $18/day (+$8)  ✓ KEEP
  FADE_MOMENTUM: Physics $15/day → +CNN $14/day (-$1)  ~ KEEP (noise)
  ...
  TOTAL:         Physics $599/day → +CNN $XXX/day

TIER REPORT (OOS-NT8 — live parity):
  CASCADE:       Physics $XX/day → +CNN $XX/day
  ...
```

If a tier's CNN hurts OOS → disabled for that tier in final config.
Final deployed model = physics + only the per-tier CNNs that earned their spot.

## Model Architecture

### Entry + Direction + Duration (at entry — shared backbone)
- Input: 91D reshaped to 6 × 15 grid (6 TFs × 12 core + 3 helper, padded)
- Conv1: 16 filters, 3×3, ReLU, BatchNorm
- Conv2: 32 filters, 3×3, ReLU, BatchNorm
- Global average pool → shared_features(64)
- Head A (entry): FC(32) → FC(2) softmax (good/bad)
- Head B (direction): FC(32) → FC(2) softmax (long/short)
- Head C (duration): FC(32) → FC(3) softmax (SHORT/MEDIUM/LONG)
- Multi-head = shared backbone trains on more data, each head specializes
- OR train 3 separate small models if multi-head overfits

### Exit + Loser ID (during trade)
- Input: 91D flat + context (bars_held, pnl, peak_pnl, entry_z, duration_pred)
- FC(128) → ReLU → FC(64) → ReLU
- Head D (exit): FC(2) softmax (HOLD/EXIT)
- Head E (loser): FC(2) softmax (RECOVER/DEAD)
- Runs every 5s while in position
- Loser head only evaluated when pnl < 0

### Size Budget
- Per tier: ~30K params (entry/direction/duration backbone + 3 heads) + ~15K (exit/loser)
- 7 tiers × 45K = ~315K total params (still tiny)

## Guard Rails

1. **Physics floor**: no entry without z extreme + vr < 1. No holding past hard stop.
2. **Per-tier disable**: `CNN_ENABLED[tier] = {'entry': True, 'direction': True, ...}`
3. **Confidence thresholds** (configurable per job):
   - `ENTRY_GATE_MIN = 0.60`
   - `DIRECTION_CONFIDENCE_MIN = 0.75`
   - `DURATION_CONFIDENCE_MIN = 0.60`
   - `EXIT_CONFIDENCE_MIN = 0.70`
   - `LOSER_CONFIDENCE_MIN = 0.80`
   - `EXIT_CONFIRMATION_BARS = 3`
   - `LOSER_CONFIRMATION_BARS = 3`
4. **A/B logging**: every trade records all CNN predictions + confidences + physics defaults
5. **Fixed seeds**: deterministic training and inference

## Files

### New
- `training/physics_labels.py` — per-tier label extraction from regret (5 label types)
- `training/cnn_entry_direction.py` — shared backbone: entry gate + direction + duration
- `training/cnn_trade_manager.py` — shared backbone: exit + loser ID

### Modify
- `training/nightmare_blended.py` — per-tier model loading, confidence-gated 5-job integration
- `training/run.py` — new pipeline phases

### Deprecate (archive)
- `training/cnn_flip.py` → replaced by per-tier direction head
- `training/cnn_hold.py` → replaced by duration head + exit head
- `training/cnn_risk.py` → replaced by loser ID head
