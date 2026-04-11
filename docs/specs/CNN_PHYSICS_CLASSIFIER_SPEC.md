# CNN Physics Classifier — Design Spec v2

## Philosophy
The regret oracle shows the PERFECT trades — optimal entry, direction, duration, exit.
The CNN's job: learn which 91D physics patterns correspond to which oracle outcomes.
The physics engine is the teacher's curriculum. The oracle is the answer key.
The CNN is the student trying to recognize the patterns that lead to A+ answers.

## Why Current CNNs Fail
The concept is correct. The execution has 3 problems:

1. **One model for all tiers** — CASCADE (96% WR, wick rejection) looks nothing like
   FADE_CALM (50% WR, no wick). A single CNN trained on all tiers learns the average
   of everything — which is useful for nothing.

2. **64.5% accuracy ≈ always-predict-majority** — The "always SAME" baseline is 64.6%.
   The CNN barely beats it. This means it hasn't actually learned the patterns.

3. **Hard binary at 50% threshold** — CNN says 51% COUNTER → flip. No confidence
   gating. Borderline predictions inject noise into deterministic physics.

## Fix: Per-Tier Oracle-Guided Classifiers

### Step 1: Oracle Labels (regret — unchanged)
Regret analysis computes the perfect trade from hindsight:
- Optimal direction (SAME or COUNTER to physics default)
- Optimal exit bar (where PnL peaked)
- Optimal PnL (what you'd get with perfect foresight)

This is the answer key. It's correct by construction.

### Step 2: Cluster by Tier
Instead of one model, train a SEPARATE classifier per physics tier:
- CASCADE model (wick + 1h aligned — high WR, directional)
- KILL_SHOT model (wick, no 1h — high WR, less directional)
- FADE_CALM model (no wick, low velocity — the hard tier)
- FADE_MOMENTUM model (no wick, high velocity)
- RIDE tiers (CNN-flipped — only exists if direction model fires)

Each tier has different physics signatures and different oracle statistics.
A per-tier model sees a tighter, more learnable distribution.

### Step 3: Three Tasks Per Tier

**Task A: Direction** (at entry)
- Input: 91D at entry (6 TFs × 15 features, reshaped to grid)
- Label: oracle's optimal direction (long/short, from regret best_action)
- Output: P(long), P(short) — continuous probability, NOT argmax
- Engine uses: if P(oracle_direction) > DIRECTION_THRESHOLD → override physics

**Task B: Duration** (at entry)
- Input: same 91D entry grid
- Label: oracle's optimal exit bar, binned into SHORT(<10)/MEDIUM(10-40)/LONG(>40)
- Output: 3-class probability
- Engine uses: sets exit patience (p_center confirmation bars, giveback tolerance)

**Task C: Exit** (during trade, every 5s bar)
- Input: 91D current state + trade context (bars_held, pnl, peak_pnl, entry_z, tier)
- Label: HOLD if current bar < oracle's optimal exit bar, EXIT if past it
- Output: P(EXIT) — continuous probability
- Engine uses: if P(EXIT) > EXIT_THRESHOLD for N consecutive bars → close

### Step 4: Confidence Gating

The critical difference from current implementation:

| | Current | New |
|---|---|---|
| Direction threshold | 50% (majority vote) | 75% minimum |
| Exit decision | Single bar argmax | 3+ consecutive bars above threshold |
| Risk cut | Binary DEAD/RECOVER | Disabled — exit CNN handles this |
| Per-tier | One model | Separate model per tier |
| Fallback | CNN overrides physics | Physics is default, CNN is optional boost |

**The rule**: if CNN confidence < threshold, physics runs unmodified.
This guarantees CNN can only HELP — if unsure, it stays silent.

## Pipeline

```
Phase 1:   Physics forward pass IS → trades with entry_91d + trade paths
Phase 1b:  Physics OOS baseline (deterministic floor)
Phase 1c:  Physics OOS-NT8 baseline (live parity floor)
Phase 2:   Regret analysis → oracle labels per trade (direction, exit bar, PnL)
Phase 3:   Split trades by tier → per-tier training sets
Phase 4a:  Train direction CNN per tier (where tier has enough trades)
Phase 4b:  Train duration CNN per tier
Phase 4c:  Train exit CNN per tier
Phase 5:   Forward pass IS with CNNs (confidence-gated)
Phase 6:   Forward pass OOS + OOS-NT8 with CNNs
Phase 7:   Report: physics vs CNN per tier, overall, per OOS dataset
```

### Phase 3 Detail: Tier Splitting
```python
# Split IS trades by entry_tier
tier_trades = {tier: [t for t in all_trades if t['entry_tier'] == tier]
               for tier in TIER_LIST}

# Minimum trades per tier for CNN training (need enough for train/val split)
MIN_TRADES_PER_TIER = 100

# Tiers with fewer trades: use physics only (no CNN)
for tier, trades in tier_trades.items():
    if len(trades) < MIN_TRADES_PER_TIER:
        print(f"  {tier}: {len(trades)} trades — physics only (too few for CNN)")
```

### Phase 4 Detail: Per-Tier Training
```python
for tier in trainable_tiers:
    trades = tier_trades[tier]
    
    # Direction labels from regret
    # regret.best_action tells us: SAME or COUNTER
    # Convert to absolute: if physics said short + regret said COUNTER → label=long
    
    # Duration labels from regret
    # regret.optimal_bar → bin into SHORT/MEDIUM/LONG
    
    # Exit labels from trade paths
    # For each bar in trade: HOLD if bar < optimal_bar, EXIT if bar >= optimal_bar
    
    # Train 3 small CNNs for this tier
    # Architecture: same as current (6x15 conv grid), but trained on 1 tier only
    # Walk-forward: train on months 1-9, validate on months 10-12
    
    # Save: training/output/nn/{tier}_direction.pt
    #        training/output/nn/{tier}_duration.pt
    #        training/output/nn/{tier}_exit.pt
```

### Phase 7 Detail: Per-Tier Reporting
```
TIER REPORT:
  CASCADE:     Physics $42/day → +CNN $48/day (+$6, CNN helps)
  KILL_SHOT:   Physics $35/day → +CNN $33/day (-$2, CNN hurts → DISABLE)
  FADE_CALM:   Physics $10/day → +CNN $18/day (+$8, CNN helps)
  FADE_MOMENTUM: Physics $15/day → +CNN $14/day (-$1, within noise → KEEP)
  ...
  TOTAL:       Physics $599/day → +CNN $XXX/day

  OOS-NT8:     Physics $XXX/day → +CNN $XXX/day (live parity)
```

If a tier's CNN hurts OOS, it gets disabled in the blended engine config.
The final deployed model uses CNN only where it's proven to help.

## Model Architecture

### Direction + Duration CNN (at entry)
Same grid architecture as current cnn_flip:
- Input: 6 × 15 grid (6 TFs × 12 core + 3 helper features, zero-padded to 15)
- Conv1: 16 filters, 3×3, ReLU, BatchNorm
- Conv2: 32 filters, 3×3, ReLU, BatchNorm  
- Global average pool → FC(128) → FC(num_classes)
- Direction: 2-class (long/short), output softmax probabilities
- Duration: 3-class (SHORT/MEDIUM/LONG), output softmax probabilities
- Can be multi-head (shared backbone, two output heads) for efficiency

### Exit CNN (during trade)
Different architecture — includes trade context:
- Input: 91D current features (flat) + 4 context scalars (bars_held, pnl, peak_pnl, entry_z)
- FC(128) → ReLU → FC(64) → ReLU → FC(2) → softmax
- Simpler than direction CNN — less spatial structure to exploit
- Runs every 5s bar while in position

### Size Budget
- Per tier: ~20K params (direction) + ~20K (duration) + ~10K (exit) = ~50K
- 7 tiers × 50K = ~350K total params
- Current: 3 models × ~16K = ~48K params
- Still tiny — fits in GPU memory trivially

## Guard Rails

1. **Physics floor**: CNN can never make the engine do something physics wouldn't allow.
   No entry without z extreme + vr < 1. No holding past hard stop.

2. **Per-tier disable**: if tier CNN hurts OOS, `use_cnn_for_tier[tier] = False`

3. **Confidence threshold**: configurable per task:
   - `DIRECTION_CONFIDENCE_MIN = 0.75`
   - `DURATION_CONFIDENCE_MIN = 0.60` (softer — wrong duration is less costly)
   - `EXIT_CONFIDENCE_MIN = 0.70`
   - `EXIT_CONFIRMATION_BARS = 3`

4. **A/B logging**: every trade records:
   - `cnn_direction_pred`, `cnn_direction_conf`, `physics_direction`
   - `cnn_duration_pred`, `cnn_duration_conf`
   - `cnn_exit_bar`, `physics_exit_bar`, `actual_exit_bar`
   - This lets you compute CNN value-add per tier in post-analysis

5. **Fixed seeds**: torch.manual_seed + numpy seed fixed per training run.
   Same data → same model → same predictions → deterministic.

## Files

### New
- `training/physics_labels.py` — extract direction/duration/exit labels from regret per tier
- `training/cnn_direction.py` — per-tier direction classifier
- `training/cnn_duration.py` — per-tier duration classifier  

### Modify
- `training/cnn_exit.py` — retrain with per-tier labels + confidence output
- `training/nightmare_blended.py` — confidence-gated CNN integration, per-tier model loading
- `training/run.py` — new pipeline phases (3-6 replace current 3-7)

### Deprecate
- `training/cnn_flip.py` — replaced by per-tier cnn_direction.py
- `training/cnn_hold.py` — replaced by cnn_duration.py (entry-time) + cnn_exit.py (during trade)
- `training/cnn_risk.py` — absorbed into cnn_exit.py (exit handles all exit decisions)
