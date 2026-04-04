# Exit Module Analysis — Core Math & Logic
> Session: 2026-04-03
> Status: REFERENCE — use this to build unified exit for new system

## MARKER: THREE EXITS TO UNIFY

The new system needs ONE exit function. These three contain the math:

### 1. Envelope Decay (`core/exits/envelope.py`)
```python
decay = exp(-ln2 * bars_held / effective_hl)    # line 104
envelope_level = floor + (initial_tp - floor) * decay
```
- **What it does**: TP target decays exponentially toward a floor
- **NN connection**: NN predicts `effective_hl` (the half-life = hold duration)
- **Modulation**: giveback ratio, band exhaustion, ADX slope, anchor patience
- **Self-tunes**: adjusts HL every 30 trades based on early/late exit ratio

### 2. Survival Stop (`core/exits/survival_stop.py`)
```python
room = max(0, 2.0 - z_score)        # how far from band wall
remain_score = room * trend_bonus * conv_bonus * align_bonus * mom_bonus
# Exit when remain_score < 0.10
```
- **What it does**: multiplicative probability that signal still has life
- **NN connection**: this IS the real-time half-life modulator
- **Key insight**: any single factor going to zero kills the score (multiplicative)
- **Factors**: z-score room, Hurst trending, conviction, F_momentum alignment

### 3. Peak Giveback (`core/exits/giveback.py`)
```python
gave_back = (peak_ticks - current_ticks) / peak_ticks
# Base threshold 65%, adaptive via volume
# Volume < 50% of peak volume → tighten to 40%
```
- **What it does**: detects when a peak is real and reversing
- **NN connection**: giveback ratio can accelerate the half-life decay
- **Self-tunes**: recalibrates threshold every 10 exits via grid search

## MARKER: UNIFIED EXIT CONCEPT

```
Initial half-life = NN prediction (from 79D state)
Every bar:
  survival_score = room * trend * conviction * alignment * momentum  
  effective_hl = initial_hl * survival_score   # score compresses HL in real time
  decay = exp(-ln2 * bars_held / effective_hl)
  envelope = floor + (peak_pnl - floor) * decay
  
  if pnl < envelope → EXIT (signal decayed past the envelope)
  if giveback > threshold → accelerate (halve the HL)
```

One function. NN sets the initial half-life. Market state modulates it live.
Giveback is the emergency accelerator. Stop loss is the hard floor (unchanged).

---

## Full Cascade (current system, for reference)

### Active in evaluate():
| Priority | Module | Trigger | Sound? |
|----------|--------|---------|--------|
| 1 | Peak Giveback | gave_back > 65% of peak, volume-adaptive | YES |
| 1b | 1m Flip | fm_1m + vol_1m both against | YES (FM threshold dead code though) |
| 2 | Survival Stop | remain_score < 0.10 (multiplicative state prob) | **HIGHLY SOUND** |
| 3 | Stop Loss | Hard price floor, intra-bar resolution | ALWAYS |
| 4 | Take Profit | Static TP from template | Sound but rigid |
| 5 | Watchdog | Session ending | Operational |
| 6 | Band Urgent | Band direction vs position, strength > 0.7 | Depends on band calc |
| 7 | Trailing Stop | MFE ratchet, sensor-adaptive width (50/65/80%) | YES |
| 7b | V-Reversal | 4+ bars since peak + BE locked + profitable | YES |
| 8 | Envelope Decay | exp(-ln2 * t / hl), HL modulated by 4 signals | **CORE MATH** |

### Built but NOT wired:
| Module | File | Why it matters |
|--------|------|---------------|
| **Fractal Exhaust** | `core/exits/fractal_exhaust.py` | ADX hook at 2-sigma wall. PF=2.44 for cascade fade. **Worth testing** |
| **Regime Decay** | `core/exits/regime_decay.py` | Hurst < 0.50 + ADX < 20 + DI cross. Trend death detector. **Worth testing** |
| **Peak State Exit** | `core/exits/peak_state_exit.py` | 4/5 TFs show DMI against + momentum decay. **Worth testing** |

### Dead:
| Module | Result | Don't resurrect |
|--------|--------|-----------------|
| Belief Flip | PF 0.02, -$378 | DI crossover at micro TF = noise |
| Tidal Wave | PF 0.00, -$387 | SE expansion at micro TF = noise |

---

## MARKER: CONFIG PARAMS TO CARRY FORWARD

### Envelope (the core exit):
- `envelope_halflife_bars=40` (default, NN will override)
- `envelope_floor_pct=0.15` (floor as % of initial TP)
- `envelope_min_bars=8` (minimum before envelope can fire)
- `envelope_early_suppress_pct=0.5` (don't fire in first half of HL)
- `envelope_force_boost=1.3` (force aligned → slow decay)
- `envelope_force_penalty=0.7` (force opposed → speed decay)

### Survival (the modulator):
- `ce_survival_min_bars=10` (activation delay)
- Threshold: `0.10` (hardcoded, no config — needs promotion)
- `room = max(0, 2.0 - |z_score|)` — "2.0" is Roche limit

### Giveback (the accelerator):
- Base threshold: `0.65` (give back 65% of peak)
- Aggressive threshold: `0.40` (when peak > 2x min MFE)
- Volume gate: tighten to 40% when vol < 50% of peak vol
- Self-tunes every 10 exits

### Stop Loss (the hard floor):
- From template SL ticks
- Uses intra-bar `worst_price` for sub-bar resolution
- Never suppressed — only exit with no gate

---

## MARKER: DEAD CODE / BUGS FOUND

1. **`FM_FLIP_THRESHOLD = 30.0`** in exit_engine.py line 523 — defined but never used.
   The 1m_flip check uses boolean `fm_1m_against` only, ignores magnitude.

2. **Watchdog has 4 dead config params** — `watchdog_tick_threshold`, 
   `watchdog_bar_threshold`, `watchdog_worker_threshold`, `watchdog_mfe_progress_pct`.
   All stored but never read after stuck-trade logic was removed.

3. **Priority cascade comments are stale** — say Death Hook / Regime Decay come 
   "before SL/TP" but they're not in evaluate() at all.

4. **Self-tuning loop adjusts 2 params from 1 signal** — envelope HL and giveback_pct 
   both adjusted from "too early / too late" which can mask one being well-calibrated.

---

## MARKER: CONNECTION TO 79D SPEC

The 79D feature vector provides exactly what the exits need:
- `z_se` at each TF → survival stop's `room` calculation
- `dmi_diff` at each TF → regime decay's DI cross detection  
- `variance_ratio` → regime classification (mean-revert vs trend)
- `hurst` → survival stop's trend memory factor
- `vol_rel` → giveback's volume-relative threshold
- `reversion_prob` → direct probability input for survival score
- `acceleration` → chop detection, modulates HL

The NN reads 79D → predicts half-life.
The exits read the SAME 79D features → modulate half-life in real time.
Same data, two uses: prediction and live adjustment.
