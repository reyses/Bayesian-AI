# Strategy NN Specification — The AI
> Session: 2026-04-03
> Status: SPEC — ready to implement

## Core Concept

The AI = NN (pattern recognition) + Bayesian Brain (learning memory).
NN provides the prior. Brain provides the evidence. Output is the posterior.

```
P(profit | state) = P(state | profit) * P(profit) / P(state)
                    ~~~~~~~~~~~~~~~~   ~~~~~~~~~~
                    brain (evidence)   NN (prior)
```

The NN is trained once. The brain updates every trade. Together they adapt.

---

## Three Layers

### Layer 1 — SETUP RECOGNITION
"I've seen these last N bars before. This is forming into something."

**Input**: Sequence of 79D states (last 10 bars at 1m anchor)
**Output**: 
- setup_id (which pattern is forming)
- p_trigger (probability Layer 2 will trigger)
- expected_entry_conditions (what the 79D should look like at trigger)

**Brain calibration**: 
- Tracks per setup_id: how often does Layer 2 actually trigger?
- Adjusts p_trigger toward observed trigger rate

**Examples**:
- "z_se climbing for 6 bars, now 2.8, acceleration positive → deep reversion forming"
- "dmi_diff aligned across 4 TFs, hurst 0.7, steady climb → trend continuation forming"
- "vol_rel spiking to 4x, z_se at 3.5 → exhaustion spike forming"

### Layer 2 — ENTRY + EXPECTED PATH
"The setup triggered. Here's what should happen."

**Input**: 79D at entry moment + setup_id from Layer 1
**Output**:
- direction: LONG / SHORT
- half_life: bars (continuous)
- expected_pnl: $ if the trade plays out
- expected_loss: $ if wrong
- p_profit: probability of profit (NN prior)
- max_drawdown: worst expected MFE on the path
- expected_path: sequence of key 79D features at bar 1, 3, 5, 8, 13
  (not every feature — just z_se, dmi_diff, vol_rel at anchor TF)

**Brain calibration**:
- Tracks per setup_id + direction: actual WR vs predicted p_profit
- Tracks actual half-life vs predicted half-life → adjusts toward reality
- Tracks actual PnL vs expected_pnl → adjusts expectations
- "NN says 8-bar HL but reality is 6 → posterior = 6"

**Expected path example (deep reversion SHORT from z_se=3.1)**:
```
bar  0 (entry): z_se=3.1, dmi=-4.2, vol_rel=1.5
bar  1:         z_se=2.8, dmi=-3.8, vol_rel=1.3
bar  3:         z_se=2.3, dmi=-2.5, vol_rel=1.1
bar  5:         z_se=1.5, dmi=-1.5, vol_rel=1.0
bar  8:         z_se=0.5, dmi=-0.5, vol_rel=0.9  ← half-life point
bar 13:         z_se=0.2, dmi= 0.0, vol_rel=0.8  ← exhaustion
```

### Layer 3 — EXIT MANAGEMENT
"Is the trade following the expected path?"

**Input**: 
- Actual 79D at current bar
- Expected 79D from Layer 2's expected_path
- bars_held, current PnL, peak PnL

**Output**: 
- action: HOLD / TIGHTEN / EXIT
- adjusted_half_life (compressed if diverging)
- envelope_level (current exit threshold)

**Core math**:
```python
divergence = distance(actual_79d_key_features, expected_path[bars_held])
effective_hl = base_hl * max(0.1, 1.0 - divergence / tolerance)
decay = exp(-ln2 * bars_held / effective_hl)
envelope = floor + (peak_pnl - floor) * decay
if pnl < envelope → EXIT
```

**Brain calibration**:
- Tracks: at what divergence level do trades typically fail?
- Adjusts tolerance toward the divergence threshold that maximizes PnL
- "Divergence 0.3 used to be OK, but last 20 trades at 0.3 → 70% went to stop"
- → Tighten tolerance to 0.25

---

## NN Architecture

```
                    ┌─────────────────────┐
 10-bar sequence    │  Layer 1: Setup     │
 of 79D states  ──→ │  (TCN or LSTM)      │──→ setup_id, p_trigger
                    │  ~10K params         │
                    └────────┬────────────┘
                             │ setup_id
                    ┌────────▼────────────┐
 79D at entry   ──→ │  Layer 2: Entry     │──→ direction, half_life,
                    │  (MLP multi-head)    │    expected_pnl, expected_loss,
                    │  ~20K params         │    p_profit, max_drawdown,
                    │                      │    expected_path[5 checkpoints]
                    └────────┬────────────┘
                             │ expected_path
                    ┌────────▼────────────┐
 actual 79D     ──→ │  Layer 3: Exit      │──→ hold/tighten/exit,
 at current bar     │  (simple MLP or     │    adjusted_half_life,
                    │   rule-based)        │    envelope_level
                    │  ~5K params          │
                    └─────────────────────┘
```

Total: ~35K parameters. Tiny. Trains fast. Runs at 5s speed easily.

Layer 3 could be rule-based (the envelope math) rather than learned.
The NN parts are Layer 1 (sequence recognition) and Layer 2 (prediction).

---

## Bayesian Brain Integration

The brain wraps OUTSIDE the NN. It doesn't modify weights — it adjusts outputs.

```python
class BayesianAI:
    def __init__(self, nn_model, brain):
        self.nn = nn_model      # frozen after training
        self.brain = brain      # updates every trade
    
    def predict(self, state_sequence, current_79d):
        # NN prior
        setup = self.nn.layer1(state_sequence)
        entry = self.nn.layer2(current_79d, setup.id)
        
        # Brain posterior
        calibrated = self.brain.calibrate(
            setup_id=setup.id,
            nn_p_profit=entry.p_profit,
            nn_half_life=entry.half_life,
            nn_expected_pnl=entry.expected_pnl,
        )
        
        return {
            'direction': entry.direction,
            'half_life': calibrated.half_life,      # brain-adjusted
            'p_profit': calibrated.p_profit,        # brain-adjusted
            'expected_pnl': calibrated.expected_pnl, # brain-adjusted
            'expected_loss': entry.expected_loss,
            'max_drawdown': entry.max_drawdown,
            'expected_path': entry.expected_path,
        }
    
    def update(self, trade_result):
        # After trade completes, brain learns
        self.brain.update(
            setup_id=trade_result.setup_id,
            predicted_hl=trade_result.predicted_hl,
            actual_hl=trade_result.actual_hl,
            predicted_pnl=trade_result.predicted_pnl,
            actual_pnl=trade_result.actual_pnl,
            predicted_p=trade_result.predicted_p,
            was_profitable=trade_result.pnl > 0,
        )
```

---

## Execution Layer — Risk-Based Trade Sizing

The AI outputs the trade specification. Execution measures risk against equity
and compresses the trade to fit.

### Equity tracking
```
equity = starting_capital + sum(realized_pnl) + unrealized_pnl
available_risk = equity - margin_requirement
```

### Risk per trade
The AI gives: expected_loss, max_drawdown, p_profit.
Execution computes:

```python
# Expected risk in dollars
risk_dollars = ai_output['max_drawdown']

# What we can afford
max_risk = equity * max_risk_pct                   # e.g., 10% of equity
max_risk = min(max_risk, equity - margin_floor)     # never go below margin

# Leash ratio: compress trade to fit bankroll
leash = min(1.0, max_risk / risk_dollars)

# Apply leash to all trade parameters
effective_hl = ai_output['half_life'] * leash       # shorter hold
stop = ai_output['max_drawdown'] * leash            # tighter stop
target = ai_output['expected_pnl'] * leash          # proportional target

# Risk-adjusted expected value
risk_adj_ev = (ai_output['p_profit'] * target) - ((1 - ai_output['p_profit']) * stop)
```

### Trade rejection (the ONE case we say no)
```python
# Only reject if risk-adjusted EV is negative after compression
if risk_adj_ev <= cost_per_trade:
    return NO_TRADE  # signal exists but math doesn't work at this equity level
```

Not rejecting because risk is too high — compressing instead.
Only rejecting when compression makes the trade unprofitable.

### Daily risk budget
```python
daily_loss_limit = equity * daily_risk_pct          # e.g., 20% of equity
daily_pnl = sum(today's realized trades)

if daily_pnl < -daily_loss_limit:
    STOP_TRADING_TODAY
    # The AI still runs, logs what it WOULD have done (shadow mode)
    # Brain still learns from the shadow trades
```

### Equity growth → leash loosens naturally
| Equity | Max Risk (10%) | Leash on $35 DD trade | Effective HL | Stop |
|--------|---------------|----------------------|-------------|------|
| $100   | $10           | 0.29x                | 2.3 bars    | $10  |
| $250   | $25           | 0.71x                | 5.7 bars    | $25  |
| $500   | $50           | 1.0x (full)           | 8 bars      | $35  |
| $2,000 | $200          | 1.0x (capped)         | 8 bars      | $35  |

### Consecutive loss protection
```python
consecutive_losses = count since last win

# After 3 consecutive losses: tighten max_risk_pct
if consecutive_losses >= 3:
    max_risk_pct *= 0.5  # half the risk budget
if consecutive_losses >= 5:
    max_risk_pct *= 0.25  # quarter the risk budget
    # Also: Brain should be questioning the AI's setup recognition
    
# Reset on next win
if trade_won:
    consecutive_losses = 0
    max_risk_pct = base_risk_pct
```

### Shadow mode (when stopped out for the day)
```python
if stopped_for_day:
    # AI continues to predict, but no orders
    ai_output = self.ai.predict(state)
    self.shadow_log.append(ai_output)
    
    # Brain STILL updates from what would have happened
    # This prevents the brain from only learning from losing streaks
    # (survivorship bias in reverse)
    simulated_result = simulate_trade(ai_output)
    self.brain.update(simulated_result, shadow=True)
```

### Full execution flow
```python
class ExecutionLayer:
    def __init__(self, starting_equity, config):
        self.equity = starting_equity
        self.max_risk_pct = config.max_risk_pct          # 0.10
        self.daily_risk_pct = config.daily_risk_pct      # 0.20
        self.margin_floor = config.margin_floor           # $50
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.stopped_for_day = False
    
    def execute(self, ai_output):
        # Daily stop check
        if self.stopped_for_day:
            self.shadow_log(ai_output)
            return None
        
        if self.daily_pnl < -(self.equity * self.daily_risk_pct):
            self.stopped_for_day = True
            return None
        
        # Risk budget
        risk_pct = self.max_risk_pct
        if self.consecutive_losses >= 3:
            risk_pct *= 0.5
        if self.consecutive_losses >= 5:
            risk_pct *= 0.25
        
        max_risk = min(
            self.equity * risk_pct,
            self.equity - self.margin_floor
        )
        
        if max_risk <= 0:
            return None  # can't afford any trade
        
        # Leash
        leash = min(1.0, max_risk / ai_output['max_drawdown'])
        
        effective_hl = ai_output['half_life'] * leash
        stop = ai_output['max_drawdown'] * leash
        target = ai_output['expected_pnl'] * leash
        
        # Risk-adjusted EV check
        ev = (ai_output['p_profit'] * target) - ((1 - ai_output['p_profit']) * stop)
        if ev <= 0.50:  # must exceed cost
            return None
        
        return TradeOrder(
            direction=ai_output['direction'],
            half_life=effective_hl,
            stop=stop,
            target=target,
            expected_path=ai_output['expected_path'],
            p_profit=ai_output['p_profit'],
            leash=leash,
            risk_dollars=stop,
            equity_at_entry=self.equity,
        )
    
    def on_trade_close(self, result):
        self.equity += result.pnl
        self.daily_pnl += result.pnl
        
        if result.pnl > 0:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
    
    def on_new_day(self):
        self.daily_pnl = 0.0
        self.stopped_for_day = False
```

---

## Training Pipeline

### Label generation (per 1m bar across 311 days):
```
For each bar:
  Compute 79D state
  Also store last 10 bars of 79D (the setup sequence)
  
  For direction in [LONG, SHORT]:
    Simulate forward at 5s resolution:
      Record PnL at checkpoints: bar 1, 3, 5, 8, 13, 21
      Record max drawdown at each checkpoint
      Record 79D features at each checkpoint (the actual path)
    
    best_checkpoint = argmax(risk_adjusted_pnl)
    
  Label = {
    setup_sequence: last 10 bars of 79D,
    entry_79d: current 79D,
    best_direction: LONG or SHORT or SKIP,
    half_life: best_checkpoint duration,
    pnl: actual PnL at best checkpoint,
    loss: actual PnL at worst checkpoint,
    max_drawdown: worst adverse excursion,
    path: 79D at each checkpoint (the expected path for this setup),
  }
```

### Training:
- Layer 1: train on setup_sequence → setup_id (unsupervised clustering or supervised)
- Layer 2: train on entry_79d + setup_id → direction, half_life, pnl, p_profit, path
- Layer 3: rule-based (envelope decay), calibrate tolerance from validation set

### Validation:
- Per-day PnL. Each day stands alone.
- Walk-forward: train on months 1-6, validate 7-8, test 9-12
- Mode of daily PnL, not mean

---

## Connection to Existing Code

| New Component | Replaces | Reuses |
|--------------|----------|--------|
| Layer 1 (setup recognition) | Pattern detection + fractal discovery | TCN architecture from `training/direction_tcn.py` |
| Layer 2 (entry + path) | Advance engine + execution engine + 9-voter direction cascade | StatePredictor architecture from `core/trade_cnn.py` |
| Layer 3 (exit management) | Exit engine 10-exit cascade | Envelope decay math from `core/exits/envelope.py` |
| Bayesian Brain | `core/bayesian_brain.py` | Same brain, new state keys (setup_id instead of template_id) |
| 79D features | 16D feature_extraction + 13D grounded | `core/feature_extraction.py` (rewritten) |
| Execution layer | N/A (new) | Trailing stop ratchet from `core/exits/breakeven.py` |

---

## What This System Does NOT Have
- No template matching (K-Means centroids)
- No pattern detection gates (cascade_detected, structure_confirmed)  
- No 9-voter direction cascade
- No cat brain regime classifier
- No peak reversal as a separate path
- No Gate 0/1/2/3/4 cascade
- No worker bypass
- No magic numbers (0.05, -0.10, 0.55, 50.0, etc.)

The AI replaces all of it. 79D → direction + half-life + expected path.
