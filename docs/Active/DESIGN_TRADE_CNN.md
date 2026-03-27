# DESIGN: Multi-Head Trade Lifecycle CNN (TradeCNN)

**Status:** Architecture design — not ready for implementation  
**Replaces:** Direction CNN (veto-only) + DMI flipper (timing + risk)  
**Goal:** Single model answers all 5 trade lifecycle questions per bar

---

## The 5 Questions

| # | Question | When Asked | Output | Training Label |
|---|----------|-----------|--------|----------------|
| 1 | Should I enter? | Flat only | P(enter) 0-1 | Was entering here profitable after N bars? |
| 2 | Which direction? | Entering | P(long) 0-1 | Was LONG more profitable than SHORT? |
| 3 | Expected outcome? | Entering | E[PnL] in ticks | Actual PnL achieved (regression) |
| 4 | How long to hold? | Entering | Bars to MFE | Actual bars to peak favorable excursion |
| 5 | Should I exit? | In trade | P(exit_now) 0-1 | Was exiting here better than holding? |

Head 5 is the hardest — it needs trade context (side, entry price, bars held,
unrealized PnL) as conditional input, not just market features.

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │     Market State Input       │
                    │  (lookback × features)       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      Shared Backbone         │
                    │  Conv1D stack (shared repr)   │
                    │  Output: latent vector [128]  │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼────────┐  ┌───────▼────────┐  ┌────────▼────────┐
    │  ENTRY HEAD      │  │  TRADE HEAD    │  │  EXIT HEAD      │
    │  (flat context)  │  │  (entry ctx)   │  │  (in-trade ctx) │
    │                  │  │                │  │                  │
    │  [128] → Dense   │  │  [128] → Dense │  │  [128 + 5]      │
    │  → P(enter)      │  │  → P(long)     │  │  → Dense         │
    │  → P(quality)    │  │  → E[pnl]      │  │  → P(exit_now)   │
    │                  │  │  → E[bars_mfe] │  │  → E[remaining]  │
    └──────────────────┘  └────────────────┘  └─────────────────┘
```

### Shared Backbone
- Input: `(batch, lookback, n_features)` — same 7D SFE features as current CNN
- Conv1D stack: 2-4 layers (reuse proven architecture)
- Output: 128-dim latent vector representing "market regime right now"
- This is the expensive part — computed once, shared across all heads

### Entry Head (active when flat)
- Input: backbone latent [128]
- Output 1: `P(enter)` — should I enter on this bar? (sigmoid)
- Output 2: `P(quality)` — confidence/quality score (sigmoid)
- Training label: 1 if entering here led to profit > 2 ticks after optimal hold
- Loss: magnitude-weighted BCE (big moves matter more, same as current CNN)

### Trade Head (active at entry decision)
- Input: backbone latent [128]
- Output 1: `P(long)` — probability LONG is the profitable side (sigmoid)
- Output 2: `E[pnl]` — expected PnL in ticks (linear, no activation)
- Output 3: `E[bars_to_mfe]` — expected bars until peak profit (ReLU)
- Training labels:
  - P(long): 1 if LONG pnl > SHORT pnl over forward window
  - E[pnl]: actual max(long_pnl, short_pnl) in ticks (best-side outcome)
  - E[bars_to_mfe]: bar index of MFE in the forward window
- Loss: BCE for direction + MSE for regression outputs

### Exit Head (active when in a trade)
- Input: backbone latent [128] **concatenated with trade context [5]**
- Trade context vector:
  - `side` (1.0 = long, -1.0 = short)
  - `unrealized_pnl_ticks` (current profit/loss)
  - `bars_held` (normalized by expected hold)
  - `pnl_vs_expected` (current pnl / predicted E[pnl])
  - `peak_capture` (current pnl / peak pnl so far, 0-1)
- Output 1: `P(exit_now)` — should I close this bar? (sigmoid)
- Output 2: `E[remaining_pnl]` — expected additional ticks if held (linear)
- Training label: 1 if exiting here captured more PnL than holding to MFE
  (accounts for drawdown — "exit" is correct if price goes further but
  draws down > SL first)
- Loss: BCE for exit signal + MSE for remaining PnL

---

## Training Data Construction

### Source
Same as current CNN: 1m ATLAS bars → SFE states → feature extraction.
But labels are richer — need full forward trajectories, not just binary.

### Label Pipeline (per bar i)

```python
# Forward trajectory (full lookahead)
future_prices = prices[i+1 : i+MAX_FORWARD+1]  # next 60 bars

# Entry labels
long_pnl = (future_prices - prices[i]) / TICK    # tick PnL if went LONG
short_pnl = (prices[i] - future_prices) / TICK   # tick PnL if went SHORT
best_pnl = np.maximum(long_pnl.max(), short_pnl.max())
best_dir = 'LONG' if long_pnl.max() >= short_pnl.max() else 'SHORT'

# Was entering profitable? (after accounting for SL)
# Entry is "good" if best-side MFE > 4 ticks (covers commission + spread)
label_enter = 1.0 if best_pnl > 4.0 else 0.0
label_quality = min(1.0, best_pnl / 20.0)  # quality score, caps at 20t

# Direction
label_long = 1.0 if best_dir == 'LONG' else 0.0

# Expected outcome
label_pnl = best_pnl  # ticks (regression target)

# Bars to MFE
if best_dir == 'LONG':
    label_bars_mfe = np.argmax(long_pnl) + 1
else:
    label_bars_mfe = np.argmax(short_pnl) + 1

# Exit labels (one per bar while "in trade")
# For each bar j after entry i:
#   Should I exit at j?
#   remaining_pnl = MFE_from_j - pnl_at_j
#   If remaining_pnl < 0 (already past peak): exit = 1
#   If drawdown from peak > SL: exit = 1 (would have been stopped out)
#   Else: exit = 0
```

### Exit Label Detail

The exit head needs **in-trade samples** — not just one label per bar,
but labels conditioned on being in a specific trade. For each training
trade:

```python
for entry_bar in range(n_bars):
    if not label_enter[entry_bar]:
        continue  # skip bars where entry wasn't good
    
    side = 'long' if label_long[entry_bar] else 'short'
    entry_px = prices[entry_bar]
    
    # Walk forward through the trade
    peak = 0.0
    for hold_bar in range(1, MAX_FORWARD):
        j = entry_bar + hold_bar
        if j >= n_bars:
            break
        
        if side == 'long':
            pnl_now = (prices[j] - entry_px) / TICK
        else:
            pnl_now = (entry_px - prices[j]) / TICK
        
        peak = max(peak, pnl_now)
        
        # Remaining potential: how much more can we get?
        future_from_j = prices[j+1:j+MAX_FORWARD]
        if side == 'long':
            remaining_mfe = (future_from_j.max() - prices[j]) / TICK if len(future_from_j) > 0 else 0
        else:
            remaining_mfe = (prices[j] - future_from_j.min()) / TICK if len(future_from_j) > 0 else 0
        
        # Drawdown from peak
        drawdown = peak - pnl_now
        
        # Exit label: exit if remaining < drawdown risk or past peak
        should_exit = 1.0 if (remaining_mfe < 2.0 or drawdown > SL * 0.5) else 0.0
        
        # Create training sample with trade context
        exit_samples.append({
            'bar_index': j,
            'features': features[j],  # market state at bar j
            'trade_context': [
                1.0 if side == 'long' else -1.0,
                pnl_now,
                hold_bar / label_bars_mfe[entry_bar],  # normalized hold
                pnl_now / label_pnl[entry_bar] if label_pnl[entry_bar] > 0 else 0,
                pnl_now / peak if peak > 0 else 0,
            ],
            'label_exit': should_exit,
            'label_remaining': remaining_mfe,
        })
```

---

## Training Strategy

### Phase 1: Train backbone + entry/trade heads (flat-context)
- Standard supervised learning on IS data
- Same train/val split as current CNN (temporal 90/10)
- Loss = `α * BCE(enter) + β * BCE(long) + γ * MSE(pnl) + δ * MSE(bars_mfe)`
- Weights: α=1.0, β=1.0, γ=0.1 (pnl in ticks, needs scaling), δ=0.01

### Phase 2: Freeze backbone, train exit head
- Uses in-trade samples (different dataset shape)
- Backbone weights frozen — only exit head trains
- Loss = `BCE(exit) + 0.1 * MSE(remaining_pnl)`
- This prevents exit head training from corrupting the shared representation

### Phase 3: Fine-tune end-to-end (optional)
- Unfreeze backbone, train all heads jointly with low LR
- Risk of catastrophic forgetting — use small learning rate (1e-5)

---

## Live Integration

### Replacing the Current Stack

| Current | TradeCNN Replacement |
|---------|---------------------|
| DMI flipper `on_bar()` → ENTER | Entry head `P(enter) > threshold` |
| DMI flipper cross direction | Trade head `P(long) > 0.5` |
| `physics_sl_ticks` / `physics_tp_ticks` | Trade head `E[pnl]` → dynamic TP, `E[bars_mfe]` → max hold |
| `check_sl_1s()` SL/TP | Exit head `P(exit) > threshold` + fixed SL safety net |
| CNN veto (current) | Eliminated — direction is now integrated |

### Live Call Pattern

```python
# Every 1m bar:
latent = backbone(features[-lookback:])

if not in_trade:
    p_enter = entry_head(latent)
    if p_enter > ENTRY_THRESHOLD:
        p_long = trade_head.direction(latent)
        e_pnl = trade_head.expected_pnl(latent)
        e_hold = trade_head.bars_to_mfe(latent)
        
        side = 'long' if p_long > 0.5 else 'short'
        dynamic_tp = max(4, int(e_pnl * 0.8))   # TP at 80% of expected
        dynamic_sl = max(4, int(e_pnl * 0.5))    # SL at 50% of expected
        max_hold = max(5, int(e_hold * 1.5))      # hold 1.5x expected
        
        enter_trade(side, dynamic_tp, dynamic_sl, max_hold)

else:  # in trade
    trade_ctx = build_trade_context(side, unrealized_pnl, bars_held, ...)
    p_exit = exit_head(latent, trade_ctx)
    e_remaining = exit_head.remaining_pnl(latent, trade_ctx)
    
    if p_exit > EXIT_THRESHOLD:
        close_position('cnn_exit')
    elif e_remaining < -2.0:  # model expects further loss
        close_position('cnn_negative_outlook')
    # Fixed SL always active as safety net (never removed)
```

### Safety: Fixed SL Always Active

The exit head is a LEARNED exit — it can be wrong. A hard SL (e.g., 40 ticks)
remains as an unconditional safety net that the model cannot override.
The model can exit BEFORE the SL, but never disable it.

---

## Feature Set Decision

### Option A: Keep 7D (low risk)
Same features as current CNN. Proven to work. Just add the multi-head
architecture on top. Lower risk of train/live skew since features are validated.

### Option B: Expand to 12D (medium risk)
Add 5 features that the exit head needs but are also useful for entry:
- `bar_range_ticks` (H-L in ticks — volatility proxy)
- `wick_ratio` (body vs range — indecision signal)  
- `adx_strength` (trend strength from SFE)
- `hurst_exponent` (mean-reversion vs trend from SFE)
- `time_of_day` (0-1, session context)

These are all available from SFE state or OHLCV bars. No new data sources.

### Recommendation: Start with 7D
Prove the architecture works with proven features. Add features in v2
once the multi-head training pipeline is validated.

---

## What This Replaces vs What It Keeps

### Eliminated
- `core/dmi_flipper.py` — entry timing, direction, TP/SL management
- CNN direction veto in `_process_1m_physics`
- `_cnn_predict()` current implementation
- `check_sl_1s()` exit checks (replaced by exit head)
- Fixed TP cycling logic

### Kept
- SFE state computation (feature source — unchanged)
- NT8 bridge + bar aggregator (data pipeline — unchanged)
- OrderManager + NT8 order lifecycle (execution — unchanged)
- Fixed SL safety net (risk management — never removed)
- SessionTracker + reporting (telemetry — unchanged)
- `ExitEngine.open_position` for position tracking (MFE, peak tracking)

### New
- `core/trade_cnn.py` — model definition + feature extraction
- `training/train_trade_cnn.py` — multi-phase training pipeline
- `live/cnn_engine.py` — live integration (replaces flipper + old CNN)

---

## Implementation Phases

### Phase 0: Data pipeline (1 session)
- Build label extraction: entry quality, direction, PnL, MFE bars, exit labels
- Validate on IS data: print label distributions, sanity check
- Output: `training/trade_cnn_labels.py`

### Phase 1: Model + entry/trade heads (1-2 sessions)  
- Define multi-head architecture in `core/trade_cnn.py`
- Train backbone + entry + trade heads on IS
- Validate direction accuracy vs current CNN (must match or beat)
- Validate entry timing: does P(enter) filter better than DMI crosses?

### Phase 2: Exit head (1 session)
- Build in-trade sample dataset
- Train exit head with frozen backbone
- Validate on OOS: does exit head capture more than fixed TP/SL?

### Phase 3: Live integration (1 session)
- Create `live/cnn_engine.py` that replaces flipper + old CNN
- Wire into `_process_1m_physics` as new engine mode (--cnn flag)
- Keep --dmi as fallback until CNN proves superior in live

### Phase 4: Fine-tune + iterate
- End-to-end fine-tuning
- Feature expansion (7D → 12D)
- Threshold tuning via live_tuning.json
- A/B comparison: --dmi vs --cnn on consecutive sessions

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Overfitting entry head (enters too often) | High | Magnitude-weighted loss + entry threshold tuning |
| Exit head learns "always hold" | High | Balanced sampling (50% exit=1, 50% exit=0) |
| Train/live skew on new features | Medium | Start with proven 7D, add features incrementally |
| Backbone representation collapse | Medium | Phase 2 freezes backbone during exit training |
| Worse than DMI flipper in live | Medium | Keep --dmi as fallback, A/B test before switching |
| Regression heads predict nonsense | Low | Clip E[pnl] to [0, 100], E[bars] to [1, 60] |

---

## Success Criteria

Before replacing DMI flipper in live:
1. Entry head filters at least as selectively as DMI crosses (same or fewer entries)
2. Direction accuracy ≥ current CNN on OOS (≥ 53%)
3. E[pnl] correlation with actual PnL > 0.3 on OOS
4. Exit head captures ≥ 60% of MFE on OOS (vs current ~40% from fixed TP)
5. Full system PnL on OOS ≥ DMI flipper PnL on same data

Only after ALL 5 criteria are met does TradeCNN go live.
