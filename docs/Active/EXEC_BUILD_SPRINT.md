# EXEC: BayesianBridge Build Sprint — Full Execution Spec

**For:** VS Code Claude  
**From:** Claude.ai (diagnosis/spec instance)  
**Scope:** Steps 1-4 in sequence. Each step has a gate — don't proceed if gate fails.

---

## Step 1: Apply DMI Fixes (live_engine.py)

**Spec file:** `SPEC_DMI_FIXES_V2.md` (already on disk)  
**Target:** `live/live_engine.py`  
**Scope:** `--dmi` mode only

### Fixes (in priority order)

1. **CNN full pre-fill** — Replace HISTORY_DONE CNN pre-fill block with sequential iteration computing all 7 features from aggregator bars + states. Search for `# Pre-fill CNN feature buffer from history`. Use 30-bar SMA for vol_avg (NOT flipper EMA). Pre-seed price/vol buffers from bars before prefill window.

2. **Flipper `_in_trade` reset** — Add `self._dmi_flipper._in_trade = False` at top of `_close_position()`, right after `self._closing_position = True`. Remove redundant scattered resets (keep CONNECTION_LOST one).

3. **Flipper entry price sync** — In `_physics_enter()`, after order sent, add: `self._dmi_flipper._entry_price = price`, `._in_trade = True`, `._last_tp_price = 0.0`. Also add fill price sync in `_pending_tp_reentry` FILL handler.

4. **Reverse TP_BANK guard** — In `_process_1m_physics()` TP_BANK handler, add: `if self._pending_tp_reentry or self._closing_position: return`

5. **CNN gate on TP re-entry** — In FILL handler `_pending_tp_reentry` block, add CNN direction check before re-entry. If CNN disagrees, cancel re-entry.

### Gate
- [ ] `--dmi --no-gui` starts without AttributeError
- [ ] After HISTORY_DONE: CNN log shows full buffer (not zeros)
- [ ] No duplicate `EXIT ->` lines within 1s in log
- [ ] Flipper `_in_trade` resets on every exit path

---

## Step 2: Apply CNN Feature Parity Fix (live_engine.py)

**Spec file:** `SPEC_CNN_FEATURE_PARITY.md` (already on disk)  
**Target:** `live/live_engine.py`  
**Scope:** `_cnn_predict()` + HISTORY_DONE pre-fill

### Changes

1. **Add CNN buffers** in `__init__` / `_init_dmi_flipper()`:
```python
self._cnn_vol_buffer = []    # 30-bar volume SMA (matches training)
self._cnn_price_buffer = []  # 60-bar price buffer for z_se + dir_vol
```

2. **Rewrite `_cnn_predict()`** — Source all features from SFE state + aggregator bars. Volume uses 30-bar SMA from `self._cnn_vol_buffer` (NOT flipper EMA). Price uses `self._cnn_price_buffer` (NOT flipper `_price_hist`). Zero flipper dependency.

3. **Rewrite HISTORY_DONE pre-fill** — Use aggregator bar volumes with 30-bar SMA. Populate `_cnn_vol_buffer` and `_cnn_price_buffer` for seamless handoff to live.

### Gate
- [ ] CNN predictions appear from bar 1 (not delayed 10 min)
- [ ] `vol_avg` logged value uses SMA, not EMA
- [ ] CNN veto rate hasn't dramatically changed (±10% is acceptable)

---

## Step 3: Build Label Pipeline (training/train_trade_cnn.py)

**Spec file:** `SPEC_TRADE_CNN_V3.md` Part 2-3  
**Target:** NEW file `training/train_trade_cnn.py`

### Build these functions

1. **`extract_features_13d(states, df)`** — 13D feature extraction from SFE states + OHLCV. 7D directional + 4D regime + 2D context. 30-bar SMA for vol. No flipper dependency. See SPEC_TRADE_CNN_V3.md Part 2.

2. **`build_state_labels(feats_7d, horizons=[1, 5, 10])`** — For each bar, label = actual 7D feature values at t+1, t+5, t+10. Returns (n_bars, 21) array. Zero failure rate. See SPEC_TRADE_CNN_V3.md Part 3.1.

3. **`build_dataset(data_root)`** — Load 1m parquet files → SFE `batch_compute_states()` → `extract_features_13d()` → `build_state_labels()`. Return feats, labels, states, df.

4. **`SlidingWindowDataset`** — PyTorch Dataset. Input: (lookback × 13) features. Label: (21,) state predictions. Lookback=10.

5. **Validation script** — Run `build_dataset` on ATLAS data. Print:
   - Feature distributions (mean, std, min, max per feature)
   - Label distributions (same)
   - Sample count
   - Any NaN/inf warnings
   - Correlation matrix between features (check for redundancy)

### Gate
- [ ] `python -m training.train_trade_cnn --phase labels` runs without error
- [ ] Feature distributions look reasonable (no collapsed variance, no NaN)
- [ ] Labels have non-zero variance for all 21 outputs
- [ ] Total samples > 500K (14 months × ~1380 bars/day)

---

## Step 4: Model A — Single-TF CNN + Walk-Forward

**Spec file:** `SPEC_TRADE_CNN_V3.md` Part 1, 4, 6-8  
**Target:** NEW files `core/trade_cnn.py` + additions to `training/train_trade_cnn.py`

### 4a: Model Definition

`core/trade_cnn.py` — StatePredictor model:
- CNNBackbone: Conv1d(13→32→64) + AdaptivePool + Linear(64)
- StatePredictor: backbone → merge → head(64→21)
- ~15K params for single-TF
- See SPEC_TRADE_CNN_V3.md Part 1.1

### 4b: Walk-Forward Training (Incremental Carry-Forward)

**CRITICAL — this is the updated flow, NOT the old min_train_days approach:**

```
Day 1:  Cold start → train from scratch on Day 1 (30 epochs) → model_v1
Day 2:  model_v1 → PREDICT Day 2 (SCORE) → fine-tune on Day 2 (5 epochs, LR=1e-4) → model_v2
Day 3:  model_v2 → PREDICT Day 3 (SCORE) → fine-tune on Day 3 → model_v3
...
Day N:  model_v(N-1) → PREDICT Day N (SCORE) → fine-tune → model_vN
```

**Key properties:**
- Score is captured BEFORE training on that day (pure OOS)
- Fine-tune LR = 1e-4 (10× lower than initial) — preserves prior knowledge
- Checkpoint every 10 days
- **WARMUP EXCEPTION:** For Model A (~15K params), Day 1 cold start on ~1,380 bars is acceptable. For larger models (>50K params), use 30-60 day bulk warmup before switching to carry-forward.

See updated `walk_forward_train()` in SPEC_TRADE_CNN_V3.md Part 4.

### 4c: Day-Level Validation

`_validate_day()` computes per scored day:
- Feature correlations (Spearman, per feature per horizon)
- Direction accuracy (sign of predicted dmi_diff_t5 vs actual)
- Trading simulation PnL (`_simulate_day_trading()`)

### 4d: Trading Simulation

`_simulate_day_trading()` — interpret predicted states into entry/exit:
- Entry: dmi_gap building across horizons + direction consistent + velocity confirming
- Exit: trend reversed OR momentum fading
- Hard SL: 40 ticks unconditional
- Commitment: 3 bars minimum hold

### 4e: Walk-Forward Report

`walk_forward_report()` — summarize across all days:
- Cumulative PnL, $/day avg
- Profitable day %, max drawdown
- Monthly breakdown
- Avg feature correlation, direction accuracy

### Execution

```bash
# Run everything
python -m training.train_trade_cnn --model A --tf single --phase all

# Expected output:
#   Day 1/290: COLD START — training v1 from scratch
#   Day 2/290: v1 → corr=0.XXX  dir=XX.X%  pnl=+Xt  cum=$XXX
#   ...
#   WALK-FORWARD SUMMARY: StatePredictor
#     $/day avg: $XXX
#     vs Baseline: $736/day
```

### Gate
- [ ] Walk-forward completes all ~290 days without crash
- [ ] Average feature correlation > 0.05 (any signal at all)
- [ ] Direction accuracy > 51% (better than random)
- [ ] Cumulative PnL > 0 (net profitable)
- [ ] Not all PnL from 1-2 lucky days (check monthly breakdown)

### If Gate Fails
- corr < 0.05 → features aren't predictive at this horizon, investigate which features have zero correlation
- dir_acc < 51% → model not learning direction, check label construction
- PnL < 0 but corr > 0 → trading simulation logic wrong, predictions are right but interpretation is wrong
- crash → data pipeline issue, check SFE batch_compute_states output format

---

## What NOT to Build Yet

These are designed but NOT in this sprint:

- ❌ Model B (multi-head) — earned by Model A results
- ❌ Model C (Transformer) — earned by Model A results
- ❌ Model D (Cascade TB-CNN) — earned by A/B/C results
- ❌ Multi-TF features — earned by single-TF results
- ❌ Pattern attention / prototypes — aspirational, not validated
- ❌ CNN autoencoder on seed data — aspirational, not validated
- ❌ Live integration (`--cnn` flag) — earned by walk-forward beating $736

---

## File Inventory (what should exist after sprint)

| File | Status | Purpose |
|------|--------|---------|
| `live/live_engine.py` | MODIFIED | Steps 1-2: DMI fixes + CNN parity |
| `training/train_trade_cnn.py` | NEW | Step 3-4: features, labels, walk-forward |
| `core/trade_cnn.py` | NEW | Step 4: StatePredictor model definition |

## Reference Docs (read before starting)

| Doc | Location | Read For |
|-----|----------|----------|
| SPEC_DMI_FIXES_V2.md | `/mnt/user-data/outputs/` | Step 1 fix details |
| SPEC_CNN_FEATURE_PARITY.md | `/mnt/user-data/outputs/` | Step 2 fix details |
| SPEC_TRADE_CNN_V3.md | `/mnt/user-data/outputs/` | Steps 3-4 model + training code |

## Baseline to Beat

```
DMI flipper + 7D CNN (OOS): $736/day
TradeCNN target:            $800/day (+8.7%)
Minimum viable:             $0/day (net profitable = worth iterating on)
```
