# Plan: Price-Aware Workers + Continuous Pressure Model

## Problem Statement

EXIT QUALITY shows ALL trades exit at 4-6 bars regardless of capture %:
```
Optimal (>=80%):  47 bars avg hold   ← rare outlier (7 trades)
Partial 20-30%:    5 bars            ← killed too early
Partial 70-80%:    6 bars            ← killed too early
Too early (<20%):  5 bars            ← killed too early (5,810 trades)
Reversed:          4 bars            ← killed same speed as winners (6,907 trades)
```

Root cause: **Workers and CST are 100% geometric, 0% price-aware.**
- Workers compute `dir_prob`, `conviction`, `wave_maturity` from 16D quantum state
- CST fires when state drifts >3σ from entry centroid
- Neither knows entry price, current P&L, or whether the trade is winning
- A +20 tick winner at bar 40 gets the IDENTICAL treatment as a -20 tick loser

## What Already Exists (from this session)

1. `Position` has 4 trade-tracking fields: `running_mfe_ticks`, `running_mae_ticks`,
   `tmpl_expected_mfe_ticks`, `tmpl_expected_hold_bars`
2. `update_trail()` tracks running MFE/MAE each bar
3. `get_exit_signal()` accepts `trade_context` dict with `profit_ticks`,
   `pct_mfe_captured`, `pct_hold_elapsed`
4. `check_structural_integrity()` accepts `profit_ticks` (but barely uses it: 3.05σ vs 3.0σ)
5. Orchestrator builds `_trade_ctx` each bar and passes to both functions
6. Report math is now fixed with exact 3-term gap decomposition

## Architecture: 3 Changes, 3 Files, ~60 Lines

### Change 1: Worker Price Awareness (timeframe_belief_network.py, ~20 lines)

**Where**: `TimeframeWorker` class + `TimeframeBeliefNetwork`

Add to `TimeframeWorker.__init__()`:
```python
self._trade_side = None           # 'long' or 'short' when in trade
self._trade_profit_ticks = 0.0    # current P&L in ticks
```

Add to `TimeframeWorker._analyze()`, after line 286 (conviction computed):
```python
# Price-aware conviction modulation (2 layers)
if self._trade_side is not None:
    _agrees = ((self._trade_side == 'long' and dir_prob > 0.5) or
               (self._trade_side == 'short' and dir_prob < 0.5))
    _winning = self._trade_profit_ticks > 0

    # Layer 1: Direction + P&L agreement
    if _agrees and _winning:
        # Worker confirms winning trade → boost conviction
        conviction = min(1.0, conviction * 1.3)
    elif not _agrees and not _winning:
        # Worker sees reversal + price confirms → strengthen exit signal
        conviction = min(1.0, conviction * 1.2)
    elif not _agrees and _winning:
        # Worker says flip but price disagrees → dampen flip signal
        conviction *= 0.7
    # (_agrees and not _winning): worker says hold but losing → no change

    # Layer 2: DMI/ADX — direct trend signal, scaled by TF reliability
    # Higher TF workers have statistically reliable DMI (14h for 1h worker
    # vs 3.5min for 15s worker). Scale signal weight by log(bars_per_update).
    _dmi_diff = getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)
    _dmi_agrees = ((self._trade_side == 'long' and _dmi_diff > 0) or
                   (self._trade_side == 'short' and _dmi_diff < 0))
    _adx = getattr(state, 'adx_strength', 0.0)
    # TF reliability: 0.0 (15s) → 1.0 (1h+)
    _tf_reliability = min(1.0, math.log(max(2, self.bars_per_update)) / math.log(240))

    if not _dmi_agrees and _adx > 25:
        # DMI flipped against trade + strong trend → penalize conviction
        # 1h worker: up to 30% penalty. 15s worker: ~3% (noise-level)
        conviction *= (1.0 - 0.3 * _tf_reliability)
    elif _dmi_agrees and _adx > 25:
        # DMI confirms trade direction + strong trend → small boost
        conviction = min(1.0, conviction * (1.0 + 0.15 * _tf_reliability))
```

Add propagation methods to `TimeframeBeliefNetwork`:
```python
def set_trade_context(self, side: str, profit_ticks: float):
    for w in self.workers.values():
        w._trade_side = side
        w._trade_profit_ticks = profit_ticks

def clear_trade_context(self):
    for w in self.workers.values():
        w._trade_side = None
        w._trade_profit_ticks = 0.0
```

### Change 2: Continuous Pressure Model (timeframe_belief_network.py, ~25 lines)

**Where**: Replace the 3 discrete rules in `get_exit_signal()` (lines 717-753)

```python
# ── Continuous hold/exit pressure ──
# Replaces discrete rules with a single scalar that drives all exit decisions.
# Positive = hold (profitable + early + aligned)
# Negative = exit (losing + late + diverging)
net_pressure = 0.0
_pressure_reason = ''

if trade_context and self._trade_expected_mfe_ticks > 0:
    _profit   = trade_context.get('profit_ticks', 0.0)
    _pct_cap  = trade_context.get('pct_mfe_captured', 0.0)
    _pct_hold = trade_context.get('pct_hold_elapsed', 0.0)
    _aligned  = 1.0 if direction_aligned else 0.0

    # Hold pressure: profitable + early + workers agree
    _profit_factor = min(1.0, max(0.0, _profit) / (self._trade_expected_mfe_ticks + 1))
    _time_factor   = max(0.0, 1.0 - _pct_hold)
    _hold = _profit_factor * _time_factor * (0.5 + 0.5 * _aligned)

    # Exit pressure: losing + late + workers disagree
    _loss_factor = min(1.0, max(0.0, -_profit) / 20.0)
    _late_factor = max(0.0, _pct_hold - 1.0)
    _exit = _loss_factor + _late_factor * (1.0 - _aligned)

    net_pressure = _hold - _exit

    # Map continuous pressure to discrete trail signals
    if net_pressure > 0.3:
        widen = True; tighten = False; _pressure_reason = 'pressure_hold'
    elif net_pressure < -0.3:
        tighten = True; widen = False; _pressure_reason = 'pressure_exit'
    elif _pct_cap >= 0.60:
        tighten = True; widen = False; _pressure_reason = 'mfe_captured_60pct'
    elif _pct_hold > 1.5 and _pct_cap < 0.10:
        urgent = True; _pressure_reason = 'time_expired_no_capture'
```

Return dict adds `'net_pressure': net_pressure`.

### Change 3: CST Pressure-Controlled (wave_rider.py, ~15 lines)

**Where**: `check_structural_integrity()` — replace profit-conditional sigma

```python
def check_structural_integrity(self, current_state, profit_ticks=0.0,
                                net_pressure: float = 0.0) -> bool:
    ...existing scaling/distance code...

    # Pressure-controlled CST
    if net_pressure > 0.3:
        return True  # CST disabled — trade is profitable + early + aligned

    # Grace period: min hold before CST can fire
    _grace = max(10, int(self.position.tmpl_expected_hold_bars / 3))
    if self.position.bars_in_trade < _grace:
        return True  # still in grace period, only hard SL active

    # Pressure-modulated sigma
    if net_pressure > 0:
        _sigma = 4.0  # slightly wider — some hold pressure
    elif net_pressure > -0.3:
        _sigma = 3.0  # neutral — standard
    else:
        _sigma = 2.0  # exit pressure — tighter

    threshold = self.position.cst_basin_mean + _sigma * self.position.cst_basin_std
    ...rest unchanged...
```

### Orchestrator Wiring (~5 lines)

**Mid-trade loop** (line ~1048): Add one call:
```python
belief_network.set_trade_context(active_side, _profit_ticks)
```

**Exit** (every exit path): Add one call:
```python
belief_network.clear_trade_context()
```

**CST call** (line ~1057): Pass net_pressure:
```python
cst_broken = not self.wave_rider.check_structural_integrity(
    current_state, profit_ticks=_profit_ticks,
    net_pressure=_exit_sig.get('net_pressure', 0.0))
```

## Data Flow (Complete)

```
ENTRY
  orchestrator → belief_network.set_trade_context(side, 0.0)
  orchestrator → wave_rider.open_position(..., tmpl_expected_mfe_ticks, tmpl_expected_hold_bars)

EACH BAR (while position open)
  1. orchestrator computes _profit_ticks from entry_price vs current price
  2. belief_network.set_trade_context(side, _profit_ticks)
     → propagates to all 10 workers
  3. workers tick: _analyze() modulates conviction using trade_side + profit_ticks
  4. get_exit_signal(side, trade_context) → continuous pressure model
     → returns net_pressure + tighten/widen/urgent
  5. check_structural_integrity(state, profit_ticks, net_pressure)
     → grace period + pressure-controlled sigma

EXIT
  orchestrator → belief_network.clear_trade_context()
  orchestrator → belief_network.clear_active_trade_timescale()
```

## Expected Impact

**Hold time**: 5 bars → 15-30 bars (3-6x increase)
- Grace period alone blocks CST for first 10+ bars
- Profitable trades get CST immunity (net_pressure > 0.3)
- Workers boost conviction on confirmed winners → wider trails

**Capture rate**: 5% → 20-40% (4-8x increase)
- Trades survive long enough to reach 20-30% of MFE
- Trail stop handles profitable exits instead of CST

**Reversal detection**: Workers flipping + losing trade → net_pressure < -0.3 → tight CST
- At bar 15-20, losing trades show clear negative P&L
- Workers that flip direction get boosted conviction (change 1)
- Aggregated belief shifts → exit pressure builds → CST fires at 2σ

**Fallback**: When template has no MFE stats, net_pressure = 0, all new code is silent.
System behaves identically to pre-change. No regression risk.

## Verification

1. Syntax check: `python -c "import training.wave_rider; import training.timeframe_belief_network; import training.orchestrator"`
2. Quick: `--forward-pass --data DATA/ATLAS_1DAY`
   - Hold bars should increase (target: >10 avg, up from 5)
   - structural_break % should decrease (target: <60%, down from 92.7%)
3. Full: `--forward-pass` on ATLAS
   - Capture rate should increase (target: >10%, up from 1.7%)
   - Too-early bucket should shrink
   - Reversed detection: net_pressure < -0.3 at exit for losing trades
