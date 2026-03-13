# DMI/ADX Gate & Direction Improvements — Task Spec for VS Code Claude

## Context
BayesianBridge trades MNQ futures on 15s bars. DMI/ADX are computed per-bar in `core/physics_utils.py` (14-period Wilder smoothing) and stored on every `MarketState`. This spec proposes improvements based on industry best practices and quantitative research.

---

## CURRENT STATE (what exists)

### Gate 0: ADX threshold (entry quality)
- Only blocks `MOMENTUM_BREAK` patterns when `adx < 25`
- Binary pass/fail — no regime awareness
- Location: `core/execution_engine.py` lines 649-656

### Direction Cascade: DMI used in 2 of 6 levels
- **P0.5 (signed_mfe regression)**: `predicted_mfe = depth * c0 + dmi_diff * c1 + intercept`
- **P4 (fallback)**: raw `dmi_diff > 0 = long` when all learned sources fail
- Location: `core/execution_engine.py` lines 1031-1044, 1083-1088

### Feature Vector: 4 of 16 dims
- `[7] adx/100`, `[9] dmi_diff/100`, `[11] parent_dmi_diff`, `[13] tf_alignment`
- In compressed/live mode, dims [11] and [13] are always 0.0

---

## PROPOSED IMPROVEMENTS

### 1. ADX Regime Classification Gate (replaces binary ADX check)

**Rationale:** Industry consensus is that ADX classifies market into distinct regimes that require different strategies. A single ADX >= 25 check is insufficient. ADX slope (rising vs falling) matters as much as absolute level — a rising ADX at 22 is more actionable than a falling ADX at 28.

**Implementation — new function in `core/execution_engine.py`:**

```python
def _classify_adx_regime(self, state) -> str:
    """Classify market regime from ADX + DMI.
    
    Returns one of: 'strong_trend', 'developing', 'exhausting', 'range', 'chop'
    """
    adx = getattr(state, 'adx_strength', 0.0)
    adx_prev = getattr(state, 'adx_prev', adx)  # requires adding to MarketState
    adx_slope = adx - adx_prev  # positive = strengthening
    
    di_plus = getattr(state, 'dmi_plus', 0.0)
    di_minus = getattr(state, 'dmi_minus', 0.0)
    di_spread = abs(di_plus - di_minus)
    
    if adx >= 40:
        if adx_slope >= 0:
            return 'strong_trend'    # strong and still building
        else:
            return 'exhausting'      # strong but fading — tighten exits
    elif adx >= 25:
        if adx_slope > 0:
            return 'developing'      # trend gaining strength — enter
        else:
            return 'exhausting'      # trend losing steam — caution
    elif adx >= 20:
        if adx_slope > 1.0:
            return 'developing'      # emerging from range — early entry
        else:
            return 'range'           # borderline — avoid trend-following
    else:
        return 'chop'                # no trend — skip or mean-revert only
```

**Gate changes:**
- `strong_trend` + `developing`: allow all pattern types, widen TP
- `exhausting`: allow entries but tighten giveback threshold and reduce max_hold_bars
- `range`: block `MOMENTUM_BREAK`, allow `BAND_REVERSAL` (currently reversed!)
- `chop`: block all entries OR only allow mean-reversion patterns with tight SL

**Requires:** Add `adx_prev` field to MarketState (previous bar's ADX value). Compute in `statistical_field_engine.py` as `adx_arr[i-1]`.

### 2. DI Spread as Entry Confidence Multiplier

**Rationale:** The absolute spread between DI+ and DI- (not just sign) indicates conviction. A DI+ of 35 vs DI- of 10 (spread=25) is far more directional than DI+ of 22 vs DI- of 20 (spread=2). Currently the system only uses the sign of `dmi_diff`, ignoring magnitude except in the regression.

**Implementation — modify direction cascade:**

In `_determine_direction()`, after computing `dmi_diff`:

```python
# DI spread confidence scaling
di_spread = abs(dmi_diff)
if di_spread < 5:
    # Weak spread: DMI signal is noise, skip to next cascade level
    pass  # fall through to velocity
elif di_spread < 15:
    # Moderate: use but with low confidence (p_long closer to 0.5)
    p_conf = 0.52 + (di_spread - 5) * 0.003  # 0.52 to 0.55
elif di_spread >= 15:
    # Strong: high confidence
    p_conf = 0.55 + min(0.10, (di_spread - 15) * 0.005)  # up to 0.65
```

This replaces the current flat `p_long = 0.55` at Priority 4.

### 3. ADX Slope for Exit Modulation

**Rationale:** A declining ADX slope during a trade signals the trend is losing momentum — even if ADX is still high. This should tighten exits. A rising ADX slope during a trade signals the trend is strengthening — widen exits to let winners run.

**Implementation — modify `core/exits/envelope.py` and `core/exits/giveback.py`:**

In `EnvelopeDecay.evaluate()`, add ADX slope modulation:
```python
# ADX slope modulation (requires adx_slope on state or passed via exit_signal)
adx_slope = exit_signal.get('adx_slope', 0.0) if exit_signal else 0.0
if adx_slope > 0:
    # Trend strengthening — slow down envelope decay
    hl_mult *= 1.0 + min(0.5, adx_slope * 0.05)  # up to 50% slower decay
elif adx_slope < -1.0:
    # Trend weakening — speed up decay
    hl_mult *= max(0.5, 1.0 + adx_slope * 0.1)   # up to 50% faster
```

In `PeakGiveback.evaluate()`, tighten threshold when trend is exhausting:
```python
adx_slope = exit_signal.get('adx_slope', 0.0) if exit_signal else 0.0
if adx_slope < -2.0:
    # Trend decelerating rapidly — tighten by 10pp
    threshold = max(0.25, threshold - 0.10)
```

**Requires:** Pass `adx_slope` through exit_signal dict from TBN's `get_exit_signal()` or from the caller.

### 4. DI Crossover Detection for Exit (Belief Flip Enhancement)

**Rationale:** When DI+ crosses below DI- during a long trade (or vice versa), it's a strong reversal signal. Currently the system's belief flip exit is based on TBN conviction only. A DI crossover against the trade should trigger tighter exit behavior.

**Implementation — add to `core/exits/belief_flip.py`:**

```python
def evaluate(self, pos, bar_close, tick_size, exit_signal=None):
    # Existing belief flip logic...
    
    # DI crossover check (new)
    di_plus = exit_signal.get('di_plus', 0.0) if exit_signal else 0.0
    di_minus = exit_signal.get('di_minus', 0.0) if exit_signal else 0.0
    di_plus_prev = exit_signal.get('di_plus_prev', di_plus) if exit_signal else di_plus
    di_minus_prev = exit_signal.get('di_minus_prev', di_minus) if exit_signal else di_minus
    
    # Detect crossover against position
    if pos.side == 'long':
        crossed_against = (di_plus_prev > di_minus_prev) and (di_minus >= di_plus)
    else:
        crossed_against = (di_minus_prev > di_plus_prev) and (di_plus >= di_minus)
    
    if crossed_against and pos.bars_held >= 3:
        # DI crossed against us — urgent exit at market
        return ExitResult(
            action=ExitAction.BELIEF_FLIP,
            exit_price=bar_close,
            reason=f"DI crossover against {pos.side}",
            pnl_ticks=...,
            bars_held=pos.bars_held,
        )
```

**Requires:** Pass `di_plus`, `di_minus`, `di_plus_prev`, `di_minus_prev` through exit_signal dict.

### 5. Multi-Timeframe ADX Alignment (TBN Integration)

**Rationale:** Multi-timeframe DMI analysis is widely regarded as one of the strongest uses of the indicator. A 15s signal in the direction of the 5m and 1h DMI trend has much higher probability than one against it. The TBN already has multi-TF workers — this just needs to surface DMI alignment.

**Implementation — add to `core/timeframe_belief_network.py`:**

```python
def get_dmi_alignment(self) -> dict:
    """Check if DMI direction agrees across timeframes.
    
    Returns:
        alignment_score: -1 to +1 (negative = bearish consensus, positive = bullish)
        aligned_tfs: count of timeframes where dmi_diff agrees with anchor
        total_tfs: total timeframes with valid DMI
    """
    anchor_dmi = 0.0
    scores = []
    for tf, worker in self.workers.items():
        b = worker.current_belief
        if b is None:
            continue
        dmi_p = getattr(b, 'dmi_plus', 0.0)
        dmi_m = getattr(b, 'dmi_minus', 0.0)
        dmi_diff = dmi_p - dmi_m
        if tf == '15s':
            anchor_dmi = dmi_diff
        if dmi_diff != 0:
            scores.append(1.0 if dmi_diff > 0 else -1.0)
    
    if not scores:
        return {'alignment_score': 0.0, 'aligned_tfs': 0, 'total_tfs': 0}
    
    avg = sum(scores) / len(scores)
    aligned = sum(1 for s in scores if (s > 0) == (anchor_dmi > 0))
    return {
        'alignment_score': avg,
        'aligned_tfs': aligned,
        'total_tfs': len(scores),
    }
```

**Gate integration:** In `_check_entry`, after direction is determined:
```python
dmi_align = self.belief_network.get_dmi_alignment()
if dmi_align['total_tfs'] >= 3:
    aligned_pct = dmi_align['aligned_tfs'] / dmi_align['total_tfs']
    if aligned_pct < 0.4:
        # Majority of timeframes disagree with trade direction
        # Either reject or reduce confidence
        gate_label = 'gate_dmi_misalign'
```

**Requires:** Store `dmi_plus` and `dmi_minus` on worker belief state (may need to add these fields to whatever object `worker.current_belief` is).

### 6. Regime-Adaptive Pattern Type Gating

**Rationale:** Currently Gate 0 blocks MOMENTUM_BREAK in low-ADX regimes but allows BAND_REVERSAL everywhere. The logic should be inverted for ranging markets: BAND_REVERSAL thrives in ranges (ADX < 20), MOMENTUM_BREAK thrives in trends (ADX > 25).

**Implementation — replace current Gate 0 logic:**

```python
regime = self._classify_adx_regime(state)
pattern = micro_pattern

# Regime-pattern compatibility matrix
ALLOWED = {
    'strong_trend':  {'MOMENTUM_BREAK', 'TREND_CONTINUATION'},
    'developing':    {'MOMENTUM_BREAK', 'TREND_CONTINUATION', 'BAND_REVERSAL'},
    'exhausting':    {'BAND_REVERSAL'},  # only mean-reversion when trend fading
    'range':         {'BAND_REVERSAL'},
    'chop':          set(),              # skip everything
}

if pattern not in ALLOWED.get(regime, set()):
    should_skip = True
    skip_label = f'gate0_regime_{regime}'
```

---

## DATA REQUIREMENTS

New fields needed on `MarketState`:
- `adx_prev` (float): previous bar's ADX value — for slope computation
- `di_plus_prev` (float): previous bar's DI+ — for crossover detection  
- `di_minus_prev` (float): previous bar's DI- — for crossover detection

These can be computed cheaply in `statistical_field_engine.py` during `batch_compute_states()` since the full arrays are already available.

## IMPLEMENTATION ORDER

1. **Add `adx_prev`, `di_plus_prev`, `di_minus_prev` to MarketState** — trivial, 0 risk
2. **ADX regime classification** (#1) — replaces Gate 0 logic
3. **Regime-pattern gating** (#6) — depends on #1
4. **DI spread confidence** (#2) — improves direction P4
5. **ADX slope exit modulation** (#3) — improves exit quality
6. **DI crossover exit** (#4) — new exit trigger
7. **Multi-TF DMI alignment** (#5) — depends on TBN worker data availability

## TESTING

Run the full pipeline: IS → OOS2 → OOS3 with the same data. Compare:
- Trade count (should decrease in `chop`/`range` regimes)
- Win rate by regime bucket
- Avg trade PnL by regime  
- Exit quality: % of correct-direction trades exiting via envelope/giveback vs stop_loss

The key metric is **fewer trades in unfavorable regimes** with **maintained or improved PnL in favorable regimes**.
