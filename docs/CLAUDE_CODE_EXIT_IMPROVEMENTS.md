# BayesianBridge Exit Engine Improvements — Adversarial Review Spec
**Source:** Structured adversarial review of OOS performance (March 2026)  
**Target:** Claude Code / Jules (VS Code)  
**Priority:** Execute before Sunday live stress test  
**Scope:** 4 independent improvements, implement in order  

---

## CONTEXT (Read First)

OOS results (Jan–Mar 2026): 1,924 trades, 78.3% WR, $21,683 PnL.

**Core problem:** The exit engine produces two failure modes simultaneously:
- **Too late (526 trades, $8 avg):** Reached peak MFE then gave back most of profit
- **Too early (250 trades, $24 avg):** Exited before move materialized

The self-tuning envelope (halflife 20→24.2, giveback 70%→55%) is converging but cannot solve this because it applies ONE set of parameters to ALL trades regardless of their fractal timescale or template characteristics.

These 4 fixes address independent root causes. Each can be implemented and validated separately.

---

## FIX 1: Hurst Exponent Validation at 15s Resolution

### Problem
Gate 0 Rule 5a (Hurst < 0.5) blocks **28.4% of all missed profitable signals** — the single largest FN gate. The question: is the Hurst calculation reliable at 15-second resolution?

Current implementation uses R/S method with `HURST_WINDOW = 100` bars. At 15s resolution, that's **25 minutes of data**. This may be too short for a meaningful persistence estimate. If the Hurst indicator is noisy at this timescale, the 0.5 threshold is rejecting valid signals based on unreliable measurements.

### File Locations
- `core/cuda_physics.py` — `compute_hurst_kernel()` (CUDA implementation)
- `core/quantum_field_engine.py` — `_compute_hurst_numpy()` (CPU fallback)
- `core/physics_utils.py` — `HURST_WINDOW = 100`, `HURST_MIN_WINDOW = 30`
- `core/execution_engine.py` — `self.hurst_min = 0.5` (gate threshold, loaded from `gate_thresholds.json`)

### Implementation

#### Step 1: Hurst Reliability Analysis Script
Create `scripts/hurst_validation.py`:

```
PURPOSE: Validate Hurst exponent accuracy at 15s resolution by comparing 
against known trending/ranging periods.

METHODOLOGY:
1. Load 15s ATLAS data for 3+ months
2. Compute Hurst at current settings (window=100, 15s bars = 25 min)
3. Identify KNOWN trending periods: 
   - ADX > 30 for 20+ consecutive bars = confirmed trend
   - These periods SHOULD produce Hurst > 0.5 (persistent)
4. Identify KNOWN ranging periods:
   - ADX < 20 for 20+ consecutive bars = confirmed range  
   - These periods SHOULD produce Hurst < 0.5 (anti-persistent)
5. Compute confusion matrix:
   - True Positive: trending AND Hurst > 0.5
   - False Negative: trending BUT Hurst < 0.5  ← THIS IS THE PROBLEM
   - True Negative: ranging AND Hurst < 0.5
   - False Positive: ranging BUT Hurst > 0.5
6. Repeat with window sizes: 50, 100, 200, 400 bars
7. Report accuracy per window size

OUTPUT: Table showing Hurst accuracy vs window size at 15s resolution
```

#### Step 2: Conditional Gate Relaxation
Based on validation results, implement ONE of:

**Option A (Hurst is unreliable at 15s):**
- Remove Hurst gate entirely from Gate 0 Rule 5a
- Keep Hurst as a SOFT scoring signal (adjust score, don't block)
- In `execution_engine.py` `_gate_check()`, change the hurst block to a score penalty:
  ```python
  # Instead of: should_skip = True; skip_label = 'gate0_hurst'
  # Do: score_penalty += 0.5  (makes trade less likely to win score competition)
  ```

**Option B (Hurst is reliable but threshold is wrong):**
- Adjust `hurst_min` in `gate_thresholds.json` based on the FN-optimal threshold
- The validation script should output the threshold that maximizes:
  `(correctly_passed_real_moves) - 0.5 * (incorrectly_passed_noise)`

**Option C (Hurst is reliable but needs larger window):**
- Increase `HURST_WINDOW` to the validated optimal size
- Update both CUDA kernel and CPU fallback
- Rerun OOS to measure FN reduction

### Validation
- Rerun OOS forward pass with fix applied
- Compare: FN count at Gate 0 Hurst should drop significantly
- WR should stay above 75% (don't pass garbage)
- Net PnL should increase (the blocked signals were profitable)

### Expected Impact
- 22,660 FN signals currently blocked by Hurst gate
- Even 10% recovery at $11/trade average = ~$25K additional PnL
- Risk: some blocked signals were correctly blocked → WR may dip 1-2%

---

## FIX 2: Per-Template Exit Timescale Calibration

### Problem
The envelope halflife is a global constant (24.2 bars) applied to every trade. But a depth-3 trade (15m macro reversal) plays out over 60+ bars, while a depth-12 trade (1s micro scalp) plays out in 3-8 bars. Using one halflife for both is fundamentally wrong.

### File Locations
- `core/fractal_clustering.py` — `_aggregate_oracle_intelligence()` — where template stats are computed
- `core/exit_engine.py` — `_check_envelope()` — where halflife is used
- `core/execution_engine.py` — `position_opened()` — where template lib_entry is passed to exit engine
- `core/timeframe_belief_network.py` — `set_active_trade_timescale()` — already exists but receives zeros
- `training/trainer.py` — forward pass trade entry block — where `set_active_trade_timescale()` is called

### Implementation

#### Step 1: Compute time-to-peak-MFE per template during Phase 2.5

In `core/fractal_clustering.py`, inside `_aggregate_oracle_intelligence()`, after the existing MFE/MAE percentile computation, add:

```python
# After: template.p75_mfe_ticks = float(np.percentile(mfe_ticks, 75))
# Add time-to-MFE computation:

# Collect bar indices where peak MFE occurred for each member
mfe_bar_indices = []
for p in patterns:
    meta = getattr(p, 'oracle_meta', {})
    if 'mfe' in meta and meta['mfe'] > 0:
        # Estimate peak bar from window_data if available
        wd = getattr(p, 'window_data', None)
        if wd is not None and len(wd) > 1:
            entry_price = p.price
            if 'high' in wd.columns:
                peak_idx = int(wd['high'].argmax())
            else:
                peak_idx = int((wd['close'] if 'close' in wd.columns else wd['price']).argmax())
            mfe_bar_indices.append(peak_idx)

if mfe_bar_indices:
    template.avg_mfe_bar = float(np.mean(mfe_bar_indices))
    template.p75_mfe_bar = float(np.percentile(mfe_bar_indices, 75))
else:
    template.avg_mfe_bar = 0.0
    template.p75_mfe_bar = 0.0
```

Add `avg_mfe_bar` and `p75_mfe_bar` fields to the `PatternTemplate` dataclass.

#### Step 2: Store in pattern_library

In `training/trainer.py` `register_template_logic()`, add:
```python
'avg_mfe_bar': getattr(template, 'avg_mfe_bar', 0.0),
'p75_mfe_bar': getattr(template, 'p75_mfe_bar', 0.0),
```

#### Step 3: Feed to belief network at trade entry

In `training/trainer.py` forward pass, after `position_opened()`, find the existing call to `belief_network.set_active_trade_timescale()` (it currently passes 0.0, 0.0). Replace with:
```python
belief_network.set_active_trade_timescale(
    avg_mfe_bar=lib_entry.get('avg_mfe_bar', 0.0),
    p75_mfe_bar=lib_entry.get('p75_mfe_bar', 0.0),
)
```

#### Step 4: Modulate envelope halflife from template timescale

In `core/exit_engine.py` `_check_envelope()`, the `base_hl` is currently `self.envelope_half_life_bars` (global 24.2). Replace with:

```python
# Use template-specific timescale if available
_template_hl = pos.max_hold_bars / 5.0  # 1/5 of max hold = natural halflife
if _template_hl > 3.0:
    base_hl = _template_hl
else:
    base_hl = self.envelope_half_life_bars  # fallback to global
```

This anchors the halflife to the template's natural timescale: a 15m pattern with max_hold=960 bars gets halflife=192; a 1s pattern with max_hold=20 gets halflife=4.

### Validation
- Rerun OOS forward pass
- Too-late count should drop (macro trades held appropriately longer)
- Too-early count should drop (micro trades exit faster)
- Per-depth PnL breakdown should improve uniformly
- Global PnL should increase

### Expected Impact
- Primary fix for the too-late/too-early split
- Depth 3-5 trades (currently $8-18 avg) should improve significantly
- Depth 9-12 trades (currently $5-9 avg) should also improve

---

## FIX 3: Tiered Giveback Threshold

### Problem
`peak_giveback` triggers at a flat 55% regardless of how high the peak was. A trade that reached 100 ticks MFE gets the same tolerance as one that reached 16 ticks. Big winners should be protected more aggressively; small winners should be given room to develop.

### File Location
- `core/exit_engine.py` — `_check_peak_giveback()`

### Current Code
```python
def _check_peak_giveback(self, pos, bar_close):
    # ... compute peak_ticks, current_ticks ...
    if peak_ticks < self.giveback_min_mfe_ticks:  # 16 ticks
        return None
    gave_back = peak_ticks - current_ticks
    if peak_ticks > 0 and gave_back / peak_ticks >= self.giveback_pct:  # 55%
        return ExitResult(...)
```

### New Code
Replace the flat `self.giveback_pct` with a tiered function:

```python
def _get_giveback_threshold(self, peak_ticks: float) -> float:
    """Tiered giveback: protect big winners aggressively, 
    give small winners room to develop.
    
    Peak MFE (ticks)  →  Giveback trigger
    ────────────────     ────────────────
    30+               →  40% (aggressive protection)
    16-30             →  55% (current default)  
    <16               →  no giveback (move hasn't proven itself)
    """
    if peak_ticks >= 30:
        return 0.40
    elif peak_ticks >= 16:
        return self.giveback_pct  # self-tuned value (currently 0.55)
    else:
        return 1.01  # effectively disabled (>100% = never triggers)
```

Then in `_check_peak_giveback()`:
```python
    threshold = self._get_giveback_threshold(peak_ticks)
    if peak_ticks > 0 and gave_back / peak_ticks >= threshold:
        return ExitResult(
            action=ExitAction.PEAK_GIVEBACK,
            reason=f"Peak giveback: peak={peak_ticks:.1f}t now={current_ticks:.1f}t "
                   f"gave_back={gave_back/peak_ticks:.0%} (tier threshold={threshold:.0%})",
            ...
        )
```

### Self-Tuning Interaction
The existing `record_trade_outcome()` self-tuning adjusts `self.giveback_pct`. Under the tiered system, this value only applies to the 16-30 tick tier. The 30+ tier is hardcoded at 40% (protect winners). This is intentional — big winners should never be given back regardless of what the self-tuner learns from small trades.

### Validation
- Rerun OOS forward pass
- The 526 too-late trades should split: big-peak trades exit earlier (more profit captured), small-peak trades hold longer (fewer premature exits)
- The too-late sub-bands (gave back 80-100%) should shrink significantly
- Total PnL from correct-direction trades should increase

### Expected Impact
- The 60 trades that gave back 90-100% of peak are the clearest target
- These currently average $2/trade; with 40% giveback on 30+ tick peaks, they should average $15-20/trade
- Conservative estimate: +$1,000 incremental PnL from this fix alone

---

## FIX 4: 30-Minute Worker Flip as Exit Tighten Signal

### Problem
OOS data shows 30m workers flip direction on 4% of losing trades vs 2% of winning trades. This is the strongest slow-worker signal that distinguishes winners from losers. When a 30m worker flips against the trade AND the trade has already captured meaningful profit, the structural trend has changed — take what you have.

### File Locations
- `core/timeframe_belief_network.py` — `get_exit_signal()` — where tighten/widen/urgent decisions are made
- `core/exit_engine.py` — `_check_peak_giveback()` — where the tightened threshold would apply

### Implementation

In `core/timeframe_belief_network.py` `get_exit_signal()`, add a new signal source after the existing band-aware exit adjustments:

```python
# ── 30m worker flip detection ──────────────────────────────────
# When the 30m worker flips direction AGAINST the active trade AND
# the trade has already reached 50%+ of template p75_mfe, tighten
# the giveback threshold. This captures structural trend changes
# that fast TF workers are too noisy to reliably detect.
_slow_flip_tighten = False
_w30m = self.workers.get(1800)  # 30m = 1800 seconds
if _w30m is not None and _w30m.current_belief is not None:
    _w30m_long = _w30m.current_belief.dir_prob > 0.5
    _trade_long = (side == 'long')
    _w30m_against = (_w30m_long != _trade_long)
    
    if _w30m_against:
        # Check if trade has already captured meaningful profit
        # (use trade_bars_held vs avg_mfe_bar as proxy)
        if (self._trade_avg_mfe_bar > 0 and 
            self._trade_bars_held >= self._trade_avg_mfe_bar * 0.5):
            _slow_flip_tighten = True

tighten = tighten or _slow_flip_tighten
```

Then update the reason string:
```python
reason = ('slow_flip_tighten' if _slow_flip_tighten else
          'band_broken'    if _band_urgent   else
          # ... rest of existing cascade
```

**How the tighten signal reaches the exit engine:**

The `exit_signal['tighten_trail']` flag is already consumed by `_check_envelope()` in exit_engine.py. But envelope decay doesn't directly use tighten_trail. The connection point is the giveback threshold. When `tighten_trail` is True AND the reason is `slow_flip_tighten`, drop the giveback threshold by 15 percentage points:

In `core/exit_engine.py` `_check_peak_giveback()`, add at the top:

```python
# Check if exit_signal requested tighten (slow worker flip)
# This is passed through PositionState or a separate mechanism
# Simplest: check if pos has a tighten flag set by the main loop
_tighten_active = getattr(pos, '_slow_flip_tighten', False)
if _tighten_active:
    # Drop giveback threshold by 15pp (e.g., 55% → 40%)
    threshold = max(0.30, self._get_giveback_threshold(peak_ticks) - 0.15)
else:
    threshold = self._get_giveback_threshold(peak_ticks)
```

The main loop (trainer.py) needs to set `pos._slow_flip_tighten` when the exit signal contains `slow_flip_tighten`. Add after the exit_signal computation:
```python
if _exit_sig.get('reason') == 'slow_flip_tighten' and _exec_engine.pos_state is not None:
    _exec_engine.pos_state._slow_flip_tighten = True
```

**Important:** Once set, `_slow_flip_tighten` stays True for the remainder of the trade. The structural trend has changed; don't relax back even if the 30m worker flips again.

### Validation
- Rerun OOS forward pass
- Count trades where slow_flip_tighten fired
- Compare avg PnL of those trades vs baseline (should be higher — exiting before full giveback)
- Overall too-late count should decrease
- Watch for false positives: trades where 30m flip was temporary and the trade would have recovered

### Expected Impact
- 30m flip rate on losers: 4% of ~420 losses = ~17 trades affected
- If even half of these exit earlier with $15 more profit each = ~$125 incremental
- Small dollar impact but important *signal validation* — confirms slow-worker flips have predictive value for future use with other slow workers (1h flip = even stronger signal, but too rare at 2% to measure reliably in current dataset)

---

## IMPLEMENTATION ORDER

```
Fix 1 (Hurst)     →  analysis script first, gate change second
Fix 2 (Timescale) →  requires Phase 2.5 rerun (adds template fields)  
Fix 3 (Giveback)  →  pure exit_engine change, no retraining needed
Fix 4 (30m Flip)  →  belief network + exit engine, no retraining needed
```

**Recommended sequence:**
1. Fix 3 first (smallest change, immediate validation, no dependencies)
2. Fix 4 second (small change, validates slow-worker signal hypothesis)
3. Fix 1 third (analysis script informs gate threshold — may not need code change)
4. Fix 2 last (requires Phase 2.5 rerun to populate avg_mfe_bar fields)

**After all 4 fixes:** Rerun full IS + OOS chain and compare against baseline:
- Baseline OOS: 1,924 trades, 78.3% WR, $21,683 PnL
- Target: >$25K PnL with WR >76% (accept slight WR drop for better capture)

---

## VALIDATION CHECKLIST

After implementing all fixes, confirm:

- [ ] Hurst validation script produced a clear accuracy vs window-size table
- [ ] Per-template avg_mfe_bar and p75_mfe_bar are populated in pattern_library.pkl
- [ ] Envelope halflife varies by trade (check log: macro trades should show HL>50, micro trades HL<10)
- [ ] Giveback threshold varies by peak MFE (check exit reasons in report: should show different tier thresholds)
- [ ] 30m worker flip tighten fires on at least 10+ trades in OOS
- [ ] Too-late count dropped from 526 baseline
- [ ] Too-early count dropped from 250 baseline
- [ ] Net PnL increased vs $21,683 baseline
- [ ] No new exit types dominating with negative avg PnL
