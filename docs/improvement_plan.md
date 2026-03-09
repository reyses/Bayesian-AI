# Exit Engine Improvements — Implementation Plan
**Created:** 2026-03-08
**Order:** Fix 3 → Fix 4 → Fix 1 → Fix 2
**Baseline OOS:** 1,924 trades, 78.3% WR, $21,683 PnL

---

## Fix 3: Tiered Giveback (no retraining)

**Problem:** Flat 55% giveback threshold treats a 100-tick peak the same as a 16-tick peak. Big winners get given back; small winners get cut too early.

**Change:** `core/exit_engine.py` only
- Add `_get_giveback_threshold(peak_ticks)`:
  - peak >= 30 ticks → 40% (protect big winners aggressively)
  - peak 16-30 ticks → self.giveback_pct (self-tuned, ~55%)
  - peak < 16 ticks → disabled (1.01, move hasn't proven itself)
- Modify `_check_peak_giveback()` to use tiered threshold
- Self-tuning still adjusts the mid-tier (16-30) value — 30+ tier hardcoded at 40%

**Validate:** `python training/trainer.py --forward-pass`
**Expected:** +$1,000 OOS PnL, too-late count drops, big winners protected
**Status:** [ ] Done

---

## Fix 4: 30m Worker Flip Tighten (no retraining)

**Problem:** 30m workers flip direction on 4% of losers vs 2% of winners. When a 30m worker flips against the trade AND the trade has captured meaningful profit, the structural trend has changed.

**Changes:**
1. `core/timeframe_belief_network.py` → `get_exit_signal()`: detect 30m worker flip against trade side, set `slow_flip_tighten = True`
2. `core/exit_engine.py` → `_check_peak_giveback()`: accept `exit_signal` param, reduce giveback threshold by 15pp when slow_flip active
3. Sticky behavior: once `_slow_flip_tighten` fires on a trade, stays True for remainder (add field to PositionState)

**Validate:** `python training/trainer.py --forward-pass`
**Expected:** +$125, validates slow-worker signal hypothesis, 10-20 fires in OOS
**Status:** [ ] Done

---

## Fix 1: Hurst Validation (analysis first, then maybe gate change)

**Problem:** Pattern Quality hurst gate blocks 28.4% of all missed profitable signals — the single largest FN source. Is Hurst reliable at 15s resolution with 100-bar window (25 min)?

**Changes:**
1. Create `scripts/hurst_validation.py`: confusion matrix at window sizes 50/100/200/400 vs ADX-confirmed trending/ranging periods
2. Based on results, one of:
   - **A (unreliable):** Convert hurst gate from hard block to score penalty
   - **B (wrong threshold):** Adjust `hurst_min` in `gate_thresholds.json`
   - **C (wrong window):** Increase `HURST_WINDOW` in `physics_utils.py`

**Validate:** Rerun `--forward-pass`, FN at hurst gate should drop, WR stays >75%
**Expected:** Up to +$25K if gate relaxed (22,660 FN signals currently blocked)
**Status:** [ ] Done

---

## Fix 2: Per-Template Exit Timescale (requires --fresh)

**Problem:** Envelope halflife is global (24.2 bars). A 15m macro trade needs HL=192, a 1s micro scalp needs HL=4. One size fits none.

**Changes:**
1. `core/fractal_clustering.py` → `_aggregate_oracle_intelligence()`: compute `avg_mfe_bar` and `p75_mfe_bar` per template
2. `training/trainer.py` → `register_template_logic()`: store in pattern_library
3. `training/trainer.py` → forward pass: pass real values to `set_active_trade_timescale()`
4. `core/exit_engine.py` → `_check_envelope()`: use `pos.max_hold_bars / 5.0` as base halflife instead of global

**Validate:** `python training/trainer.py --fresh --forward-pass`
**Expected:** Primary too-late/too-early fix. Per-depth PnL should improve uniformly.
**Status:** [ ] Done

---

## Summary

| Fix | Files | Retrain? | Risk | Impact |
|-----|-------|----------|------|--------|
| 3 Tiered Giveback | exit_engine.py | No | Very low | +$1K |
| 4 30m Worker Flip | TBN + exit_engine | No | Low | +$125, signal validation |
| 1 Hurst Validation | NEW script + maybe execution_engine | No | Medium | Up to +$25K |
| 2 Template Timescale | clustering + trainer + exit_engine | Yes | Medium | Primary exit fix |
