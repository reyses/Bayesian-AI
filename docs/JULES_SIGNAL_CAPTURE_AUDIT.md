# Jules Task: Signal Capture Diagnostic & Fix

## Problem Statement

The system captures only ~2.6% of its theoretical ideal PnL. The oracle proves
real moves exist — we just fail to trade them or exit too early. This task is a
**diagnostic + targeted fix** across the detection funnel.

Last run stats (10 months, 2025-01 to 2025-10):
- Oracle ideal (parallel): $1,825,968
- Actual PnL: ~$48,192 (2.6% capture)
- 2,805 missed profitable signals worth $804,732
- 78,057 score losers worth $11.76M (sequential constraint — not fixable here)

## Reference Files (committed to repo)

Jules: READ THESE FILES FIRST before modifying any code.

- `docs/checkpoint_reference/SCHEMAS.md` — **All object schemas** (PatternTemplate,
  TradeOutcome, Position, BeliefState, BayesianBrain, ThreeBodyQuantumState, 16D feature
  vector, CSV column layouts). This is your primary reference for data structures.
- `docs/checkpoint_reference/depth_weights.json` — Current per-depth filtering config
- `docs/checkpoint_reference/pipeline_state.json` — Pipeline status (105 templates, 370 optimized)
- `docs/checkpoint_reference/run_snapshot.json` — **Latest run metrics snapshot**
- `docs/checkpoint_reference/discovery_levels.json` — Discovery depth levels
- `docs/checkpoint_reference/depth_analytics.txt` — Per-depth performance breakdown
- `docs/checkpoint_reference/oos_report.txt` — OOS forward pass report
- `docs/checkpoint_reference/oos_analytics.txt` — OOS analytics summary
- `docs/checkpoint_reference/trade_analytics.txt` — Trade quality analytics
- `docs/checkpoint_reference/template_tiers.pkl` — Tier map (small pkl, 1.7 KB)
- `docs/checkpoint_reference/sample_oracle_trade_log.csv` — Sample rows (5) showing exact columns
- `docs/checkpoint_reference/sample_fn_oracle_log.csv` — Sample FN rows (5)
- `docs/checkpoint_reference/sample_pid_oracle_log.csv` — Sample PID rows (5)
- `docs/checkpoint_reference/sample_signal_log.csv` — Sample signal log rows (3)
- `reports/phase4_report.txt` — Full Phase 4 report (latest forward pass)
- `reports/oos_report.txt` — Full OOS report

**Latest run_snapshot.json** (current state of the system):
```json
{
  "trades": 517,
  "win_rate": 25.7,
  "total_pnl": -1929.5,
  "ideal_pnl": 20728730.0,
  "capture_pct": -0.01,
  "fn_missed_pnl": 20663168.5,
  "gate0_skip": 101279,
  "gate1_skip": 0,
  "gate2_skip": 520,
  "gate3_skip": 391
}
```

NOTE: gate0_skip=101,279 is the dominant blocker — dwarfs all other gates combined.

## Detection Funnel (Current State)

```
96,719 candidates evaluated
  │
  ├─ GATE 0 (structural rules):     4,228 rejected (4.4%)
  │     └─ 1,309 were REAL moves (46.7% of all missed $)
  │
  ├─ GATE 1 (cluster distance):     1,190 rejected (1.2%)
  │     └─ 836 were REAL moves (29.8% of all missed $)
  │
  ├─ GATE 2 (brain P(Win)):           236 rejected (0.2%)
  ├─ GATE 3 (conviction):               0 rejected (0.0%)
  ├─ SCORE COMPETITION:            78,057 score losers (sequential constraint)
  └─ TRADED:                        4,837 entries (5.0%)
        ├─ 86 optimal captures (≥80% of MFE)
        ├─ 602 too-early exits (<20% of MFE, left $139 avg on table)
        └─ 1,445 reversals (correct entry, market flipped, -$23 avg)
```

## The 5 Root Causes (Ranked by $ Impact)

### 1. Gate 0 Rule 4 — Extreme Zone Over-Rejection ($354K missed, 44.1% of FN)

**File**: `training/orchestrator.py`, the gate-0 structural headroom section
(search for `STRUCTURAL_DRIVE` extreme zone logic)

**Current rule**: If `macro_z >= 3.0` → skip ALL `STRUCTURAL_DRIVE` continuations.
Rationale was "no headroom for continuation at extreme macro."

**Problem**: 1,309 of these were confirmed real moves by the oracle. The rule is
too blunt — it assumes macro extremes always mean reversal, but continuation
patterns at extreme z often produce the BIGGEST moves (momentum breakouts).

**Diagnostic task**:
1. Add a counter that logs every Gate 0 Rule 4 rejection with its oracle outcome:
   ```python
   # In the Gate 0 rule 4 block, when rejecting:
   gate0_r4_rejections.append({
       'timestamp': ts_raw,
       'z_score': z_score,
       'macro_z': macro_z,
       'pattern_type': pattern_type,
       'oracle_marker': oracle_marker,
       'oracle_mfe': oracle_mfe,
       'oracle_mae': oracle_mae,
       'hurst': hurst,
       'adx': adx,
       'dmi_diff': dmi_diff,
       'tunnel_prob': tunnel_prob,
   })
   ```
2. At end of forward pass, write `gate0_rule4_analysis.csv` with these records.
3. Add a report section "GATE 0 RULE 4 ANALYSIS" showing:
   - Total rejected, % that were real moves, $ left on table
   - Distribution of macro_z values (histogram bins: 3.0-3.5, 3.5-4.0, 4.0-5.0, 5.0+)
   - Win rate by macro_z bin IF we had traded them (oracle MFE > MAE)
   - Mean MFE/MAE ratio by bin
   - Hurst and ADX distributions of rejected signals

**Fix suggestion** (implement if diagnostic confirms):
- Replace hard cutoff `macro_z >= 3.0` with graduated filter:
  - macro_z 3.0-4.0: Allow if `hurst >= 0.6 AND adx >= 35` (strong trend confirmation)
  - macro_z 4.0-5.0: Allow if `hurst >= 0.65 AND adx >= 40 AND tunnel_prob >= 0.5`
  - macro_z > 5.0: Keep blocking (true nightmare field)
- Alternative: If oracle_marker is MEGA → always allow (but this uses lookahead,
  so only valid for diagnostic, not live)

### 2. Gate 1 — Cluster Distance / Novel Patterns ($240K missed, 29.8% of FN)

**File**: `training/orchestrator.py`, Gate 1 cluster distance section
**File**: `training/fractal_clustering.py`

**Current rule**: If nearest centroid distance > 4.5 → reject (no template match).
Worker bypass at conviction >= 0.65 is rarely triggered.

**Problem**: 836 rejected signals were real moves (90.9% of no-match signals!).
These are statistically rare extreme-z patterns the clustering never learned.

**Diagnostic task**:
1. Add logging for Gate 1 rejections:
   ```python
   gate1_rejections.append({
       'timestamp': ts_raw,
       'z_score': z_score,
       'nearest_dist': dist,
       'nearest_tid': nearest_template_id,
       'feature_vector': feature_vec.tolist(),
       'oracle_marker': oracle_marker,
       'oracle_mfe': oracle_mfe,
       'conviction': belief_conviction,
   })
   ```
2. Write `gate1_nomatch_analysis.csv` at end of forward pass.
3. Add report section "GATE 1 NO-MATCH ANALYSIS" showing:
   - Total rejected, % real moves, $ left on table
   - Distance distribution (4.5-5.0, 5.0-6.0, 6.0-8.0, 8.0+)
   - Oracle WR by distance bin
   - Which feature dimensions are most different from nearest centroid
     (compute per-dimension z-score of rejected vs centroid)
   - Conviction distribution of rejected signals (how many had conviction > 0.55?)

**Fix suggestion** (implement if diagnostic confirms):
- Raise distance threshold from 4.5 to 6.0 for MEGA oracle markers only (diagnostic)
- For live: raise threshold to 5.5 universally, but add a "novelty penalty"
  to the score competition (novel patterns score worse, so known patterns win ties)
- Consider adding a "catch-all" template for dist > 4.5 signals that uses
  belief network conviction + physics as the sole decision (no cluster stats)

### 3. Exit Quality — Too-Early Exits ($83K left on table, 602 trades)

**File**: `training/wave_rider.py` — `update_trail()` method
**File**: `training/orchestrator.py` — exit mechanics section

**Current behavior**:
- Trail stop: `mean_mae * 1.1` distance from HWM
- Trail activation: `p25_mae * 0.3` profit needed
- Dynamic tightening: If conviction < 0.5 → multiply trail by 0.5 (very tight)
- MAX_HOLD: Forces exit after N bars even if trade is profitable

**Problem**: 602 trades exited with < 20% capture, leaving $139 avg on table.
The trail is too eager to lock in small gains. MAX_HOLD cuts winners short.

**Diagnostic task**:
1. For every trade that exits "too early" (oracle says > 50% more MFE remained):
   ```python
   early_exits.append({
       'entry_time': entry_time,
       'exit_time': exit_time,
       'exit_reason': exit_reason,
       'actual_pnl': pnl,
       'oracle_mfe': oracle_mfe,
       'remaining_mfe': oracle_mfe - actual_capture_ticks,
       'bars_held': bars_held,
       'max_hold_limit': max_hold_bars,
       'exit_conviction': exit_conviction,
       'exit_wave_maturity': exit_wave_maturity,
       'trail_distance_at_exit': trail_distance,
       'hwm_at_exit': high_water_mark,
       'entry_depth': depth,
   })
   ```
2. Write `early_exit_analysis.csv`.
3. Add report section "EARLY EXIT ANALYSIS" showing:
   - Exit reason breakdown: trail_stop vs max_hold vs watchdog
   - For MAX_HOLD exits: how many were still profitable and trending?
   - For trail_stop exits: what was conviction at exit? Was trail tightened?
   - Average bars remaining (oracle MFE bar - actual exit bar)
   - Conviction at exit vs conviction at entry (did it actually drop?)

**Fix suggestion**:
- MAX_HOLD: Don't force exit if `conviction >= 0.55 AND trade is profitable`.
  Extend hold by 50% once. Only force after extended hold.
- Trail tightening: Only tighten if conviction < 0.45 (not 0.50) AND wave_maturity < 0.3.
  Currently 0.50 is too close to neutral.
- Add "let winners run" rule: If HWM > 2x initial SL distance AND conviction rising,
  widen trail to 1.5x instead of tightening.

### 4. Direction Errors — Wrong Side (1,827 trades, -$9.9K)

**File**: `training/orchestrator.py` — direction gating hierarchy

**Current hierarchy**:
1. Oracle marker (uses lookahead — not available live)
2. Per-cluster logistic regression P(LONG)
3. Template aggregate bias
4. DMI trend-following (last resort)

**Problem**: When oracle marker is unavailable (noise markers, or live trading),
the system falls through to logistic regression or DMI. DMI is unreliable at
entry — it's a lagging indicator. 15s worker direction agreement is only +0.14
edge on wins vs losses.

**Diagnostic task**:
1. For every wrong-direction trade (oracle says opposite direction was correct):
   ```python
   wrong_dir.append({
       'timestamp': ts_raw,
       'chosen_direction': direction,
       'oracle_direction': 'LONG' if oracle_marker > 0 else 'SHORT',
       'direction_source': direction_source,  # which priority level decided
       'logit_p_long': logit_p_long,
       'template_long_bias': template.long_bias,
       'template_short_bias': template.short_bias,
       'dmi_diff': dmi_diff,
       'z_score': z_score,
       'belief_conviction': conviction,
       'belief_direction': belief_direction,
       'worker_1s_direction': w1_dir,
       'worker_5m_direction': w5m_dir,
       'worker_15m_direction': w15m_dir,
   })
   ```
2. Write `wrong_direction_analysis.csv`.
3. Add report section "DIRECTION ERROR ANALYSIS" showing:
   - Which priority level made the decision (logit vs bias vs DMI vs velocity)
   - Error rate per priority level
   - Worker agreement patterns on wrong-direction trades
   - z_score sign vs oracle direction agreement rate
   - Could belief network direction have saved it? (compare belief_dir vs oracle_dir)

### 5. Reversal Trades — Correct Entry, Market Flipped (1,445 trades, -$33K)

**File**: `training/wave_rider.py` — position management
**File**: `training/timeframe_belief_network.py` — exit signal generation

**Problem**: 1,445 trades entered correctly (oracle confirmed direction) but the
market reversed before TP was reached. The system held through the reversal.
Belief flip exit triggered 0 times — the belief network is too "sticky."

**Diagnostic task**:
1. For every reversal trade (entered correct direction, exited at loss):
   ```python
   reversals.append({
       'entry_time': entry_time,
       'exit_time': exit_time,
       'hold_bars': bars_held,
       'pnl': pnl,
       'oracle_mfe': oracle_mfe,  # how far it went in our favor first
       'peak_unrealized_pnl': (hwm - entry_price) * point_value,
       'entry_conviction': entry_conviction,
       'exit_conviction': exit_conviction,
       'conviction_at_hwm': conviction_at_peak,  # need to track this
       'bars_after_hwm': exit_bar - hwm_bar,  # how long we held after peak
       'z_score_at_exit': z_at_exit,
       'z_score_at_hwm': z_at_hwm,
       'worker_5m_flipped': did_5m_flip_direction,
   })
   ```
2. Write `reversal_analysis.csv`.
3. Add report section "REVERSAL ANALYSIS" showing:
   - How far did the trade go in our favor before reversing? (peak unrealized PnL)
   - How many bars did we hold AFTER the peak? (holding too long?)
   - Did conviction drop before the reversal? (early warning signal?)
   - Did the 5m worker flip direction before exit? (unused exit signal?)
   - Distribution of hold-after-peak: if > 5 bars on average, trail is too wide

## Implementation Guidelines

### File Organization
- All diagnostic CSV files go in `output/diagnostics/` directory
- All new report sections go at the END of the existing report (don't reorganize)
- Use the existing `_log_info()` / `_log_warn()` patterns for console output

### Data Collection Pattern
Use lists to accumulate records during the forward pass, then write CSVs at the end:
```python
# At start of run_forward_pass():
gate0_r4_rejections = []
gate1_rejections = []
early_exits = []
wrong_directions = []
reversals = []

# ... during forward pass, append to lists ...

# At end, after report generation:
import os
diag_dir = os.path.join(self.output_dir, 'diagnostics')
os.makedirs(diag_dir, exist_ok=True)
if gate0_r4_rejections:
    pd.DataFrame(gate0_r4_rejections).to_csv(
        os.path.join(diag_dir, 'gate0_rule4_analysis.csv'), index=False)
# ... same for others ...
```

### Report Sections
Add after the existing "FALSE NEGATIVE ANALYSIS" section. Format:
```python
lines.append("")
lines.append("=" * 80)
lines.append("  GATE 0 RULE 4 DIAGNOSTIC")
lines.append("=" * 80)
# ... histogram bins, percentages, oracle WR by bin ...
```

### What NOT to Change
- Do NOT change any gate thresholds yet — this task is DIAGNOSTIC FIRST
- Do NOT change the score competition logic
- Do NOT change the clustering algorithm
- Do NOT add new dependencies
- Do NOT modify the 1s inner loop (just committed, needs testing first)

### What TO Change (After Diagnostics)
After the diagnostic CSVs are generated and we analyze results, THEN implement
the graduated Gate 0 Rule 4 fix (the biggest bang for buck):
- Replace hard `macro_z >= 3.0` cutoff with graduated Hurst+ADX thresholds
- Add the "catch-all conviction-only" path for Gate 1 dist > 4.5

## Expected Output

After running `--fresh` forward pass with these diagnostics:
1. `output/diagnostics/gate0_rule4_analysis.csv` — every Rule 4 rejection with oracle truth
2. `output/diagnostics/gate1_nomatch_analysis.csv` — every cluster distance rejection
3. `output/diagnostics/early_exit_analysis.csv` — trades that left money on table
4. `output/diagnostics/wrong_direction_analysis.csv` — trades that picked wrong side
5. `output/diagnostics/reversal_analysis.csv` — correct entry, market flipped
6. Updated report with 5 new diagnostic sections at the end

## Success Criteria

The diagnostic is successful if:
- We can see the exact macro_z distribution where Rule 4 kills real moves
- We can see which feature dimensions cause Gate 1 no-match
- We can quantify how much money early exits and reversals cost
- We have enough data to confidently set new graduated thresholds
- The forward pass runs without errors and produces all 5 CSV files

## Files to Modify

1. `training/orchestrator.py` — Add diagnostic logging in forward pass + report sections
   (This is the ONLY file that needs changes. All gates, exits, and reporting live here.)

## Priority

**HIGH** — This diagnostic directly informs the next round of threshold tuning.
Without it, we're guessing at where the money is leaking.
