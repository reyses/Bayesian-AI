# Jules Task: Signal Capture Diagnostic — 5 CSV Outputs

> **Relationship to JULES_PERFORMANCE_TARGETS.md**: This is the DIAGNOSTIC step.
> PERFORMANCE_TARGETS defines the fixes (Phase A/B/C). This task generates the
> data needed to calibrate those fixes. Do this FIRST.

## Problem Statement

The system captures near-zero of its theoretical ideal PnL. The oracle proves
real moves exist — we just fail to trade them or exit too early.

**Current baseline** (IS, 10 months, 3,754 trades):
```
Win Rate:           37.5%
Total PnL:          $5,834
Avg PnL/trade:      $1.55
Direction correct:  45.2%
Direction wrong:    46.1%
Capture rate:       0.1% mean

Exit breakdown:
  profit_target:      1,135 trades  100.0% WR   $18.58 avg   capture: 37.8%
  trail_stop:         2,221 trades    0.9% WR   -$9.02 avg   capture: -21.7%
  belief_flip:          283 trades   58.0% WR   $10.08 avg   capture:  9.5%
  structural_break:     115 trades   78.3% WR   $16.70 avg   capture: 23.8%

CRITICAL FINDING: 99.9% of trail stops never reach activation threshold.
Trail stop is acting as a raw stop-loss, killing trades in 2-3 bars.
```

## Reference Files

READ THESE FIRST before modifying any code:
- `docs/checkpoint_reference/SCHEMAS.md` — All object schemas
- `docs/checkpoint_reference/run_snapshot.json` — Latest metrics snapshot
- `docs/JULES_PERFORMANCE_TARGETS.md` — The fix plan that uses this diagnostic data

## Task: Add 5 Diagnostic CSV Outputs

Instrument `training/orchestrator.py` to collect diagnostic data during the
forward pass. **Do NOT change any thresholds or logic** — this is measurement only.

### 1. Gate 0 Rule 4 Rejections — `output/diagnostics/gate0_rule4_analysis.csv`

Log every Gate 0 extreme-zone rejection:
```python
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

Report section "GATE 0 RULE 4 DIAGNOSTIC":
- Total rejected, % that were real moves, $ left on table
- macro_z distribution bins: 3.0-3.5, 3.5-4.0, 4.0-5.0, 5.0+
- Oracle WR by macro_z bin (MFE > MAE = would have won)
- Mean MFE/MAE ratio by bin

### 2. Gate 1 No-Match Rejections — `output/diagnostics/gate1_nomatch_analysis.csv`

Log every cluster distance rejection:
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

Report section "GATE 1 NO-MATCH DIAGNOSTIC":
- Total rejected, % real moves, $ left on table
- Distance distribution bins: 4.5-5.0, 5.0-6.0, 6.0-8.0, 8.0+
- Oracle WR by distance bin

### 3. Early Exits — `output/diagnostics/early_exit_analysis.csv`

Log every trade where oracle says >50% more MFE remained at exit:
```python
early_exits.append({
    'entry_time': entry_time,
    'exit_time': exit_time,
    'exit_reason': exit_reason,
    'actual_pnl': pnl,
    'oracle_mfe': oracle_mfe,
    'remaining_mfe': oracle_mfe - actual_capture,
    'bars_held': bars_held,
    'exit_conviction': exit_conviction,
    'exit_wave_maturity': exit_wave_maturity,
    'trail_activated': trail_was_activated,  # True/False
    'entry_depth': depth,
})
```

Report section "EARLY EXIT DIAGNOSTIC":
- Exit reason breakdown (trail_stop vs max_hold vs belief_flip)
- % where trail never activated
- Avg bars remaining (oracle MFE bar - actual exit bar)
- Conviction at exit vs entry

### 4. Wrong Direction — `output/diagnostics/wrong_direction_analysis.csv`

Log every trade where oracle says opposite direction was correct:
```python
wrong_dir.append({
    'timestamp': ts_raw,
    'chosen_direction': direction,
    'oracle_direction': 'LONG' if oracle_marker > 0 else 'SHORT',
    'direction_source': direction_source,
    'logit_p_long': logit_p_long,
    'template_long_bias': template_long_bias,
    'template_short_bias': template_short_bias,
    'dmi_diff': dmi_diff,
    'z_score': z_score,
    'belief_conviction': conviction,
})
```

Report section "DIRECTION ERROR DIAGNOSTIC":
- Which priority level made the decision (logit vs bias vs DMI)
- Error rate per priority level
- SHORT vs LONG error rates (currently SHORT is 65% wrong)

### 5. Reversals — `output/diagnostics/reversal_analysis.csv`

Log every trade that entered correct direction but exited at a loss:
```python
reversals.append({
    'entry_time': entry_time,
    'exit_time': exit_time,
    'hold_bars': bars_held,
    'pnl': pnl,
    'oracle_mfe': oracle_mfe,
    'peak_unrealized_pnl': peak_pnl,
    'entry_conviction': entry_conviction,
    'exit_conviction': exit_conviction,
    'bars_after_hwm': bars_after_peak,
})
```

Report section "REVERSAL DIAGNOSTIC":
- Peak unrealized PnL before reversal
- Bars held after peak (are we holding too long?)
- Did conviction drop before reversal? (early warning?)

## Implementation Pattern

```python
# At start of run_forward_pass():
gate0_r4_rejections = []
gate1_rejections = []
early_exits = []
wrong_directions = []
reversals = []

# ... during forward pass, append to lists at each gate/exit ...

# At end, after report generation:
import os
diag_dir = os.path.join(self.output_dir, 'diagnostics')
os.makedirs(diag_dir, exist_ok=True)
for name, data in [('gate0_rule4_analysis', gate0_r4_rejections),
                    ('gate1_nomatch_analysis', gate1_rejections),
                    ('early_exit_analysis', early_exits),
                    ('wrong_direction_analysis', wrong_directions),
                    ('reversal_analysis', reversals)]:
    if data:
        pd.DataFrame(data).to_csv(os.path.join(diag_dir, f'{name}.csv'), index=False)
```

Report sections go at the END of the existing Phase 4 report. Use existing
`lines.append()` pattern with `"=" * 80` headers.

## Files to Modify

1. `training/orchestrator.py` — ONLY this file. All gates, exits, and reporting live here.

## What NOT to Change

- Do NOT change any gate thresholds — this is DIAGNOSTIC ONLY
- Do NOT change the score competition logic
- Do NOT change the clustering algorithm
- Do NOT add new dependencies
- Do NOT change exit logic or trail parameters

## Expected Output

After running `python -m training.orchestrator --forward-pass`:
1. 5 CSV files in `output/diagnostics/`
2. 5 new report sections at end of `reports/is/phase4_report.txt`
3. Zero behavior changes — identical trade results, just more data

## Success Criteria

- Forward pass runs without errors and produces all 5 CSV files
- We can see the exact macro_z distribution where Rule 4 kills real moves
- We can quantify trail activation failure rate
- We can see which direction source (logit vs DMI) has highest error rate
- Data is sufficient to calibrate the fixes in JULES_PERFORMANCE_TARGETS.md

## Priority

**HIGH** — This diagnostic feeds directly into the Phase A/B/C implementation.
