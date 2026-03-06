# MASTER EXECUTION PLAN — BayesianBridge Pipeline Fix
# All specs sequenced with dependency gates and rollback points
# Date: March 6, 2026

## EXECUTION ORDER (dependencies flow downward)

```
PHASE 1: Unified Exit Engine        ← foundation (no dependencies)
   ↓ GATE 1: forward pass produces identical trade count
PHASE 2: Band Context               ← requires: exit engine (band_context param)
   ↓ GATE 2: band_direction logged in oracle_trade_log.csv
PHASE 3: Oracle Direction Learning   ← requires: exit engine + band context
   ↓ GATE 3: pattern_library.pkl shows corrected biases
PHASE 4: Terminology Refactor        ← requires: all logic stable (rename only)
   ↓ GATE 4: grep returns zero hits on old terminology
```

**WHY THIS ORDER:**
- Exit engine FIRST because band context feeds into it (band_context param in evaluate())
- Band context SECOND because oracle learning measures direction accuracy using band signals
- Oracle learning THIRD because it writes to pattern_library.pkl — do this after exit logic is stable
- Terminology LAST because it touches every file — do it when nothing else is changing

---

## PHASE 1: UNIFIED EXIT ENGINE
**Spec:** `CLAUDE_CODE_UNIFIED_EXIT_ENGINE.md`
**New file:** `core/exit_engine.py` (~450 lines)
**Modifies:** `training/orchestrator.py`, `live/live_engine.py`
**Deletes:** scattered exit logic in both files (~200 lines removed)

### Steps
```
1.1  Create core/exit_engine.py (full module from spec)
1.2  In training/orchestrator.py:
     - Import ExitEngine at top
     - Initialize _exit_engine = ExitEngine(mode='training', ...) before day loop
     - Replace position opening logic with _exit_engine.open_position()
     - Replace bar-by-bar exit checks with _exit_engine.evaluate()
     - Keep old exit code COMMENTED (not deleted) until Gate 1 passes
1.3  In live/live_engine.py:
     - Import ExitEngine at top
     - Initialize self._exit_engine in __init__ or _load_checkpoints
     - Replace _manage_trail, _check_envelope_decay, _check_band_exit, _check_watchdog
       with self._exit_engine.evaluate()
     - Keep old exit code COMMENTED (not deleted) until Gate 1 passes
1.4  Run forward pass
```

### GATE 1 — Exit Engine Validation
```bash
# 1A: Forward pass completes without errors
python training/orchestrator.py --forward-pass --data DATA/ATLAS
# Must complete. Any exception = rollback (uncomment old code, debug exit_engine.py)

# 1B: Trade count within 15% of previous run
# Old run trade count: [FILL FROM LAST RUN]
# New run trade count: should be within ±15%
# If >15% difference, the exit logic is triggering differently — inspect exit_action breakdown

# 1C: Exit action breakdown makes sense
# In oracle_trade_log.csv or report, check distribution:
#   stop_loss:    should be ~similar to old SL count
#   take_profit:  should be ~similar to old TP count  
#   trail_stop:   should exist (was already present)
#   envelope_decay: NEW — should be small but nonzero
#   band_urgent:  will be 0 (band context not implemented yet — that's Phase 2)
#   watchdog:     should be ~similar to old
#   max_hold:     should be ~similar

# 1D: Live dry run starts
python -m live --dry-run --no-gui
# Must start without import errors or crashes

# 1E: PnL is within 20% of previous run
# Some difference expected (cluster-fitted stops vs ATR, bar H/L vs close-only)
# If PnL drops >20%, inspect which exit_action is causing the divergence
```

### GATE 1 PASS CRITERIA
- [ ] Forward pass completes without exception
- [ ] Trade count within ±15% of baseline
- [ ] Exit action distribution has no single category >80%
- [ ] Live dry run starts without error
- [ ] PnL within ±20% of baseline

### GATE 1 FAIL → ROLLBACK
```bash
# Uncomment old exit logic in orchestrator.py and live_engine.py
# Comment out ExitEngine calls
# Debug core/exit_engine.py against specific failing cases
# Common failures:
#   - sl_ticks = 0 → p25_mae missing from library → check lib_entry keys
#   - tp_ticks = 0 → same issue
#   - trail never activates → trail_activation_ticks too high
#   - all trades exit immediately → sl_ticks too tight (check floor/ceiling)
```

### After Gate 1 passes:
```bash
# DELETE old commented exit code (not before)
# Commit: "Phase 1: Unified exit engine — training/live parity"
git add core/exit_engine.py training/orchestrator.py live/live_engine.py
git commit -m "Phase 1: Unified exit engine replaces scattered exit logic"
```

---

## PHASE 2: STANDARD ERROR BAND MULTI-TF CONTEXT
**Spec:** `CLAUDE_CODE_BAND_CONTEXT.md`
**New code:** ~100 lines across 3 files
**Modifies:** `training/timeframe_belief_network.py`, `live/live_engine.py`, `training/orchestrator.py`
**Depends on:** Phase 1 (exit engine accepts band_context parameter)

### Steps
```
2.1  Add BandContext dataclass to training/timeframe_belief_network.py
2.2  Populate band_context in each worker's _analyze() method
     - Extract from existing state fields (z_score, sigma, center)
     - Attach to WorkerBelief output
2.3  Add get_band_confluence() to TimeframeBeliefNetwork class
     - Aggregates band positions across all 11 workers
     - Returns: {direction, strength, support_score, resistance_score, band_summary, per_tf}
2.4  In training/orchestrator.py:
     - Call self.tbn.get_band_confluence() each bar
     - Pass result to _exit_engine.evaluate(band_context=_band_ctx)
     - Insert band confluence into direction cascade (Priority 3.5)
     - Add band_direction, band_strength, band_summary to oracle_trade_log.csv
2.5  In live/live_engine.py:
     - Call self._tbn.get_band_confluence() each bar
     - Pass result to self._exit_engine.evaluate(band_context=_band_ctx)
     - Insert band confluence into _determine_direction (Priority 3.5)
2.6  Run forward pass
```

### GATE 2 — Band Context Validation
```bash
# 2A: Forward pass completes
python training/orchestrator.py --forward-pass --data DATA/ATLAS

# 2B: band_direction column exists in oracle_trade_log.csv
head -1 oracle_trade_log.csv | tr ',' '\n' | grep -n band
# Should show band_direction, band_strength, band_summary columns

# 2C: band_direction is not all None
python -c "
import pandas as pd
df = pd.read_csv('oracle_trade_log.csv')
print('band_direction value counts:')
print(df['band_direction'].value_counts(dropna=False))
print(f'Non-null: {df[\"band_direction\"].notna().sum()}/{len(df)}')
"
# At least 50% of trades should have a band_direction

# 2D: band_direction accuracy vs oracle
python -c "
import pandas as pd
df = pd.read_csv('oracle_trade_log.csv')
df = df[df['band_direction'].notna() & df['oracle_label'].notna()]
df['band_correct'] = (
    ((df['band_direction'] == 'long') & (df['oracle_label'] > 0)) |
    ((df['band_direction'] == 'short') & (df['oracle_label'] < 0))
)
accuracy = df['band_correct'].mean()
print(f'Band direction accuracy vs oracle: {accuracy:.1%}')
print(f'  Needs to be >50% to be useful, >55% = strong signal')
"

# 2E: LONG/SHORT ratio is more balanced than before
python -c "
import pandas as pd
df = pd.read_csv('oracle_trade_log.csv')
print('Direction distribution:')
print(df['direction'].value_counts())
ratio = df['direction'].value_counts().get('LONG', 0) / max(df['direction'].value_counts().get('SHORT', 1), 1)
print(f'LONG/SHORT ratio: {ratio:.2f} (1.0 = balanced)')
"
# Old ratio was heavily SHORT-biased. New should be closer to 1.0

# 2F: Exit engine now shows band_urgent exits
python -c "
import pandas as pd
df = pd.read_csv('oracle_trade_log.csv')
if 'exit_action' in df.columns:
    print(df['exit_action'].value_counts())
else:
    print('exit_action column not found — check logging')
"
# band_urgent should be nonzero (small but present)

# 2G: Live dry run
python -m live --dry-run --no-gui
```

### GATE 2 PASS CRITERIA
- [ ] Forward pass completes without exception
- [ ] band_direction column populated (>50% non-null)
- [ ] band_direction accuracy vs oracle >50%
- [ ] LONG/SHORT ratio closer to 1.0 than baseline
- [ ] Live dry run starts without error

### GATE 2 FAIL → ROLLBACK
```bash
# If band_direction is all None:
#   → get_band_confluence() isn't being called or workers don't have z_score
#   → Check: print(worker.state.z_score) in _analyze() — is it populated?

# If accuracy <50%:
#   → Band confluence is anti-correlated with oracle
#   → Check: are higher TF weights inverted? (daily should dominate)
#   → Check: z_score sign convention (z>0 = above center = resistance, not support)

# If LONG/SHORT ratio unchanged:
#   → Band confluence direction isn't reaching the cascade
#   → Check: is Priority 3.5 insertion correct? Is side still None at that point?
```

### After Gate 2 passes:
```bash
git add training/timeframe_belief_network.py training/orchestrator.py live/live_engine.py
git commit -m "Phase 2: Multi-TF band context — direction via SE band confluence"
```

---

## PHASE 3: ORACLE DIRECTION LEARNING
**Spec:** `CLAUDE_CODE_ORACLE_LEARNING.md`
**New code:** ~180 lines across 2 files
**Modifies:** `training/orchestrator.py`, `live/live_engine.py`
**Depends on:** Phase 1 (exit engine stable), Phase 2 (band_direction in trade log)

### Steps
```
3.1  In training/orchestrator.py — after forward pass day loop:
     - Add direction correction accumulator (~80 lines)
     - Processes all oracle_trade_records
     - Computes per-template: long_correct, long_wrong, short_correct, short_wrong
     - Blends with original Phase 2.5 biases (70% forward pass, 30% original)
     - Trains signed MFE regression per template (if ≥15 samples)
     - Saves updated pattern_library.pkl with 'direction_source': 'oracle_corrected'
     
3.2  In training/orchestrator.py — direction cascade:
     - Add Priority 0.5: signed MFE regression prediction (~15 lines)
     - Add Priority 1.5: brain dir_table win rate check (~10 lines)
     
3.3  In training/orchestrator.py — after pattern library save:
     - Save forward pass brain as pattern_forward_brain.pkl (~5 lines)
     
3.4  In training/orchestrator.py — report section:
     - Add direction learning summary table (~50 lines)
     
3.5  In live/live_engine.py — _determine_direction():
     - Add Priority 0.5: brain dir_table check (~8 lines)
     
3.6  In live/live_engine.py — _load_checkpoints():
     - Prefer pattern_forward_brain.pkl over training base brain (~10 lines)
     
3.7  Run forward pass TWICE (first builds corrections, second uses them)
```

### GATE 3 — Oracle Learning Validation
```bash
# 3A: Forward pass completes (Run 1 — builds corrections)
python training/orchestrator.py --forward-pass --data DATA/ATLAS

# 3B: Direction corrections section appears in report
# Look for "DIRECTION LEARNING (oracle corrections absorbed)" in output
# Should show:
#   - Templates with direction corrections: N (should be >20)
#   - Templates with signed MFE regression: M (should be >5)
#   - TOP 15 DIRECTION CORRECTIONS table

# 3C: Pattern library has corrected biases
python -c "
import pickle
with open('checkpoints/pattern_library.pkl', 'rb') as f:
    lib = pickle.load(f)
corrected = sum(1 for v in lib.values() if v.get('direction_source') == 'oracle_corrected')
smfe = sum(1 for v in lib.values() if v.get('signed_mfe_coeff') is not None)
print(f'Total templates: {len(lib)}')
print(f'Oracle-corrected direction: {corrected}')
print(f'Signed MFE regression fitted: {smfe}')

# Show a few examples
for tid, v in sorted(lib.items())[:10]:
    src = v.get('direction_source', 'original')
    lb = v.get('long_bias', 0)
    sb = v.get('short_bias', 0)
    has_smfe = 'yes' if v.get('signed_mfe_coeff') else 'no'
    print(f'  T{tid}: long={lb:.3f} short={sb:.3f} src={src} smfe={has_smfe}')
"

# 3D: Forward pass brain saved
ls -la checkpoints/pattern_forward_brain.pkl
# Should exist and be >0 bytes

# 3E: Run 2 — uses corrected biases
python training/orchestrator.py --forward-pass --data DATA/ATLAS
# Compare Run 2 vs Run 1:
#   - LONG/SHORT ratio should shift (fewer wrong-direction trades)
#   - PnL should improve (direction corrections reduce losses)
#   - Direction accuracy in report should be higher

# 3F: Velocity fallback usage dropped
# In Run 2's report or trade log, check how many trades used
# velocity_sign as direction source vs signed_mfe or brain_dir or band_confluence
# Target: velocity_sign < 10% of direction decisions (was ~40%)

# 3G: Live loads forward brain
python -m live --dry-run --no-gui
# Log should show: "Brain: pattern_forward_brain.pkl (N states, M dir pairs) — IS-learned directions"
```

### GATE 3 PASS CRITERIA
- [ ] Direction corrections section in report (>20 templates corrected)
- [ ] pattern_library.pkl has oracle_corrected entries
- [ ] pattern_forward_brain.pkl exists
- [ ] Run 2 PnL >= Run 1 PnL (corrections helped)
- [ ] Velocity fallback usage <15% of direction decisions
- [ ] Live loads forward brain without error

### GATE 3 FAIL → ROLLBACK
```bash
# If 0 templates corrected:
#   → oracle_trade_records is empty or oracle_label field missing
#   → Check: print(len(oracle_trade_records)) after day loop
#   → Check: print(oracle_trade_records[0].keys()) for field names

# If Run 2 PnL WORSE than Run 1:
#   → Corrected biases are wrong direction
#   → Check: the 70/30 blend — is fp_long_correct counting correctly?
#   → Check: oracle_label sign convention (+1=LONG, -1=SHORT — verify)

# If velocity fallback still >30%:
#   → Priority 0.5 and 1.5 aren't resolving direction
#   → Check: signed_mfe_coeff is populated but prediction is below 0.5 threshold
#   → Consider lowering threshold from 0.5 to 0.25

# Safe rollback: the old pattern_library.pkl is overwritten.
# BEFORE running Phase 3, back it up:
cp checkpoints/pattern_library.pkl checkpoints/pattern_library_pre_phase3.pkl
```

### After Gate 3 passes:
```bash
git add training/orchestrator.py live/live_engine.py
git commit -m "Phase 3: Oracle direction learning — IS forward pass absorbs oracle corrections"
```

---

## PHASE 4: TERMINOLOGY REFACTOR
**Spec:** `CLAUDE_CODE_TERMINOLOGY_REFACTOR.md`
**Modifies:** ~25 files (rename only, no logic changes)
**Depends on:** Phases 1-3 complete and stable

### Steps
```
4.1  Create core/market_state.py (copy + rename from three_body_state.py)
4.2  Add backward-compat alias in three_body_state.py
4.3  Create core/field_engine.py (copy + rename from quantum_field_engine.py)
4.4  Add backward-compat alias in quantum_field_engine.py
4.5  Rename risk_engine class
4.6  Rename bayesian_brain methods
4.7  Update consumers one file at a time (25 files, test after each)
4.8  Verify pickle compatibility (load existing checkpoints)
4.9  Remove backward-compat aliases
4.10 Final grep verification
```

### GATE 4 — Terminology Validation
```bash
# 4A: All tests pass
pytest tests/ -v

# 4B: No old terminology
grep -rn "ThreeBodyQuantumState\|QuantumFieldEngine\|ROCHE_SNAP\|particle_position\|event_horizon\|lagrange\|tunnel_prob\|escape_prob\|Nightmare" \
  core/ training/ live/ --include="*.py" | grep -v "DEPRECATED\|backward compat\|alias\|# renamed from"
# Should return 0 lines

# 4C: Forward pass produces identical results to Phase 3
python training/orchestrator.py --forward-pass --data DATA/ATLAS
# Trade count, PnL, direction distribution must match Phase 3 Run 2 exactly
# Any difference = a rename broke something

# 4D: Saved brains still load
python -c "
from core.bayesian_brain import BayesianBrain
b = BayesianBrain()
b.load('checkpoints/pattern_forward_brain.pkl')
print(f'Loaded {len(b.table)} states, {len(b.dir_table)} dir pairs')
"

# 4E: Live dry run
python -m live --dry-run --no-gui
```

### GATE 4 PASS CRITERIA
- [ ] All tests pass
- [ ] grep returns 0 hits on old terminology
- [ ] Forward pass output matches Phase 3 exactly (same PnL, same trades)
- [ ] Saved checkpoints load correctly
- [ ] Live dry run starts without error

### After Gate 4 passes:
```bash
git add -A
git commit -m "Phase 4: Terminology refactor — quantum metaphors → statistical language"
```

---

## BACKUP PROTOCOL (before starting ANY phase)

```bash
# Run ONCE before Phase 1:
mkdir -p backups/pre_refactor
cp -r core/ backups/pre_refactor/core/
cp -r training/ backups/pre_refactor/training/
cp -r live/ backups/pre_refactor/live/
cp -r checkpoints/ backups/pre_refactor/checkpoints/

# Before each phase:
cp checkpoints/pattern_library.pkl checkpoints/pattern_library_pre_phase${N}.pkl
cp checkpoints/live_brain.pkl checkpoints/live_brain_pre_phase${N}.pkl 2>/dev/null || true

# Nuclear rollback (any phase):
cp -r backups/pre_refactor/core/ core/
cp -r backups/pre_refactor/training/ training/
cp -r backups/pre_refactor/live/ live/
```

---

## SPEC FILE REFERENCE

| Phase | Spec File | Priority |
|-------|-----------|----------|
| 1 | `CLAUDE_CODE_UNIFIED_EXIT_ENGINE.md` | CRITICAL — foundation |
| 2 | `CLAUDE_CODE_BAND_CONTEXT.md` | HIGH — fixes short bias |
| 3 | `CLAUDE_CODE_ORACLE_LEARNING.md` | HIGH — pipeline learns |
| 4 | `CLAUDE_CODE_TERMINOLOGY_REFACTOR.md` | MEDIUM — readability |

All specs are in `/mnt/user-data/outputs/`.

---

## EXPECTED OUTCOMES AFTER ALL 4 PHASES

| Metric | Before | After |
|--------|--------|-------|
| Training/live exit parity | ~40% match | 100% same code |
| LONG/SHORT ratio | SHORT-biased (~0.4) | Balanced (~0.8-1.2) |
| Velocity fallback usage | ~40% of trades | <5% |
| Direction accuracy vs oracle | ~50% (coin flip) | >60% |
| Templates with corrected bias | 0 | >50% of active |
| Codebase grep "quantum" | 200+ hits | 0 |
| Forward pass PnL | X | Closer to live reality |

---

## TIMELINE ESTIMATE

| Phase | Effort | Calendar |
|-------|--------|----------|
| Phase 1: Exit Engine | 3-4 hours | Day 1 |
| Gate 1 validation | 1 hour | Day 1 |
| Phase 2: Band Context | 2-3 hours | Day 2 |
| Gate 2 validation | 1 hour | Day 2 |
| Phase 3: Oracle Learning | 2-3 hours | Day 3 |
| Gate 3 validation | 2 hours (2 runs) | Day 3 |
| Phase 4: Terminology | 3-4 hours | Day 4 |
| Gate 4 validation | 1 hour | Day 4 |
| **Total** | **~16-20 hours** | **4 days** |

Do NOT compress into fewer days. Each gate needs a clean forward pass
(30-60 min runtime) and careful inspection of results. Rushing past a
gate means the next phase builds on a broken foundation.
