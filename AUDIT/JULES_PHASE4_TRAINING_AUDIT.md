# JULES TASK: Phase 4 - Training Loop Audit

## Objective
Validate training orchestration, iteration logic, and learning convergence.

## Scope
**FILES TO AUDIT:**
- `training/orchestrator.py` (training loop)
- `core/engine_core.py` (tick processing)
- `execution/wave_rider.py` (position management)

**Time Estimate:** 30-40 minutes

## Tasks

### Task 4.1: Orchestrator Iteration Loop ✓
**File:** `training/orchestrator.py`

**Check:**
1. run_training() iterates correctly ✓
2. Daily P&L resets per iteration ✓
3. Trade count accumulates ✓
4. Progress callback invoked ✓
5. Model saved after training ✓

**Acceptance Criteria:**
- [ ] Loops `iterations` times
- [ ] Calls engine.on_tick() for each tick
- [ ] Resets engine.daily_pnl = 0.0 at start of each iteration
- [ ] Calls on_progress(metrics) if provided
- [ ] Saves probability_table.pkl to output_dir

**Output:** Document in `AUDIT_FINDINGS_PHASE4.md`

---

### Task 4.2: Engine Core - Tick Processing ✓
**File:** `core/engine_core.py`

**Check:**
1. on_tick() adds to aggregator ✓
2. Computes current state ✓
3. Exit management checked first ✓
4. Entry logic only if no position ✓
5. Updates probability table on exit ✓

**Acceptance Criteria:**
- [ ] aggregator.add_tick(tick_data) called
- [ ] current_data = aggregator.get_current_data()
- [ ] If position exists: check exit first, skip entry
- [ ] If no position: check entry conditions
- [ ] On exit: brain.update(outcome) called

**Output:** Add to `AUDIT_FINDINGS_PHASE4.md`

---

### Task 4.3: Wave Rider - Position Management ✓
**File:** `execution/wave_rider.py`

**Check:**
1. open_position() creates Position correctly ✓
2. update_trail() adjusts stop dynamically ✓
3. Adaptive trail: 10 → 20 → 30 ticks based on profit ✓
4. Exit triggers: stop hit OR structure break ✓
5. P&L calculation correct ✓

**Acceptance Criteria:**
- [ ] Position initialized with entry_price, stop, side, state
- [ ] High water mark updated (min for short, max for long)
- [ ] Trail distance: <$50=10ticks, <$150=20ticks, else 30ticks
- [ ] Returns {'should_exit': True, 'pnl': X, 'exit_reason': Y}
- [ ] Structure break: L7_pattern changed OR L8_confirm=False

**Output:** Add to `AUDIT_FINDINGS_PHASE4.md`

---

### Task 4.4: Learning Convergence ✓

**Check:**
1. Probability table grows over iterations ✓
2. Win rate improves over time (if edge exists) ✓
3. High-confidence states accumulate ✓
4. No infinite loops or hangs ✓
5. Memory usage bounded ✓

**Trace:**
- Run `test_training_validation.py` or inspect logs
- Check `brain.table` size growth
- Verify `get_summary()` metrics

**Acceptance Criteria:**
- [ ] brain.table size increases with iterations
- [ ] brain.get_summary()['high_probability_states'] > 0 after sufficient data
- [ ] Win rate converges (not oscillating wildly)
- [ ] No memory leaks (check ring buffer max_ticks enforced)

**Output:** Add to `AUDIT_FINDINGS_PHASE4.md`

---

### Task 4.5: Error Handling ✓

**Check:**
1. Handles missing data gracefully ✓
2. Handles insufficient bars for layers ✓
3. Handles CUDA failures (fallback to CPU) ✓
4. Handles pickle save/load errors ✓
5. Logs errors clearly ✓

**Acceptance Criteria:**
- [ ] No crashes on empty DataFrame
- [ ] Returns null_state() or skips tick if insufficient data
- [ ] CUDA unavailable → CPU fallback without crash
- [ ] save() failure → error logged
- [ ] All exceptions caught with try/except

**Output:** Add to `AUDIT_FINDINGS_PHASE4.md`

---

## Deliverable

**FILE TO CREATE:** `AUDIT_FINDINGS_PHASE4.md`

**Template:**
```markdown
# Phase 4 Audit Findings - Training Loop

## Summary
- **Files Audited:** 3
- **Loop Logic:** VALIDATED
- **Learning:** CONVERGES / ISSUES
- **Issues Found:** X

## Orchestrator Loop
### ✓ PASS: Iteration Logic
- run_training() loops correctly
- Daily P&L resets per iteration
- Model saved after training

### ✓ PASS: Progress Tracking
- on_progress() callback invoked
- Metrics updated: iteration, pnl, win_rate, etc.

### ⚠ ISSUE: [If any]
...

## Engine Core
### ✓ PASS: Tick Processing
- aggregator.add_tick() called
- State computed via layer_engine
- Exit checked before entry

### ✓ PASS: Learning Update
- brain.update() called on trade completion
- Probability table grows

### ⚠ ISSUE: [If any]
...

## Wave Rider
### ✓ PASS: Position Management
- open_position() correct
- Adaptive trailing stop logic
- Exit triggers: stop OR structure break

### ✓ PASS: P&L Calculation
- calculate_pnl() uses tick_value
- Side (long/short) handled correctly

### ⚠ ISSUE: [If any]
...

## Learning Convergence
### ✓ PASS: Table Growth
- brain.table size: 0 → 48 → 127 states (example)
- High-confidence states accumulate

### ✓ PASS: Win Rate
- Converges after 500+ trades
- No wild oscillations

### ⚠ ISSUE: [If any]
...

## Error Handling
### ✓ PASS: Graceful Degradation
- Empty data → null_state()
- CUDA fail → CPU fallback
- Save fail → logged

### ⚠ ISSUE: [If any]
...

## Recommendations
1. [If any]

## Next Steps
- Proceed to Phase 5: Test Coverage Audit
```

---

## Git Commit

```bash
git add AUDIT_FINDINGS_PHASE4.md
git commit -m "audit: Phase 4 - Training loop validation complete

Audited:
- Orchestrator: Iteration logic & progress
- Engine Core: Tick processing & learning
- Wave Rider: Position & exit management
- Learning convergence verified

Findings: [X issues found]
Status: PASS/NEEDS_FIX"

git push
```

---

## Notes for Jules
- Can run actual training with small dataset (10 iterations)
- Monitor brain.table size growth
- Check for hangs or infinite loops
- Verify memory doesn't grow unbounded
- Time box: 40 minutes
