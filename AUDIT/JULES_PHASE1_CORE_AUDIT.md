# JULES TASK: Phase 1 - Core Architecture Audit

## Objective
Audit the 3 core modules (StateVector, BayesianBrain, LayerEngine) for correctness and document findings.

## Scope
**FILES TO AUDIT:**
- `core/state_vector.py` (131 lines)
- `core/bayesian_brain.py` (167 lines)  
- `core/layer_engine.py` (315 lines)

**Time Estimate:** 30-45 minutes

## Tasks

### Task 1.1: StateVector Validation ✓
**File:** `core/state_vector.py`

**Check:**
1. Hash function excludes timestamp/price metadata ✓
2. Equality operator matches hash logic ✓
3. All 9 layers present in dataclass ✓
4. Frozen=True enforced (immutability) ✓
5. null_state() returns valid defaults ✓

**Acceptance Criteria:**
- [ ] Hash uses only L1-L9 layers (not timestamp/price)
- [ ] Two states with same L1-L9 but different metadata hash equal
- [ ] StateVector is immutable (frozen=True)

**Output:** Document any issues in `AUDIT_FINDINGS_PHASE1.md`

---

### Task 1.2: BayesianBrain Validation ✓
**File:** `core/bayesian_brain.py`

**Check:**
1. Laplace smoothing formula correct: `(wins+1)/(total+2)` ✓
2. Confidence calculation: `min(total/30.0, 1.0)` ✓
3. should_fire() checks both probability AND confidence ✓
4. save/load preserve table structure ✓
5. get_all_states_above_threshold() filters correctly ✓

**Acceptance Criteria:**
- [ ] Laplace smoothing prevents 0% and 100% probabilities
- [ ] Confidence requires 30 samples for 100%
- [ ] should_fire() default thresholds: prob=0.80, conf=0.30
- [ ] Pickle save/load works (test with dummy data)

**Output:** Add findings to `AUDIT_FINDINGS_PHASE1.md`

---

### Task 1.3: LayerEngine Static Context ✓
**File:** `core/layer_engine.py`

**Check:**
1. initialize_static_context() computes L1-L4 ✓
2. L1 (90d bias): `>+5% = bull, <-5% = bear, else range` ✓
3. L2 (30d regime): `recent_range > avg*1.5 = trending` ✓
4. L3 (1wk swing): higher_highs/lower_lows logic ✓
4. L4 (zone): killzone/support/resistance/mid_range ✓

**Acceptance Criteria:**
- [ ] Static layers computed once, cached in self.static_context
- [ ] L1-L4 logic matches spec above
- [ ] Kill zone detection uses tolerance (default 5.0)

**Output:** Add findings to `AUDIT_FINDINGS_PHASE1.md`

---

### Task 1.4: LayerEngine Fluid Layers
**File:** `core/layer_engine.py`

**Check:**
1. compute_current_state() combines static + fluid ✓
2. L5 (4hr trend): 3-bar sequential logic ✓
3. L6 (1hr structure): bullish/bearish count ratio ✓
4. L7-L9 call CUDA modules correctly ✓
5. Returns complete StateVector with all 9 layers ✓

**Acceptance Criteria:**
- [ ] Fluid layers (L5-L9) computed per tick
- [ ] L7-L9 delegate to cuda_modules (pattern_detector, confirmation, velocity_gate)
- [ ] StateVector returned with timestamp/price metadata

**Output:** Add findings to `AUDIT_FINDINGS_PHASE1.md`

---

## Deliverable

**FILE TO CREATE:** `AUDIT_FINDINGS_PHASE1.md`

**Template:**
```markdown
# Phase 1 Audit Findings - Core Architecture

## Summary
- **Files Audited:** 3
- **Issues Found:** X
- **Critical:** Y
- **Minor:** Z

## StateVector (core/state_vector.py)
### ✓ PASS: Hash Logic
- Hash excludes timestamp/price correctly
- Equality operator consistent with hash

### ⚠ ISSUE: [If any found]
- **Description:** ...
- **Severity:** CRITICAL | MAJOR | MINOR
- **Fix:** ...

## BayesianBrain (core/bayesian_brain.py)
### ✓ PASS: Laplace Smoothing
- Formula: (wins+1)/(total+2) ✓
- Prevents edge case probabilities ✓

### ⚠ ISSUE: [If any found]
...

## LayerEngine (core/layer_engine.py)
### ✓ PASS: Static Context
- L1-L4 computed correctly
- Cached after initialization

### ✓ PASS: Fluid Layers
- L5-L9 computed per tick
- CUDA delegation working

## Recommendations
1. [If any improvements suggested]
2. ...

## Next Steps
- Proceed to Phase 2: CUDA Module Audit
```

---

## Git Commit

After completing all tasks:

```bash
git add AUDIT_FINDINGS_PHASE1.md
git commit -m "audit: Phase 1 - Core architecture validation complete

Audited:
- StateVector: Hash/equality logic
- BayesianBrain: Laplace smoothing & confidence
- LayerEngine: Static (L1-L4) and Fluid (L5-L9) layers

Findings: [X issues found]
Status: PASS/NEEDS_FIX"

git push
```

---

## Notes for Jules
- This is a READ-ONLY audit
- Do NOT modify code yet
- Focus on documenting issues
- If critical issues found, flag for Phase 6 (fixes)
- Use code inspection only - no test execution needed
- Time box: 45 minutes maximum
