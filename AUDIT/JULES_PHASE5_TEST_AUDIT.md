# JULES TASK: Phase 5 - Test Coverage Audit

## Objective
Audit test suite completeness and identify gaps in test coverage.

## Scope
**FILES TO AUDIT:**
- All files in `tests/` directory
- Test execution results
- Coverage gaps

**Time Estimate:** 20-30 minutes

## Tasks

### Task 5.1: Test Inventory ✓

**Check:**
1. List all test files ✓
2. Identify test type (unit/integration/system) ✓
3. Map tests to source modules ✓
4. Check for orphaned tests ✓

**Acceptance Criteria:**
- [ ] Complete list of test files with descriptions
- [ ] Each test mapped to source file(s)
- [ ] Test types categorized

**Output:** Create table in `AUDIT_FINDINGS_PHASE5.md`

**Template:**
```
| Test File | Type | Tests | Source Files | Status |
|-----------|------|-------|--------------|--------|
| test_phase1.py | Integration | Core components | state_vector, bayesian_brain, layer_engine | PASS |
| test_cuda_*.py | Unit | CUDA modules | cuda_modules/*.py | PASS |
| ... | ... | ... | ... | ... |
```

---

### Task 5.2: Core Module Coverage ✓

**Check which core modules have tests:**
- [ ] state_vector.py
- [ ] bayesian_brain.py
- [ ] layer_engine.py
- [ ] data_aggregator.py
- [ ] engine_core.py

**For each module, check:**
1. Unit tests exist?
2. Integration tests exist?
3. Edge cases tested?
4. Error conditions tested?

**Acceptance Criteria:**
- [ ] All core modules have at least 1 test
- [ ] Critical functions tested
- [ ] Edge cases covered

**Output:** Add coverage matrix to `AUDIT_FINDINGS_PHASE5.md`

---

### Task 5.3: CUDA Module Coverage ✓

**Check:**
- [ ] test_cuda_pattern.py
- [ ] test_cuda_confirmation.py
- [ ] test_cuda_imports_and_init.py

**For each:**
1. Tests CPU fallback?
2. Tests GPU kernel (if available)?
3. Tests error handling?
4. Tests singleton pattern?

**Acceptance Criteria:**
- [ ] All 3 CUDA modules tested
- [ ] CPU fallback verified
- [ ] Error handling tested

**Output:** Add to `AUDIT_FINDINGS_PHASE5.md`

---

### Task 5.4: Integration Test Coverage ✓

**Check:**
- [ ] test_phase1.py (core integration)
- [ ] test_phase2.py (CUDA integration)
- [ ] test_full_system.py (end-to-end)
- [ ] test_training_validation.py (training loop)

**For each:**
1. Tests multiple components together?
2. Tests data flow?
3. Tests realistic scenarios?

**Acceptance Criteria:**
- [ ] Integration tests cover component interactions
- [ ] End-to-end flow tested
- [ ] Training loop validated

**Output:** Add to `AUDIT_FINDINGS_PHASE5.md`

---

### Task 5.5: Identify Coverage Gaps ✓

**Check which are NOT tested:**
- [ ] execution/wave_rider.py
- [ ] training/databento_loader.py
- [ ] visualization/*.py
- [ ] scripts/*.py

**Prioritize gaps:**
1. CRITICAL: Core logic untested
2. HIGH: Integration points untested
3. MEDIUM: Utility functions untested
4. LOW: Visualization/reporting untested

**Acceptance Criteria:**
- [ ] List of untested files
- [ ] Prioritized by criticality
- [ ] Recommendations for new tests

**Output:** Add to `AUDIT_FINDINGS_PHASE5.md`

---

## Deliverable

**FILE TO CREATE:** `AUDIT_FINDINGS_PHASE5.md`

**Template:**
```markdown
# Phase 5 Audit Findings - Test Coverage

## Summary
- **Total Test Files:** X
- **Test Coverage:** Y% (estimated)
- **Critical Gaps:** Z

## Test Inventory

| Test File | Type | LOC | Source Coverage | Status |
|-----------|------|-----|-----------------|--------|
| test_phase1.py | Integration | 150 | core/*.py (3 files) | ✓ PASS |
| test_cuda_pattern.py | Unit | 80 | pattern_detector.py | ✓ PASS |
| test_full_system.py | System | 120 | ALL | ✓ PASS |
| ... | ... | ... | ... | ... |

**Total:** X test files, Y total tests

## Coverage Matrix

### Core Modules
| Module | Unit Tests | Integration Tests | Edge Cases | Status |
|--------|------------|-------------------|------------|--------|
| state_vector.py | ✓ | ✓ | ✓ | GOOD |
| bayesian_brain.py | ✓ | ✓ | ✓ | GOOD |
| layer_engine.py | ✓ | ✓ | ⚠ Partial | NEEDS_IMPROVEMENT |
| data_aggregator.py | ✗ | ⚠ Partial | ✗ | CRITICAL_GAP |
| engine_core.py | ⚠ Partial | ✓ | ✗ | NEEDS_IMPROVEMENT |

### CUDA Modules
| Module | CPU Fallback | GPU Kernel | Error Handling | Status |
|--------|--------------|------------|----------------|--------|
| pattern_detector.py | ✓ | ✓ | ✓ | GOOD |
| confirmation.py | ✓ | ✓ | ✓ | GOOD |
| velocity_gate.py | ✓ | ⚠ Partial | ✓ | NEEDS_IMPROVEMENT |

### Integration Tests
| Test | Components Tested | Scenarios | Status |
|------|-------------------|-----------|--------|
| test_phase1.py | Core 3 modules | Basic flow | ✓ |
| test_full_system.py | All components | End-to-end | ✓ |
| test_training_validation.py | Training loop | Real data | ✓ |

## Critical Gaps

### PRIORITY 1 - CRITICAL
1. **data_aggregator.py** - No dedicated unit tests
   - Ring buffer wrap-around not tested
   - Resampling edge cases not covered
   - Recommendation: Create `test_data_aggregator.py`

2. **wave_rider.py** - No tests found
   - Position management untested
   - Adaptive trailing stop not verified
   - Recommendation: Create `test_wave_rider.py`

### PRIORITY 2 - HIGH
1. **databento_loader.py** - Minimal testing
   - Only tested via integration tests
   - File format edge cases not covered
   - Recommendation: Add `test_databento_loader.py`

2. **orchestrator.py** - Partial coverage
   - Multi-file loading not tested
   - Progress callbacks not verified
   - Recommendation: Expand `test_training_validation.py`

### PRIORITY 3 - MEDIUM
1. **visualization/*.py** - No tests
   - Not critical for core functionality
   - Recommendation: Manual QA sufficient

2. **scripts/*.py** - No automated tests
   - Utility scripts
   - Recommendation: Manual testing OK

## Recommendations

### Immediate Actions
1. Create `test_data_aggregator.py` (CRITICAL)
2. Create `test_wave_rider.py` (CRITICAL)
3. Expand `test_databento_loader.py` (HIGH)

### New Tests Needed
```python
# tests/test_data_aggregator.py
def test_ring_buffer_wrap()
def test_resampling_ohlc()
def test_empty_data_handling()

# tests/test_wave_rider.py
def test_open_position()
def test_adaptive_trail()
def test_exit_triggers()
def test_pnl_calculation()
```

## Test Execution Summary
```
PASSED: X tests
FAILED: Y tests
SKIPPED: Z tests
Coverage: ~W%
```

## Next Steps
- Proceed to Phase 6: Issue Resolution
- Prioritize CRITICAL gaps for immediate fix
```

---

## Git Commit

```bash
git add AUDIT_FINDINGS_PHASE5.md
git commit -m "audit: Phase 5 - Test coverage analysis complete

Analyzed:
- Test inventory: X test files
- Coverage matrix: Core & CUDA modules
- Critical gaps: 2 identified (data_aggregator, wave_rider)
- Recommendations: 3 new test files needed

Status: COVERAGE_GAPS_IDENTIFIED"

git push
```

---

## Notes for Jules
- Can run `pytest --collect-only` to list all tests
- Check test execution with `pytest -v`
- Estimate coverage by file (manual inspection OK)
- Focus on identifying CRITICAL gaps
- Time box: 30 minutes
