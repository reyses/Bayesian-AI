# JULES TASK: Phase 6 - Issue Resolution & Fixes

## Objective
Fix critical issues identified in Phases 1-5 with small, atomic commits.

## Scope
**Issues from Previous Phases:**
- Critical bugs from Phase 1-4 audits
- Critical test gaps from Phase 5
- Any blockers preventing production use

**Time Estimate:** 60-90 minutes (split into sub-tasks)

## Prerequisites
- Complete Phases 1-5 first
- Have `AUDIT_FINDINGS_PHASE*.md` files ready
- Prioritize CRITICAL issues only

---

## Task 6.1: Triage Issues ✓

**Create priority list:**

1. Review all `AUDIT_FINDINGS_PHASE*.md` files
2. Extract all issues marked CRITICAL or MAJOR
3. Rank by impact:
   - BLOCKER: Prevents system from running
   - CRITICAL: Causes incorrect results
   - MAJOR: Causes crashes or data loss
   - MINOR: Cosmetic or optimization

**Acceptance Criteria:**
- [ ] Complete list of issues with severity
- [ ] Estimated fix time for each
- [ ] Dependencies identified (fix X before Y)

**Output:** Create `ISSUE_TRIAGE.md`

**Template:**
```markdown
# Issue Triage - Fix Priority

## BLOCKER (Fix Immediately)
1. [ISSUE-001] StateVector hash excludes metadata
   - Source: AUDIT_FINDINGS_PHASE1.md
   - Impact: HashMap lookups fail
   - Fix: Update __hash__() method
   - Time: 5 minutes
   - Files: core/state_vector.py

## CRITICAL (Fix This Session)
1. [ISSUE-002] DataAggregator ring buffer overflow
   - Source: AUDIT_FINDINGS_PHASE3.md
   - Impact: Data corruption after 10k ticks
   - Fix: Add bounds checking
   - Time: 10 minutes
   - Files: core/data_aggregator.py

## MAJOR (Fix Soon)
1. [ISSUE-003] Missing test for wave_rider
   - Source: AUDIT_FINDINGS_PHASE5.md
   - Impact: Untested critical component
   - Fix: Create test file
   - Time: 20 minutes
   - Files: tests/test_wave_rider.py (new)

## MINOR (Backlog)
...
```

---

## Task 6.2: Fix Critical Bugs ✓

**For each CRITICAL issue:**

1. Create branch: `fix/issue-XXX-short-description`
2. Make minimal fix
3. Add test if missing
4. Verify fix
5. Commit with template below
6. Push and move to next

**Commit Template:**
```
fix: [SHORT DESCRIPTION]

Issue: [ISSUE-XXX]
Source: AUDIT_FINDINGS_PHASEX.md

Problem:
- [What was wrong]

Fix:
- [What changed]

Verification:
- [How tested]

Impact: BLOCKER|CRITICAL|MAJOR
```

**Example Fix Flow:**
```bash
# Issue: StateVector hash bug
git checkout -b fix/issue-001-statevector-hash

# Edit core/state_vector.py
# Run test
pytest tests/test_phase1.py -k test_state_vector

# Commit
git add core/state_vector.py
git commit -m "fix: StateVector hash excludes timestamp/price

Issue: ISSUE-001
Source: AUDIT_FINDINGS_PHASE1.md

Problem:
- __hash__() included timestamp and price
- HashMap lookups failed for same state at different times

Fix:
- Updated __hash__() to exclude timestamp/price
- Only hash L1-L9 layers

Verification:
- test_state_vector_equality passes
- Two states with same layers now hash equal

Impact: BLOCKER"

git push origin fix/issue-001-statevector-hash
```

---

## Task 6.3: Add Critical Tests ✓

**For each missing critical test:**

1. Create test file
2. Write minimum viable tests:
   - Basic functionality
   - Edge cases
   - Error handling
3. Run and verify
4. Commit

**Example:**
```bash
git checkout -b test/wave-rider-coverage

# Create tests/test_wave_rider.py
cat > tests/test_wave_rider.py << 'EOF'
"""Tests for WaveRider position management"""
import pytest
from execution.wave_rider import WaveRider, Position
from config.symbols import MNQ
from core.state_vector import StateVector

def test_open_position():
    rider = WaveRider(MNQ)
    state = StateVector.null_state()
    
    rider.open_position(100.0, 'short', state)
    
    assert rider.position is not None
    assert rider.position.entry_price == 100.0
    assert rider.position.side == 'short'
    assert rider.position.stop_loss == 105.0  # 20 ticks

def test_adaptive_trail():
    rider = WaveRider(MNQ)
    state = StateVector.null_state()
    
    rider.open_position(100.0, 'short', state)
    
    # Profit < $50: 10 tick trail
    decision = rider.update_trail(98.0, state)
    assert not decision['should_exit']
    
    # Profit > $50: 20 tick trail
    rider.position.high_water_mark = 90.0
    decision = rider.update_trail(91.0, state)
    assert decision['should_exit']
EOF

# Run
pytest tests/test_wave_rider.py -v

# Commit
git add tests/test_wave_rider.py
git commit -m "test: Add WaveRider position management tests

Added tests for:
- open_position(): Entry creation
- update_trail(): Adaptive trailing stop
- Exit triggers: Stop hit and structure break

Coverage: WaveRider now has 80%+ test coverage"

git push origin test/wave-rider-coverage
```

---

## Task 6.4: Update Documentation ✓

**After fixes:**

1. Update CURRENT_STATUS.md with fix summary
2. Update any affected README sections
3. Mark issues as RESOLVED in triage doc

**Acceptance Criteria:**
- [ ] CURRENT_STATUS.md reflects fixes
- [ ] ISSUE_TRIAGE.md updated with resolution status
- [ ] All fixes documented

**Commit:**
```bash
git add CURRENT_STATUS.md ISSUE_TRIAGE.md
git commit -m "docs: Update status after Phase 6 fixes

Resolved:
- ISSUE-001: StateVector hash bug (BLOCKER)
- ISSUE-002: DataAggregator bounds checking (CRITICAL)
- ISSUE-003: WaveRider test coverage (MAJOR)

Remaining:
- X MINOR issues in backlog

Status: PHASE_6_COMPLETE"
```

---

## Deliverables

**FILES TO CREATE/UPDATE:**
1. `ISSUE_TRIAGE.md` (new)
2. Updated source files (fixes)
3. New test files (coverage)
4. `CURRENT_STATUS.md` (updated)

---

## Git Workflow

```bash
# For each fix:
git checkout main
git pull
git checkout -b fix/issue-XXX-description
# Make fix
git add [files]
git commit -m "fix: ..."
git push origin fix/issue-XXX-description

# Create PR or merge
gh pr create --title "Fix: Issue XXX" --body "See commit message"
# OR
git checkout main
git merge fix/issue-XXX-description
git push
```

---

## Notes for Jules
- Fix one issue at a time
- Keep commits atomic (1 issue = 1 commit)
- Run tests after each fix
- Don't combine unrelated fixes
- Target: 3-5 critical fixes this phase
- Time box: 90 minutes total
- If stuck on a fix >20 min, skip and document
