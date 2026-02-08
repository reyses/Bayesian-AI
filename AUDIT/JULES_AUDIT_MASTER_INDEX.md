# JULES AUDIT MASTER INDEX

## Overview
Complete system audit broken into 6 bite-sized phases. Each phase takes 25-90 minutes and produces a single commit.

**Total Time:** 3-4 hours (can be split across multiple sessions)

---

## Phase Execution Order

### Phase 1: Core Architecture Audit
**File:** `JULES_PHASE1_CORE_AUDIT.md`  
**Time:** 30-45 minutes  
**Deliverable:** `AUDIT_FINDINGS_PHASE1.md`  
**Focus:**
- StateVector: Hash/equality logic
- BayesianBrain: Laplace smoothing & confidence
- LayerEngine: Static (L1-L4) and Fluid (L5-L9) layers

**Commit:**
```bash
git commit -m "audit: Phase 1 - Core architecture validation complete"
```

---

### Phase 2: CUDA Modules Audit
**File:** `JULES_PHASE2_CUDA_AUDIT.md`  
**Time:** 30-40 minutes  
**Deliverable:** `AUDIT_FINDINGS_PHASE2.md`  
**Focus:**
- Pattern Detector (L7): CPU/GPU logic parity
- Confirmation (L8): Volume spike detection
- Velocity Gate (L9): Cascade detection
- Verification: 3-stage audit system

**Commit:**
```bash
git commit -m "audit: Phase 2 - CUDA modules validation complete"
```

---

### Phase 3: Data Pipeline Audit
**File:** `JULES_PHASE3_DATA_AUDIT.md`  
**Time:** 25-35 minutes  
**Deliverable:** `AUDIT_FINDINGS_PHASE3.md`  
**Focus:**
- Databento Loader: File format support
- Data Aggregator: Ring buffer & resampling
- Orchestrator: Multi-file loading
- End-to-end data flow

**Commit:**
```bash
git commit -m "audit: Phase 3 - Data pipeline validation complete"
```

---

### Phase 4: Training Loop Audit
**File:** `JULES_PHASE4_TRAINING_AUDIT.md`  
**Time:** 30-40 minutes  
**Deliverable:** `AUDIT_FINDINGS_PHASE4.md`  
**Focus:**
- Orchestrator: Iteration logic & progress
- Engine Core: Tick processing & learning
- Wave Rider: Position & exit management
- Learning convergence

**Commit:**
```bash
git commit -m "audit: Phase 4 - Training loop validation complete"
```

---

### Phase 5: Test Coverage Audit
**File:** `JULES_PHASE5_TEST_AUDIT.md`  
**Time:** 20-30 minutes  
**Deliverable:** `AUDIT_FINDINGS_PHASE5.md`  
**Focus:**
- Test inventory and categorization
- Coverage matrix (core, CUDA, integration)
- Critical gaps identification
- Recommendations for new tests

**Commit:**
```bash
git commit -m "audit: Phase 5 - Test coverage analysis complete"
```

---

### Phase 6: Issue Resolution
**File:** `JULES_PHASE6_FIXES.md`  
**Time:** 60-90 minutes (split into multiple sub-commits)  
**Deliverables:** 
- `ISSUE_TRIAGE.md`
- Fixed source files
- New test files
- Updated docs

**Commits:** (Multiple, one per fix)
```bash
git commit -m "fix: [Issue XXX] - [Description]"
git commit -m "test: Add [Component] test coverage"
git commit -m "docs: Update status after Phase 6 fixes"
```

---

## Quick Start

### Option 1: Sequential Execution
Run all phases in order, one per session:

```bash
# Day 1
claude: "Execute JULES_PHASE1_CORE_AUDIT.md"
# Commit AUDIT_FINDINGS_PHASE1.md

# Day 2
claude: "Execute JULES_PHASE2_CUDA_AUDIT.md"
# Commit AUDIT_FINDINGS_PHASE2.md

# ... etc
```

### Option 2: Batch Execution
Run multiple phases in one session:

```bash
# Morning: Phases 1-3 (Core, CUDA, Data)
claude: "Execute Phases 1, 2, and 3 back-to-back"
# 3 commits

# Afternoon: Phases 4-5 (Training, Tests)
claude: "Execute Phases 4 and 5"
# 2 commits

# Next day: Phase 6 (Fixes)
claude: "Execute Phase 6 with top 3 critical issues"
# Multiple commits
```

---

## Expected Outputs

After all phases complete, you'll have:

```
Bayesian-AI/
├── AUDIT_FINDINGS_PHASE1.md  ← Core architecture report
├── AUDIT_FINDINGS_PHASE2.md  ← CUDA modules report
├── AUDIT_FINDINGS_PHASE3.md  ← Data pipeline report
├── AUDIT_FINDINGS_PHASE4.md  ← Training loop report
├── AUDIT_FINDINGS_PHASE5.md  ← Test coverage report
├── ISSUE_TRIAGE.md           ← Prioritized fix list
├── CURRENT_STATUS.md         ← Updated with audit results
└── [Fixed source files]
└── [New test files]
```

---

## Progress Tracking

Use this checklist:

- [ ] Phase 1: Core Architecture (30-45 min)
- [ ] Phase 2: CUDA Modules (30-40 min)
- [ ] Phase 3: Data Pipeline (25-35 min)
- [ ] Phase 4: Training Loop (30-40 min)
- [ ] Phase 5: Test Coverage (20-30 min)
- [ ] Phase 6: Issue Resolution (60-90 min)

**Total:** 195-280 minutes (3.25-4.67 hours)

---

## Communication with Jules

### Starting a Phase
```
User: "Jules, execute Phase 1: Core Architecture Audit. 
Follow JULES_PHASE1_CORE_AUDIT.md exactly. 
Commit when done."
```

### Checking Status
```
User: "Jules, what phase are we on? Show progress."

Jules: "Currently on Phase 3 (Data Pipeline). 
Completed: Phases 1, 2
Remaining: Phases 3, 4, 5, 6
Time invested: ~75 minutes"
```

### Handling Issues
```
User: "Jules, you found a critical bug in Phase 1. 
Document it in AUDIT_FINDINGS_PHASE1.md. 
Don't fix yet - that's Phase 6."
```

---

## Phase Dependencies

```
Phase 1 (Core) ──┐
                 ├──> Phase 4 (Training)
Phase 2 (CUDA) ──┤                      ├──> Phase 5 (Tests) ──> Phase 6 (Fixes)
                 └──> Phase 3 (Data) ───┘
```

- Phases 1-3 can run in parallel
- Phase 4 requires 1-3 complete
- Phase 5 requires all previous phases
- Phase 6 requires Phase 5

---

## Tips for Jules

1. **Read the entire phase doc first** before starting
2. **Time box each task** - don't get stuck
3. **Document issues, don't fix** (until Phase 6)
4. **Use templates provided** - they're tested
5. **Commit early and often** - one phase = one commit
6. **Ask for help if stuck** >20 minutes on one task

---

## Quality Gates

Each phase must pass these before commit:

**Phase 1-4:**
- [ ] All sections completed
- [ ] All ✓ checks documented
- [ ] All ⚠ issues documented with severity
- [ ] Recommendations provided
- [ ] Next steps clear

**Phase 5:**
- [ ] Test inventory complete
- [ ] Coverage matrix filled
- [ ] Gaps prioritized
- [ ] Recommendations specific

**Phase 6:**
- [ ] Issues triaged by severity
- [ ] Each fix has test
- [ ] Each commit atomic
- [ ] Docs updated

---

## Success Metrics

After all 6 phases:
- ✓ Complete system audit documented
- ✓ All critical issues identified
- ✓ Top 3-5 critical issues fixed
- ✓ Test coverage improved
- ✓ 6+ commits pushed
- ✓ System more production-ready

---

## Support

If Jules gets stuck:
1. Check phase doc for guidance
2. Skip stuck task, document why
3. Move to next task
4. Flag for human review
5. Don't let one task block entire phase

---

## Final Deliverable

After Phase 6, create summary:

**FILE:** `AUDIT_COMPLETE_SUMMARY.md`
```markdown
# Complete System Audit - Summary

## Phases Completed: 6/6 ✓

### Issues Found & Fixed
- Critical: X found, Y fixed
- Major: X found, Y fixed
- Minor: X found, Y backlogged

### Test Coverage
- Before: ~40%
- After: ~65%
- Gaps: [List remaining]

### Commits
- Audit reports: 5 commits
- Fixes: X commits
- Tests: Y commits
- Total: Z commits

### Production Readiness
- Before: 60%
- After: 85%
- Blockers: 0

## Recommendations
1. [Priority 1]
2. [Priority 2]
3. [Priority 3]
```

Commit:
```bash
git add AUDIT_COMPLETE_SUMMARY.md
git commit -m "audit: Complete 6-phase system audit summary

Phases: 6/6 complete
Issues fixed: X critical, Y major
Coverage improved: 40% → 65%
Production readiness: 85%"
```
