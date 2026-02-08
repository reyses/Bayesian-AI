# JULES TASK: Phase 2 - CUDA Module Audit

## Objective
Validate CUDA acceleration modules (L7-L9) for correctness and CPU fallback reliability.

## Scope
**FILES TO AUDIT:**
- `cuda_modules/pattern_detector.py` (L7)
- `cuda_modules/confirmation.py` (L8)
- `cuda_modules/velocity_gate.py` (L9)
- `cuda_modules/hardened_verification.py`

**Time Estimate:** 30-40 minutes

## Tasks

### Task 2.1: Pattern Detector (L7) ✓
**File:** `cuda_modules/pattern_detector.py`

**Check:**
1. CPU fallback exists and works ✓
2. GPU kernel logic matches CPU logic ✓
3. Pattern priority: compression > wedge > breakdown ✓
4. Singleton pattern implemented correctly ✓
5. Error handling for CUDA unavailable ✓

**Acceptance Criteria:**
- [ ] `_detect_cpu()` and `detect_pattern_kernel()` produce same results
- [ ] Compression: `recent_range < prev_range * 0.7`
- [ ] Wedge: `lows[-1] > lows[-5] AND highs[-1] < highs[-5]`
- [ ] Breakdown: `lows[-1] < min(lows[-5:])`
- [ ] Returns tuple: (pattern_name: str, confidence: float)

**Output:** Document in `AUDIT_FINDINGS_PHASE2.md`

---

### Task 2.2: Confirmation Engine (L8) ✓
**File:** `cuda_modules/confirmation.py`

**Check:**
1. Volume spike detection: `current > mean*1.2` ✓
2. Requires L7 pattern active ✓
3. CPU fallback exists ✓
4. GPU kernel matches CPU logic ✓
5. Singleton pattern ✓

**Acceptance Criteria:**
- [ ] Returns False if L7_pattern == 'none'
- [ ] Returns True if volume spike detected
- [ ] `_confirm_cpu()` matches `confirm_kernel()` logic
- [ ] Error handling for missing 'volume' column

**Output:** Add to `AUDIT_FINDINGS_PHASE2.md`

---

### Task 2.3: Velocity Gate (L9) ✓
**File:** `cuda_modules/velocity_gate.py`

**Check:**
1. Cascade threshold: 10 points default ✓
2. Time window: 0.5 seconds default ✓
3. Handles DataFrame, ndarray, and list inputs ✓
4. CPU fallback exists ✓
5. GPU kernel logic correct ✓

**Acceptance Criteria:**
- [ ] Cascade = `abs(max-min) >= 10.0 AND time_elapsed <= 0.5s`
- [ ] Processes last 50-200 ticks (LOOKBACK optimization)
- [ ] `_cpu_detect()` matches `detect_cascade_kernel()` logic
- [ ] Timestamp conversion handles float/datetime/int

**Output:** Add to `AUDIT_FINDINGS_PHASE2.md`

---

### Task 2.4: CUDA Verification System ✓
**File:** `cuda_modules/hardened_verification.py`

**Check:**
1. 3-stage audit: Handshake → Injection → Handoff ✓
2. Stage A: GPU detection ✓
3. Stage B: CPU→GPU→CPU verification ✓
4. Stage C: GPU→GPU data passing ✓
5. Logs to CUDA_Debug.log ✓

**Acceptance Criteria:**
- [ ] run_audit() returns True on success
- [ ] Gracefully handles missing CUDA
- [ ] Uses real/synthetic data for Stage B
- [ ] Verifies deterministic results (expected vs actual)

**Output:** Add to `AUDIT_FINDINGS_PHASE2.md`

---

### Task 2.5: Cross-Module Integration ✓

**Check:**
1. LayerEngine correctly calls get_pattern_detector() ✓
2. LayerEngine correctly calls get_confirmation_engine() ✓
3. LayerEngine correctly calls get_velocity_gate() ✓
4. use_gpu flag propagates correctly ✓
5. Fallback chain works: CUDA fail → CPU ✓

**Files to check:**
- `core/layer_engine.py` (imports and calls)
- All 3 cuda_modules (singleton getters)

**Acceptance Criteria:**
- [ ] LayerEngine.__init__() accepts use_gpu parameter
- [ ] Singletons prevent multiple instances
- [ ] CPU fallback doesn't crash if CUDA unavailable
- [ ] Error messages logged clearly

**Output:** Add to `AUDIT_FINDINGS_PHASE2.md`

---

## Deliverable

**FILE TO CREATE:** `AUDIT_FINDINGS_PHASE2.md`

**Template:**
```markdown
# Phase 2 Audit Findings - CUDA Modules

## Summary
- **Files Audited:** 4
- **GPU Tests:** Can run? [YES/NO]
- **CPU Fallback:** Working? [YES/NO]
- **Issues Found:** X

## Pattern Detector (L7)
### ✓ PASS: CPU Fallback
- Logic matches GPU kernel
- Patterns detected correctly

### ✓ PASS: GPU Kernel
- Compression priority correct
- Confidence values: compression=0.85, wedge=0.75, breakdown=0.90

### ⚠ ISSUE: [If any]
...

## Confirmation (L8)
### ✓ PASS: Volume Detection
- Threshold: 1.2x mean
- Requires L7 pattern

### ⚠ ISSUE: [If any]
...

## Velocity Gate (L9)
### ✓ PASS: Cascade Detection
- Threshold: 10 points / 0.5 sec
- Handles multiple input types

### ⚠ ISSUE: [If any]
...

## Verification System
### ✓ PASS: 3-Stage Audit
- Stage A: Handshake works
- Stage B: Injection verified
- Stage C: Handoff confirmed

## Integration
### ✓ PASS: LayerEngine Calls
- All 3 modules imported correctly
- Singleton pattern working
- use_gpu flag propagates

## Recommendations
1. [If any]

## Next Steps
- Proceed to Phase 3: Data Pipeline Audit
```

---

## Git Commit

```bash
git add AUDIT_FINDINGS_PHASE2.md
git commit -m "audit: Phase 2 - CUDA modules validation complete

Audited:
- Pattern Detector (L7): CPU/GPU logic
- Confirmation (L8): Volume spike detection
- Velocity Gate (L9): Cascade detection
- Verification: 3-stage audit system

Findings: [X issues found]
Status: PASS/NEEDS_FIX"

git push
```

---

## Notes for Jules
- Can run code if CUDA available (optional)
- If no GPU: Audit CPU fallback only
- Focus on logic correctness, not performance
- Check determinism: same input → same output
- Time box: 40 minutes
