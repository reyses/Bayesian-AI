# Issue Triage - Fix Priority

## BLOCKER (Fix Immediately)
1. **[ISSUE-001] StateVector hash incorrectly includes metadata**
   - **Source:** JULES_PHASE6_FIXES.md / CURRENT_STATUS.md
   - **Impact:** `StateVector` objects with the same logical state but different timestamps hash differently. This breaks the `BayesianBrain` lookup, resulting in "Unique States Learned: 0".
   - **Fix:** Update `__hash__()` in `core/state_vector.py` to exclude timestamp and price.
   - **Time:** 5 minutes
   - **Files:** `core/state_vector.py`

## CRITICAL (Fix This Session)
1. **[ISSUE-002] DataAggregator ring buffer overflow**
   - **Source:** JULES_PHASE6_FIXES.md
   - **Impact:** Lack of bounds checking causes data corruption after ~10k ticks, crashing long training runs.
   - **Fix:** Add bounds checking logic in `core/data_aggregator.py`.
   - **Time:** 10 minutes
   - **Files:** `core/data_aggregator.py`

## MAJOR (Fix Soon)
1. **[ISSUE-003] Missing test for WaveRider**
   - **Source:** JULES_PHASE6_FIXES.md
   - **Impact:** The execution logic (`WaveRider`) is a critical component but currently has 0% test coverage.
   - **Fix:** Create `tests/test_wave_rider.py`.
   - **Time:** 20 minutes
   - **Files:** `tests/test_wave_rider.py` (New File)

## MINOR (Backlog)
1. **[ISSUE-004] DOE Optimization Not Implemented**
   - **Source:** CURRENT_STATUS.md
   - **Impact:** Statistical validation relies on manual iteration.