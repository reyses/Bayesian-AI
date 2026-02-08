# JULES TASK: Phase 3 - Data Pipeline Audit

## Objective
Validate data loading, aggregation, and preprocessing for correctness.

## Scope
**FILES TO AUDIT:**
- `training/databento_loader.py`
- `core/data_aggregator.py`
- `training/orchestrator.py` (data loading portions)

**Time Estimate:** 25-35 minutes

## Tasks

### Task 3.1: Databento Loader ✓
**File:** `training/databento_loader.py`

**Check:**
1. Loads .dbn and .dbn.zst files ✓
2. Loads .parquet files ✓
3. Standardizes columns: timestamp, price, volume, type ✓
4. Handles missing columns gracefully ✓
5. Converts timestamp to float seconds ✓

**Acceptance Criteria:**
- [ ] Returns DataFrame with required columns: ['timestamp', 'price', 'volume', 'type']
- [ ] Timestamp is float (seconds since epoch)
- [ ] filter_trades=True removes non-trade events
- [ ] Raises FileNotFoundError if file missing
- [ ] Preserves OHLC columns if present

**Output:** Document in `AUDIT_FINDINGS_PHASE3.md`

---

### Task 3.2: Data Aggregator - Ring Buffer ✓
**File:** `core/data_aggregator.py`

**Check:**
1. Ring buffer implemented correctly ✓
2. Max ticks enforced (default 10,000) ✓
3. add_tick() handles missing data ✓
4. Timestamp forward-fill if missing ✓
5. Prevents buffer overflow ✓

**Acceptance Criteria:**
- [ ] Buffer wraps at max_ticks (index = (idx+1) % max_ticks)
- [ ] _get_ordered_data() returns chronological order
- [ ] Handles None/NaN values safely
- [ ] Forward-fills price if missing

**Output:** Add to `AUDIT_FINDINGS_PHASE3.md`

---

### Task 3.3: Data Aggregator - Resampling ✓
**File:** `core/data_aggregator.py`

**Check:**
1. get_current_data() returns all timeframes ✓
2. Resamples to: 15s, 5m, 15m, 1h, 4hr ✓
3. OHLC aggregation correct ✓
4. Volume summation correct ✓
5. Handles empty DataFrame gracefully ✓

**Acceptance Criteria:**
- [ ] Returns dict with keys: 'price', 'timestamp', 'ticks', 'bars_15s', 'bars_5m', 'bars_15m', 'bars_1h', 'bars_4hr'
- [ ] Resampling uses correct rules: '15s', '5min', '15min', '1h', '4h'
- [ ] Returns None for bars if insufficient data
- [ ] Caches DataFrame to avoid repeated conversion

**Output:** Add to `AUDIT_FINDINGS_PHASE3.md`

---

### Task 3.4: Orchestrator - Data Loading ✓
**File:** `training/orchestrator.py`

**Check:**
1. get_data_source() handles .dbn and .parquet ✓
2. load_data_from_directory() finds all files ✓
3. Concatenates multiple files correctly ✓
4. TrainingOrchestrator accepts data parameter ✓
5. Initializes engine with data ✓

**Acceptance Criteria:**
- [ ] Supports extensions: .dbn, .dbn.zst, .parquet
- [ ] Raises ValueError if unsupported format
- [ ] load_data_from_directory() scans recursively
- [ ] Concatenates with ignore_index=True
- [ ] TrainingOrchestrator.data is DataFrame

**Output:** Add to `AUDIT_FINDINGS_PHASE3.md`

---

### Task 3.5: Data Flow Validation ✓

**Check end-to-end flow:**
1. File → DatabentoLoader → DataFrame ✓
2. DataFrame → DataAggregator → Ring Buffer ✓
3. Ring Buffer → get_current_data() → Dict ✓
4. Dict → LayerEngine → StateVector ✓
5. StateVector → BayesianBrain → Learning ✓

**Trace through:**
- `training/orchestrator.py` line ~550: `get_data_source()`
- `core/data_aggregator.py` line ~50: `add_tick()`
- `core/layer_engine.py` line ~150: `compute_current_state()`

**Acceptance Criteria:**
- [ ] Data types consistent at each stage
- [ ] No data loss in transformations
- [ ] Timestamps preserved correctly
- [ ] Column names standardized

**Output:** Add to `AUDIT_FINDINGS_PHASE3.md`

---

## Deliverable

**FILE TO CREATE:** `AUDIT_FINDINGS_PHASE3.md`

**Template:**
```markdown
# Phase 3 Audit Findings - Data Pipeline

## Summary
- **Files Audited:** 3
- **Data Flow:** END_TO_END_TRACED
- **Issues Found:** X

## Databento Loader
### ✓ PASS: File Format Support
- .dbn, .dbn.zst, .parquet all supported
- Standardizes to ['timestamp', 'price', 'volume', 'type']

### ✓ PASS: Column Handling
- Missing columns handled gracefully
- Timestamp converted to float seconds

### ⚠ ISSUE: [If any]
...

## Data Aggregator
### ✓ PASS: Ring Buffer
- Max 10,000 ticks enforced
- Wraps correctly at limit
- Forward-fills missing data

### ✓ PASS: Resampling
- All timeframes generated: 15s, 5m, 15m, 1h, 4hr
- OHLC aggregation correct
- Volume summation correct

### ⚠ ISSUE: [If any]
...

## Orchestrator
### ✓ PASS: Data Loading
- Multi-file concatenation works
- Directory scanning finds all files
- Error handling for missing files

## Data Flow
### ✓ PASS: End-to-End
File → Loader → DataFrame → Aggregator → Dict → LayerEngine → StateVector

### Data Type Trace:
1. File (bytes) → DataFrame (timestamp: float, price: float)
2. DataFrame → add_tick() → Ring buffer (numpy arrays)
3. Ring buffer → get_current_data() → Dict (ticks: DataFrame, bars: DataFrame)
4. Dict → compute_current_state() → StateVector (frozen dataclass)

## Recommendations
1. [If any optimizations]

## Next Steps
- Proceed to Phase 4: Training Loop Audit
```

---

## Git Commit

```bash
git add AUDIT_FINDINGS_PHASE3.md
git commit -m "audit: Phase 3 - Data pipeline validation complete

Audited:
- Databento Loader: File format support
- Data Aggregator: Ring buffer & resampling
- Orchestrator: Multi-file loading
- End-to-end data flow traced

Findings: [X issues found]
Status: PASS/NEEDS_FIX"

git push
```

---

## Notes for Jules
- Can test with sample data if available
- Trace data types at each stage
- Check for data loss or corruption
- Verify timestamp handling carefully
- Time box: 35 minutes
