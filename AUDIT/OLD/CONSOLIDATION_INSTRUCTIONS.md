# CODEBASE CONSOLIDATION INSTRUCTIONS

## Mission

You've built a lot of new components. Now consolidate them into a clean, production-ready system with a single entry point and no obsolete code.

## Current State Assessment Needed

**Before refactoring, audit:**

1. **List all files in the project** - what exists now?
2. **Identify duplicates** - same functionality in multiple places?
3. **Find obsolete files** - old versions that are superseded?
4. **Map dependencies** - what imports what?

**Then report back with:**
- Files to keep (core system)
- Files to consolidate (merge functionality)
- Files to obsolete (move to archive/)
- Missing components (need to build)

## Consolidation Objectives

### Objective 1: Single Entry Point

**Goal:** User runs ONE file that does everything

**Target structure:**
```
python orchestrator.py
```

**Should:**
1. Load configuration
2. Initialize all components (brain, engine, detector, generator)
3. Launch live visualization dashboard
4. Run walk-forward training
5. Save checkpoints
6. Display results

**Current problem:** 
- `orchestrator.py` exists but is superseded
- `walk_forward_trainer.py` was created separately
- User doesn't know which to run

**Required:** 
- Merge walk-forward logic INTO orchestrator.py
- Make orchestrator.py the master runner
- Deprecate/remove walk_forward_trainer.py as standalone

### Objective 2: Clean Module Structure

**Organize by responsibility:**

```
core/                           # Core trading logic
├── state_vector.py            [KEEP] State definition
├── bayesian_brain.py          [KEEP] Probability learning
├── quantum_field_engine.py    [KEEP] Layer computation
├── context_detector.py        [KEEP] Context identification
└── adaptive_confidence.py     [KEEP] Phase management

training/
├── doe_parameter_generator.py [KEEP] Parameter generation
├── orchestrator.py            [REFACTOR] Single entry point
└── [DELETE walk_forward_trainer.py - merge into orchestrator]

execution/
├── integrated_statistical_system.py [KEEP] Validation
└── wave_rider.py              [KEEP] Position management

config/
└── symbols.py                 [KEEP] Asset profiles

visualization/                  # NEW FOLDER
└── live_dashboard.py          [CREATE] Real-time charts

checkpoints/                    # Runtime data
archive/                        # OLD CODE
└── orchestrator_old.py        [MOVE old version here]
```

### Objective 3: Eliminate Redundancy

**Check for duplicated functionality:**

1. **Statistical validation** - is it in multiple places?
   - `integrated_statistical_system.py`
   - `bayesian_brain.should_fire_validated()`
   - Consolidate if duplicated

2. **Data loading** - multiple versions?
   - Ensure ONE function handles Parquet loading
   - Remove any duplicates

3. **Checkpoint management** - scattered across files?
   - Centralize save/load logic
   - Single checkpoint format

4. **Progress tracking** - different implementations?
   - Standardize on tqdm + file logging
   - Remove any console spam code

### Objective 4: Add Missing Component - Live Visualization

**File to create:** `visualization/live_dashboard.py`

**Requirements:**
- Opens automatically when orchestrator starts
- Shows 2x2 grid: P&L, Win Rate, Sharpe, Current Metrics
- Updates after each day completes
- Uses matplotlib interactive mode (plt.ion())
- Non-blocking (training continues while charts update)
- Stays open when training completes

**Integration:**
```python
# In orchestrator.py
from visualization.live_dashboard import LiveDashboard

def main():
    # Initialize dashboard
    dashboard = LiveDashboard()
    dashboard.launch()
    
    # Run training
    for day in days:
        results = train_day(day)
        dashboard.update(results)  # Update charts
    
    # Keep dashboard open
    dashboard.finalize()
```

## Refactoring Plan

### Phase 1: Audit
1. List all .py files in project
2. Map dependencies (what imports what)
3. Identify duplicates and conflicts
4. Create consolidation plan

### Phase 2: Archive Old Code
1. Create `archive/` folder
2. Move superseded files to archive
3. Document what was archived and why

### Phase 3: Consolidate
1. Merge walk_forward_trainer.py logic into orchestrator.py
2. Consolidate any duplicated validation logic
3. Centralize checkpoint management
4. Remove redundant functions

### Phase 4: Add Visualization
1. Create `visualization/` folder
2. Build `live_dashboard.py`
3. Integrate into orchestrator.py
4. Test dashboard opens and updates

### Phase 5: Clean Up
1. Update all imports to reflect new structure
2. Remove unused imports
3. Add docstrings to all public functions
4. Run tests to verify nothing broke

## Success Criteria

**When done:**

✅ User can run: `python orchestrator.py`
✅ Dashboard opens automatically
✅ Training runs day-by-day
✅ Charts update in real-time
✅ No obsolete files in main directories
✅ No duplicated functionality
✅ All imports work correctly
✅ Clear folder structure by responsibility

## Critical Rules

**DO:**
- Keep all working components (brain, engine, detector, etc.)
- Preserve checkpoint data if it exists
- Test after each consolidation step
- Document what you changed

**DON'T:**
- Delete anything before moving to archive/
- Break existing functionality
- Change core algorithms (StateVector, BayesianBrain logic)
- Add new dependencies without checking

## Report Format

After auditing, provide:

```markdown
## AUDIT RESULTS

### Files Identified: XX total

**Core Components (Keep):**
- core/state_vector.py ✓
- core/bayesian_brain.py ✓
- ...

**To Consolidate:**
- training/orchestrator.py + training/walk_forward_trainer.py
  → Merge into single orchestrator.py
  → Reason: Duplicated training loop logic

**To Archive:**
- training/orchestrator_old.py
  → Reason: Superseded by new walk-forward implementation

**Missing Components:**
- visualization/live_dashboard.py
  → Reason: Required for real-time feedback

### Consolidation Plan:
1. Step 1...
2. Step 2...
3. ...

### Risk Assessment:
- Low risk: Moving files to archive
- Medium risk: Merging orchestrator + walk_forward_trainer
- High risk: (none identified)
```

## Start Here

1. **Run the audit first** - don't refactor until you know what exists
2. **Report the audit results** - let user review before proceeding
3. **Get approval** - wait for user confirmation
4. **Execute consolidation** - follow the approved plan
5. **Test** - verify everything still works

**Do NOT immediately start refactoring.** Audit and report first.

---

**USER WANTS:** Clean, consolidated system with single entry point and live visualization before running any tests.
