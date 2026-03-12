# Project Audit Report — 2026-03-12

## 1. Documentation & File Structure Status

### Incomplete Items from Previous Audit (2026-03-08):
- **0a. Create `CLAUDE.md`**: Completed (exists at root).
- **0b. Create `AGENTS.md`**: Completed (exists at root).
- **0c. Surgical update `docs/memory/MEMORY.md`**: Completed. `quantum_field_engine` and `ThreeBodyQuantumState` were successfully removed.
- **1. Update `README.md`**: Completed. No `physics` references remain.
- **2-5. Document Archive Restructuring**: Incomplete. Missing directories like `docs/active` and `docs/specs`, though some movement happened. The files `CLAUDE_CODE_REMAINING_WORK.md` and `RESEARCH_SPEC_V_TO_FF.md` are missing from `docs/active`.
- **6. Create `tools/archive/`**: Completed.
- **7. `.gitignore` generated files**: Completed (`reports/is/*.csv`, `reports/oos/*.csv`, `reports/training/*.csv`, `tools/plots/` are ignored).
- **8. `git rm --cached`**: Verified that generated CSV files are no longer tracked.
- **9. Fix metaphor remnants**: Completed in `tools/research/tbn_trade_aware.py`.
- **11. Create `scripts/research/`**: Completed. Scaffolding exists.

### Current File Structure Evaluation:
- **Debug Scripts Location**: The `scripts/debug/` folder exists but is empty.
- **Plotting Scripts Location**: The `tools/plots/` directory is missing. Plotting scripts (e.g., `session_overlay.py`, `trade_visualizer.py`, `visualize_template.py`) are still located in `tools/` instead of `tools/plots/`.
- **Cleanliness**: The root directory and `tools/` directory are cluttered with many one-off standalone scripts. These should be moved to appropriate subdirectories like `tools/plots/` or `scripts/debug/`.

## 2. Code Functionality & Project Rules Adherence

### `logger.py` vs `print()` Usage:
- **Violation**: The project memory explicitly requires using `core/logger.py` over raw `print()` statements across the codebase.
- **Finding**: There are currently **481 instances** of `print()` usage outside of debug and tools scripts. Notable offenders include:
  - `core/fractal_clustering.py`
  - `core/timeframe_belief_network.py`
  - `core/bayesian_brain.py`
  - `core/execution_engine.py`
  - `live/atlas_loader.py`
  - `live/launcher.py`
  - `live/history_replay.py`
  - `training/fractal_discovery_agent.py`
  - `training/anova_analyzer.py`

### Magic Numbers (`docs/JULES_MAGIC_NUMBER_REFACTOR.md`):
- **Finding**: Some magic numbers exist in `core/statistical_field_engine.py` and `core/cuda_pattern_detector.py` (e.g., `MIN_CUDA_LEN = 1024`, `threads_per_block = 256`, `self.residual_window = 500`). These should ideally be refactored into a configuration file.

## 3. Recommended Actions for Next Session

**Prompt for Jules to Execute Improvements:**

```markdown
1. Move plotting scripts (e.g., `session_overlay.py`, `trade_visualizer.py`, `visualize_template.py`) into `tools/plots/`. Ensure imports and paths are updated accordingly.
2. Refactor raw `print()` statements in `core/`, `live/`, and `training/` modules to use the central logger from `core/logger.py`. Ensure that background thread writes or high-frequency loops are properly optimized when logging.
3. Review and complete any remaining tasks from the previous audit that were missed (such as setting up the final remaining `docs/active` and `docs/specs` structure if required by ROADMAP).
4. Review hardcoded magic numbers (like `MIN_CUDA_LEN = 1024` or `threads_per_block = 256`) in `core/statistical_field_engine.py` and extract them into constants or configuration files per the refactoring spec.
```
