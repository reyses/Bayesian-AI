# System Audit — 2026-03-14

---

## 1. Previous Audit Verification

The previous audits `DOC_STATUS_AUDIT (1).md` and `DOC_STATUS_AUDIT (2).md` were verified and successfully archived into `AUDIT/OLD/`.

From `DOC_STATUS_AUDIT (2).md`, the following tasks were defined but still require action from Jules:
- **`CLAUDE.md` and `AGENTS.md`**: Need to be created in the repository root (they were missing).
- **`MEMORY.md`**: Requires a surgical update to correct file paths (e.g., `quantum_field_engine.py` → `statistical_field_engine.py`) while preserving the historical timeline of changes.
- **Documentation Overhaul**: `ARCHITECTURE.md` must be regenerated from the actual codebase. `SYSTEM_DESCRIPTION.md` needs a rewrite to purge the physics metaphors and use statistical language. `CHANGELOG.md` needs an update with the V3.0 section. `ROADMAP.md` needs to be updated with completed sections and active research.
- **Tracker Cleanup**: A large volume (~533MB) of generated files (`tools/plots/`, `reports/is/*.csv`, etc.) need to be ignored and untracked.

---

## 2. File Structure Cleanliness

A full traversal of the repository found several files that are misplaced or require structural cleanup.

### Misplaced Files:
- **`tools/research_pattern_audit.py`**: A research tool script that should be organized under `scripts/research/` or `tools/research/` instead of cluttering the root `tools/` directory.
- **`scripts/research/`**: Contains various analyses scripts (e.g., `analysis_ff_conviction_audit.py`). These are correctly placed per the research spec. However, some tools in `tools/` like `research_belief_flip.py`, `research_golden_physics.py`, `research_signal_fire.py`, and `research_tf_mixing.py` still reside directly under `tools/` and should likely be moved.

### Missing Directories:
- **`scripts/debug/`**: A proper debugging folder didn't exist, though guidelines suggest all debug scripts belong there. I've created the `scripts/debug/` directory to facilitate proper placement.
- **`debug_outputs/`**: Not found in the root directory structure, though the logger uses it for output storage. The system should ensure `debug_outputs/` is properly created and `.gitignore`d.

---

## 3. Functionality & Debug Files

No actual active debug files were found directly cluttering the repository root or other structural folders. However:
- We expect `run_logs/` and `debug_outputs/` to accumulate operational output. Right now, `run_logs/` exists and contains a `.gitkeep`, which is correct.
- Scripts in `scripts/debug/` are non-existent. Any future debug scripts need to be strictly placed here.

---

## PROMPT FOR JULES: Actionable Improvements

Jules, please execute the following improvements based on this audit:

1. **Create Root Agent Files**: Create `CLAUDE.md` and `AGENTS.md` in the root repository exactly as prescribed in `DOC_STATUS_AUDIT (2).md`.
2. **Surgical MEMORY Update**: Update `docs/memory/MEMORY.md` to fix outdated file paths/class names while appending (not overwriting) historical context and new implemented features.
3. **Repository Tracker Cleanup**: Add generated tracking artifacts (`tools/plots/`, `reports/is/*.csv`, `reports/oos/*.csv`, `reports/is/shards/`) to `.gitignore`, and untrack them using `git rm --cached`.
4. **Tool Directory Cleanup**: Move misplaced research scripts (e.g., `tools/research_pattern_audit.py`, `tools/research_belief_flip.py`, etc.) to the appropriate `scripts/research/` or `tools/research/` subdirectory.
5. **Update Documentation**: Regenerate `ARCHITECTURE.md`, rewrite `SYSTEM_DESCRIPTION.md` to remove physics metaphors, and update `CHANGELOG.md` (V3.0/V6.0/V7.0) and `ROADMAP.md` to reflect the current state of the project.
6. **Ensure Debug Setup**: Verify the `scripts/debug/` and `debug_outputs/` folders exist and are appropriately configured (e.g., ignored if needed).