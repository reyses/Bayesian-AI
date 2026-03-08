# System Audit - 2026-03-08

## Previous Audit Status
- Checked `AUDIT/OLD` directory.
- **Result:** No previous audit was found in the repository.

## Functionality Evaluation
- Executed core scripts:
  - `tools/standalone_research.py --help`
  - `live/launcher.py --help`
- **Result:** Missing `numpy` and other packages on first run. After running `pip install -r requirements.txt`, `tools/standalone_research.py` executed successfully.
- **Issue Discovered:** There was a namespace collision between `live/config.py` and the root `config/` directory causing `live/launcher.py` to crash.
- **Resolution during Audit:** Renamed `live/config.py` to `live/live_config.py`, updated imports across the `live/` module, and created an empty `config/__init__.py`. After these fixes, `live/launcher.py --help` successfully executes.
- **Overall Functionality:** Core systems run without immediate dependency or syntax failures once the environment is properly set up.

## File Structure Cleanliness
- Overall, directories are reasonably defined (`core/`, `tools/`, `live/`, `training/`).
- `tools/` directory contains many scripts but lacks internal categorization for specific tool types, aside from `tools/research/` and `tools/plots/`.
- Some standalone plots exist under `tools/plots/standalone/`, which seems appropriate given memory context ("standalone plot scripts in `tools/plots/`").
- Standalone logs and outputs are placed in appropriate folders like `run_logs/`, `reports/`, and `DATA/`.

## Debug Files Location
- The rule in memory states: *"Debug scripts in `scripts/debug/` and standalone plot scripts in `tools/plots/` rather than cluttering the root or `tools/` directory."*
- Currently, there is **no `scripts/debug/` folder**.
- Some files inside `tools/` function as debugging or analytics scripts (e.g., `tools/checkpoint_viewer.py`, `tools/gate_interaction_matrix.py`, `tools/analyze_gates.py`, `tools/analyze_wrong_dir.py`). These might need to be evaluated and categorized as pure debug scripts or moved accordingly.

---

## Prompt for Jules (Execution Items)

**To Jules (the AI Agent):**
Please execute the following improvements based on this audit:

1. **Move Debug Scripts:** Review the files currently in the `tools/` directory (e.g., `checkpoint_viewer.py`, `analyze_wrong_dir.py`, `analyze_gates.py`, `gate_interaction_matrix.py`). Move the ones that act purely as debug scripts into a new `scripts/debug/` directory to adhere to the memory guideline. Ensure any import paths or shell execution references are updated if necessary.
2. **Review Codebase Cleanup:** The root directory has some stray `.txt` files (e.g., in `reports/live/`). If any are debug artifacts, move them to the proper debug output folder.
3. **Commit Instructions:** After completing these refactors, commit the changes using the `submit` tool, documenting the cleanup of the `live/config.py` collision and the directory restructurings.
