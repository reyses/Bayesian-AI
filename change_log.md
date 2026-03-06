
### Changes Implemented (Phase 1: Unified Exit Engine)
- Created `core/exit_engine.py` following the spec to unify exit logic between training and live execution.
- Updated `training/orchestrator.py` to utilize `ExitEngine` for entering and exiting positions, replacing fragmented logic like `check_stops_hilo` and custom trailing stops.
- Updated `live/live_engine.py` to similarly use `ExitEngine` for all exit evaluations.

### Changes Implemented (Phase 3: Oracle Direction Learning)
- Modified `training/orchestrator.py` to add a direction correction accumulator at the end of the forward pass.
- Added signed MFE features into the regression step to better gauge directional profitability and output to `pattern_library.pkl`.
- Updated direction determination in `training/orchestrator.py` and `live/live_engine.py` to rely on the MFE regression model and brain direction-specific win rates.
- Enabled saving the learned directions to `pattern_forward_brain.pkl` and added logic in `live/live_engine.py` to prioritize loading this checkpoint if it exists.
