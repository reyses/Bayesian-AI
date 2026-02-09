# Archive - Old Code

**Date Archived:** February 9, 2026
**Reason:** Consolidation to single-entry-point system

## Files Archived

### old_core/ (5 files)
- `exploration_mode.py` - Replaced by context_detector.py + DOE parameter generation
- `unconstrained_explorer.py` - Replaced by DOE systematic exploration strategies
- `engine_core.py` - Lower-level engine superseded by quantum_field_engine.py
- `fractal_three_body.py` - Theoretical implementation covered in quantum_field_engine.py
- `resonance_cascade.py` - Specific detector covered in context_detector.py

### old_training/ (3 files)
- `cuda_backtest.py` - CUDA-specific backtest (not used in walk-forward)
- `run_optimizer.py` - Old optimizer superseded by DOE parameter generator
- `test_progress_display.py` - Testing script no longer needed

### old_scripts/ (8 files)
- `build_executable.py`
- `generate_dashboard.py` - Superseded by live_training_dashboard.py
- `generate_status_report.py` - Superseded by enhanced reporting
- `inspect_results.py` - Superseded by pattern analyzer
- `manifest_integrity_check.py`
- `sentinel_bridge.py`
- `setup_test_data.py`
- `verify_environment.py`

**Total Archived:** 16 files

## Status

These files are kept for reference only. They can be safely deleted after March 1, 2026 if no issues are discovered with the new consolidated system.

## Restoration

If needed, files can be restored with:
```bash
cp archive/old_core/[file].py core/
```

---

**New System:** All functionality consolidated into `training/orchestrator.py` as single entry point.
