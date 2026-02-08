# CURRENT STATUS REPORT

### 1. METADATA
- **Timestamp:** 2026-02-08 17:09:38
- **Git Branch:** HEAD
- **Last Commit:** a7f9f56463fd7cbaac243260fa4b1e3c4aa4c60f
- **Build Status:** (See GitHub Actions Badge)

### 1A. ARCHITECTURE STATUS
- **Current State:** TRANSITIONAL (Dual Architecture)
- **Active Engine:** 9-Layer Hierarchy (Legacy)
- **Experimental Engine:** Fractal Three-Body Quantum (Inactive)
- **Details:** See `AUDIT_REPORT.md`

### 2. CHANGELOG
#### Last 10 Commits
```
a7f9f56 - Merge a0024c275cbd1c4f998f82147c45c452feb20a59 into 63006a05a7ac51372e385cf5fc01c6150a489aec (reyses)
a0024c2 - fix(core): handle missing open column in quantum engine (google-labs-jules[bot])
63006a0 - docs: auto-update CURRENT_STATUS.md [skip ci] (github-actions[bot])
eb0fd38 - Initual run after mayor changes (reyses)
b783d28 - initial run after mayor changes (reyses)
d61045e - docs: auto-update CURRENT_STATUS.md [skip ci] (github-actions[bot])
88101bc - Merge pull request #70 from reyses/feat/orchestrator-integration-dashboard-15158105287559044499 (reyses)
c9c576c - docs: auto-update CURRENT_STATUS.md [skip ci] (github-actions[bot])
ff7a6dd - Merge 824b6584c4b1257a52fc6e946ae0dc5fdaac550d into d638a91c2744369e88904e5327eb12a34443a626 (reyses)
824b658 - chore: Verify dependencies in requirements.txt (google-labs-jules[bot])
```

### 3. FILE STRUCTURE
```
Bayesian-AI/
│   ├── AGENTS.md
│   ├── CUDA_Debug.log.processed_20260208_170934
│   ├── CURRENT_STATUS.md
│   ├── ISSUE_TRIAGE.md
│   ├── README.md
│   ├── requirements.txt
│   ├── AUDIT/
│   │   ├── INTEGRATION_GUIDE.md
│   │   ├── JULES_AUDIT_MASTER_INDEX.md
│   │   ├── JULES_PHASE1_CORE_AUDIT.md
│   │   ├── JULES_PHASE2_CUDA_AUDIT.md
│   │   ├── JULES_PHASE3_DATA_AUDIT.md
│   │   ├── JULES_PHASE4_TRAINING_AUDIT.md
│   │   ├── JULES_PHASE5_TEST_AUDIT.md
│   │   ├── JULES_PHASE6_FIXES.md
│   ├── DATA/
│   │   ├── RAW/
│   │   │   ├── condition.json
│   │   │   ├── glbx-mdp3-20250730.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250731.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250801.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250803.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250804.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250805.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250806.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250807.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250808.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250810.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250811.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250812.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250813.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250814.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250815.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250817.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250818.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250819.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250820.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250821.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250822.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250824.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250825.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250826.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250827.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250828.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250829.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250831.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250901.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250902.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250903.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250904.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250905.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250907.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250908.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250909.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250910.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250911.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250912.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250914.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250915.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250916.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250917.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250918.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250919.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250921.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250922.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250923.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250924.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250925.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250926.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250928.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250929.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250930.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251001.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251002.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251003.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251005.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251006.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251007.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251008.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251009.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251010.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251012.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251013.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251014.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251015.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251016.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251017.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251019.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251020.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251021.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251022.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251023.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251024.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251026.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251027.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251028.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251029.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251030.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251031.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251102.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251103.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251104.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251105.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251106.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251107.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251109.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251110.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251111.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251112.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251113.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251114.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251116.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251117.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251118.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251119.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251120.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251121.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251123.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251124.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251125.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251126.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251127.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251128.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251130.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251201.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251202.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251203.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251204.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251205.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251207.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251208.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251209.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251210.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251211.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251212.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251214.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251215.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251216.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251217.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251218.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251219.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251221.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251222.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251223.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251224.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251225.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251226.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251228.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251229.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251230-20260129.ohlcv-1s.dbn.zst
│   │   │   ├── glbx-mdp3-20251230.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251231.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260101.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260102.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260104.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260105.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260106.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260107.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260108.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260109.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260111.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260112.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260113.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260114.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260115.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260116.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260118.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260119.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260120.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260121.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260122.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260123.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260125.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260126.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260127.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260128.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20260129.trades.0000.dbn.zst
│   │   │   ├── manifest.json
│   │   │   ├── metadata.json
│   │   │   ├── ohlcv-1s.parquet
│   │   │   ├── symbology.json
│   │   │   ├── trades.parquet
│   ├── config/
│   │   ├── __init__.py [COMPLETE]
│   │   ├── settings.py [COMPLETE]
│   │   ├── symbols.py [COMPLETE]
│   │   ├── workflow_manifest.json
│   ├── core/
│   │   ├── __init__.py [COMPLETE]
│   │   ├── adaptive_confidence.py [COMPLETE]
│   │   ├── bayesian_brain.py [COMPLETE]
│   │   ├── data_aggregator.py [COMPLETE]
│   │   ├── engine_core.py [COMPLETE]
│   │   ├── exploration_mode.py [COMPLETE]
│   │   ├── fractal_three_body.py [COMPLETE]
│   │   ├── layer_engine.py [COMPLETE]
│   │   ├── logger.py [COMPLETE]
│   │   ├── quantum_field_engine.py [COMPLETE]
│   │   ├── resonance_cascade.py [COMPLETE]
│   │   ├── state_vector.py [COMPLETE]
│   │   ├── three_body_state.py [COMPLETE]
│   │   ├── unconstrained_explorer.py [COMPLETE]
│   ├── cuda_modules/
│   │   ├── __init__.py [COMPLETE]
│   │   ├── confirmation.py [COMPLETE]
│   │   ├── hardened_verification.py [COMPLETE]
│   │   ├── pattern_detector.py [COMPLETE]
│   │   ├── velocity_gate.py [COMPLETE]
│   ├── debug_outputs/
│   │   ├── test_phase0.log
│   ├── docs/
│   │   ├── CHANGELOG.md
│   │   ├── DASHBOARD_GUIDE.md
│   │   ├── README_DASHBOARD.md
│   │   ├── TECHNICAL_MANUAL.md
│   ├── execution/
│   │   ├── __init__.py [COMPLETE]
│   │   ├── wave_rider.py [TESTED]
│   ├── notebooks/
│   │   ├── CUDA_Debug.log
│   │   ├── dashboard.ipynb
│   │   ├── debug_outputs/
│   ├── scripts/
│   │   ├── build_executable.py [COMPLETE]
│   │   ├── generate_dashboard.py [COMPLETE]
│   │   ├── generate_status_report.py [WIP]
│   │   ├── inspect_results.py [COMPLETE]
│   │   ├── manifest_integrity_check.py [COMPLETE]
│   │   ├── sentinel_bridge.py [COMPLETE]
│   │   ├── setup_test_data.py [TESTED]
│   │   ├── verify_environment.py [COMPLETE]
│   ├── tests/
│   │   ├── glbx-mdp3-20250730.trades.0000.dbn.zst
│   │   ├── math_verify.py [COMPLETE]
│   │   ├── test_cuda_confirmation.py [TESTED]
│   │   ├── test_cuda_imports_and_init.py [TESTED]
│   │   ├── test_cuda_pattern.py [TESTED]
│   │   ├── test_dashboard_controls.py [TESTED]
│   │   ├── test_dashboard_metrics.py [TESTED]
│   │   ├── test_databento_loading.py [TESTED]
│   │   ├── test_doe.py [TESTED]
│   │   ├── test_full_system.py [TESTED]
│   │   ├── test_phase0.py [TESTED]
│   │   ├── test_phase1.py [TESTED]
│   │   ├── test_phase2.py [TESTED]
│   │   ├── test_quantum_system.py [TESTED]
│   │   ├── test_real_data_velocity.py [TESTED]
│   │   ├── test_training_validation.py [TESTED]
│   │   ├── test_wave_rider.py [TESTED]
│   │   ├── topic_build.py [COMPLETE]
│   │   ├── topic_diagnostics.py [COMPLETE]
│   │   ├── topic_math.py [COMPLETE]
│   │   ├── utils.py [COMPLETE]
│   │   ├── Testing DATA/
│   │   │   ├── glbx-mdp3-20250730.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250731.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250801.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250803.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251230-20260129.ohlcv-1s.dbn.zst
│   ├── training/
│   │   ├── __init__.py [COMPLETE]
│   │   ├── cuda_backtest.py [TESTED]
│   │   ├── databento_loader.py [COMPLETE]
│   │   ├── orchestrator.py [COMPLETE]
│   │   ├── test_progress_display.py [TESTED]
│   │   ├── training_progress.json
│   ├── visualization/
│   │   ├── __init__.py [COMPLETE]
│   │   ├── live_training_dashboard.py [COMPLETE]
│   │   ├── visualization_module.py [COMPLETE]

```

### 4. CODE STATISTICS
- **Python Files:** 60
- **Total Lines of Code:** 8161

### 5. CRITICAL INTEGRATION POINTS
- **Databento API:**
- API_KEY: NO
- DatabentoLoader: YES
- **Training Connection:**
- DatabentoLoader: YES
- pd.read_parquet: YES

### 6. DEPENDENCIES
#### requirements.txt
```
aiohappyeyeballs==2.6.1
aiohttp==3.13.3
aiosignal==1.4.0
attrs==25.4.0
certifi==2026.1.4
charset-normalizer==3.4.4
databento==0.70.0
databento-dbn==0.48.0
frozenlist==1.8.0
idna==3.11
multidict==6.7.1
numpy==1.26.4
pandas==2.2.3
propcache==0.4.1
pyarrow==23.0.0
python-dateutil==2.9.0.post0
requests==2.32.5
six==1.17.0
urllib3==2.6.3
yarl==1.22.0
zstandard==0.25.0
numba==0.63.1
numba-cuda
llvmlite==0.46.0
pyinstaller==6.18.0
pytest
jupyter
plotly
ipywidgets
tqdm
scipy
matplotlib
colorama

```
- **Installation:** `pip install -r requirements.txt`

### 7. EXECUTION READINESS
- **Entry Point:** `python -m core.engine_core`
- **Exists:** YES
- **Expected Runtime:** Long-running process (Server/Loop)

### 8. CODE VALIDATION CHECKLIST
#### bayesian_brain.py
- Laplace: YES
- save: YES
- load: YES

#### layer_engine.py
- L1: YES
- L9: YES
- CUDA: YES

#### orchestrator.py
- DOE: NO
- grid: NO
- Walk-forward: NO
- Monte Carlo: NO
- iterations: YES

### 9. TESTING STATUS
- **Tests Directory:** YES
- **Test Files Count:** 18

### 10. FILES MODIFIED (Last Commit)
```

```

### 11. REVIEWER CHECKLIST
- [ ] Architectural Review
- [ ] Potential Bugs
- [ ] Missing Features
- [ ] Performance Concerns

### 12. LOGIC CORE VALIDATION

- **Status:** PASS
- **Command:** `pytest tests/topic_math.py`
- **Summary:** 4 passed in 0.12s


### 13. TRAINING VALIDATION METRICS
| Metric | Value | Status |
| :--- | :--- | :--- |
| Training Status | SUCCESS | ✓ |
| Iterations Completed | 10 | ✓ |
| Runtime | 2.53s | - |
| Data Files Tested | 1 | ✓ |
| Total Ticks (Sample) | 0 | - |
| Unique States Learned | 10 | - |
| High-Confidence States (80%+) | 0 | ✓ |

**Top 5 States by Probability (Sample):**
None

### 14. DOE OPTIMIZATION STATUS
- [ ] Parameter Grid Generator
- [ ] Latin Hypercube Sampling
- [ ] ANOVA Analysis Module
- [ ] Walk-Forward Test Harness
- [ ] Monte Carlo Bootstrap
- [ ] Response Surface Optimizer

**Current Status:** NOT IMPLEMENTED
**Estimated Implementation Time:** 1-2 weeks
**Priority:** HIGH (required for statistical validation)

QC VALIDATION SNAPSHOT
======================

Topic 1: Executable Build
PASS: All 16 manifest files exist.
PASS: All 17 modules imported successfully.
PASS: OPERATIONAL_MODE is valid: LEARNING

Topic 2: Math and Logic
PASS: Logic Core verified

Topic 3: Diagnostics
PASS: Required files found in DATA/RAW

Manifest Integrity
PASS: Manifest Integrity Check Passed
