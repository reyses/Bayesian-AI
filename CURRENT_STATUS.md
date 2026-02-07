# CURRENT STATUS REPORT

### 1. METADATA
- **Timestamp:** 2026-02-07 14:51:10
- **Git Branch:** jules-3382033469114581180-8936eecb
- **Last Commit:** b1e01d0923ee71a9e944439479a083c14ed10afa
- **Build Status:** (See GitHub Actions Badge)

### 1A. ARCHITECTURE STATUS
- **Current State:** TRANSITIONAL (Dual Architecture)
- **Active Engine:** 9-Layer Hierarchy (Legacy)
- **Experimental Engine:** Fractal Three-Body Quantum (Inactive)
- **Details:** See `AUDIT_REPORT.md`

### 2. CHANGELOG
#### Last 10 Commits
```
b1e01d0 - Merge branch 'main' of https://github.com/reyses/Bayesian-AI (reyses)
```

### 3. FILE STRUCTURE
```
Bayesian-AI/
│   ├── AGENTS.md
│   ├── AUDIT_REPORT.md
│   ├── COMPLETE_IMPLEMENTATION_SPEC.md
│   ├── CUDA_Debug.log
│   ├── CURRENT_STATUS.md
│   ├── README.md
│   ├── Training Orchestrator.txt
│   ├── __init__.py [COMPLETE]
│   ├── engine_core.py [COMPLETE]
│   ├── nconstrained Exploration.txt
│   ├── requirements.txt
│   ├── requirements_dashboard.txt
│   ├── requirements_notebook.txt
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
│   │   ├── probability_table.pkl
│   │   ├── test_phase0.log
│   ├── docs/
│   │   ├── CHANGELOG.md
│   │   ├── LEARNING_DASHBOARD_GUIDE.md
│   │   ├── README_DASHBOARD.md
│   │   ├── TECHNICAL_MANUAL.md
│   │   ├── archive/
│   │   │   ├── JULES_OUTPUT_SNAPSHOT.txt
│   │   │   ├── PHASE1_COMPLETE.md
│   │   │   ├── UNIFIED_MASTER_DIRECTIVE.md
│   │   │   ├── all_requirements.txt
│   │   │   ├── project_update.txt
│   ├── execution/
│   │   ├── __init__.py [COMPLETE]
│   │   ├── wave_rider.py [COMPLETE]
│   ├── notebooks/
│   │   ├── CUDA_Debug.log
│   │   ├── master_dashboard.ipynb
│   │   ├── debug_outputs/
│   ├── scripts/
│   │   ├── build_executable.py [COMPLETE]
│   │   ├── generate_master_dashboard.py [COMPLETE]
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
│   │   ├── test_databento_loading.py [TESTED]
│   │   ├── test_doe.py [TESTED]
│   │   ├── test_full_system.py [TESTED]
│   │   ├── test_phase0.py [TESTED]
│   │   ├── test_phase1.py [TESTED]
│   │   ├── test_phase2.py [TESTED]
│   │   ├── test_quantum_system.py [TESTED]
│   │   ├── test_real_data_velocity.py [TESTED]
│   │   ├── test_training_validation.py [TESTED]
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
│   │   ├── training_progress.json
│   ├── visualization/
│   │   ├── __init__.py [COMPLETE]
│   │   ├── live_training_dashboard.py [COMPLETE]
│   │   ├── visualization_module.py [COMPLETE]

```

### 4. CODE STATISTICS
- **Python Files:** 57
- **Total Lines of Code:** 7804

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

```
- **Installation:** `pip install -r requirements.txt`

### 7. EXECUTION READINESS
- **Entry Point:** `python engine_core.py`
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
- **Test Files Count:** 15

### 10. FILES MODIFIED (Last Commit)
```
A	.Jules/palette.md
A	.github/workflows/unified_test_pipeline.yml
A	.gitignore
A	AGENTS.md
A	AUDIT_REPORT.md
A	COMPLETE_IMPLEMENTATION_SPEC.md
A	CUDA_Debug.log
A	CURRENT_STATUS.md
A	DATA/RAW/condition.json
A	DATA/RAW/glbx-mdp3-20250730.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250731.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250801.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250803.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250804.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250805.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250806.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250807.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250808.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250810.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250811.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250812.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250813.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250814.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250815.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250817.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250818.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250819.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250820.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250821.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250822.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250824.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250825.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250826.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250827.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250828.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250829.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250831.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250901.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250902.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250903.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250904.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250905.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250907.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250908.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250909.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250910.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250911.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250912.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250914.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250915.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250916.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250917.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250918.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250919.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250921.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250922.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250923.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250924.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250925.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250926.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250928.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250929.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20250930.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251001.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251002.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251003.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251005.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251006.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251007.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251008.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251009.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251010.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251012.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251013.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251014.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251015.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251016.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251017.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251019.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251020.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251021.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251022.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251023.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251024.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251026.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251027.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251028.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251029.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251030.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251031.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251102.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251103.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251104.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251105.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251106.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251107.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251109.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251110.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251111.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251112.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251113.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251114.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251116.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251117.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251118.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251119.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251120.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251121.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251123.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251124.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251125.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251126.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251127.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251128.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251130.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251201.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251202.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251203.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251204.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251205.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251207.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251208.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251209.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251210.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251211.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251212.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251214.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251215.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251216.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251217.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251218.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251219.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251221.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251222.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251223.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251224.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251225.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251226.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251228.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251229.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251230-20260129.ohlcv-1s.dbn.zst
A	DATA/RAW/glbx-mdp3-20251230.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20251231.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260101.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260102.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260104.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260105.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260106.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260107.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260108.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260109.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260111.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260112.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260113.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260114.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260115.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260116.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260118.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260119.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260120.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260121.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260122.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260123.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260125.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260126.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260127.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260128.trades.0000.dbn.zst
A	DATA/RAW/glbx-mdp3-20260129.trades.0000.dbn.zst
A	DATA/RAW/manifest.json
A	DATA/RAW/metadata.json
A	DATA/RAW/ohlcv-1s.parquet
A	DATA/RAW/symbology.json
A	DATA/RAW/trades.parquet
A	README.md
A	Training Orchestrator.txt
A	__init__.py
A	config/__init__.py
A	config/settings.py
A	config/symbols.py
A	config/workflow_manifest.json
A	core/__init__.py
A	core/adaptive_confidence.py
A	core/bayesian_brain.py
A	core/data_aggregator.py
A	core/exploration_mode.py
A	core/fractal_three_body.py
A	core/layer_engine.py
A	core/logger.py
A	core/quantum_field_engine.py
A	core/resonance_cascade.py
A	core/state_vector.py
A	core/three_body_state.py
A	core/unconstrained_explorer.py
A	cuda_modules/__init__.py
A	cuda_modules/confirmation.py
A	cuda_modules/hardened_verification.py
A	cuda_modules/pattern_detector.py
A	cuda_modules/velocity_gate.py
A	debug_outputs/probability_table.pkl
A	debug_outputs/test_phase0.log
A	docs/CHANGELOG.md
A	docs/LEARNING_DASHBOARD_GUIDE.md
A	docs/README_DASHBOARD.md
A	docs/TECHNICAL_MANUAL.md
A	docs/archive/JULES_OUTPUT_SNAPSHOT.txt
A	docs/archive/PHASE1_COMPLETE.md
A	docs/archive/UNIFIED_MASTER_DIRECTIVE.md
A	docs/archive/all_requirements.txt
A	docs/archive/project_update.txt
A	engine_core.py
A	execution/__init__.py
A	execution/wave_rider.py
A	nconstrained Exploration.txt
A	notebooks/CUDA_Debug.log
A	notebooks/debug_dashboard.ipynb
A	notebooks/debug_outputs/mini_run/probability_table.pkl
A	notebooks/learning_dashboard.ipynb
A	requirements.txt
A	requirements_dashboard.txt
A	requirements_notebook.txt
A	scripts/build_executable.py
A	scripts/generate_debug_dashboard.py
A	scripts/generate_learning_dashboard.py
A	scripts/generate_status_report.py
A	scripts/inspect_results.py
A	scripts/manifest_integrity_check.py
A	scripts/sentinel_bridge.py
A	scripts/setup_test_data.py
A	scripts/verify_environment.py
A	tests/Testing DATA/glbx-mdp3-20250730.trades.0000.dbn.zst
A	tests/Testing DATA/glbx-mdp3-20250731.trades.0000.dbn.zst
A	tests/Testing DATA/glbx-mdp3-20250801.trades.0000.dbn.zst
A	tests/Testing DATA/glbx-mdp3-20250803.trades.0000.dbn.zst
A	tests/Testing DATA/glbx-mdp3-20251230-20260129.ohlcv-1s.dbn.zst
A	tests/glbx-mdp3-20250730.trades.0000.dbn.zst
A	tests/math_verify.py
A	tests/test_cuda_confirmation.py
A	tests/test_cuda_imports_and_init.py
A	tests/test_cuda_pattern.py
A	tests/test_databento_loading.py
A	tests/test_doe.py
A	tests/test_full_system.py
A	tests/test_phase0.py
A	tests/test_phase1.py
A	tests/test_phase2.py
A	tests/test_quantum_system.py
A	tests/test_real_data_velocity.py
A	tests/test_training_validation.py
A	tests/topic_build.py
A	tests/topic_diagnostics.py
A	tests/topic_math.py
A	tests/utils.py
A	training/__init__.py
A	training/cuda_backtest.py
A	training/databento_loader.py
A	training/orchestrator.py
A	training/training_progress.json
A	visualization/__init__.py
A	visualization/live_training_dashboard.py
A	visualization/visualization_module.py
```

### 11. REVIEWER CHECKLIST
- [ ] Architectural Review
- [ ] Potential Bugs
- [ ] Missing Features
- [ ] Performance Concerns

### 12. LOGIC CORE VALIDATION

- **Status:** PASS
- **Command:** `pytest tests/topic_math.py`
- **Summary:** 4 passed in 0.23s


### 13. TRAINING VALIDATION METRICS
| Metric | Value | Status |
| :--- | :--- | :--- |
| Training Status | SUCCESS | ✓ |
| Iterations Completed | 10 | ✓ |
| Runtime | 6.51s | - |
| Data Files Tested | 1 | ✓ |
| Total Ticks (Sample) | 5 | - |
| Unique States Learned | 0 | - |
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
