# CURRENT STATUS REPORT

### 1. METADATA
- **Timestamp:** 2026-02-24 08:26:50
- **Git Branch:** jules-4696302846440702408-7a10b972
- **Last Commit:** 907e4323a682b6c6bf3aa004bd97726d6237ccfb
- **Build Status:** (See GitHub Actions Badge)

### 1A. ARCHITECTURE STATUS
- **Current State:** UNKNOWN
- **Active Engine:** 9-Layer Hierarchy (Legacy)
- **Experimental Engine:** Fractal Three-Body Quantum (Inactive)
- **Details:** See `AUDIT_REPORT.md`

### 2. CHANGELOG
#### Last 10 Commits
```
907e432 - Merge pull request #214 from reyses/jules-shape-clustering-cst-4999145118411150632 (reyses)
```

### 3. FILE STRUCTURE
```
Bayesian-AI/
│   ├── =3.5.0
│   ├── AGENTS.md
│   ├── CURRENT_STATUS.md
│   ├── README.md
│   ├── requirements.txt
│   ├── run_test_workflow.py [TESTED]
│   ├── AUDIT/
│   │   ├── AUDIT_REPORT_2026_02_23.md
│   │   ├── OLD/
│   │   │   ├── ADD_PROGRESS_UPDATES regret.md
│   │   │   ├── ADD_PROGRESS_UPDATES.md
│   │   │   ├── AUDIT_FINDINGS_PHASE1.md
│   │   │   ├── AUDIT_REPORT_2025_02_16.md
│   │   │   ├── AUDIT_REPORT_2026_02_10.md
│   │   │   ├── AUDIT_REPORT_2026_02_12.md
│   │   │   ├── AUDIT_REPORT_2026_02_16.md
│   │   │   ├── AUDIT_REPORT_2026_02_19.md
│   │   │   ├── AUDIT_REPORT_2026_02_20.md
│   │   │   ├── AUDIT_REPORT_2026_02_21.md
│   │   │   ├── AUDIT_REPORT_2026_02_22.md
│   │   │   ├── CONSOLIDATION_AUDIT_REPORT.md
│   │   │   ├── CONSOLIDATION_COMPLETE.md
│   │   │   ├── CONSOLIDATION_INSTRUCTIONS.md
│   │   │   ├── CUDA_AI_READINESS_REPORT.md
│   │   │   ├── CUDA_INTEGRATION_REPORT.md
│   │   │   ├── EXECUTE_CONSOLIDATION_NOW.md
│   │   │   ├── EXPERT_LOGIC_REVIEW_2026.md
│   │   │   ├── FULL_CASCADE_ARCHITECTURE.md
│   │   │   ├── ISSUE_TRIAGE.md
│   │   │   ├── JULES_INDICATORS.md
│   │   │   ├── JULES_PHASE4.md
│   │   │   ├── MASTER_CONTEXT_VS_CODE.md
│   │   │   ├── SYSTEM_ANALYSIS_REPORT.md
│   │   │   ├── SYSTEM_OVERHAUL_SUMMARY.md
│   ├── DATA/
│   │   ├── RAW/
│   │   │   ├── ohlcv-1s.dbn.zst
│   │   │   ├── ohlcv-1s.parquet
│   │   │   ├── trades.dbn.zst
│   │   │   ├── trades.parquet
│   ├── archive/
│   │   ├── README.md
│   │   ├── __init__.py [COMPLETE]
│   │   ├── layer_engine.py [COMPLETE]
│   │   ├── orchestrator_pre_consolidation.py [COMPLETE]
│   │   ├── training_progress.json
│   │   ├── walk_forward_trainer_merged.py [COMPLETE]
│   │   ├── cuda_modules/
│   │   │   ├── __init__.py [COMPLETE]
│   │   │   ├── confirmation.py [COMPLETE]
│   │   │   ├── hardened_verification.py [COMPLETE]
│   │   │   ├── pattern_detector.py [COMPLETE]
│   │   │   ├── velocity_gate.py [COMPLETE]
│   │   ├── old_core/
│   │   │   ├── engine_core.py [COMPLETE]
│   │   │   ├── exploration_mode.py [TESTED]
│   │   │   ├── fractal_three_body.py [COMPLETE]
│   │   │   ├── resonance_cascade.py [COMPLETE]
│   │   │   ├── unconstrained_explorer.py [COMPLETE]
│   │   ├── old_scripts/
│   │   │   ├── build_executable.py [COMPLETE]
│   │   │   ├── generate_dashboard.py [COMPLETE]
│   │   │   ├── generate_status_report.py [WIP]
│   │   │   ├── inspect_results.py [COMPLETE]
│   │   │   ├── manifest_integrity_check.py [COMPLETE]
│   │   │   ├── sentinel_bridge.py [COMPLETE]
│   │   │   ├── setup_test_data.py [TESTED]
│   │   │   ├── verify_environment.py [COMPLETE]
│   │   ├── old_training/
│   │   │   ├── cuda_backtest.py [TESTED]
│   │   │   ├── run_optimizer.py [COMPLETE]
│   │   │   ├── test_progress_display.py [TESTED]
│   ├── config/
│   │   ├── __init__.py [COMPLETE]
│   │   ├── oracle_config.py [COMPLETE]
│   │   ├── settings.py [COMPLETE]
│   │   ├── symbols.py [COMPLETE]
│   │   ├── workflow_manifest.json
│   ├── core/
│   │   ├── __init__.py [COMPLETE]
│   │   ├── adaptive_confidence.py [COMPLETE]
│   │   ├── bayesian_brain.py [TESTED]
│   │   ├── context_detector.py [COMPLETE]
│   │   ├── cuda_pattern_detector.py [COMPLETE]
│   │   ├── cuda_physics.py [TESTED]
│   │   ├── data_aggregator.py [COMPLETE]
│   │   ├── dynamic_binner.py [TESTED]
│   │   ├── exploration_mode.py [TESTED]
│   │   ├── logger.py [COMPLETE]
│   │   ├── multi_timeframe_context.py [COMPLETE]
│   │   ├── pattern_utils.py [COMPLETE]
│   │   ├── physics_utils.py [COMPLETE]
│   │   ├── quantum_field_engine.py [TESTED]
│   │   ├── risk_engine.py [COMPLETE]
│   │   ├── state_vector.py [TESTED]
│   │   ├── three_body_state.py [COMPLETE]
│   ├── docs/
│   │   ├── ARCHITECTURE.md
│   │   ├── CHANGELOG.md
│   │   ├── JULES_AUTO_DOCS.md
│   │   ├── JULES_SIGNAL_CAPTURE_AUDIT.md
│   │   ├── JULES_SNOWFLAKE_BASELINE.md
│   │   ├── JULES_SPECTRAL_GATES.md
│   │   ├── JULES_TASK_1_2_IMPLEMENTATION.md
│   │   ├── JULES_TEMPLATE_TIMESCALE.md
│   │   ├── SYSTEM_STATUS.md
│   │   ├── OLD/
│   │   │   ├── COMPLETE_IMPLEMENTATION_SPEC.md
│   │   │   ├── FRACTAL_PATTERN_RECOGNITION_SPEC.md
│   │   │   ├── JULES_5S_1S_WORKERS.md
│   │   │   ├── JULES_CPU_PHYSICS.md
│   │   │   ├── JULES_DYNAMIC_EXIT.md
│   │   │   ├── JULES_FRACTAL_DNA_TREE.md
│   │   │   ├── JULES_GOLDEN_PATH_ORACLE.md
│   │   │   ├── JULES_IMPLEMENTATION_DETAILS.md
│   │   │   ├── JULES_INSTRUCTIONS.md
│   │   │   ├── JULES_MONTE_CARLO.md
│   │   │   ├── JULES_NUMBA_SIM.md
│   │   │   ├── JULES_ORACLE_ENGINE.md
│   │   │   ├── JULES_PID_OPTIMIZER.md
│   │   │   ├── JULES_PID_OSCILLATION.md
│   │   │   ├── JULES_SHAPE_CLUSTERING.md
│   │   │   ├── JULES_SNOWFLAKE_CLUSTERS.md
│   │   │   ├── New logic.txt
│   │   │   ├── README_DASHBOARD.md
│   │   │   ├── TECHNICAL_MANUAL.md
│   │   │   ├── THE NIGHTMARE FIELD EQUATIOn.pdf
│   │   │   ├── Training Orchestrator.txt
│   │   │   ├── Unconstrained_Exploration.txt
│   │   │   ├── evaluation_legacy_pattern_detector.md
│   ├── models/
│   │   ├── quantum_probability_table.pkl
│   ├── notebooks/
│   │   ├── dashboard.ipynb
│   │   ├── debug_outputs/
│   │   ├── models/
│   ├── run_logs/
│   │   ├── fn_signal_log_2025_01.csv
│   │   ├── fn_signal_log_2025_02.csv
│   │   ├── fn_signal_log_2025_03.csv
│   │   ├── fn_signal_log_2025_04.csv
│   │   ├── fn_signal_log_2025_05.csv
│   │   ├── fn_signal_log_2025_06.csv
│   │   ├── fn_signal_log_2025_07.csv
│   │   ├── fn_signal_log_2025_08.csv
│   │   ├── fn_signal_log_2025_09.csv
│   │   ├── fn_signal_log_2025_10.csv
│   │   ├── oracle_trade_log_2025_01.csv
│   │   ├── oracle_trade_log_2025_02.csv
│   │   ├── oracle_trade_log_2025_03.csv
│   │   ├── oracle_trade_log_2025_04.csv
│   │   ├── oracle_trade_log_2025_05.csv
│   │   ├── oracle_trade_log_2025_06.csv
│   │   ├── oracle_trade_log_2025_07.csv
│   │   ├── oracle_trade_log_2025_08.csv
│   │   ├── oracle_trade_log_2025_09.csv
│   │   ├── oracle_trade_log_2025_10.csv
│   │   ├── pid_signal_log_2025_01.csv
│   │   ├── pid_signal_log_2025_02.csv
│   │   ├── pid_signal_log_2025_03.csv
│   │   ├── pid_signal_log_2025_04.csv
│   │   ├── pid_signal_log_2025_05.csv
│   │   ├── pid_signal_log_2025_06.csv
│   │   ├── pid_signal_log_2025_07.csv
│   │   ├── pid_signal_log_2025_08.csv
│   │   ├── pid_signal_log_2025_09.csv
│   │   ├── pid_signal_log_2025_10.csv
│   │   ├── signal_log_2025_01.csv
│   │   ├── signal_log_2025_02.csv
│   │   ├── signal_log_2025_03.csv
│   │   ├── signal_log_2025_04.csv
│   │   ├── signal_log_2025_05.csv
│   │   ├── signal_log_2025_06.csv
│   │   ├── signal_log_2025_07.csv
│   │   ├── signal_log_2025_08.csv
│   │   ├── signal_log_2025_09.csv
│   │   ├── signal_log_2025_10.csv
│   │   ├── trade_analytics.txt
│   ├── scripts/
│   │   ├── benchmark_regression.py [COMPLETE]
│   │   ├── build_executable.py [COMPLETE]
│   │   ├── fix_cuda.py [COMPLETE]
│   │   ├── generate_status_report.py [WIP]
│   │   ├── gpu_health_check.py [COMPLETE]
│   │   ├── manifest_integrity_check.py [COMPLETE]
│   │   ├── sentinel_bridge.py [COMPLETE]
│   │   ├── setup_oos_atlas.py [COMPLETE]
│   │   ├── setup_test_data.py [TESTED]
│   │   ├── split_atlas_to_daily.py [COMPLETE]
│   │   ├── verify_clustering.py [COMPLETE]
│   │   ├── verify_cuda_readiness.py [COMPLETE]
│   │   ├── debug/
│   │   │   ├── benchmark_extract_features.py [COMPLETE]
│   │   │   ├── debug_databento.py [COMPLETE]
│   │   │   ├── debug_utils.py [COMPLETE]
│   │   │   ├── verify_databento_loader.py [COMPLETE]
│   ├── tests/
│   │   ├── conftest.py [TESTED]
│   │   ├── math_verify.py [COMPLETE]
│   │   ├── test_batch_regret_analyzer.py [TESTED]
│   │   ├── test_bayesian_brain.py [TESTED]
│   │   ├── test_clustering_integration.py [TESTED]
│   │   ├── test_cpu_physics.py [TESTED]
│   │   ├── test_cuda_imports_and_init.py [TESTED]
│   │   ├── test_cuda_pattern.py [TESTED]
│   │   ├── test_cuda_physics.py [TESTED]
│   │   ├── test_databento_loading.py [TESTED]
│   │   ├── test_doe_features.py [TESTED]
│   │   ├── test_doe_regret.py [TESTED]
│   │   ├── test_dynamic_binner.py [TESTED]
│   │   ├── test_exploration_mode.py [TESTED]
│   │   ├── test_fractal_atlas.py [TESTED]
│   │   ├── test_fractal_dashboard.py [TESTED]
│   │   ├── test_integration_quantum.py [TESTED]
│   │   ├── test_legacy_layer_engine.py [TESTED]
│   │   ├── test_multi_timeframe_doe.py [TESTED]
│   │   ├── test_pattern_recognition.py [TESTED]
│   │   ├── test_performance_optimizations.py [TESTED]
│   │   ├── test_phase0.py [TESTED]
│   │   ├── test_pid_analyzer.py [TESTED]
│   │   ├── test_quantum_field_engine.py [TESTED]
│   │   ├── test_real_data_velocity.py [TESTED]
│   │   ├── test_risk_engine_optimization.py [TESTED]
│   │   ├── test_state_vector.py [TESTED]
│   │   ├── test_timeframe_belief_network.py [TESTED]
│   │   ├── test_torch_pattern.py [TESTED]
│   │   ├── test_training_validation.py [TESTED]
│   │   ├── test_wave_rider.py [TESTED]
│   │   ├── test_wave_rider_playbook.py [TESTED]
│   │   ├── topic_build.py [COMPLETE]
│   │   ├── topic_diagnostics.py [COMPLETE]
│   │   ├── topic_math.py [COMPLETE]
│   │   ├── utils.py [COMPLETE]
│   │   ├── verify_regret_fallback.py [COMPLETE]
│   │   ├── Testing DATA/
│   │   │   ├── glbx-mdp3-20250730.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250731.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250801.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20250803.trades.0000.dbn.zst
│   │   │   ├── glbx-mdp3-20251230-20260129.ohlcv-1s.dbn.zst
│   ├── training/
│   │   ├── STOP
│   │   ├── __init__.py [COMPLETE]
│   │   ├── anova_analyzer.py [COMPLETE]
│   │   ├── batch_regret_analyzer.py [TESTED]
│   │   ├── cuda_kmeans.py [COMPLETE]
│   │   ├── data_loading_optimizer.py [COMPLETE]
│   │   ├── databento_loader.py [COMPLETE]
│   │   ├── dbn_to_parquet.py [COMPLETE]
│   │   ├── doe_parameter_generator.py [COMPLETE]
│   │   ├── fractal_atlas_builder.py [COMPLETE]
│   │   ├── fractal_clustering.py [COMPLETE]
│   │   ├── fractal_discovery_agent.py [COMPLETE]
│   │   ├── fractal_dna_tree.py [COMPLETE]
│   │   ├── integrated_statistical_system.py [COMPLETE]
│   │   ├── monte_carlo_engine.py [COMPLETE]
│   │   ├── orchestrator.py [COMPLETE]
│   │   ├── orchestrator_worker.py [COMPLETE]
│   │   ├── pattern_analyzer.py [COMPLETE]
│   │   ├── pid_oscillation_analyzer.py [COMPLETE]
│   │   ├── pipeline_checkpoint.py [COMPLETE]
│   │   ├── progress_reporter.py [COMPLETE]
│   │   ├── thompson_refiner.py [COMPLETE]
│   │   ├── timeframe_belief_network.py [TESTED]
│   │   ├── trade_analytics.py [COMPLETE]
│   │   ├── wave_rider.py [TESTED]
│   ├── visualization/
│   │   ├── __init__.py [COMPLETE]
│   │   ├── live_training_dashboard.py [COMPLETE]
│   │   ├── visualization_module.py [COMPLETE]

```

### 4. CODE STATISTICS
- **Python Files:** 127
- **Total Lines of Code:** 30311

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
--extra-index-url https://download.pytorch.org/whl/cu121
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
numpy<2
pandas>=2.2.0
propcache==0.4.1
pyarrow==23.0.0
python-dateutil==2.9.0.post0
requests==2.32.5
six==1.17.0
urllib3==2.6.3
yarl==1.22.0
zstandard==0.25.0
numba==0.61.2
llvmlite==0.44.0
pyinstaller==6.18.0
pytest
plotly
tqdm
scipy==1.13.1
matplotlib>=3.8.0
colorama
anywidget
torch>=2.1.0,<2.6
git+https://github.com/MerlinR/Pandas-ta-fork.git#egg=pandas_ta
hurst
QuantLib
scikit-learn
optuna>=3.5.0

```
- **Installation:** `pip install -r requirements.txt`

### 7. EXECUTION READINESS
- **Entry Point:** `python -m core.engine_core`
- **Exists:** NO
- **Expected Runtime:** Long-running process (Server/Loop)

### 8. CODE VALIDATION CHECKLIST
#### bayesian_brain.py
- Laplace: NO
- save: YES
- load: YES

#### layer_engine.py
File core/layer_engine.py not found

#### orchestrator.py
- DOE: YES
- grid: YES
- Walk-forward: YES
- Monte Carlo: YES
- iterations: YES

### 9. TESTING STATUS
- **Tests Directory:** YES
- **Test Files Count:** 33

### 10. FILES MODIFIED (Last Commit)
```
A	.Jules/bolt.md
A	.Jules/palette.md
A	.claude/settings.json
A	.claude/settings.local.json
A	.github/workflows/github_workflows_data_preprocessing.yml.disabled
A	.github/workflows/github_workflows_parallel_preprocessing.yml.disabled
A	.github/workflows/unified_test_pipeline.yml.disabled
A	.gitignore
A	.jules/bolt.md
A	.vscode/settings.json
A	=3.5.0
A	AGENTS.md
A	AUDIT/AUDIT_REPORT_2026_02_23.md
A	AUDIT/OLD/ADD_PROGRESS_UPDATES regret.md
A	AUDIT/OLD/ADD_PROGRESS_UPDATES.md
A	AUDIT/OLD/AUDIT_FINDINGS_PHASE1.md
A	AUDIT/OLD/AUDIT_REPORT_2025_02_16.md
A	AUDIT/OLD/AUDIT_REPORT_2026_02_10.md
A	AUDIT/OLD/AUDIT_REPORT_2026_02_12.md
A	AUDIT/OLD/AUDIT_REPORT_2026_02_16.md
A	AUDIT/OLD/AUDIT_REPORT_2026_02_19.md
A	AUDIT/OLD/AUDIT_REPORT_2026_02_20.md
A	AUDIT/OLD/AUDIT_REPORT_2026_02_21.md
A	AUDIT/OLD/AUDIT_REPORT_2026_02_22.md
A	AUDIT/OLD/CONSOLIDATION_AUDIT_REPORT.md
A	AUDIT/OLD/CONSOLIDATION_COMPLETE.md
A	AUDIT/OLD/CONSOLIDATION_INSTRUCTIONS.md
A	AUDIT/OLD/CUDA_AI_READINESS_REPORT.md
A	AUDIT/OLD/CUDA_INTEGRATION_REPORT.md
A	AUDIT/OLD/EXECUTE_CONSOLIDATION_NOW.md
A	AUDIT/OLD/EXPERT_LOGIC_REVIEW_2026.md
A	AUDIT/OLD/FULL_CASCADE_ARCHITECTURE.md
A	AUDIT/OLD/ISSUE_TRIAGE.md
A	AUDIT/OLD/JULES_INDICATORS.md
A	AUDIT/OLD/JULES_PHASE4.md
A	AUDIT/OLD/MASTER_CONTEXT_VS_CODE.md
A	AUDIT/OLD/SYSTEM_ANALYSIS_REPORT.md
A	AUDIT/OLD/SYSTEM_OVERHAUL_SUMMARY.md
A	CURRENT_STATUS.md
A	README.md
A	archive/README.md
A	archive/__init__.py
A	archive/cuda_modules/__init__.py
A	archive/cuda_modules/confirmation.py
A	archive/cuda_modules/hardened_verification.py
A	archive/cuda_modules/pattern_detector.py
A	archive/cuda_modules/velocity_gate.py
A	archive/layer_engine.py
A	archive/old_core/engine_core.py
A	archive/old_core/exploration_mode.py
A	archive/old_core/fractal_three_body.py
A	archive/old_core/resonance_cascade.py
A	archive/old_core/unconstrained_explorer.py
A	archive/old_scripts/build_executable.py
A	archive/old_scripts/generate_dashboard.py
A	archive/old_scripts/generate_status_report.py
A	archive/old_scripts/inspect_results.py
A	archive/old_scripts/manifest_integrity_check.py
A	archive/old_scripts/sentinel_bridge.py
A	archive/old_scripts/setup_test_data.py
A	archive/old_scripts/verify_environment.py
A	archive/old_training/cuda_backtest.py
A	archive/old_training/run_optimizer.py
A	archive/old_training/test_progress_display.py
A	archive/orchestrator_pre_consolidation.py
A	archive/training_progress.json
A	archive/walk_forward_trainer_merged.py
A	config/__init__.py
A	config/oracle_config.py
A	config/settings.py
A	config/symbols.py
A	config/workflow_manifest.json
A	core/__init__.py
A	core/adaptive_confidence.py
A	core/bayesian_brain.py
A	core/context_detector.py
A	core/cuda_pattern_detector.py
A	core/cuda_physics.py
A	core/data_aggregator.py
A	core/dynamic_binner.py
A	core/exploration_mode.py
A	core/logger.py
A	core/multi_timeframe_context.py
A	core/pattern_utils.py
A	core/physics_utils.py
A	core/quantum_field_engine.py
A	core/risk_engine.py
A	core/state_vector.py
A	core/three_body_state.py
A	docs/ARCHITECTURE.md
A	docs/CHANGELOG.md
A	docs/JULES_AUTO_DOCS.md
A	docs/JULES_SIGNAL_CAPTURE_AUDIT.md
A	docs/JULES_SNOWFLAKE_BASELINE.md
A	docs/JULES_SPECTRAL_GATES.md
A	docs/JULES_TASK_1_2_IMPLEMENTATION.md
A	docs/JULES_TEMPLATE_TIMESCALE.md
A	docs/OLD/COMPLETE_IMPLEMENTATION_SPEC.md
A	docs/OLD/FRACTAL_PATTERN_RECOGNITION_SPEC.md
A	docs/OLD/JULES_5S_1S_WORKERS.md
A	docs/OLD/JULES_CPU_PHYSICS.md
A	docs/OLD/JULES_DYNAMIC_EXIT.md
A	docs/OLD/JULES_FRACTAL_DNA_TREE.md
A	docs/OLD/JULES_GOLDEN_PATH_ORACLE.md
A	docs/OLD/JULES_IMPLEMENTATION_DETAILS.md
A	docs/OLD/JULES_INSTRUCTIONS.md
A	docs/OLD/JULES_MONTE_CARLO.md
A	docs/OLD/JULES_NUMBA_SIM.md
A	docs/OLD/JULES_ORACLE_ENGINE.md
A	docs/OLD/JULES_PID_OPTIMIZER.md
A	docs/OLD/JULES_PID_OSCILLATION.md
A	docs/OLD/JULES_SHAPE_CLUSTERING.md
A	docs/OLD/JULES_SNOWFLAKE_CLUSTERS.md
A	docs/OLD/New logic.txt
A	docs/OLD/README_DASHBOARD.md
A	docs/OLD/TECHNICAL_MANUAL.md
A	docs/OLD/THE NIGHTMARE FIELD EQUATIOn.pdf
A	docs/OLD/Training Orchestrator.txt
A	docs/OLD/Unconstrained_Exploration.txt
A	docs/OLD/evaluation_legacy_pattern_detector.md
A	docs/SYSTEM_STATUS.md
A	models/quantum_probability_table.pkl
A	notebooks/dashboard.ipynb
A	notebooks/debug_outputs/mini_run/probability_table.pkl
A	notebooks/models/production_learning/quantum_probability_table.pkl
A	requirements.txt
A	run_logs/.gitkeep
A	run_logs/fn_signal_log_2025_01.csv
A	run_logs/fn_signal_log_2025_02.csv
A	run_logs/fn_signal_log_2025_03.csv
A	run_logs/fn_signal_log_2025_04.csv
A	run_logs/fn_signal_log_2025_05.csv
A	run_logs/fn_signal_log_2025_06.csv
A	run_logs/fn_signal_log_2025_07.csv
A	run_logs/fn_signal_log_2025_08.csv
A	run_logs/fn_signal_log_2025_09.csv
A	run_logs/fn_signal_log_2025_10.csv
A	run_logs/oracle_trade_log_2025_01.csv
A	run_logs/oracle_trade_log_2025_02.csv
A	run_logs/oracle_trade_log_2025_03.csv
A	run_logs/oracle_trade_log_2025_04.csv
A	run_logs/oracle_trade_log_2025_05.csv
A	run_logs/oracle_trade_log_2025_06.csv
A	run_logs/oracle_trade_log_2025_07.csv
A	run_logs/oracle_trade_log_2025_08.csv
A	run_logs/oracle_trade_log_2025_09.csv
A	run_logs/oracle_trade_log_2025_10.csv
A	run_logs/pid_signal_log_2025_01.csv
A	run_logs/pid_signal_log_2025_02.csv
A	run_logs/pid_signal_log_2025_03.csv
A	run_logs/pid_signal_log_2025_04.csv
A	run_logs/pid_signal_log_2025_05.csv
A	run_logs/pid_signal_log_2025_06.csv
A	run_logs/pid_signal_log_2025_07.csv
A	run_logs/pid_signal_log_2025_08.csv
A	run_logs/pid_signal_log_2025_09.csv
A	run_logs/pid_signal_log_2025_10.csv
A	run_logs/signal_log_2025_01.csv
A	run_logs/signal_log_2025_02.csv
A	run_logs/signal_log_2025_03.csv
A	run_logs/signal_log_2025_04.csv
A	run_logs/signal_log_2025_05.csv
A	run_logs/signal_log_2025_06.csv
A	run_logs/signal_log_2025_07.csv
A	run_logs/signal_log_2025_08.csv
A	run_logs/signal_log_2025_09.csv
A	run_logs/signal_log_2025_10.csv
A	run_logs/trade_analytics.txt
A	run_test_workflow.py
A	scripts/benchmark_regression.py
A	scripts/build_executable.py
A	scripts/debug/benchmark_extract_features.py
A	scripts/debug/debug_databento.py
A	scripts/debug/debug_utils.py
A	scripts/debug/verify_databento_loader.py
A	scripts/fix_cuda.py
A	scripts/generate_status_report.py
A	scripts/gpu_health_check.py
A	scripts/manifest_integrity_check.py
A	scripts/sentinel_bridge.py
A	scripts/setup_oos_atlas.py
A	scripts/setup_test_data.py
A	scripts/split_atlas_to_daily.py
A	scripts/verify_clustering.py
A	scripts/verify_cuda_readiness.py
A	tests/Testing DATA/glbx-mdp3-20250730.trades.0000.dbn.zst
A	tests/Testing DATA/glbx-mdp3-20250731.trades.0000.dbn.zst
A	tests/Testing DATA/glbx-mdp3-20250801.trades.0000.dbn.zst
A	tests/Testing DATA/glbx-mdp3-20250803.trades.0000.dbn.zst
A	tests/Testing DATA/glbx-mdp3-20251230-20260129.ohlcv-1s.dbn.zst
A	tests/conftest.py
A	tests/math_verify.py
A	tests/test_batch_regret_analyzer.py
A	tests/test_bayesian_brain.py
A	tests/test_clustering_integration.py
A	tests/test_cpu_physics.py
A	tests/test_cuda_imports_and_init.py
A	tests/test_cuda_pattern.py
A	tests/test_cuda_physics.py
A	tests/test_databento_loading.py
A	tests/test_doe_features.py
A	tests/test_doe_regret.py
A	tests/test_dynamic_binner.py
A	tests/test_exploration_mode.py
A	tests/test_fractal_atlas.py
A	tests/test_fractal_dashboard.py
A	tests/test_integration_quantum.py
A	tests/test_legacy_layer_engine.py
A	tests/test_multi_timeframe_doe.py
A	tests/test_pattern_recognition.py
A	tests/test_performance_optimizations.py
A	tests/test_phase0.py
A	tests/test_pid_analyzer.py
A	tests/test_quantum_field_engine.py
A	tests/test_real_data_velocity.py
A	tests/test_risk_engine_optimization.py
A	tests/test_state_vector.py
A	tests/test_timeframe_belief_network.py
A	tests/test_torch_pattern.py
A	tests/test_training_validation.py
A	tests/test_wave_rider.py
A	tests/test_wave_rider_playbook.py
A	tests/topic_build.py
A	tests/topic_diagnostics.py
A	tests/topic_math.py
A	tests/utils.py
A	tests/verify_regret_fallback.py
A	training/STOP
A	training/__init__.py
A	training/anova_analyzer.py
A	training/batch_regret_analyzer.py
A	training/cuda_kmeans.py
A	training/data_loading_optimizer.py
A	training/databento_loader.py
A	training/dbn_to_parquet.py
A	training/doe_parameter_generator.py
A	training/fractal_atlas_builder.py
A	training/fractal_clustering.py
A	training/fractal_discovery_agent.py
A	training/fractal_dna_tree.py
A	training/integrated_statistical_system.py
A	training/monte_carlo_engine.py
A	training/orchestrator.py
A	training/orchestrator_worker.py
A	training/pattern_analyzer.py
A	training/pid_oscillation_analyzer.py
A	training/pipeline_checkpoint.py
A	training/progress_reporter.py
A	training/thompson_refiner.py
A	training/timeframe_belief_network.py
A	training/trade_analytics.py
A	training/wave_rider.py
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
- **Summary:** 4 passed in 1.03s


### 13. TRAINING VALIDATION METRICS
| Metric | Value | Status |
| :--- | :--- | :--- |
| Training Status | SKIPPED | ✗ |
| Iterations Completed | ? | ✗ |
| Runtime | ?s | - |
| Data Files Tested | 1 | ✗ |
| Total Ticks (Sample) | 0 | - |
| Unique States Learned | 0 | - |
| High-Confidence States (80%+) | 0 | ✗ |

**Top 5 States by Probability (Sample):**
None

**Error Details:**
```
CUDA not available
```
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
PASS: All 12 manifest files exist.
PASS: All 13 modules imported successfully.
PASS: OPERATIONAL_MODE is valid: LEARNING

Topic 2: Math and Logic
PASS: Logic Core verified

Topic 3: Diagnostics
PASS: Required files found in DATA/RAW

Manifest Integrity
PASS: Manifest Integrity Check Passed
