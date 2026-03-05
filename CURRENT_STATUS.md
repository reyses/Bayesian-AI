# CURRENT STATUS REPORT

### 1. METADATA
- **Timestamp:** 2026-03-05 08:48:59
- **Git Branch:** jules-4823296224407295797-1237c7c0
- **Last Commit:** e51ce901ac546fb5c73dd24837d290775e383059
- **Build Status:** (See GitHub Actions Badge)

### 1A. ARCHITECTURE STATUS
- **Current State:** UNKNOWN
- **Active Engine:** 9-Layer Hierarchy (Legacy)
- **Experimental Engine:** Fractal Three-Body Quantum (Inactive)
- **Details:** See `AUDIT_REPORT.md`

### 2. CHANGELOG
#### Last 10 Commits
```
e51ce90 - s (reyses)
```

### 3. FILE STRUCTURE
```
Bayesian-AI/
│   ├── AGENTS.md
│   ├── CURRENT_STATUS.md
│   ├── README.md
│   ├── requirements.txt
│   ├── run_test_workflow.py [TESTED]
│   ├── tmp_b64_part2.txt
│   ├── AUDIT/
│   │   ├── AUDIT_REPORT_2026_02_22.md
│   │   ├── NinjaTrader Grid 2026-03-03 02-30 PM.csv
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
│   │   ├── ATLAS_1MONTH/
│   │   ├── RAW/
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
│   │   ├── keep_awake.py [COMPLETE]
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
│   │   ├── JULES_EXECUTION_REPORT.md
│   │   ├── JULES_HYPERVOLUME_CLUSTERING.md
│   │   ├── JULES_PERFORMANCE_TARGETS.md
│   │   ├── JULES_SIGNAL_CAPTURE_AUDIT.md
│   │   ├── JULES_SPECTRAL_GATES.md
│   │   ├── JULES_WAVEFORM_SEED_INTEGRATION.md
│   │   ├── NT8_BayesianBridge.cs
│   │   ├── SYSTEM_DESCRIPTION.md
│   │   ├── OLD/
│   │   │   ├── COMPLETE_IMPLEMENTATION_SPEC.md
│   │   │   ├── FRACTAL_PATTERN_RECOGNITION_SPEC.md
│   │   │   ├── JULES_5S_1S_WORKERS.md
│   │   │   ├── JULES_CPU_PHYSICS.md
│   │   │   ├── JULES_DYNAMIC_EXIT.md
│   │   │   ├── JULES_FRACTAL_DNA_TREE.md
│   │   │   ├── JULES_GOLDEN_PATH_ORACLE.md
│   │   │   ├── JULES_INSTRUCTIONS.md
│   │   │   ├── JULES_MONTE_CARLO.md
│   │   │   ├── JULES_NUMBA_SIM.md
│   │   │   ├── JULES_ORACLE_ENGINE.md
│   │   │   ├── JULES_PID_OPTIMIZER.md
│   │   │   ├── JULES_PID_OSCILLATION.md
│   │   │   ├── JULES_SNOWFLAKE_BASELINE.md
│   │   │   ├── JULES_SNOWFLAKE_CLUSTERS.md
│   │   │   ├── New logic.txt
│   │   │   ├── README_DASHBOARD.md
│   │   │   ├── TECHNICAL_MANUAL.md
│   │   │   ├── THE NIGHTMARE FIELD EQUATIOn.pdf
│   │   │   ├── Training Orchestrator.txt
│   │   │   ├── Unconstrained_Exploration.txt
│   │   │   ├── evaluation_legacy_pattern_detector.md
│   │   ├── checkpoint_reference/
│   │   │   ├── SCHEMAS.md
│   │   │   ├── depth_analytics.txt
│   │   │   ├── depth_weights.json
│   │   │   ├── discovery_levels.json
│   │   │   ├── oos_analytics.txt
│   │   │   ├── oos_report.txt
│   │   │   ├── pipeline_state.json
│   │   │   ├── run_snapshot.json
│   │   │   ├── sample_fn_oracle_log.csv
│   │   │   ├── sample_oracle_trade_log.csv
│   │   │   ├── sample_pid_oracle_log.csv
│   │   │   ├── sample_signal_log.csv
│   │   │   ├── template_tiers.pkl
│   │   │   ├── trade_analytics.txt
│   │   ├── memory/
│   │   │   ├── MEMORY.md
│   │   │   ├── ROADMAP.md
│   │   ├── old_jules/
│   │   │   ├── JULES_SNOWFLAKE_BASELINE.md
│   │   │   ├── JULES_SPECTRAL_GATES.md
│   │   │   ├── JULES_TASK_1_2_IMPLEMENTATION.md
│   │   │   ├── JULES_TEMPLATE_TIMESCALE.md
│   │   │   ├── PLAN_PRICE_AWARE_WORKERS.md
│   ├── live/
│   │   ├── __init__.py [COMPLETE]
│   │   ├── __main__.py [COMPLETE]
│   │   ├── bar_aggregator.py [COMPLETE]
│   │   ├── config.py [COMPLETE]
│   │   ├── launcher.py [COMPLETE]
│   │   ├── live_engine.py [COMPLETE]
│   │   ├── nt8_client.py [COMPLETE]
│   │   ├── order_manager.py [COMPLETE]
│   │   ├── protocol.py [COMPLETE]
│   ├── models/
│   │   ├── quantum_probability_table.pkl
│   ├── notebooks/
│   │   ├── dashboard.ipynb
│   │   ├── debug_outputs/
│   │   ├── models/
│   ├── reports/
│   │   ├── oos_report.txt
│   │   ├── pattern_map_2025_01.png
│   │   ├── phase4_report.txt
│   │   ├── trades_2025_01.png
│   │   ├── is/
│   │   │   ├── fn_oracle_log.csv
│   │   │   ├── oracle_trade_log.csv
│   │   │   ├── pid_oracle_log.csv
│   │   │   ├── signal_log.csv
│   │   │   ├── signal_log_prev.csv
│   │   │   ├── trade_analytics.txt
│   │   ├── live/
│   │   │   ├── session_20260302_195935.txt
│   │   │   ├── session_20260302_203709.txt
│   │   │   ├── session_20260302_211659.txt
│   │   │   ├── session_20260302_212039.txt
│   │   │   ├── session_20260303_142227.txt
│   │   │   ├── session_20260303_161855.txt
│   │   │   ├── session_20260303_163639.txt
│   │   │   ├── session_20260303_164507.txt
│   │   │   ├── session_20260303_170214.txt
│   │   │   ├── session_20260303_170559.txt
│   │   │   ├── session_20260303_203004.txt
│   │   │   ├── session_20260303_204448.txt
│   │   │   ├── session_20260303_204449.txt
│   │   │   ├── session_20260303_211242.txt
│   │   │   ├── session_20260303_211243.txt
│   │   │   ├── session_20260303_215434.txt
│   │   │   ├── session_20260303_221810.txt
│   │   │   ├── session_20260303_222910.txt
│   │   │   ├── session_20260303_225751.txt
│   │   │   ├── session_20260303_232130.txt
│   │   │   ├── session_20260303_232748.txt
│   │   │   ├── session_20260304_054311.txt
│   │   │   ├── session_20260304_060333.txt
│   │   │   ├── session_20260304_061251.txt
│   │   │   ├── session_20260304_061255.txt
│   │   │   ├── session_20260304_061839.txt
│   │   │   ├── session_20260304_061842.txt
│   │   │   ├── session_20260304_064516.txt
│   │   │   ├── session_20260304_070241.txt
│   │   │   ├── session_20260304_075415.txt
│   │   │   ├── session_20260304_082425.txt
│   │   │   ├── session_20260304_082427.txt
│   │   │   ├── session_20260304_083028.txt
│   │   │   ├── session_20260304_083204.txt
│   │   │   ├── session_20260304_083428.txt
│   │   │   ├── session_20260304_091358.txt
│   │   │   ├── session_20260304_114309.txt
│   │   │   ├── session_20260304_115150.txt
│   │   │   ├── session_20260304_115153.txt
│   │   │   ├── session_20260304_115921.txt
│   │   │   ├── session_20260304_115926.txt
│   │   │   ├── session_20260304_121303.txt
│   │   │   ├── session_20260304_121356.txt
│   │   ├── oos/
│   │   │   ├── fn_oracle_log.csv
│   │   │   ├── pid_oracle_log.csv
│   │   │   ├── signal_log.csv
│   │   │   ├── signal_log_prev.csv
│   │   ├── snowflake/
│   │   │   ├── oos_report.txt
│   │   │   ├── phase4_report.txt
│   │   ├── training/
│   │   │   ├── training_log.txt
│   ├── run_logs/
│   ├── runs/
│   │   ├── 2026-02-22_pre-depth-gate/
│   │   │   ├── depth_analytics.txt
│   │   │   ├── depth_weights.json
│   │   │   ├── fn_oracle_log.csv
│   │   │   ├── oracle_trade_log.csv
│   │   │   ├── phase4_report.txt
│   │   │   ├── phase5_report.txt
│   │   │   ├── pid_oracle_log.csv
│   │   │   ├── run_snapshot.json
│   │   │   ├── signal_log_2025_Q1.csv
│   │   │   ├── signal_log_2025_Q2.csv
│   │   │   ├── signal_log_2025_Q3.csv
│   │   │   ├── signal_log_2025_Q4.csv
│   │   │   ├── training_log.txt
│   ├── scripts/
│   │   ├── benchmark_regression.py [COMPLETE]
│   │   ├── build_executable.py [COMPLETE]
│   │   ├── fix_cuda.py [COMPLETE]
│   │   ├── generate_status_report.py [WIP]
│   │   ├── gpu_health_check.py [COMPLETE]
│   │   ├── manifest_integrity_check.py [COMPLETE]
│   │   ├── monthly_pnl_chart.py [COMPLETE]
│   │   ├── sentinel_bridge.py [COMPLETE]
│   │   ├── setup_oos_atlas.py [COMPLETE]
│   │   ├── setup_test_data.py [TESTED]
│   │   ├── split_atlas_to_daily.py [COMPLETE]
│   │   ├── verify_clustering.py [COMPLETE]
│   │   ├── verify_cuda_readiness.py [COMPLETE]
│   │   ├── debug/
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
│   │   ├── test_dashboard_ux.py [TESTED]
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
│   │   ├── test_tbn_optimization.py [TESTED]
│   │   ├── test_three_body_exits.py [TESTED]
│   │   ├── test_timeframe_belief_network.py [TESTED]
│   │   ├── test_torch_pattern.py [TESTED]
│   │   ├── test_training_validation.py [TESTED]
│   │   ├── test_wave_rider.py [TESTED]
│   │   ├── test_wave_rider_features.py [TESTED]
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
│   ├── tools/
│   │   ├── __init__.py [COMPLETE]
│   │   ├── checkpoint_viewer.py [COMPLETE]
│   │   ├── pattern_map.py [COMPLETE]
│   │   ├── screening_plots.py [COMPLETE]
│   │   ├── trade_visualizer.py [COMPLETE]
│   │   ├── waveform_screening.py [COMPLETE]
│   │   ├── waveform_standalone.py [COMPLETE]
│   │   ├── plots/
│   │   │   ├── page1_pattern_library.png
│   │   │   ├── page2_direction_depth.png
│   │   │   ├── page3_brain.png
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
│   │   ├── integrated_statistical_system.py [COMPLETE]
│   │   ├── monte_carlo_engine.py [COMPLETE]
│   │   ├── orchestrator.py [COMPLETE]
│   │   ├── orchestrator_worker.py [COMPLETE]
│   │   ├── pattern_analyzer.py [COMPLETE]
│   │   ├── pid_oscillation_analyzer.py [COMPLETE]
│   │   ├── pipeline_checkpoint.py [COMPLETE]
│   │   ├── progress_reporter.py [COMPLETE]
│   │   ├── run_analytics.py [COMPLETE]
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
- **Python Files:** 148
- **Total Lines of Code:** 46177

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
- **Test Files Count:** 37

### 10. FILES MODIFIED (Last Commit)
```
A	.Jules/palette.md
A	.claude/settings.json
A	.claude/settings.local.json
A	.github/workflows/github_workflows_data_preprocessing.yml.disabled
A	.github/workflows/github_workflows_parallel_preprocessing.yml.disabled
A	.github/workflows/unified_test_pipeline.yml.disabled
A	.gitignore
A	.vscode/settings.json
A	AGENTS.md
A	AUDIT/AUDIT_REPORT_2026_02_22.md
A	AUDIT/NinjaTrader Grid 2026-03-03 02-30 PM.csv
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
A	DATA/ATLAS_1MONTH/15m/2025_01.parquet
A	DATA/ATLAS_1MONTH/15s/2025_01.parquet
A	DATA/ATLAS_1MONTH/1D/2025_01.parquet
A	DATA/ATLAS_1MONTH/1W/2025_01.parquet
A	DATA/ATLAS_1MONTH/1h/2025_01.parquet
A	DATA/ATLAS_1MONTH/1m/2025_01.parquet
A	DATA/ATLAS_1MONTH/1s/2025_01.parquet
A	DATA/ATLAS_1MONTH/2m/2025_01.parquet
A	DATA/ATLAS_1MONTH/30m/2025_01.parquet
A	DATA/ATLAS_1MONTH/30s/2025_01.parquet
A	DATA/ATLAS_1MONTH/3m/2025_01.parquet
A	DATA/ATLAS_1MONTH/4h/2025_01.parquet
A	DATA/ATLAS_1MONTH/5m/2025_01.parquet
A	DATA/ATLAS_1MONTH/5s/2025_01.parquet
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
A	core/keep_awake.py
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
A	docs/JULES_EXECUTION_REPORT.md
A	docs/JULES_HYPERVOLUME_CLUSTERING.md
A	docs/JULES_PERFORMANCE_TARGETS.md
A	docs/JULES_SIGNAL_CAPTURE_AUDIT.md
A	docs/JULES_SPECTRAL_GATES.md
A	docs/JULES_WAVEFORM_SEED_INTEGRATION.md
A	docs/NT8_BayesianBridge.cs
A	docs/OLD/COMPLETE_IMPLEMENTATION_SPEC.md
A	docs/OLD/FRACTAL_PATTERN_RECOGNITION_SPEC.md
A	docs/OLD/JULES_5S_1S_WORKERS.md
A	docs/OLD/JULES_CPU_PHYSICS.md
A	docs/OLD/JULES_DYNAMIC_EXIT.md
A	docs/OLD/JULES_FRACTAL_DNA_TREE.md
A	docs/OLD/JULES_GOLDEN_PATH_ORACLE.md
A	docs/OLD/JULES_INSTRUCTIONS.md
A	docs/OLD/JULES_MONTE_CARLO.md
A	docs/OLD/JULES_NUMBA_SIM.md
A	docs/OLD/JULES_ORACLE_ENGINE.md
A	docs/OLD/JULES_PID_OPTIMIZER.md
A	docs/OLD/JULES_PID_OSCILLATION.md
A	docs/OLD/JULES_SNOWFLAKE_BASELINE.md
A	docs/OLD/JULES_SNOWFLAKE_CLUSTERS.md
A	docs/OLD/New logic.txt
A	docs/OLD/README_DASHBOARD.md
A	docs/OLD/TECHNICAL_MANUAL.md
A	docs/OLD/THE NIGHTMARE FIELD EQUATIOn.pdf
A	docs/OLD/Training Orchestrator.txt
A	docs/OLD/Unconstrained_Exploration.txt
A	docs/OLD/evaluation_legacy_pattern_detector.md
A	docs/SYSTEM_DESCRIPTION.md
A	docs/checkpoint_reference/SCHEMAS.md
A	docs/checkpoint_reference/depth_analytics.txt
A	docs/checkpoint_reference/depth_weights.json
A	docs/checkpoint_reference/discovery_levels.json
A	docs/checkpoint_reference/oos_analytics.txt
A	docs/checkpoint_reference/oos_report.txt
A	docs/checkpoint_reference/pipeline_state.json
A	docs/checkpoint_reference/run_snapshot.json
A	docs/checkpoint_reference/sample_fn_oracle_log.csv
A	docs/checkpoint_reference/sample_oracle_trade_log.csv
A	docs/checkpoint_reference/sample_pid_oracle_log.csv
A	docs/checkpoint_reference/sample_signal_log.csv
A	docs/checkpoint_reference/template_tiers.pkl
A	docs/checkpoint_reference/trade_analytics.txt
A	docs/memory/MEMORY.md
A	docs/memory/ROADMAP.md
A	docs/old_jules/JULES_SNOWFLAKE_BASELINE.md
A	docs/old_jules/JULES_SPECTRAL_GATES.md
A	docs/old_jules/JULES_TASK_1_2_IMPLEMENTATION.md
A	docs/old_jules/JULES_TEMPLATE_TIMESCALE.md
A	docs/old_jules/PLAN_PRICE_AWARE_WORKERS.md
A	live/__init__.py
A	live/__main__.py
A	live/bar_aggregator.py
A	live/config.py
A	live/launcher.py
A	live/live_engine.py
A	live/nt8_client.py
A	live/order_manager.py
A	live/protocol.py
A	models/quantum_probability_table.pkl
A	notebooks/dashboard.ipynb
A	notebooks/debug_outputs/mini_run/probability_table.pkl
A	notebooks/models/production_learning/quantum_probability_table.pkl
A	reports/is/fn_oracle_log.csv
A	reports/is/oracle_trade_log.csv
A	reports/is/pid_oracle_log.csv
A	reports/is/shards/fn_oracle_log_2025_01.csv
A	reports/is/shards/fn_oracle_log_2025_02.csv
A	reports/is/shards/fn_oracle_log_2025_03.csv
A	reports/is/shards/fn_oracle_log_2025_04.csv
A	reports/is/shards/fn_oracle_log_2025_05.csv
A	reports/is/shards/fn_oracle_log_2025_06.csv
A	reports/is/shards/fn_oracle_log_2025_07.csv
A	reports/is/shards/fn_oracle_log_2025_08.csv
A	reports/is/shards/fn_oracle_log_2025_09.csv
A	reports/is/shards/fn_oracle_log_2025_10.csv
A	reports/is/shards/fn_oracle_log_2025_11.csv
A	reports/is/shards/fn_oracle_log_2025_12.csv
A	reports/is/shards/oracle_trade_log_2025_08.csv
A	reports/is/shards/pid_oracle_log_2025_01.csv
A	reports/is/shards/pid_oracle_log_2025_02.csv
A	reports/is/shards/pid_oracle_log_2025_03.csv
A	reports/is/shards/pid_oracle_log_2025_04.csv
A	reports/is/shards/pid_oracle_log_2025_05.csv
A	reports/is/shards/pid_oracle_log_2025_06.csv
A	reports/is/shards/pid_oracle_log_2025_07.csv
A	reports/is/shards/pid_oracle_log_2025_08.csv
A	reports/is/shards/pid_oracle_log_2025_09.csv
A	reports/is/shards/pid_oracle_log_2025_10.csv
A	reports/is/shards/pid_oracle_log_2025_11.csv
A	reports/is/shards/pid_oracle_log_2025_12.csv
A	reports/is/shards/signal_log_2025_01.csv
A	reports/is/shards/signal_log_2025_02.csv
A	reports/is/shards/signal_log_2025_03.csv
A	reports/is/shards/signal_log_2025_04.csv
A	reports/is/shards/signal_log_2025_05.csv
A	reports/is/shards/signal_log_2025_06.csv
A	reports/is/shards/signal_log_2025_07.csv
A	reports/is/shards/signal_log_2025_08.csv
A	reports/is/shards/signal_log_2025_09.csv
A	reports/is/shards/signal_log_2025_10.csv
A	reports/is/shards/signal_log_2025_11.csv
A	reports/is/shards/signal_log_2025_12.csv
A	reports/is/signal_log.csv
A	reports/is/signal_log_prev.csv
A	reports/is/trade_analytics.txt
A	reports/live/session_20260302_195935.txt
A	reports/live/session_20260302_203709.txt
A	reports/live/session_20260302_211659.txt
A	reports/live/session_20260302_212039.txt
A	reports/live/session_20260303_142227.txt
A	reports/live/session_20260303_161855.txt
A	reports/live/session_20260303_163639.txt
A	reports/live/session_20260303_164507.txt
A	reports/live/session_20260303_170214.txt
A	reports/live/session_20260303_170559.txt
A	reports/live/session_20260303_203004.txt
A	reports/live/session_20260303_204448.txt
A	reports/live/session_20260303_204449.txt
A	reports/live/session_20260303_211242.txt
A	reports/live/session_20260303_211243.txt
A	reports/live/session_20260303_215434.txt
A	reports/live/session_20260303_221810.txt
A	reports/live/session_20260303_222910.txt
A	reports/live/session_20260303_225751.txt
A	reports/live/session_20260303_232130.txt
A	reports/live/session_20260303_232748.txt
A	reports/live/session_20260304_054311.txt
A	reports/live/session_20260304_060333.txt
A	reports/live/session_20260304_061251.txt
A	reports/live/session_20260304_061255.txt
A	reports/live/session_20260304_061839.txt
A	reports/live/session_20260304_061842.txt
A	reports/live/session_20260304_064516.txt
A	reports/live/session_20260304_070241.txt
A	reports/live/session_20260304_075415.txt
A	reports/live/session_20260304_082425.txt
A	reports/live/session_20260304_082427.txt
A	reports/live/session_20260304_083028.txt
A	reports/live/session_20260304_083204.txt
A	reports/live/session_20260304_083428.txt
A	reports/live/session_20260304_091358.txt
A	reports/live/session_20260304_114309.txt
A	reports/live/session_20260304_115150.txt
A	reports/live/session_20260304_115153.txt
A	reports/live/session_20260304_115921.txt
A	reports/live/session_20260304_115926.txt
A	reports/live/session_20260304_121303.txt
A	reports/live/session_20260304_121356.txt
A	reports/oos/fn_oracle_log.csv
A	reports/oos/pid_oracle_log.csv
A	reports/oos/shards/fn_oracle_log_2026_01.csv
A	reports/oos/shards/fn_oracle_log_2026_02.csv
A	reports/oos/shards/pid_oracle_log_2026_01.csv
A	reports/oos/shards/pid_oracle_log_2026_02.csv
A	reports/oos/shards/signal_log_2026_01.csv
A	reports/oos/shards/signal_log_2026_02.csv
A	reports/oos/signal_log.csv
A	reports/oos/signal_log_prev.csv
A	reports/oos_report.txt
A	reports/pattern_map_2025_01.png
A	reports/phase4_report.txt
A	reports/snowflake/oos_report.txt
A	reports/snowflake/phase4_report.txt
A	reports/trades_2025_01.png
A	reports/training/training_log.txt
A	requirements.txt
A	run_logs/.gitkeep
A	run_test_workflow.py
A	runs/2026-02-22_pre-depth-gate/depth_analytics.txt
A	runs/2026-02-22_pre-depth-gate/depth_weights.json
A	runs/2026-02-22_pre-depth-gate/fn_oracle_log.csv
A	runs/2026-02-22_pre-depth-gate/oracle_trade_log.csv
A	runs/2026-02-22_pre-depth-gate/phase4_report.txt
A	runs/2026-02-22_pre-depth-gate/phase5_report.txt
A	runs/2026-02-22_pre-depth-gate/pid_oracle_log.csv
A	runs/2026-02-22_pre-depth-gate/run_snapshot.json
A	runs/2026-02-22_pre-depth-gate/signal_log_2025_Q1.csv
A	runs/2026-02-22_pre-depth-gate/signal_log_2025_Q2.csv
A	runs/2026-02-22_pre-depth-gate/signal_log_2025_Q3.csv
A	runs/2026-02-22_pre-depth-gate/signal_log_2025_Q4.csv
A	runs/2026-02-22_pre-depth-gate/training_log.txt
A	scripts/benchmark_regression.py
A	scripts/build_executable.py
A	scripts/debug/debug_databento.py
A	scripts/debug/debug_utils.py
A	scripts/debug/reproduce_loader_error.py
A	scripts/fix_cuda.py
A	scripts/generate_status_report.py
A	scripts/gpu_health_check.py
A	scripts/manifest_integrity_check.py
A	scripts/monthly_pnl_chart.py
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
A	tests/test_dashboard_ux.py
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
A	tests/test_tbn_optimization.py
A	tests/test_three_body_exits.py
A	tests/test_timeframe_belief_network.py
A	tests/test_torch_pattern.py
A	tests/test_training_validation.py
A	tests/test_wave_rider.py
A	tests/test_wave_rider_features.py
A	tests/test_wave_rider_playbook.py
A	tests/topic_build.py
A	tests/topic_diagnostics.py
A	tests/topic_math.py
A	tests/utils.py
A	tests/verify_regret_fallback.py
A	tmp_b64_part2.txt
A	tools/__init__.py
A	tools/checkpoint_viewer.py
A	tools/pattern_map.py
A	tools/plots/page1_pattern_library.png
A	tools/plots/page2_direction_depth.png
A	tools/plots/page3_brain.png
A	tools/plots/standalone/1y/0_price_imr.png
A	tools/plots/standalone/1y/0b_regime_summary.png
A	tools/plots/standalone/1y/0c_stacked_shapes.png
A	tools/plots/standalone/1y/0d_regime_audit.png
A	tools/plots/standalone/1y/0e_laplacian_signatures.png
A	tools/plots/standalone/1y/0f_shape_clusters.png
A	tools/plots/standalone/1y/0g_seed_templates.png
A	tools/plots/standalone/1y/0h_seed_matches.png
A	tools/plots/standalone/1y/0i_corr_distribution.png
A	tools/plots/standalone/1y/0j_noise_audit.png
A	tools/plots/standalone/1y/0k_sub_classification.png
A	tools/plots/standalone/1y/0l_back_skewed_down_raw.png
A	tools/plots/standalone/1y/0l_back_skewed_up_raw.png
A	tools/plots/standalone/1y/0l_damped_oscillator_raw.png
A	tools/plots/standalone/1y/0l_expand_oscillator_raw.png
A	tools/plots/standalone/1y/0l_exponential_down_raw.png
A	tools/plots/standalone/1y/0l_exponential_up_raw.png
A	tools/plots/standalone/1y/0l_front_skewed_down_raw.png
A	tools/plots/standalone/1y/0l_front_skewed_up_raw.png
A	tools/plots/standalone/1y/0l_linear_down_raw.png
A	tools/plots/standalone/1y/0l_linear_up_raw.png
A	tools/plots/standalone/1y/0l_logarithmic_down_raw.png
A	tools/plots/standalone/1y/0l_logarithmic_up_raw.png
A	tools/plots/standalone/1y/0l_rounded_u_down_raw.png
A	tools/plots/standalone/1y/0l_rounded_u_up_raw.png
A	tools/plots/standalone/1y/0l_sine_wave_raw.png
A	tools/plots/standalone/1y/0l_step_down_raw.png
A	tools/plots/standalone/1y/0l_step_up_raw.png
A	tools/plots/standalone/1y/0l_symmetric_v_down_raw.png
A	tools/plots/standalone/1y/0l_symmetric_v_up_raw.png
A	tools/plots/standalone/1y/0m_direction_prediction.png
A	tools/plots/standalone/1y/0n_signed_mfe_direction.png
A	tools/plots/standalone/1y/0o_price_direction_overlay.png
A	tools/plots/standalone/1y/0p_next_price_forecast.png
A	tools/plots/standalone/1y/0q_next_price_overlay.png
A	tools/plots/standalone/1y/0r_delta_direct_forecast.png
A	tools/plots/standalone/1y/0s_delta_direct_overlay.png
A	tools/plots/standalone/1y/0t_stepwise_direction.png
A	tools/plots/standalone/1y/0u_paired_point_direction.png
A	tools/plots/standalone/1y/1_imr_key_features.png
A	tools/plots/standalone/1y/2_i_heatmap.png
A	tools/plots/standalone/1y/3_mr_heatmap.png
A	tools/plots/standalone/1y/4_imr_correlation.png
A	tools/plots/standalone/1y/analysis_m_predictions.csv
A	tools/plots/standalone/1y/analysis_n_delta_predictions.csv
A	tools/plots/standalone/1y/analysis_o_stepwise_predictions.csv
A	tools/plots/standalone/1y/analysis_p_paired_points.csv
A	tools/plots/standalone/1y/analysis_q_signed_bins.csv
A	tools/plots/standalone/1y/analysis_q_signed_histogram.png
A	tools/plots/standalone/1y/waveform_report.txt
A	tools/screening_plots.py
A	tools/trade_visualizer.py
A	tools/waveform_screening.py
A	tools/waveform_standalone.py
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
A	training/integrated_statistical_system.py
A	training/monte_carlo_engine.py
A	training/orchestrator.py
A	training/orchestrator_worker.py
A	training/pattern_analyzer.py
A	training/pid_oscillation_analyzer.py
A	training/pipeline_checkpoint.py
A	training/progress_reporter.py
A	training/run_analytics.py
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
- **Summary:** 4 passed in 1.73s


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
PASS: Check passed (no details)

Manifest Integrity
PASS: Manifest Integrity Check Passed
