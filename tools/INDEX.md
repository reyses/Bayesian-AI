# Tools Index
> Quick reference for all analysis/research tools. Update when adding new tools.

## Gate & Entry Analysis
| Tool | Description |
|------|-------------|
| `analyze_gates.py` | Oracle-driven gate threshold analysis, `--apply` writes JSON |
| `gate_interaction_matrix.py` | C&E matrix empirical validation (Spearman/Kruskal) |
| `research_signal_fire.py` | Gate cascade analysis and loosening experiments |
| `research_pattern_selection.py` | Score competition analysis for tied candidates |
| `research_pattern_audit.py` | End-to-end audit of pattern recognition pipeline |
| `research_tf_mixing.py` | Quantify timeframe mixing in K-Means clustering |
| `pattern_map.py` | Where every detected pattern sits on price waveform |

## Direction & Belief
| Tool | Description |
|------|-------------|
| `research_belief_flip.py` | Side-by-side TBN comparison with trade context |
| `dmi_crossover_validation.py` | DMI+/DMI- crossovers: can they identify $10+ regime moves? |
| `dmi_reversal_simulator.py` | Multi-TF reversal alignment and testing |

## Risk & Sizing
| Tool | Description |
|------|-------------|
| `equity_risk_simulator.py` | Equity growth simulation (flat vs dynamic sizing) |
| `l2_risk_budget.py` | MAE/MFE of $30+ segments from 1s data |

## Price Structure & Regimes
| Tool | Description |
|------|-------------|
| `golden_path.py` | Y10/Y11/Y12 chord length metrics from 1s data |
| `imr_golden_path.py` | I-MR control chart + golden path overlay (1m SPC) |
| `imr_regime_segments.py` | Identify tradeable regimes from price structure |
| `research_golden_physics.py` | What physics state separates real moves from noise |
| `session_overlay.py` | Map trades onto 1h + 1m price with adaptive Fibs |
| `regime_labeler.py` | Step through I-MR regime segments for labeling |

## Templates & Shapes
| Tool | Description |
|------|-------------|
| `shape_primitive_builder.py` | Two-stage entry+exit primitives via UMAP+HDBSCAN |
| `shape_primitive_report.py` | Read shape_primitives.pkl and write text report |
| `inspect_templates.py` | Dump human-readable template report from checkpoint |
| `visualize_template.py` | Multi-panel visualization of templates |
| `checkpoint_viewer.py` | Inspect pattern_library.pkl + brain |

## Seeds & Swings
| Tool | Description |
|------|-------------|
| `auto_swing_marker.py` | ZigZag-based swing detector calibrated from human seeds |
| `seed_oracle_trainer.py` | Extract 192D features from manual seeds, train classifier |
| `seed_pattern_analyzer.py` | Extract actual price waveforms from ATLAS for seeds |
| `seed_inspector.py` | Step through I-MR auto-seeds on price chart for QA |
| `imr_to_seeds.py` | Convert auto-detected I-MR regimes into seed JSON |
| `swing_inspector.py` | Grade continuous swing groups on price chart |
| `trade_marker.py` | Manually mark trades on price chart with crosshair |

## Trade Review & Visualization
| Tool | Description |
|------|-------------|
| `trade_review.py` | Comprehensive post-run trade analysis |
| `trade_visualizer.py` | Price waveform with entry/exit markers |
| `oos_chain_chart.py` | PnL curves + trade correlation across OOS1/OOS2/OOS3 |
| `oos_parity_overlay.py` | OOS2 vs OOS3 trade overlay (match by entry_price+side) |

## Pipeline & Data
| Tool | Description |
|------|-------------|
| `standalone_research.py` | Thin CLI orchestrator for research modules (A-R) |
| `run_analytics.py` | Re-run analytics suite on existing checkpoint data |
| `setup_oos_atlas.py` | Reorganize DATA/ATLAS into clean train/test split |
| `nt8_to_parquet.py` | Convert NT8 tick exports to ATLAS parquet |

## Subpackage: `tools/research/`
| Module | Description |
|--------|-------------|
| `data.py` | Data loading helpers for research modules |
| `imr.py` | I-MR control chart computations |
| `screening.py` | Template screening utilities |
| `seeds.py` | Seed file I/O |
| `plots.py` | Shared plotting helpers |
| `tbn_trade_aware.py` | Trade-aware TBN analysis |

## Archive (orphaned/superseded — do NOT use)

Recent archive activity (2026-04-30 cleanup):

| Subdir | Why archived |
|---|---|
| `tools/archive/2026-04-29_overnight_research/` | One-shot scripts from overnight EDA cycle. Outputs preserved in `reports/findings/`. 7 files. |
| `tools/archive/superseded_v_x_runners/` | v1.0.4 / v1.0.6 / v1.2-RC backtest + parity scripts. Strategies retired. 6 files. |
| `tools/archive/v15rc_chop_specialist/` | v1.5-RC chop-specialist family (v15_*, nt8_bleed_harvest, tier9_bleed, tier_day_*, sweep_phantom, zigzag_5s_perfect_fill). Strategy archived. 12 files. |

If you need one of these back, `git mv` it from the archive subdir — history is intact.
