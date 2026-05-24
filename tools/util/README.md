# tools/_util — Utilities & Support

> Virtual folder. Files in `tools/` root.

## Checkpoints / snapshots
| Tool | Purpose |
|---|---|
| [`checkpoint_viewer.py`](../checkpoint_viewer.py) | Human-readable checkpoint dump |
| [`inspect_templates.py`](../inspect_templates.py) | Template checkpoint → human-readable report |
| [`build_checkpoint.py`](../build_checkpoint.py) | Build checkpoint.json at specific end-date cutoff |
| [`shard_reader.py`](../shard_reader.py) | Summary of in-progress signal-log shards |
| [`tree_health.py`](../tree_health.py) | Check tree leaf distinctness |
| [`precompute_live_states.py`](../precompute_live_states.py) | Pre-compute MarketState per TF |

## Golden path / risk
| Tool | Purpose |
|---|---|
| [`golden_path.py`](../golden_path.py) | Y10/Y11/Y12 computation on 1s ATLAS. **Hub** |
| [`l2_risk_budget.py`](../l2_risk_budget.py) | MFE vs MAE cost for $30+ trades. **Hub** |
| [`equity_risk_simulator.py`](../equity_risk_simulator.py) | Dynamic position sizing from $10 floor |

## Strategy discovery / research orchestration
| Tool | Purpose |
|---|---|
| [`strategy_miner.py`](../strategy_miner.py) | Data-driven entry discovery, threshold scans |
| [`standalone_research.py`](../standalone_research.py) | Orchestrator for `tools/research/` subpackage |
| [`run_analytics.py`](../run_analytics.py) | Re-run analytics suite on existing checkpoints |
| [`hypothesis_test.py`](../hypothesis_test.py) | Apply candidate filters → measure $/day impact |
| [`template_instruction_test.py`](../template_instruction_test.py) | Apply template direction + duration rule |
| [`seed_funnel_test.py`](../seed_funnel_test.py) | Feature-trajectory funnel → template-match entry |
| [`seed_funnel_oos.py`](../seed_funnel_oos.py) | OOS seed funnel |
| [`seed_pattern_analyzer.py`](../seed_pattern_analyzer.py) | Seed waveform shape + cross-TF nesting analysis |
| [`z_range_filter_backtest.py`](../z_range_filter_backtest.py) | 1h z-range filter validation |
