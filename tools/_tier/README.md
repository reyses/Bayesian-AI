# tools/_tier — Tier-Specific Analysis & Runners

> Virtual folder. Files in `tools/` root.

## Runners
| Tool | Purpose |
|---|---|
| [`run_tier_isolated.py`](../run_tier_isolated.py) | Run each tier in isolation (only one tier fires) |
| [`run_iso_tiers_isolated.py`](../run_iso_tiers_isolated.py) | Isolated run for 5 NMP tiers |
| [`iso_tier_audit.py`](../iso_tier_audit.py) | Measure each nn_v2 tier on post-lookahead-fix features |
| [`iso_tier_eda.py`](../iso_tier_eda.py) | Max-fill tier forward pass — every tier evaluates every bar |

## EDAs
| Tool | Purpose |
|---|---|
| [`tier_eda.py`](../tier_eda.py) | Segment / separator / peak / regime-shift EDA per tier |
| [`tier_eda_killshot.py`](../tier_eda_killshot.py) | KILL_SHOT-specific EDA |
| [`tier_exit_physics.py`](../tier_exit_physics.py) | Full physics dump for Q2/Q3 EDA per tier |
| [`tier_daily_concentration.py`](../tier_daily_concentration.py) | Per-tier Pareto of daily PnL |
| [`tier_day_classifier.py`](../tier_day_classifier.py) | Day-level features separating BLEED from HARVEST |
| [`tier_day_rule_backtest.py`](../tier_day_rule_backtest.py) | Apply combined day rule, measure $ lift |
| [`tier_lookback_eda.py`](../tier_lookback_eda.py) | 10 min pre-entry physics, winners vs losers |
| [`tier_segment_diagnostic.py`](../tier_segment_diagnostic.py) | Chronological IS split — tier stability |
| [`tier_sequence_analysis.py`](../tier_sequence_analysis.py) | Tier firings predict other firings? |
| [`tier_signal_conflicts.py`](../tier_signal_conflicts.py) | Multi-tier same-bar conflicts |
| [`tune_tier_thresholds.py`](../tune_tier_thresholds.py) | Suggest threshold changes to improve WR |
| [`regret_on_isolated.py`](../regret_on_isolated.py) | Per-tier regret: actual vs optimal, capture % |
| [`corrected_regime_discovery.py`](../corrected_regime_discovery.py) | CART on best_action labels |
| [`maxfill_regret.py`](../maxfill_regret.py) | Oracle-optimal PnL per tier on max-fill trades |
