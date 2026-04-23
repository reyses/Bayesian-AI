# tools/_engines — Engine Variants & Entry Tests

> Virtual folder. Files in `tools/` root.

| Tool | Purpose |
|---|---|
| [`blended_test.py`](../blended_test.py) | Blended engine test (one NMP + tiered exits) |
| [`exnmp_trio_test.py`](../exnmp_trio_test.py) | All 3 ExNMP strategies (KillShot, Overshoot, Cascade) sequentially |
| [`killshot_test.py`](../killshot_test.py) | NMP+wick-rejection (KillShot) on IS/OOS |
| [`overshoot_test.py`](../overshoot_test.py) | NMP base + hold through mean for momentum overshoot |
| [`wick_overshoot_test.py`](../wick_overshoot_test.py) | KillShot entry + overshoot exit + breakeven regret |
| [`saturation_sim.py`](../saturation_sim.py) | Fixed TP/SL/timeout on every trade — saturation strategy |
| [`eda_new_entries.py`](../eda_new_entries.py) | Standalone EDA of new entry strategies |
| [`regret_new_entries.py`](../regret_new_entries.py) | Regret analysis on new entries |
| [`cascade_order_optimizer.py`](../cascade_order_optimizer.py) | Find TIER_PRIORITY that maximizes cascade PnL |
