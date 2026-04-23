# tools/_rm_pivot — RM Pivot Research (current topic)

> Virtual folder. See also `research/rm_pivot/` for project docs + cycles.

## Signal measurement (Cycles 1-3)
| Tool | Purpose |
|---|---|
| [`measure_rm_pivot_direction_cohen_d.py`](../measure_rm_pivot_direction_cohen_d.py) | Cycle 1: residual-sign Cohen d at RM zigzag pivots |
| [`forward_pass_rm_pivot.py`](../forward_pass_rm_pivot.py) | Cycle 2: naive pivot-to-pivot forward pass |
| [`measure_rm_pivot_entry_direction.py`](../measure_rm_pivot_entry_direction.py) | Cycle 3: signal portfolio (Q1–Q5) |

## Sweeps + diagnostics
| Tool | Purpose |
|---|---|
| [`sweep_phantom.py`](../sweep_phantom.py) | Phantom-entry params sweep (wait_bars, min_favorable, SL) |
| [`time_to_wrong.py`](../time_to_wrong.py) | Seconds to first ±$X threshold (winners vs losers) |
| [`pivot_daily_distribution.py`](../pivot_daily_distribution.py) | Per-day PnL distribution for pivot sim |
| [`pivot_zero_day_diagnosis.py`](../pivot_zero_day_diagnosis.py) | Classify zero-PnL days (no-trade / wash / small-pos) |
| [`seed_variance_check.py`](../seed_variance_check.py) | Slippage RNG seed variance on fixed config |

## Cord analysis (ceilings + realistic capture)
| Tool | Purpose |
|---|---|
| [`cord_length_1m.py`](../cord_length_1m.py) | 1m price cord length = oracle upper bound at threshold R |
| [`cord_length_regression.py`](../cord_length_regression.py) | RM cord length = honest ceiling (smoother, lower) |
| [`cord_tradeable.py`](../cord_tradeable.py) | Realistic capture after 2R retracement tax |
| [`chord_ratio_analysis.py`](../chord_ratio_analysis.py) | reg_chord / price_path ratio → NOISE/TREND classifier |
| [`pivot_accuracy_stratified.py`](../pivot_accuracy_stratified.py) | Pivot accuracy stratified by chord ratio + wick |

## NN training + application (Cycle NN)
| Tool | Purpose |
|---|---|
| [`train_pivot_direction_nn.py`](../train_pivot_direction_nn.py) | CNN on 91D at RM pivots → P(win). Walk-forward |
| [`train_tier_direction_nn.py`](../train_tier_direction_nn.py) | CNN on tier trades (iso_is.pkl). More regularized |
| [`apply_pivot_nn_filter.py`](../apply_pivot_nn_filter.py) | Post-hoc filter: TAKE/FLIP/SKIP regime classification |

## Visualization
| Tool | Purpose |
|---|---|
| [`chart_rm_trades.py`](../chart_rm_trades.py) | Price + RM + trade segments per day |
| [`chart_strategy_comparison.py`](../chart_strategy_comparison.py) | CURRENT vs IDEAL (cusp → mean cross) side-by-side |
| [`chart_regression_z.py`](../chart_regression_z.py) | Regression mean + z overlay (zoomed-out) |
| [`chart_1s_trades.py`](../chart_1s_trades.py) | 1s-pivot forward-pass trades per day |
