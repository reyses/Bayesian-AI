# tools/_eda — Feature & Physics EDA

> Virtual folder. Files in `tools/` root.

## Feature analysis
| Tool | Purpose |
|---|---|
| [`feature_eda.py`](../feature_eda.py) | Multi-TF feature EDA (modules E1–E7) |
| [`feature_price_relationship.py`](../feature_price_relationship.py) | 12 features vs next-bar price (scatter/binned/correlation) |
| [`feature_response_surface.py`](../feature_response_surface.py) | Which features + combos separate winners from losers |
| [`mv_response_surface.py`](../mv_response_surface.py) | Multivariate per-tier feature interactions |
| [`gate_interaction_matrix.py`](../gate_interaction_matrix.py) | C&E matrix — X parameters to Y responses |
| [`path_features_eda.py`](../path_features_eda.py) | 12 5s bars preceding entry — path metrics |

## Bar physics
| Tool | Purpose |
|---|---|
| [`bar_color_payoff.py`](../bar_color_payoff.py) | Green/red bar color vs next-bar payoff |
| [`bar_directional_wick.py`](../bar_directional_wick.py) | Upper vs lower wick — rejection/support |
| [`bar_flip_size.py`](../bar_flip_size.py) | Size of bar flips across TFs |
| [`bar_wick_continuation.py`](../bar_wick_continuation.py) | Big-body (conviction) vs big-wick (rejection) continuation |

## Direction / movement
| Tool | Purpose |
|---|---|
| [`movement_direction_eda.py`](../movement_direction_eda.py) | Direction prediction EDA (single + polynomial features) |
| [`movement_z_stratified.py`](../movement_z_stratified.py) | Direction stratified by z-score (mean-reversion pockets) |
| [`tag_15_movements.py`](../tag_15_movements.py) | Tag every $15 / 8-min move in raw 5s. **Hub** |
| [`minimum_prediction_window.py`](../minimum_prediction_window.py) | At what N does sign prediction exceed 50%? |

## I-MR / regression kinematics
| Tool | Purpose |
|---|---|
| [`imr_analysis.py`](../imr_analysis.py) | I-MR on trade replays |
| [`imr_golden_path.py`](../imr_golden_path.py) | I-MR chart + golden-path overlay. **Hub** |
| [`imr_trade_chart.py`](../imr_trade_chart.py) | Single trade chart w/ MFE/MAE + physics |
| [`slope_eda.py`](../slope_eda.py) | β (slope) + γ (curvature) via 30-sample OLS |
| [`long_trade_path_eda.py`](../long_trade_path_eda.py) | 79D trajectory through 18+ min trades |

## Context / conditions
| Tool | Purpose |
|---|---|
| [`exit_physics_eda.py`](../exit_physics_eda.py) | 79D at regret's optimal exit per pattern |
