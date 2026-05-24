# tools/_pivot — Pivot Forward-Pass Variants

> Virtual folder. Files in `tools/` root.

## Physics-exit pivots
| Tool | Purpose |
|---|---|
| [`pivot_physics_exit.py`](../pivot_physics_exit.py) | 1s pivot entry + 1m reg-flip exit + 30s sniper. No SL. **Hub: imported by many** |
| [`pivot_physics_chains.py`](../pivot_physics_chains.py) | + chain multiplier (concurrent positions) |
| [`pivot_entry_variants.py`](../pivot_entry_variants.py) | Test 3 entry variants with identical physics exit |
| [`pivot_inspector.py`](../pivot_inspector.py) | Visualize REAL vs FAKEOUT pivots |

## Residual-based pivots
| Tool | Purpose |
|---|---|
| [`pivot_residual_forward.py`](../pivot_residual_forward.py) | Zigzag + residual direction, real-time (no lookahead). TP/SL sweep |
| [`pivot_residual_sim.py`](../pivot_residual_sim.py) | Oracle-pivot sim (with lookahead) — edge ceiling |
| [`pivot_forward_1s.py`](../pivot_forward_1s.py) | Pivot-residual + 1s slippage resolution (OHLC intra-bar) |
| [`pivot_1s_forward.py`](../pivot_1s_forward.py) | Full 1s: both detection + execution at 1s resolution |

## Regression-line pivots
| Tool | Purpose |
|---|---|
| [`regression_line_cohen_d.py`](../regression_line_cohen_d.py) | Cohen d at zigzag pivots on smoothed signal. **Hub** |
| [`regression_line_cohen_d_sr.py`](../regression_line_cohen_d_sr.py) | + S/R features from 5 prior business days |
| [`regression_pivot_forward.py`](../regression_pivot_forward.py) | RM-pivot forward pass with 1s slippage |
| [`residual_by_sr_context.py`](../residual_by_sr_context.py) | Residual signal stratified by S/R proximity |
