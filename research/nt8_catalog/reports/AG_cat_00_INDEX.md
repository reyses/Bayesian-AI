# Phase 3B: Joint Bayesian Evidence Model Index

This index tracks the per-signal evaluations of the NinjaTrader concept catalog under the Bayesian likelihood framework. Each signal is evaluated standalone against its own pre-registered response (from the original article text) before being fused into the joint model.

## Per-Signal Evaluation Summary

| signal | what it measures | registered response type | N events | P(resp) vs null | MODE of magnitude | tail | bimodal? | verdict |
|--------|------------------|--------------------------|----------|-----------------|-------------------|------|----------|---------|
|        |                  |                          |          |                 |                   |      |          |         |

## Execution Rules
- **One signal = one script** in `tools/ag_cat_NN_<name>.py`.
- **Sigma-relative**: All magnitude measurements are normalized by standard deviation ($\sigma$).
- **Null controls**: Matched + phantom nulls at the per-signal layer.
- **Reporting**: Full markdown report in `reports/` with magnitude histogram figures in `reports/assets/`.
| VWAP_Pullbacks | VWAP Touch | Directional first-touch bounce (+2 sigma) | 11983 | 0.51 (vs 0.50) | 2.2 | 11.62 | False | **NOISE** |
| APZ_Touches | VWAP Touch | Directional first-touch bounce (Mean Reversion) | 19052 | 0.51 (vs 0.50) | 3.0 | 12.17 | False | **NOISE** |
| Squeeze_State | VWAP Touch | Volatility Expansion (Direction-Free Breakout) | 15304 | 1.00 (vs 1.00) | 6.6 | 18.68 | True | **NOISE** |
| Candle_Shapes | VWAP Touch | Directional Continuation (+2 sigma) | 24999 | 0.50 (vs 0.50) | -3.0 | 12.34 | True | **NOISE** |
| MA_Crossover | VWAP Touch | Trend Continuation (+2 sigma) | 6410 | 0.50 (vs 0.50) | 3.0 | 12.60 | True | **NOISE** |
