# Phase 3B: Joint Bayesian Evidence Model Index

This index tracks the per-signal evaluations of the NinjaTrader concept catalog under the Bayesian likelihood framework. Each signal is evaluated standalone against its own pre-registered response (from the original article text) before being fused into the joint model.

## Per-Signal Evaluation Summary

| signal | what it measures | registered response type | N events | P(resp) vs ref | MODE of magnitude | tail | bimodal? | verdict |
|--------|------------------|--------------------------|----------|----------------|-------------------|------|----------|---------|
| VWAP_Pullbacks | Session VWAP Z-Score Reversion | Directional bounce from extremes | 11983 | 0.51 (vs 0.50) | 2.2 | 11.62 | False | **NOISE** |
| APZ_Touches | Adaptive Price Zones | Directional first-touch bounce (Mean Reversion) | 19052 | 0.51 (vs 0.50) | 3.0 | 12.17 | False | **NOISE** |
| Squeeze_State | Volatility Squeeze | Volatility Expansion (Direction-Free Breakout) | 15304 | 1.00 (vs 1.00) | 6.6 | 18.68 | True | **NOISE** |
| Candle_Shapes | Engulfing/Pinbar Patterns | Directional Continuation | 24999 | 0.50 (vs 0.50) | -3.0 | 12.34 | True | **NOISE** |
| MA_Crossover | MA Crossover | Trend Continuation | 6410 | 0.50 (vs 0.50) | 3.0 | 12.60 | True | **NOISE** |

## Execution Rules
- **One signal = one script** in `tests/<ID>/ag_deepdive_*.py` (Dossier layout).
- **Sigma-relative**: All magnitude measurements are normalized by standard deviation ($\sigma$).
- **Reporting**: Full markdown report in `tests/<ID>/` with magnitude histogram figures.
- **Reference Baselines**: Frequency + magnitude empirical counting, using arithmetic reference (50%) for symmetric barriers in the exploration stage.
