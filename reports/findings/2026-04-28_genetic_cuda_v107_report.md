# CUDA Genetic optimization report — ZigzagRunner_v107

Generated: 2026-04-28 22:16

## Setup

- Atlas: `DATA/ATLAS`
- Train: through `2025-12-31` (314,716 1m bars)
- Holdout: from `2026-01-01` (76,798 1m bars)
- Optimizer: numba.cuda kernel + manual rand/1/bin DE on GPU
- Search dim: 9, popsize_mult: 40 (actual P=360), maxiter: 50, seed: 42
- DE wall time: 6.8 sec
- Best train PnL (float32 sim): $+20,284.57

## Top-5 candidates (CPU float64 verification + holdout)

```
 atr_lookback  atr_multiplier  min_r_points  max_r_points  max_loss_pts  mfe_cut_bars  mfe_cut_usd  trail_activate_pts  trail_giveback_pct  pnl_net  pnl_per_day  n_trades  win_rate  holdout_pnl_net  holdout_pnl_per_day  holdout_n_trades  holdout_win_rate  pnl_drop_pct  robust_flag
          127           10.31          5.00        195.72        144.87             3         0.00                0.00                0.50 20284.80        87.81      1148      0.26          4512.30                79.16               258              0.35         77.76        False
          129           11.75          7.75        230.50         54.52            30         0.00                0.00                0.50 20265.60        88.50       936      0.31          5597.80                98.21               208              0.34         72.38        False
          124           11.62          8.17        249.58         48.79             0        46.02                0.00                0.50 19096.40        83.39       939      0.31          5325.10                93.42               206              0.36         72.11        False
          240           15.00         26.67        236.46         36.01            24        12.66                0.00                0.50 18984.10        84.00       711      0.25          1446.90                25.38               164              0.23         92.38        False
          172           11.98         16.72        237.06         53.15             6         0.00                0.00                0.05 18953.10        82.05       926      0.26          1392.30                24.43               203              0.29         92.65        False
```


## Decision matrix

| Outcome | Action |
|---|---|
| ≥1 row with `robust_flag=True` | Pick that combo. Holdout PnL ≥ $30/day AND drop < 30% from train. |
| All rows fail `robust_flag` | Strategy overfits the train window. **Stay on v1.0.4 baseline.** |
| Best holdout drop > 50% | Severe overfit. Re-run with stricter `--maxiter` or different `--seed`. |

## Caveats

- CUDA sim uses **float32** for speed; CPU verification uses **float64**. PnL drift typically $<10$ over 314k bars.
- 1m-resolution sim — hard SL fires at 1m close, not 1s like real NT8.
- Python ATR uses SMA, NT8 uses Wilder smoothing. Small but non-zero discrepancy.
- Prior parity work showed Python sim has ~2× trade count vs NT8 SA. Use rank-order to pick combos.
