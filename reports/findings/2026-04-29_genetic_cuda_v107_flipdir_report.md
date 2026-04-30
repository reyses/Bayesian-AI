# CUDA Genetic optimization report — ZigzagRunner_v107

Generated: 2026-04-29 06:05

## Setup

- Atlas: `DATA/ATLAS`
- Train: through `2025-12-31` (314,716 1m bars)
- Holdout: from `2026-01-01` (76,798 1m bars)
- Optimizer: numba.cuda kernel + manual rand/1/bin DE on GPU
- Search dim: 9, popsize_mult: 40 (actual P=360), maxiter: 50, seed: 42
- DE wall time: 6.9 sec
- Best train PnL (float32 sim): $+7,144.54

## Top-5 candidates (CPU float64 verification + holdout)

```
 atr_lookback  atr_multiplier  min_r_points  max_r_points  max_loss_pts  mfe_cut_bars  mfe_cut_usd  trail_activate_pts  trail_giveback_pct  pnl_net  pnl_per_day  n_trades  win_rate  holdout_pnl_net  holdout_pnl_per_day  holdout_n_trades  holdout_win_rate  pnl_drop_pct  robust_flag
           20           15.00         21.10        286.83        150.00            23        16.18               27.13                0.05  7134.00        32.58       615      0.61         -1146.20               -20.47               148              0.54        116.07        False
           20           15.00         29.46        280.68        150.00            15        16.44               27.27                0.07  7114.90        32.49       619      0.59         -2201.60               -39.31               149              0.48        130.94        False
           20           15.00         24.78        279.11        147.12            17        17.44               28.25                0.26  6757.50        30.86       620      0.59         -2262.10               -40.39               149              0.49        133.48        False
           20           15.00          5.00        277.18        150.00            29         9.24               28.38                0.10  6717.50        30.67       620      0.65         -1685.60               -30.10               149              0.60        125.09        False
           20           15.00         13.29        273.22        150.00            27        17.75               24.28                0.05  6722.50        30.70       620      0.62         -2020.60               -36.08               149              0.54        130.06        False
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
