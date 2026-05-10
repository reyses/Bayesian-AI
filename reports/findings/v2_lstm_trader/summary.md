# V2 LSTM trader - 2026-05-04 02:14 UTC

## Config

- seq_len=50, hidden=64, layers=1
- forward_n=12, flat_thresh=8.0 ticks
- hold_bars=12, conf_threshold=0.5
- params=47,971, best_epoch=10 (val_acc=0.4474)

## OOS classification

- 3-class accuracy: 44.7%

## OOS trading

- Trades: 957
- Total PnL: $-2243.50
- $/trade: $-2.34
- $/calendar day: $-31.60
- Count-WR: 49.9%
- PF-WR: -0.05
- Sharpe (annualized): -1.00
- Max drawdown: $-4037.00

## Bayesian-pair handoff

`oos_predictions.csv` contains [p_short, p_flat, p_long] per OOS bar — these are categorical likelihoods ready for a Dirichlet-conjugate Bayesian update.
