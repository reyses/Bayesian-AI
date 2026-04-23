# Pivot-residual FORWARD PASS with 1s slippage

Exits resolved at 1-second granularity (no same-bar TP/SL ambiguity).

r_confirm=$8.0 TP=$50.0 SL=$3.0 min_res=0.5

## Results

| Dataset | Days | Pivots | Trades | $/day | $/trade | WR | $WR | Total | Hold (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 277 | 72,285 | 52,369 | $+56 | $+0.30 | 6.4% | +11% | $+15,558 | 70 |
| OOS | 68 | 20,742 | 15,068 | $+64 | $+0.29 | 6.3% | +10% | $+4,354 | 65 |

## Exit breakdown

| Reason | IS N | IS % | OOS N | OOS % |
|---|---:|---:|---:|---:|
| SL | 48,994 | 93.6% | 13,831 | 91.8% |
| TP | 3,215 | 6.1% | 909 | 6.0% |
| entry_past_day_end | 44 | 0.1% | 304 | 2.0% |
| eod | 116 | 0.2% | 24 | 0.2% |
