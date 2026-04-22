# Pivot-residual FORWARD PASS with 1s slippage

Exits resolved at 1-second granularity (no same-bar TP/SL ambiguity).

r_confirm=$5.0 TP=$30.0 SL=$10.0 min_res=0.5
**Velocity filter**: skipping pivots where price_vel agrees with prediction (8-18% WR zone).
**Velocity flip**: flipping direction when price_vel strongly agrees with prediction (92% WR inverted).

## Results

| Dataset | Days | Pivots | Trades | $/day | $/trade | WR | $WR | Total | Hold (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 277 | 95,082 | 64,449 | $-722 | $-3.10 | 17.4% | -38% | $-200,011 | 25 |
| OOS | 68 | 26,304 | 18,055 | $-796 | $-3.00 | 17.4% | -37% | $-54,141 | 6 |

## Exit breakdown

| Reason | IS N | IS % | OOS N | OOS % |
|---|---:|---:|---:|---:|
| SL | 53,144 | 82.5% | 14,611 | 80.9% |
| TP | 10,998 | 17.1% | 3,056 | 16.9% |
| entry_past_day_end | 0 | 0.0% | 342 | 1.9% |
| eod | 307 | 0.5% | 46 | 0.3% |
