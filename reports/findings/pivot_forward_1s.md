# Pivot-residual FORWARD PASS with 1s slippage

Exits resolved at 1-second granularity (no same-bar TP/SL ambiguity).

r_confirm=$5.0 TP=$50.0 SL=$3.0 min_res=0.5
**Velocity filter**: skipping pivots where price_vel agrees with prediction (8-18% WR zone).
**Velocity flip**: flipping direction when price_vel strongly agrees with prediction (92% WR inverted).

## Results

| Dataset | Days | Pivots | Trades | $/day | $/trade | WR | $WR | Total | Hold (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 277 | 95,082 | 64,449 | $+48 | $+0.21 | 6.2% | +7% | $+13,410 | 71 |
| OOS | 68 | 26,304 | 18,055 | $+80 | $+0.30 | 6.3% | +11% | $+5,471 | 66 |

## Exit breakdown

| Reason | IS N | IS % | OOS N | OOS % |
|---|---:|---:|---:|---:|
| SL | 60,403 | 93.7% | 16,578 | 91.8% |
| TP | 3,847 | 6.0% | 1,095 | 6.1% |
| entry_past_day_end | 49 | 0.1% | 355 | 2.0% |
| eod | 150 | 0.2% | 27 | 0.1% |
