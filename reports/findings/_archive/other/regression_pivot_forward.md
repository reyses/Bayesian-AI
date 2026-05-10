# Regression-pivot forward pass

r_reg=$8.0 TP=$30.0 SL=$15.0 window=60

Zigzag applied to the REGRESSION LINE (1h-window OLS fitted values), not to price. Entry at pivot confirmation bar (1m close). Exit resolved at 1s intra-bar granularity.

| Dataset | Days | Reg pivots | Trades | $/day | $/trade | WR | $WR | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 277 | 3,554 | 3,554 | $+5 | $+0.37 | 34.5% | +4% | $+1,329 |
| OOS | 68 | 1,180 | 1,180 | $-6 | $-0.33 | 32.8% | -3% | $-386 |

## Exit breakdown

| Reason | IS N | IS % | OOS N | OOS % |
|---|---:|---:|---:|---:|
| SL | 2,303 | 64.8% | 775 | 65.7% |
| TP | 1,190 | 33.5% | 373 | 31.6% |
| entry_past_day_end | 2 | 0.1% | 24 | 2.0% |
| eod | 59 | 1.7% | 8 | 0.7% |

## Cord-capture context

| | Reg cord ceiling | System $/day | Capture % |
|---|---:|---:|---:|
| IS | $2,504 | $+5 | 0.2% |
| OOS | $2,874 | $-6 | -0.2% |
