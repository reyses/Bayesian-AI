# Physics-only exit simulator

r_entry=$2.0  r_reg=$8.0  min_res=0.5  sniper=0s

Entry: 1s zigzag pivot + residual direction.
Exit signal: 1m regression direction flipped AND residual sign flipped vs entry.
Exit execution: 30-second sniper window, take running extreme - 1 tick slip.
No stop-loss. EOD force-close at 20:55 UTC.

| Dataset | Days | Trades | $/day | $/trade | WR | $WR | Mean hold | p90 hold | Max win | Max loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS | 277 | 3,908 | $+150 | $+10.60 | 48.5% | +29% | 4416s | 5153s | $+2,836 | $-1,577 |
| OOS | 68 | 1,270 | $+274 | $+14.65 | 52.2% | +46% | 3278s | 4614s | $+766 | $-721 |

## Exit breakdown

| Reason | IS N | IS % | OOS N | OOS % |
|---|---:|---:|---:|---:|
| day_end | 6 | 0.2% | 1 | 0.1% |
| eod | 220 | 5.6% | 55 | 4.3% |
| sniper | 3,239 | 82.9% | 1,054 | 83.0% |
| thesis_broken | 443 | 11.3% | 160 | 12.6% |
