# Per-day PnL distribution — pivot_physics_chains

Config: r_entry=$2.0  r_reg=$8.0  min_res=0.5  sniper=30s  mode_bin=$25.0

| Config | Days | Mean | Median | Mode ($25 bin) | p05 | p25 | p75 | p95 | Min | Max | Win-day% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IS chains=1 | 277 | $+273.3 | $+142.0 | $+12 (18%) | $-488 | $+0 | $+539 | $+1,131 | $-1,937 | $+4,914 | 62% |
| OOS chains=1 | 68 | $+437.8 | $+332.0 | $+12 (18%) | $-403 | $+0 | $+865 | $+1,273 | $-660 | $+1,950 | 68% |

## How to read

- **Mode**: most common $25-bucket. Shows the typical day-PnL a trader experiences.
- **Median** < **Mean**: right-tail skew (rare big wins lift the average).
- **Win-day%**: fraction of days with net > $0.
- **p05**: 5th percentile — your worst reasonably-likely day.
