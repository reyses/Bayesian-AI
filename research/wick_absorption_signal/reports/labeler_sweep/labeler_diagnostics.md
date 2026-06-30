# Aperiodic Turn-Labeler Diagnostics

**Timeframe:** 1-minute
**Volatility Filter:** 50-bar trailing standard deviation of log-returns (causal).
**Turn Logic:** Plain centered cubic regression (slope=0, curv sign) + ZigZag accumulation filter.

## Table 1: Resolution Stability (N-Sweep)
We fix the magnitude threshold $k=3.0$ and vary the centered cubic resolution $N$. 
If the located turn set is stable, $N$ is just a resolution slider, and we can safely lock $N=20$.

| Resolution (N) | Raw Turns | Filtered Turns (k=3.0) | Median Spacing (Mins) | Mean Spacing (Mins) |
| --- | --- | --- | --- | --- |
| 12 | 32139 | 4020 | 24.0 | 44.1 |
| 20 | 20301 | 3264 | 30.0 | 54.4 |
| 40 | 12447 | 2226 | 42.0 | 79.7 |

## Table 2: Economic Scale (k-Sweep)
We fix the resolution $N=20$ and vary the magnitude threshold $k$. 
Choose $k$ based on the **Median Swing Magnitude** that corresponds to the minimum move size the strategy intends to capture.

| Magnitude Threshold (k) | Filtered Turns | Median Swing Mag (%) | Mean Swing Mag (%) | Median Spacing (Mins) |
| --- | --- | --- | --- | --- |
| 0.5 | 7444 | 0.068% | 82.697% | 13.0 |
| 1.0 | 6317 | 0.077% | 9.983% | 15.0 |
| 2.0 | 4584 | 0.105% | 22.475% | 21.0 |
| 3.0 | 3264 | 0.136% | 44.195% | 30.0 |
| 4.0 | 2460 | 0.166% | 54.123% | 40.0 |
| 5.0 | 1922 | 0.199% | 69.121% | 52.0 |
| 6.0 | 1510 | 0.239% | 74.360% | 67.0 |
