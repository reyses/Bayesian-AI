# Teacher effectiveness — tiered full-depth run (2026-07-23)
N = 156 episodes (156 packets; missing-truth skipped: 0). Teacher exited in 130/156 episodes (median exit minute 7; oracle median minute 15).

| policy | mean pts | median pts | PF-Trade-WR | capture ratio vs oracle |
|---|---|---|---|---|
| ORACLE (ceiling) | 87.39 | 70.00 | 509.664 | 1.00 |
| TEACHER (qwen tiered) | 26.31 | 13.00 | 3.370 | 0.301 |
| RIDE-TO-END (never-bail) | 38.87 | 27.50 | 3.804 | 0.445 |
| FIXED-5m | 24.73 | 19.00 | 8.483 | 0.283 |

**Teacher − RIDE-TO-END**: -12.56 pts/episode, 95% CI [-22.00, -3.39] — SIGNIFICANT (N=156, 4000 resamples)
**Teacher − FIXED-5m**:   +1.59 pts/episode, 95% CI [-7.91, +10.40] — NOT significant (CI includes 0) (N=156)

## Caveats (honest)
- Points-from-entry per truth drift paths; NOT $-net-of-costs; no CI on the oracle ratio.
- gen-0 teacher = 3 seed genome rules — this is the BASELINE generation, not a tuned one.
- Teacher exit = first p_exit>0.5; threshold sensitivity unexplored (labels are continuous).
- Single run, single seed, deterministic; 7/2884 frames ctx-tainted (0.24%, random).

## Per-type breakdown
- **midflip** (N=81): teacher 17.3 vs ride-end 8.7 vs oracle 62.9
- **winner** (N=75): teacher 36.0 vs ride-end 71.5 vs oracle 113.8
