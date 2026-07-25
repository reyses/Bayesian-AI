# Ground-truth (oracle) labels vs ours — F-space
156 episodes. Oracle exit = true peak of favorable px.

## 1. What a perfect exit was worth (the exit-head ceiling here)
- peak−final (points left by never-bail): mean +58.8, median +54.0, p90 +109.5 pts/ep
- friction floor: ~0.9 pts RT — the ceiling is 65x friction on average.

## 2. Distance of our labels from the oracle
- NEVER-BAIL: leaves the full peak−final on the table (above) but pays zero churn.
- GEN-0 (N=130 exited eps): median 8 min EARLY vs oracle; median +56.9 pts left vs peak (exits before the move develops — worse than never-bail both ways).

## 3. Is the top VISIBLE in F-space? (features at oracle peaks vs all frames)
| feature | mean@peak | mean@all | d |
|---|---|---|---|
| price_accel_30 | -0.192 | +0.004 | -0.25 |
| z_se_30 | -0.259 | -0.011 | -0.20 |
| hurst_30 | +0.632 | +0.606 | +0.18 |
| band_pos_30 | +0.274 | +0.478 | -0.18 |
| z_21 | -0.123 | +0.021 | -0.12 |
| reversion_prob_30 | +0.920 | +0.932 | -0.08 |
| vr_exact | +0.425 | +0.429 | -0.02 |
| price_velocity_30 | -0.440 | -0.404 | -0.01 |

Reading: large |d| = the true top has an observable signature these features capture (an exit head CAN learn it). All |d| small = tops are F-space-invisible here and never-bail wins by information default — the curriculum cannot teach exits.
