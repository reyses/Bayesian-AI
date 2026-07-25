# F-space label divergence — gen-0 exits vs gen-1 response
gen-0: 156 eps, 734 EXIT frames. gen-1 overlap: 112 paired exit-frames, 594 paired non-exit frames.

## 1. F-space signature of gen-0 exits (Cohen's d, exit vs all)
| feature | mean@exit | mean@all | d |
|---|---|---|---|
| giveback_pct | +64.490 | +34.120 | +0.56 |
| vr_exact | +0.346 | +0.430 | -0.37 |
| leg_amp | +50.579 | +67.879 | -0.32 |
| z_21 | +0.208 | +0.020 | +0.16 |
| hurst_30 | +0.629 | +0.606 | +0.16 |
| z_se_30 | +0.147 | -0.013 | +0.13 |
| band_pos_30 | +0.600 | +0.476 | +0.11 |
| leg_age_m | +4.183 | +4.551 | -0.09 |
| price_velocity_30 | -0.765 | -0.404 | -0.09 |
| reversion_prob_30 | +0.941 | +0.932 | +0.07 |
| price_accel_30 | +0.028 | +0.005 | +0.03 |

## 2. Gen-1 response on the SAME frames (paired)
- median p_exit on gen-0-EXIT frames: 0.0043 (p90 0.1704)
- median p_exit elsewhere:            0.0000 (p90 0.0023)

Reading: if gen-1 p_exit is materially ELEVATED on the frames where gen-0 pulled the trigger, the two labelers share a decision boundary and the genome only damped the gain. If flat, the boundary MOVED — education/rules changed WHERE the teacher looks, not just how hard it fires.
