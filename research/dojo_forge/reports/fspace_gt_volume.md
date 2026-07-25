# Volume progression around oracle peaks (within-episode z-scores)
156 episodes aligned at their true peak (offset 0).

| offset (min) | vol_velocity_1b | vol_velocity_30 | vol_accel_30 | price_velocity_30 | n |
|---|---|---|---|---|---|
| -5 | -0.14 | -0.10 | -0.17 | +0.14 | 147 |
| -4 | -0.07 | -0.06 | +0.05 | +0.08 | 152 |
| -3 | -0.02 | -0.02 | -0.01 | -0.03 | 153 |
| -2 | +0.20 | +0.15 | +0.21 | -0.11 | 154 |
| -1 | +0.20 | +0.18 | +0.01 | -0.09 | 155 |
| +0 **PEAK** | +0.06 | +0.10 | -0.09 | -0.21 | 156 |
| +1 | +0.06 | +0.26 | +0.13 | -0.36 | 153 |
| +2 | -0.10 | +0.13 | -0.11 | -0.28 | 152 |
| +3 | -0.06 | +0.16 | +0.05 | -0.22 | 150 |
| +4 | -0.11 | +0.02 | -0.12 | -0.15 | 148 |
| +5 | -0.04 | -0.07 | -0.06 | -0.03 | 147 |

Reading: values are z-scores vs the episode's own distribution. A volume CLIMAX at tops shows vol metrics spiking at offset 0; EXHAUSTION/divergence shows price velocity fading while volume drains into the peak; post-peak columns show what confirmation the first minutes after the top offer.
