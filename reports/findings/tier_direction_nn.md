# Tier direction NN

Generated: 2026-04-22T20:35:11
Trades source: `training_iso/output/trades/iso_is.pkl`
Valid: 5244. Train 3733 / Val 731 / Test 780.
Best val AUC: 0.5210
Test AUC: 0.5020  accuracy: 0.604

## Calibration (test)

| P(win) bucket | N | actual win% |
|---|---:|---:|
| 0.5–0.6 | 723 | 60.0% |
| 0.6–0.7 | 57 | 64.9% |

## Per-tier test accuracy

| Tier | N | Accuracy |
|---|---:|---:|
| NMP_FADE | 539 | 61.6% |
| NMP_RIDE | 85 | 50.6% |
| FADE_AGAINST | 60 | 50.0% |
| TREND_FOLLOWER | 32 | 78.1% |
| KILL_SHOT_CALM | 31 | 83.9% |
| RIDE_AGAINST | 10 | 40.0% |
| CASCADE | 6 | 66.7% |
| KILL_SHOT_ACTIVE | 6 | 66.7% |
| MTF_EXHAUSTION | 6 | 16.7% |
| MTF_BREAKOUT | 5 | 40.0% |

## Tier mix (train + val + test)

| Tier | N |
|---|---:|
| NMP_FADE | 3521 |
| NMP_RIDE | 619 |
| FADE_AGAINST | 484 |
| KILL_SHOT_CALM | 265 |
| TREND_FOLLOWER | 139 |
| RIDE_AGAINST | 68 |
| CASCADE | 66 |
| KILL_SHOT_ACTIVE | 36 |
| MTF_BREAKOUT | 26 |
| MTF_EXHAUSTION | 20 |

## Reproduction

```
python tools/train_tier_direction_nn.py --trades training_iso/output/trades/iso_is.pkl
```