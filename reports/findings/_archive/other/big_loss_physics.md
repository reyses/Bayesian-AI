# BIG_LOSS Physics — when does the bleed commit?

BIG_LOSS (pnl < -$50): **2,219 trades**, total $-355,948. Largest single drain on engine PnL.

## Tier source of BIG_LOSS

| Tier | N | avg $/trade | total |
|---|---:|---:|---:|
| NMP_FADE | 1296 | $-134 | $-173,348 |
| RIDE_AGAINST | 337 | $-99 | $-33,431 |
| FADE_AGAINST | 135 | $-270 | $-36,417 |
| MTF_BREAKOUT | 124 | $-169 | $-20,932 |
| NMP_RIDE | 107 | $-377 | $-40,298 |
| TREND_FOLLOWER | 105 | $-177 | $-18,560 |
| MTF_EXHAUSTION | 57 | $-332 | $-18,896 |
| CASCADE | 36 | $-339 | $-12,187 |
| KILL_SHOT_INVERSE | 22 | $-85 | $-1,879 |

## Median MAE by bar (big_loss vs winners vs mild_loss)

| bar | BIG_LOSS MAE | Winners MAE | Mild Losers MAE | Gap (BL − W) |
|---:|---:|---:|---:|---:|
| 1 | $-4 | $+0 | $-2 | $-4 |
| 2 | $-10 | $-2 | $-5 | $-8 |
| 3 | $-16 | $-3 | $-8 | $-13 |
| 5 | $-25 | $-6 | $-12 | $-20 |
| 7 | $-32 | $-8 | $-16 | $-25 |
| 10 | $-43 | $-10 | $-20 | $-34 |
| 15 | $-62 | $-11 | $-24 | $-50 |
| 20 | $-58 | $-12 | $-24 | $-46 |
| 30 | $-75 | $-14 | $-30 | $-61 |

## When does BIG_LOSS commit? (MAE-threshold crossing)

| MAE threshold | % BIG_LOSS crossed | Median bar | % winners crossed |
|---:|---:|---:|---:|
| $-10 | 100% | 2 | 45% |
| $-20 | 100% | 4 | 28% |
| $-30 | 100% | 7 | 19% |
| $-40 | 100% | 9 | 14% |
| $-50 | 100% | 12 | 11% |

**Rule candidate**: if winners rarely dip to -$X but BIG_LOSS regularly does, a hard MAE stop at -$X catches the bleeders without hitting winners.
