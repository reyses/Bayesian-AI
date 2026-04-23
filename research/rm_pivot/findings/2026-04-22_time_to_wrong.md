# Time-to-wrong diagnostic

Generated: 2026-04-22T12:46:26
Trades analyzed: 45592 (winners 6942, losers 38077)

## Seconds to first adverse threshold

(How fast does price cross below entry by N dollars, counting 1s bar lows)

| Threshold | N winners crossed | Win median (s) | Win p25 | Win p75 | N losers crossed | Lose median (s) | Lose p25 | Lose p75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| −$1 | 6124 | 2 | 0 | 29 | 36954 | 0 | 0 | 5 |
| −$3 | 5182 | 25 | 3 | 147 | 35403 | 5 | 0 | 29 |
| −$5 | 4319 | 66 | 12 | 349 | 33862 | 14 | 2 | 60 |
| −$10 | 2457 | 371 | 85 | 1014 | 30084 | 43 | 10 | 129 |

## Seconds to first favorable threshold

| Threshold | N winners crossed | Win median (s) | Win p25 | Win p75 | N losers crossed | Lose median (s) | Lose p25 | Lose p75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| +$1 | 6931 | 0 | 0 | 3 | 36196 | 0 | 0 | 7 |
| +$3 | 6805 | 5 | 1 | 22 | 33882 | 6 | 1 | 36 |
| +$5 | 6604 | 14 | 3 | 55 | 31675 | 17 | 3 | 70 |
| +$10 | 6023 | 47 | 11 | 137 | 26701 | 52 | 12 | 152 |

## Interpretation

If losers cross −$X much faster than winners cross +$X, we have an **early-detection signal**: "if down $X in Y seconds, flip."
