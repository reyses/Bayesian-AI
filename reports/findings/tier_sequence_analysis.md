# Tier Sequence Analysis (window: 15 bars)

For each A→B pair where B fires same direction within 15 bars of A (and different tier), compare A-trade PnL when followed vs A-alone (no follower).

## Per-tier baseline (all primaries, any follow-up)

| Tier | N total | N solo (no follower) | $/tr total | $/tr solo |
|---|---:|---:|---:|---:|
| CASCADE | 113 | 13 | $+42.84 | $-67.23 |
| KILL_SHOT_CALM | 361 | 100 | $+1.96 | $+1.92 |
| NMP_RIDE | 299 | 214 | $+40.01 | $+24.95 |
| MTF_EXHAUSTION | 154 | 103 | $+43.78 | $+82.77 |
| KILL_SHOT_ACTIVE | 103 | 64 | $+12.29 | $+14.42 |
| FADE_AGAINST | 326 | 46 | $+23.19 | $-58.71 |
| NMP_FADE | 10195 | 6428 | $+1.23 | $-0.26 |
| MTF_BREAKOUT | 731 | 675 | $+0.30 | $+3.94 |
| TREND_FOLLOWER | 780 | 472 | $+0.82 | $-5.63 |
| RIDE_AGAINST | 4306 | 3543 | $+0.83 | $+2.16 |

## Amplifier pairs (A-follows-B boosts A's WR / $/tr)

Top pairs ranked by **A-trade $/tr lift when B follows**. If A-then-B $/tr > A-solo $/tr by >= $2, B firing is a positive confirmation signal for A.

| A | B | Pairs | A-solo $/tr | A-then-B $/tr | Lift | A WR solo | A WR then-B |
|---|---|---:|---:|---:|---:|---:|---:|
| CASCADE | RIDE_AGAINST | 68 | $-67.23 | $+89.43 | $+156.66 ** | 38% | 60% |
| CASCADE | NMP_FADE | 80 | $-67.23 | $+40.51 | $+107.74 ** | 38% | 55% |
| FADE_AGAINST | NMP_FADE | 286 | $-58.71 | $+41.53 | $+100.23 ** | 43% | 49% |
| NMP_RIDE | NMP_FADE | 69 | $+24.95 | $+79.18 | $+54.23 ** | 50% | 52% |
| NMP_FADE | MTF_EXHAUSTION | 31 | $-0.26 | $+51.11 | $+51.37 ** | 56% | 90% |
| NMP_FADE | KILL_SHOT_ACTIVE | 23 | $-0.26 | $+43.48 | $+43.74 ** | 56% | 91% |
| RIDE_AGAINST | MTF_EXHAUSTION | 21 | $+2.16 | $+38.50 | $+36.34 ** | 65% | 95% |
| NMP_RIDE | RIDE_AGAINST | 29 | $+24.95 | $+59.55 | $+34.60 ** | 50% | 45% |
| NMP_FADE | NMP_RIDE | 57 | $-0.26 | $+30.68 | $+30.94 ** | 56% | 91% |
| RIDE_AGAINST | NMP_RIDE | 37 | $+2.16 | $+23.03 | $+20.87 ** | 65% | 78% |
| TREND_FOLLOWER | RIDE_AGAINST | 247 | $-5.63 | $+13.79 | $+19.42 ** | 66% | 72% |
| NMP_FADE | RIDE_AGAINST | 3332 | $-0.26 | $+9.06 | $+9.32 ** | 56% | 62% |
| KILL_SHOT_CALM | RIDE_AGAINST | 93 | $+1.92 | $+3.90 | $+1.98 | 56% | 70% |
| NMP_FADE | MTF_BREAKOUT | 547 | $-0.26 | $+0.82 | $+1.08 | 56% | 55% |
| KILL_SHOT_CALM | NMP_FADE | 235 | $+1.92 | $+1.93 | $+0.02 | 56% | 63% |
| RIDE_AGAINST | MTF_BREAKOUT | 389 | $+2.16 | $-0.36 | $-2.52 | 65% | 68% |
| NMP_FADE | KILL_SHOT_CALM | 29 | $-0.26 | $-5.38 | $-5.12 | 56% | 48% |
| KILL_SHOT_ACTIVE | NMP_FADE | 34 | $+14.42 | $+8.24 | $-6.19 | 73% | 74% |
| RIDE_AGAINST | NMP_FADE | 278 | $+2.16 | $-9.30 | $-11.46 | 65% | 64% |
| TREND_FOLLOWER | MTF_BREAKOUT | 89 | $-5.63 | $-25.16 | $-19.53 | 66% | 65% |

## Dampener pairs (B following A HURTS A's outcome)

| A | B | Pairs | A-solo $/tr | A-then-B $/tr | Drop |
|---|---|---:|---:|---:|---:|
| MTF_EXHAUSTION | NMP_FADE | 41 | $+82.77 | $-65.04 | $-147.80 |
| NMP_FADE | TREND_FOLLOWER | 411 | $-0.26 | $-53.54 | $-53.28 |
| MTF_BREAKOUT | RIDE_AGAINST | 20 | $+3.94 | $-48.30 | $-52.24 |
| RIDE_AGAINST | TREND_FOLLOWER | 98 | $+2.16 | $-43.90 | $-46.06 |
| MTF_BREAKOUT | NMP_FADE | 35 | $+3.94 | $-40.61 | $-44.55 |
| NMP_FADE | FADE_AGAINST | 20 | $-0.26 | $-33.10 | $-32.84 |
| TREND_FOLLOWER | MTF_BREAKOUT | 89 | $-5.63 | $-25.16 | $-19.53 |
| RIDE_AGAINST | NMP_FADE | 278 | $+2.16 | $-9.30 | $-11.46 |
| KILL_SHOT_ACTIVE | NMP_FADE | 34 | $+14.42 | $+8.24 | $-6.19 |
| NMP_FADE | KILL_SHOT_CALM | 29 | $-0.26 | $-5.38 | $-5.12 |
| RIDE_AGAINST | MTF_BREAKOUT | 389 | $+2.16 | $-0.36 | $-2.52 |
