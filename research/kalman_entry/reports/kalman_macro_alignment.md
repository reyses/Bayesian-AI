# STOPPED = against-macro? (with/against 1h & 4h trend at entry) — CLEAN, trade-level

with_macro = sign(price move over lookback into entry) matches trade direction.

## OOS
### (a) % AGAINST-macro by archetype (hypothesis: STOPPED is high)
| archetype | n | % against 1h | % against 4h |
|---|---|---|---|
| STOPPED | 2041 | 28% | 41% |
| CHOP | 217 | 21% | 36% |
| GAVE_BACK | 1863 | 27% | 40% |
| SMALL_LOSS | 147 | 25% | 43% |
| SMALL_WIN | 886 | 25% | 40% |
| CLEAN_RIDE | 553 | 23% | 41% |

### (b) outcomes: with vs against macro  |  (c) entry-filter effect
| macro | filter | trades | net $/tr | PF | stop-rate | mfe |
|---|---|---|---|---|---|---|
| (all) | 6069 | -1.23 | 0.97 | 34% | 55 |
| 1h WITH | 4180 | +1.17 | 1.03 | 34% | 56 |
| 1h AGAINST | 1481 | -8.13 | 0.84 | 36% | 54 |
| 4h WITH | 3253 | -2.72 | 0.94 | 33% | 53 |
| 4h AGAINST | 2158 | -3.26 | 0.93 | 35% | 54 |
| 1h&4h WITH | 2591 | -2.61 | 0.94 | 33% | 54 |

## IS
### (a) % AGAINST-macro by archetype (hypothesis: STOPPED is high)
| archetype | n | % against 1h | % against 4h |
|---|---|---|---|
| STOPPED | 230 | 18% | 32% |
| CHOP | 25 | 4% | 29% |
| GAVE_BACK | 239 | 18% | 36% |
| SMALL_LOSS | 13 | 17% | 30% |
| SMALL_WIN | 114 | 20% | 36% |
| CLEAN_RIDE | 76 | 15% | 38% |

### (b) outcomes: with vs against macro  |  (c) entry-filter effect
| macro | filter | trades | net $/tr | PF | stop-rate | mfe |
|---|---|---|---|---|---|---|
| (all) | 711 | +6.49 | 1.15 | 32% | 61 |
| 1h WITH | 567 | +5.44 | 1.12 | 32% | 60 |
| 1h AGAINST | 119 | +3.83 | 1.09 | 34% | 61 |
| 4h WITH | 381 | +3.66 | 1.08 | 32% | 58 |
| 4h AGAINST | 198 | +14.31 | 1.34 | 29% | 65 |
| 1h&4h WITH | 343 | +2.23 | 1.05 | 32% | 57 |
