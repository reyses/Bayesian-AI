# Time-gated flip — let the dip work T min, flip if not recovered

N=23,378. never-bail mean abs = +9.36 t/ep. Arm at dip<=-DSMALL, flip at +T if still underwater.

| DSMALL | T (min) | flip mean | Δ vs never-bail | %flipped | wrong-flip% |
|---|---|---|---|---|---|
| 4 | 3 | +6.19 | -3.17 | 48% | 74% |
| 4 | 5 | +7.79 | -1.57 | 40% | 80% |
| 4 | 7 | +8.68 | -0.68 | 36% | 85% |
| 4 | 10 | +7.97 | -1.39 | 32% | 90% |
| 4 | 15 | +8.15 | -1.21 | 29% | 96% |
| 8 | 3 | +6.40 | -2.96 | 50% | 76% |
| 8 | 5 | +7.26 | -2.10 | 43% | 82% |
| 8 | 7 | +8.91 | -0.45 | 39% | 87% |
| 8 | 10 | +7.83 | -1.53 | 35% | 92% |
| 8 | 15 | +8.10 | -1.26 | 33% | 96% |

## Best: DSMALL=8, T=7min (mean +8.91 vs never-bail +9.36)
| class | never-bail | best time-flip |
|---|---|---|
| good_clean | +291.6 | +291.6 |
| good_dipped | +221.5 | +154.7 |
| wrong | -198.6 | -160.3 |
| dead_band | -2.9 | -39.4 |

Read: if the best time-flip beats never-bail, the recovery-TIME horizon discriminates good_dipped (recover fast, held) from runaways (never bounce, flipped) where depth could not. wrong-flip% = purity of what we flip. If it still loses, good_dipped fast-recovery overlaps runaways too much even in time.
