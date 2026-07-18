# Macro context via R-CURVE slope (1h & 4h trailing regression) — CLEAN, trade-level

with_macro = R-curve slope sign agrees with trade direction.

## OOS — % AGAINST macro by archetype (R-curve)
| archetype | n | %against 1h-R | %against 4h-R |
|---|---|---|---|
| STOPPED | 1551 | 37% | 42% |
| GAVE_BACK | 1438 | 38% | 40% |
| CHOP | 190 | 40% | 43% |
| SMALL_LOSS | 123 | 44% | 50% |
| SMALL_WIN | 686 | 36% | 44% |
| CLEAN_RIDE | 432 | 34% | 40% |

## OOS — expectancy: with vs against R-curve macro (trades | net $/tr | PF)
- all OOS:        4749 | -2.00 | 0.96
- 1h-R WITH:      2578 | -1.07 | 0.98
- 1h-R AGAINST:   1748 | -7.14 | 0.85
- 4h-R WITH:      1820 | -1.39 | 0.97
- 4h-R AGAINST:   1991 | -3.28 | 0.93
- 1h&4h-R WITH:   1219 | -0.57 | 0.99

## SUB-PERIOD VALIDATION — net $/tr (PF) per period: all vs 1h-R-WITH vs 4h-R-WITH
| period | all | 1h-R WITH | 4h-R WITH | 1h&4h WITH |
|---|---|---|---|---|
| IS_2024 | 2031 | +3.26 | 1.07 | 1186 | +3.83 | 1.08 | 722 | -1.30 | 0.97 | 521 | +1.57 | 1.03 |
| 2025H1 | 2370 | -0.44 | 0.99 | 1297 | -0.91 | 0.98 | 927 | +0.85 | 1.02 | 614 | +0.50 | 1.01 |
| 2025H2 | 1307 | -0.69 | 0.99 | 684 | -2.45 | 0.95 | 471 | +0.65 | 1.01 | 310 | +0.94 | 1.02 |
| 2026 | 1072 | -7.04 | 0.85 | 597 | +0.19 | 1.00 | 422 | -8.58 | 0.83 | 295 | -4.39 | 0.91 |

Read: the with-macro edge is REAL only if 1h-R-WITH (or 4h) beats 'all' in EVERY OOS sub-period. If it flips sign across periods, it's period-luck, not an edge.