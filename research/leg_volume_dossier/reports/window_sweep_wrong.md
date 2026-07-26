# Window sweep — when does the wrong-direction signal switch on?
composite (no_recovery+mostly_under+low_ER-coil) >= 2, day-block CI.

| minute W | eps eligible | fired | P(wrong|on) | base | lift | 95% CI | sig |
|---|---|---|---|---|---|---|---|
| 3 | 156 | 13 | 31% | 37% | -6% | [-30%, +15%] |  |
| 5 | 156 | 20 | 60% | 37% | +23% | [+7%, +43%] | YES |
| 7 | 156 | 16 | 75% | 37% | +38% | [+21%, +58%] | YES |
| 8 | 156 | 15 | 53% | 37% | +17% | [-9%, +41%] |  |
| 10 | 156 | 21 | 76% | 37% | +40% | [+23%, +56%] | YES |
| 12 | 156 | 24 | 83% | 37% | +47% | [+33%, +62%] | YES |
| 15 | 147 | 7 | 86% | 34% | +52% | [+31%, +74%] | YES |

**Signal switches on at minute 5** (first W where CI clears 0). Note: eligible-episode count falls as W grows (short episodes drop out) — survivorship, read with the n column.
