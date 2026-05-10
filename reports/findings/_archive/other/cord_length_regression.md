# Cord length — regression-line perspective

Regression window: 60 1m bars (= 60 min).

**Price cord** = zigzag path on closes (noise-inclusive).
**Regression cord** = zigzag path on the smoothed regression-line fitted values.
**Efficiency** = regression_cord / price_cord. Higher = cleaner trend; lower = more noise.

## IS aggregate

| R | Price $/day | Price legs/day | Reg $/day | Reg legs/day | Efficiency |
|---:|---:|---:|---:|---:|---:|
| all bars | $10,474 | 1135 | **$2,633** | 1076 | 25.1% |
| $5 | $9,976 | 343 | **$2,578** | 28 | 25.8% |
| $10 | $9,140 | 223 | **$2,541** | 23 | 27.8% |
| $15 | $8,375 | 160 | **$2,504** | 20 | 29.9% |
| $20 | $7,702 | 120 | **$2,470** | 18 | 32.1% |
| $30 | $6,672 | 78 | **$2,398** | 15 | 35.9% |

## OOS aggregate

| R | Price $/day | Price legs/day | Reg $/day | Reg legs/day | Efficiency |
|---:|---:|---:|---:|---:|---:|
| all bars | $12,400 | 1128 | **$3,009** | 1069 | 24.3% |
| $5 | $11,983 | 387 | **$2,945** | 29 | 24.6% |
| $10 | $11,146 | 267 | **$2,908** | 24 | 26.1% |
| $15 | $10,285 | 196 | **$2,874** | 21 | 27.9% |
| $20 | $9,518 | 151 | **$2,836** | 19 | 29.8% |
| $30 | $8,263 | 100 | **$2,765** | 16 | 33.5% |

## Interpretation

Regression cord = upper bound for a strategy that ONLY captures smooth trend moves (no intra-bar noise). Price cord − regression cord = the "unextractable" noise component.

At R=$15: IS regression cord = $2,504/day (vs price cord $8,375/day, efficiency 29.9%).
           OOS regression cord = $2,874/day (vs price cord $10,285/day, efficiency 27.9%).

Current NMP engine: +$311/day IS / +$67/day OOS.
NMP captures 12.4% of IS regression cord.
NMP captures 2.3% of OOS regression cord.
