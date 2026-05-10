# Pivot direction accuracy — stratified by chord + wick

Zigzag threshold $15.0. Only pivots with |residual|>=0.5.

IS events: 31,674 | OOS events: 9,528

**Baseline accuracy (no stratification)**
- IS: **68.4%**
- OOS: **65.1%**

## 1D: by chord ratio (noise vs trend regime)

| Regime | IS N | IS acc | OOS N | OOS acc |
|---|---:|---:|---:|---:|
| VERY_NOISE (<0.05) | 2,681 | 62.1% | 855 | 60.0% |
| NOISE (0.05-0.15) | 5,049 | 63.3% | 1,603 | 60.9% |
| MIXED (0.15-0.30) | 7,250 | 64.0% | 2,209 | 62.0% |
| TREND (0.30-0.50) | 8,151 | 68.0% | 2,424 | 65.5% |
| STRONG_TREND (>0.50) | 8,543 | 77.4% | 2,437 | 72.2% |

## 1D: by rejection-wick % at pivot bar

"Rejection wick" = the wick in the direction of the predicted bounce. LONG pred → lower wick. SHORT pred → upper wick.

| Rejection wick | IS N | IS acc | OOS N | OOS acc |
|---|---:|---:|---:|---:|
| NONE (<0.15) | 12,970 | 58.9% | 4,121 | 57.0% |
| MILD (0.15-0.30) | 8,536 | 70.7% | 2,551 | 67.3% |
| MED (0.30-0.50) | 6,453 | 75.6% | 1,856 | 72.0% |
| STRONG (>0.50) | 3,715 | 83.6% | 1,000 | 80.4% |

## 2D: chord × wick (IS)

| Chord \ Wick | NONE (<0.15) | MILD (0.15-0.30) | MED (0.30-0.50) | STRONG (>0.50) |
|---|---:|---:|---:|---:|
| VERY_NOISE (<0.05) | **51.1%** (n=1265) | **66.8%** (n=687) | **73.8%** (n=455) | **81.0%** (n=274) |
| NOISE (0.05-0.15) | **53.3%** (n=2191) | **66.5%** (n=1354) | **72.1%** (n=953) | **80.0%** (n=551) |
| MIXED (0.15-0.30) | **55.0%** (n=3075) | **66.5%** (n=1989) | **71.5%** (n=1440) | **80.4%** (n=746) |
| TREND (0.30-0.50) | **57.9%** (n=3395) | **71.2%** (n=2163) | **75.8%** (n=1627) | **83.4%** (n=966) |
| STRONG_TREND (>0.50) | **71.3%** (n=3044) | **77.4%** (n=2343) | **80.6%** (n=1978) | **88.1%** (n=1178) |

## 2D: chord × wick (OOS)

| Chord \ Wick | NONE (<0.15) | MILD (0.15-0.30) | MED (0.30-0.50) | STRONG (>0.50) |
|---|---:|---:|---:|---:|
| VERY_NOISE (<0.05) | **51.0%** (n=435) | **63.8%** (n=213) | **68.1%** (n=138) | **88.4%** (n=69) |
| NOISE (0.05-0.15) | **50.7%** (n=738) | **64.2%** (n=436) | **73.2%** (n=269) | **78.8%** (n=160) |
| MIXED (0.15-0.30) | **53.1%** (n=963) | **62.8%** (n=573) | **71.3%** (n=443) | **79.6%** (n=230) |
| TREND (0.30-0.50) | **58.0%** (n=1054) | **69.0%** (n=658) | **71.1%** (n=474) | **77.7%** (n=238) |
| STRONG_TREND (>0.50) | **67.6%** (n=931) | **72.7%** (n=671) | **73.9%** (n=532) | **82.2%** (n=303) |

## 1D: by volume ratio (pivot volume / mean 20-bar lookback)

| Vol regime | IS N | IS acc | OOS N | OOS acc |
|---|---:|---:|---:|---:|
| LOW (<0.7) | 5,656 | 65.3% | 1,669 | 61.2% |
| NORMAL (0.7-1.3) | 14,998 | 64.2% | 4,812 | 62.0% |
| ELEVATED (1.3-2) | 6,714 | 73.4% | 1,938 | 71.3% |
| SPIKE (>2) | 4,306 | 79.2% | 1,109 | 73.8% |

## 1D: by predicted-direction price velocity (pts/bar over last 5 bars)

"WITH prediction" = price velocity sign matches predicted direction. Positive values mean market already moving in the predicted direction at entry.

| Bucket | IS N | IS acc | OOS N | OOS acc |
|---|---:|---:|---:|---:|
| AGAINST prediction (vel opposite) | 26,030 | 81.1% | 7,671 | 78.5% |
| NEUTRAL (|vel|<0.01 pts/bar) | 94 | 35.1% | 27 | 33.3% |
| WITH prediction (vel small) | 877 | 18.2% | 300 | 17.7% |
| WITH prediction (vel strong) | 4,673 | 7.8% | 1,530 | 8.0% |

## 1D: by predicted-direction regression velocity (pts/bar)

| Bucket | IS N | IS acc | OOS N | OOS acc |
|---|---:|---:|---:|---:|
| AGAINST prediction (vel opposite) | 26,926 | 73.1% | 8,077 | 69.6% |
| NEUTRAL (|vel|<0.01 pts/bar) | 65 | 55.4% | 11 | — |
| WITH prediction (vel small) | 1,401 | 46.3% | 366 | 46.2% |
| WITH prediction (vel strong) | 3,282 | 39.3% | 1,074 | 38.0% |

## 2D: wick × volume (inflection combo — IS)

| Wick \ Vol | LOW (<0.7) | NORMAL (0.7-1.3) | ELEVATED (1.3-2) | SPIKE (>2) |
|---|---:|---:|---:|---:|
| NONE (<0.15) | **62.5%** (n=2357) | **54.5%** (n=6154) | **60.9%** (n=2774) | **66.9%** (n=1685) |
| MILD (0.15-0.30) | **67.4%** (n=1507) | **65.2%** (n=4188) | **78.1%** (n=1798) | **84.8%** (n=1043) |
| MED (0.30-0.50) | **66.1%** (n=1155) | **72.6%** (n=3026) | **82.9%** (n=1331) | **86.7%** (n=941) |
| STRONG (>0.50) | **69.2%** (n=637) | **82.8%** (n=1630) | **90.4%** (n=811) | **91.7%** (n=637) |

## 2D: wick × volume (inflection combo — OOS)

| Wick \ Vol | LOW (<0.7) | NORMAL (0.7-1.3) | ELEVATED (1.3-2) | SPIKE (>2) |
|---|---:|---:|---:|---:|
| NONE (<0.15) | **60.8%** (n=663) | **52.0%** (n=2087) | **61.5%** (n=899) | **64.8%** (n=472) |
| MILD (0.15-0.30) | **60.0%** (n=465) | **64.5%** (n=1354) | **75.7%** (n=448) | **79.6%** (n=284) |
| MED (0.30-0.50) | **62.5%** (n=355) | **72.0%** (n=920) | **79.0%** (n=366) | **76.3%** (n=215) |
| STRONG (>0.50) | **63.4%** (n=186) | **80.5%** (n=451) | **89.3%** (n=225) | **88.4%** (n=138) |
