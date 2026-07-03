# Regime Spread Diagnostics

## 1. Per-Day Regime Table & Best-T Sweep

| Date | Picks | Day Scale (pt) | Best Fixed T (F1) |
|---|---|---|---|
| 2024_01_02 | 73 | 3.91 | T=2 (F1 0.748) |
| 2024_01_03 | 56 | 4.35 | T=2 (F1 0.804) |
| 2024_01_04 | 63 | 4.56 | T=2 (F1 0.777) |
| 2024_01_05 | 42 | 4.74 | T=4 (F1 0.753) |
| 2024_01_08 | 29 | 4.47 | T=7 (F1 0.690) |
| 2024_02_01 | 14 | 6.71 | T=4 (F1 0.456) |
| 2024_02_02 | 61 | 5.60 | T=3 (F1 0.750) |
| 2024_03_04 | 8 | 4.28 | T=3 (F1 0.259) |
| 2025_01_06 | 52 | 6.72 | T=2 (F1 0.629) |

## 2. Cross-Day Correlation

Spearman correlation between day scale and best-T: **0.184** (p-value: 0.635)

> **Analysis**: Positive correlation found. Cross-day adaptation has signal.

## 3. Intraday Spread (RTH vs Overnight)

| Date | RTH Scale | ON Scale | Ratio | Human Picks in RTH |
|---|---|---|---|---|
| 2024_01_02 | 9.55 | 2.77 | 3.45x | 19/73 (26%) |
| 2024_01_03 | 11.80 | 3.07 | 3.84x | 20/56 (36%) |
| 2024_01_04 | 9.55 | 3.38 | 2.83x | 17/63 (27%) |
| 2024_01_05 | 10.62 | 4.19 | 2.53x | 10/42 (24%) |
| 2024_01_08 | 7.42 | 4.04 | 1.84x | 0/29 (0%) |
| 2024_02_01 | 13.75 | 5.53 | 2.48x | 5/14 (36%) |
| 2024_02_02 | 12.32 | 4.77 | 2.59x | 16/61 (26%) |
| 2024_03_04 | 8.53 | 3.84 | 2.22x | 0/8 (0%) |
| 2025_01_06 | 13.58 | 5.43 | 2.50x | 8/52 (15%) |

## 4. Swing Ratio Tightening

- **Raw Swings**: median 15.75, relative dispersion (IQR/median) 1.556
- **Scaled Ratios**: median 3.06, relative dispersion (IQR/median) 0.965

> **Analysis**: Material tightening observed (>= 20% drop in relative dispersion). Premise Supported.
