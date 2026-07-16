# Combiner preview — pooled P(right) across all dossier streams
- pooled N = 713926 fires (312596 train 2024 / 401330 test 2025+26) across 38 streams
- pooled OOS AUC **0.689** (test base agreement 0.550)
- coefs (standardized): pivot_age_min=+0.113, sig_with_leg=+0.376, tod=+0.104, inter=-0.152, consensus=+0.385, is_ADX08=+0.008, is_ATR09=-0.213, is_CROSS11=+0.003, is_CURVE=+0.020, is_DOW19=-0.140, is_FIB17=-0.010, is_HNS22=-0.004, is_MACD07=-0.363, is_NMP=-0.169, is_NMPLAMBDA=-0.006, is_NMPTCASCADE=-0.027, is_NMPTFADEAGN=-0.018, is_NMPTFADECALM=-0.119, is_NMPTFREIGHT=+0.031, is_NMPTKILLSHOT=-0.036, is_NMPTMTFBRK=+0.052, is_NMPTMTFEXH=+0.031, is_NMPTRIDEAGN=+0.018, is_OHLC01=-0.008, is_ORB02=+0.084, is_PIVOT16=-0.074, is_PTRNENGULF=+0.043, is_PTRNHAMMER=-0.006, is_RENKO24=+0.105, is_ROUND05=+0.077, is_RSI06=-0.448, is_SAR23=-0.087, is_SCALP18=-0.031, is_SEASON12=-0.002, is_SQZ04=+0.005, is_TMPL0=+0.295, is_TUNNEL20=+0.037, is_VA13=-0.016, is_VP01=-0.011, is_VWAP03=-0.100, is_VWMA10=+0.011, is_ZIGZAG=+0.229, is_ZONE21=+0.034

| P-decile | N | mean P | observed agreement | day-block 95% CI |
|---|---|---|---|---|
| 0 | 40133 | 0.21 | 0.20 | [0.19,0.21] |
| 1 | 40133 | 0.41 | 0.39 | [0.37,0.40] |
| 2 | 40133 | 0.48 | 0.44 | [0.43,0.45] |
| 3 | 40133 | 0.53 | 0.50 | [0.49,0.51] |
| 4 | 40133 | 0.59 | 0.55 | [0.54,0.56] |
| 5 | 40133 | 0.63 | 0.60 | [0.59,0.61] |
| 6 | 40133 | 0.68 | 0.64 | [0.63,0.65] |
| 7 | 40133 | 0.72 | 0.68 | [0.67,0.69] |
| 8 | 40134 | 0.77 | 0.72 | [0.71,0.73] |
| 9 | 40132 | 0.85 | 0.78 | [0.76,0.79] |

## Consensus effect (test set, raw)
| same-direction co-fires (+-3min) | N | agreement |
|---|---|---|
| 0 | 331 | 0.42 |
| 1-2 | 7303 | 0.43 |
| 3-5 | 61398 | 0.46 |
| 6+ | 332298 | 0.57 |