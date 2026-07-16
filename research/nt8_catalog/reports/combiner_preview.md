# Combiner preview — pooled P(right) across all dossier streams
- pooled N = 556813 fires (238697 train 2024 / 318116 test 2025+26) across 37 streams
- pooled OOS AUC **0.677** (test base agreement 0.516)
- coefs (standardized): pivot_age_min=+0.092, sig_with_leg=+0.472, tod=+0.035, inter=-0.142, consensus=+0.000, is_ADX08=+0.021, is_ATR09=-0.234, is_CROSS11=+0.014, is_CURVE=+0.064, is_DOW19=-0.113, is_FIB17=-0.005, is_HNS22=+0.001, is_MACD07=-0.385, is_NMP=-0.161, is_NMPLAMBDA=+0.040, is_NMPTCASCADE=-0.024, is_NMPTFADEAGN=-0.010, is_NMPTFADECALM=-0.075, is_NMPTFREIGHT=+0.086, is_NMPTKILLSHOT=-0.022, is_NMPTMTFBRK=+0.083, is_NMPTMTFEXH=+0.060, is_NMPTRIDEAGN=+0.085, is_OHLC01=+0.006, is_ORB02=+0.109, is_PIVOT16=-0.074, is_PTRNENGULF=+0.116, is_PTRNHAMMER=+0.016, is_RENKO24=+0.140, is_ROUND05=+0.172, is_RSI06=-0.478, is_SAR23=-0.055, is_SCALP18=-0.032, is_SEASON12=+0.005, is_SQZ04=+0.008, is_TUNNEL20=+0.091, is_VA13=-0.008, is_VP01=-0.006, is_VWAP03=-0.066, is_VWMA10=+0.026, is_ZIGZAG=+0.276, is_ZONE21=+0.058

| P-decile | N | mean P | observed agreement | day-block 95% CI |
|---|---|---|---|---|
| 0 | 31812 | 0.18 | 0.18 | [0.17,0.18] |
| 1 | 31812 | 0.38 | 0.37 | [0.36,0.38] |
| 2 | 31811 | 0.46 | 0.46 | [0.44,0.47] |
| 3 | 31812 | 0.48 | 0.47 | [0.46,0.49] |
| 4 | 31811 | 0.50 | 0.50 | [0.49,0.51] |
| 5 | 31812 | 0.54 | 0.52 | [0.51,0.53] |
| 6 | 31811 | 0.63 | 0.60 | [0.59,0.61] |
| 7 | 31812 | 0.66 | 0.64 | [0.62,0.65] |
| 8 | 31811 | 0.68 | 0.68 | [0.67,0.69] |
| 9 | 31812 | 0.76 | 0.75 | [0.74,0.76] |

## Consensus effect (test set, raw)
| same-direction co-fires (+-3min) | N | agreement |
|---|---|---|
| 0 | 729 | 0.53 |
| 1-2 | 15389 | 0.55 |
| 3-5 | 87721 | 0.52 |
| 6+ | 214277 | 0.51 |