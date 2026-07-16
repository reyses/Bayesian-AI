# Combiner preview — pooled P(right) across all dossier streams
- pooled N = 154760 fires (68104 train 2024 / 86656 test 2025+26) across 12 streams
- pooled OOS AUC **0.687** (test base agreement 0.533)
- coefs (standardized): pivot_age_min=+0.097, sig_with_leg=+0.454, tod=+0.069, inter=-0.172, consensus=+0.097, is_ATR09=-0.462, is_CROSS11=+0.017, is_DOW19=-0.258, is_OHLC01=-0.004, is_ORB02=+0.196, is_PIVOT16=-0.152, is_ROUND05=+0.222, is_SEASON12=-0.003, is_TUNNEL20=+0.084, is_VWAP03=-0.179, is_VWMA10=+0.033, is_ZIGZAG=+0.481

| P-decile | N | mean P | observed agreement | day-block 95% CI |
|---|---|---|---|---|
| 0 | 8666 | 0.28 | 0.25 | [0.23,0.26] |
| 1 | 8666 | 0.35 | 0.36 | [0.34,0.38] |
| 2 | 8665 | 0.44 | 0.43 | [0.42,0.44] |
| 3 | 8666 | 0.49 | 0.46 | [0.45,0.48] |
| 4 | 8665 | 0.53 | 0.51 | [0.50,0.53] |
| 5 | 8666 | 0.57 | 0.54 | [0.53,0.56] |
| 6 | 8665 | 0.61 | 0.58 | [0.56,0.60] |
| 7 | 8666 | 0.67 | 0.66 | [0.65,0.68] |
| 8 | 8665 | 0.72 | 0.73 | [0.71,0.74] |
| 9 | 8666 | 0.83 | 0.80 | [0.79,0.82] |

## Consensus effect (test set, raw)
| same-direction co-fires (+-3min) | N | agreement |
|---|---|---|
| 0 | 15787 | 0.49 |
| 1-2 | 48238 | 0.54 |
| 3-5 | 18903 | 0.56 |
| 6+ | 3728 | 0.51 |