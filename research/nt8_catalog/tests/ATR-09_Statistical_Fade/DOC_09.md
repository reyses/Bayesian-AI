# Document ID: AG-DOC-ATR-09 (LOGISTIC REGRESSION VERIFIED)
**Title:** Deep Dive #9: Statistical ATR fade (True 14-day ATR Sweep)
**Status:** Completed (Dual-Year Validated)
**Ruleset:** Bespoke Exit (Revert 50% ATR or 10pt Stop). 14-day True ATR calculation.

## LR: Unnormalized Expected Value (EV)
> *Note: Magnitudes are in raw points. Win Rate is binary (%).*

### Results for 2024
| Setup (Threshold) | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 50.0% | Bearish Fade | 108 | 0.13 | -10.88 | **1.52** | [-5.31, 9.02] | No |
| 50.0% | Bullish Fade | 100 | 0.07 | -11.24 | **-6.00** | [-10.82, -0.24] | Yes |
| 75.0% | Bearish Fade | 76 | 0.13 | -10.00 | **-0.39** | [-7.76, 8.51] | No |
| 75.0% | Bullish Fade | 74 | 0.09 | -10.80 | **-3.04** | [-10.07, 5.54] | No |
| 100.0% | Bearish Fade | 28 | 0.14 | -10.01 | **0.02** | [-10.93, 13.06] | No |
| 100.0% | Bullish Fade | 46 | 0.11 | -10.35 | **4.42** | [-7.98, 19.14] | No |

### Results for 2025
| Setup (Threshold) | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 50.0% | Bearish Fade | 92 | 0.12 | -9.18 | **-1.76** | [-9.01, 7.02] | No |
| 50.0% | Bullish Fade | 91 | 0.09 | -11.43 | **1.39** | [-7.77, 12.17] | No |
| 75.0% | Bearish Fade | 52 | 0.08 | -9.94 | **-4.28** | [-11.97, 5.57] | No |
| 75.0% | Bullish Fade | 61 | 0.05 | -11.80 | **-8.64** | [-13.95, -1.43] | Yes |
| 100.0% | Bearish Fade | 24 | 0.04 | -10.90 | **-6.76** | [-17.31, 9.46] | No |
| 100.0% | Bullish Fade | 47 | 0.06 | -9.96 | **-2.62** | [-13.34, 11.13] | No |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-ATR-09_distributions.png)