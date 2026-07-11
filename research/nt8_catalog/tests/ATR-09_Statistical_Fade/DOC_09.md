# Document ID: AG-DOC-ATR-09 (LOGISTIC REGRESSION VERIFIED)
**Title:** Deep Dive #9: Statistical ATR fade (True 14-day ATR Sweep)
**Status:** Completed (Dual-Year Validated)
**Ruleset:** Bespoke Exit (Revert 50% ATR or 10pt Stop). 14-day True ATR calculation.

## LR: Unnormalized Expected Value (EV)
> *Note: Magnitudes are in raw points. Win Rate is binary (%).*

### Results for 2024
| Setup (Threshold) | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 50.0% | Bearish Fade | 108 | 0.13 | -10.88 | **1.47** | [-5.05, 8.67] | No |
| 50.0% | Bullish Fade | 100 | 0.07 | -11.24 | **-5.90** | [-10.70, -0.31] | Yes |
| 75.0% | Bearish Fade | 76 | 0.13 | -10.00 | **-0.34** | [-7.56, 8.12] | No |
| 75.0% | Bullish Fade | 74 | 0.09 | -10.80 | **-3.04** | [-10.16, 5.41] | No |
| 100.0% | Bearish Fade | 28 | 0.14 | -10.01 | **-0.15** | [-11.04, 13.94] | No |
| 100.0% | Bullish Fade | 46 | 0.11 | -10.35 | **4.51** | [-7.98, 19.91] | No |

### Results for 2025
| Setup (Threshold) | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 50.0% | Bearish Fade | 92 | 0.12 | -9.18 | **-1.91** | [-8.96, 6.77] | No |
| 50.0% | Bullish Fade | 91 | 0.09 | -11.43 | **1.23** | [-7.94, 11.81] | No |
| 75.0% | Bearish Fade | 52 | 0.08 | -9.94 | **-4.23** | [-12.08, 6.46] | No |
| 75.0% | Bullish Fade | 61 | 0.05 | -11.80 | **-8.71** | [-14.02, -1.24] | Yes |
| 100.0% | Bearish Fade | 24 | 0.04 | -10.90 | **-6.89** | [-17.44, 9.53] | No |
| 100.0% | Bullish Fade | 47 | 0.06 | -9.96 | **-2.68** | [-13.39, 11.13] | No |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-ATR-09_distributions.png)