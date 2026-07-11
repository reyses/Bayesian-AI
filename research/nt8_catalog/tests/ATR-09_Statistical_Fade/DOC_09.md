# Document ID: AG-DOC-ATR-09 (LOGISTIC REGRESSION VERIFIED)
**Title:** Deep Dive #9: Statistical ATR fade (True 14-day ATR Sweep)
**Status:** Completed (Dual-Year Validated)
**Ruleset:** Bespoke Exit (Revert 50% ATR or 10pt Stop). 14-day True ATR calculation.

## LR: Unnormalized Expected Value (EV)
> *Note: Magnitudes are in raw points. Win Rate is binary (%).*

### Results for 2024
| Setup (Threshold) | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 50.0% | Bearish Fade | 108 | 0.13 | -10.88 | **1.48** | [-4.90, 8.99] | No |
| 50.0% | Bullish Fade | 100 | 0.07 | -11.24 | **-6.03** | [-10.82, -0.40] | Yes |
| 75.0% | Bearish Fade | 76 | 0.13 | -10.00 | **-0.46** | [-7.66, 7.79] | No |
| 75.0% | Bullish Fade | 74 | 0.09 | -10.80 | **-2.99** | [-9.91, 5.42] | No |
| 100.0% | Bearish Fade | 28 | 0.14 | -10.01 | **-0.03** | [-11.33, 13.70] | No |
| 100.0% | Bullish Fade | 46 | 0.11 | -10.35 | **4.46** | [-7.92, 19.39] | No |

### Results for 2025
| Setup (Threshold) | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 50.0% | Bearish Fade | 92 | 0.12 | -9.18 | **-2.00** | [-9.26, 6.51] | No |
| 50.0% | Bullish Fade | 91 | 0.09 | -11.43 | **1.25** | [-7.76, 12.34] | No |
| 75.0% | Bearish Fade | 52 | 0.08 | -9.94 | **-4.24** | [-11.68, 5.74] | No |
| 75.0% | Bullish Fade | 61 | 0.05 | -11.80 | **-8.73** | [-13.99, -1.43] | Yes |
| 100.0% | Bearish Fade | 24 | 0.04 | -10.90 | **-6.57** | [-17.18, 9.47] | No |
| 100.0% | Bullish Fade | 47 | 0.06 | -9.96 | **-2.70** | [-13.27, 10.84] | No |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-ATR-09_distributions.png)