# Document ID: AG-DOC-VWAP-03 (LOGISTIC REGRESSION VERIFIED)
**Title:** Deep Dive #3: Session VWAP Z-Score Mean Reversion
**Status:** Completed (Dual-Year Validated)
**Ruleset:** Bespoke Exit (Mean Revert to VWAP or 3.0$\sigma$ Stop). Z-turn confirmed entry.

## LR: Unnormalized Expected Value (EV)
> *Note: Magnitudes are in raw points. Win Rate is binary (%).*

### Results for 2024
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Bearish Bounce (Short from +2z) | 128 | 0.59 | 4.54 | **1.48** | [-0.62, 3.44] | No |
| 2 | Bullish Bounce (Long from -2z) | 130 | 0.63 | 7.39 | **1.69** | [-0.72, 3.85] | No |

### Results for 2025
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Bearish Bounce (Short from +2z) | 118 | 0.60 | 15.17 | **2.62** | [-0.55, 5.78] | No |
| 2 | Bullish Bounce (Long from -2z) | 109 | 0.65 | 10.79 | **2.59** | [-0.12, 5.24] | No |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-VWAP-03_distributions.png)