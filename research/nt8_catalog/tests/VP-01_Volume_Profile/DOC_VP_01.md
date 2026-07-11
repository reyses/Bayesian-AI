# Document ID: AG-DOC-VP-01 (LOGISTIC REGRESSION VERIFIED)
**Title:** Deep Dive #1: Volume Profile Trading Strategies
**Status:** Completed (Dual-Year Validated)
**Ruleset:** Bespoke Exit (Target VAH/VAL or 20pt/10pt Stop). Unclamped Magnitude.

## LR: Unnormalized Expected Value (EV)
> *Note: Magnitudes are in raw points. Win Rate is binary (%).*

### Results for 2024
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Naked POC Test | 23 | 0.22 | -10.89 | **-5.15** | [-10.48, 1.38] | No |
| 2 | Naked POC Test | 14 | 0.36 | -9.94 | **0.01** | [-7.77, 9.34] | No |
| 3 | Trend Runner | 98 | 0.28 | -10.68 | **-3.05** | [-6.27, 0.21] | No |

### Results for 2025
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Naked POC Test | 20 | 0.25 | -13.23 | **-5.52** | [-11.94, 3.39] | No |
| 2 | Naked POC Test | 7 | 0.43 | -16.14 | **17.93** | [-5.86, 44.97] | No |
| 3 | Trend Runner | 71 | 0.27 | -11.01 | **-4.11** | [-7.80, -0.30] | Yes |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-VP-01_distributions.png)