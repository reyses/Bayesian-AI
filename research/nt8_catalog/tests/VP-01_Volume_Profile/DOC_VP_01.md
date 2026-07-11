# Document ID: AG-DOC-VP-01 (LOGISTIC REGRESSION VERIFIED)
**Title:** Deep Dive #1: Volume Profile Trading Strategies
**Status:** Completed (Dual-Year Validated)
**Ruleset:** Bespoke Exit (Target VAH/VAL or 20pt/10pt Stop). Unclamped Magnitude.

## LR: Unnormalized Expected Value (EV)
> *Note: Magnitudes are in raw points. Win Rate is binary (%).*

### Results for 2024
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Naked POC Test | 23 | 0.22 | -10.89 | **-5.19** | [-10.48, 1.18] | No |
| 2 | Naked POC Test | 14 | 0.36 | -9.94 | **0.06** | [-8.16, 9.43] | No |
| 3 | Trend Runner | 98 | 0.28 | -10.68 | **-3.10** | [-6.25, 0.08] | No |

### Results for 2025
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Naked POC Test | 20 | 0.25 | -13.23 | **-5.56** | [-11.94, 3.39] | No |
| 2 | Naked POC Test | 7 | 0.43 | -16.14 | **17.98** | [-5.75, 46.04] | No |
| 3 | Trend Runner | 71 | 0.27 | -11.01 | **-4.15** | [-7.71, -0.38] | Yes |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-VP-01_distributions.png)