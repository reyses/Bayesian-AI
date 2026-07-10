# Document ID: AG-DOC-VP-01 (LOGISTIC REGRESSION VERIFIED)
**Title:** Deep Dive #1: Volume Profile Trading Strategies
**Status:** Completed (Dual-Year Validated)
**Ruleset:** Bespoke Exit (Target VAH/VAL or 20pt/10pt Stop). Unclamped Magnitude.

## LR: Unnormalized Expected Value (EV)
> *Note: Magnitudes are in raw points. Win Rate is binary (%).*

### Results for 2024
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Naked POC Test | 23 | 0.22 | -10.89 | **-5.16** | [-10.51, 1.32] | No |
| 2 | Naked POC Test | 14 | 0.36 | -9.94 | **-0.06** | [-7.93, 9.38] | No |
| 3 | Trend Runner | 98 | 0.28 | -10.68 | **-3.06** | [-6.18, 0.18] | No |

### Results for 2025
| Setup | Description | N | WR% | Mag (Mode) | EV (Mean Points) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Naked POC Test | 20 | 0.25 | -13.23 | **-5.55** | [-11.76, 2.63] | No |
| 2 | Naked POC Test | 7 | 0.43 | -16.14 | **18.32** | [-5.75, 43.65] | No |
| 3 | Trend Runner | 71 | 0.27 | -11.01 | **-4.21** | [-7.88, -0.39] | Yes |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-VP-01_distributions.png)