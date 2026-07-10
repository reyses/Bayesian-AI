# Document ID: AG-DOC-VWAP-03 (FABLE-5 VERIFIED)
**Title:** Deep Dive #3: Session VWAP Z-Score Mean Reversion
**Status:** Completed (Dual-Year Validated)
**Ruleset:** Symmetric 2.0$\sigma$ barriers. Open = 08:30 CT. MFE Clamped.

## PQ: Empirical Expectation (EV)
> *Note: For symmetric barriers ($\pm 2\sigma$), a random walk baseline expectation is a PF-WR of 0.00.*

### Results for 2024
| Setup | Description | N | PF-WR | Mag (Mode) | EV (Mean $\sigma$) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Bullish Pullback | 125 | 0.16 | 1.95$\sigma$ | **0.14** | [-0.21, 0.50] | No |
| 2 | Bearish Pullback | 133 | 0.18 | 1.95$\sigma$ | **0.17** | [-0.20, 0.50] | No |

### Results for 2025
| Setup | Description | N | PF-WR | Mag (Mode) | EV (Mean $\sigma$) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Bullish Pullback | 110 | -0.14 | -2.05$\sigma$ | **-0.15** | [-0.51, 0.22] | No |
| 2 | Bearish Pullback | 117 | -0.28 | -2.05$\sigma$ | **-0.33** | [-0.67, 0.02] | No |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-VWAP-03_distributions.png)