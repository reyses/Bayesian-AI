# Document ID: AG-DOC-SQZ-04 (FABLE-5 VERIFIED)
**Title:** Deep Dive #4: Volatility Squeeze Breakout Strategies
**Status:** Completed (Dual-Year Validated)
**Ruleset:** Symmetric 2.0$\sigma$ barriers. Open = 08:30 CT. MFE Clamped.

## PQ: Empirical Expectation (EV)
> *Note: For symmetric barriers ($\pm 2\sigma$), a random walk baseline expectation is a PF-WR of 1.00.*

### Results for 2024
| Setup | Description | N | PF-WR | Mag (Mode) | EV (Mean $\sigma$) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Bullish Squeeze Breakout | 44 | 0.10 | 1.95$\sigma$ | **0.09** | [-0.55, 0.73] | No |
| 2 | Bearish Squeeze Breakout | 26 | -0.38 | -2.05$\sigma$ | **-0.46** | [-1.23, 0.31] | No |

### Results for 2025
| Setup | Description | N | PF-WR | Mag (Mode) | EV (Mean $\sigma$) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | Bullish Squeeze Breakout | 30 | -0.12 | -2.05$\sigma$ | **-0.14** | [-0.80, 0.53] | No |
| 2 | Bearish Squeeze Breakout | 30 | 0.73 | 1.95$\sigma$ | **0.54** | [-0.13, 1.20] | No |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-SQZ-04_distributions.png)