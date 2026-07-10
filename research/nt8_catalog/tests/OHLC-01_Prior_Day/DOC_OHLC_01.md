# Document ID: AG-DOC-OHLC-01 (FABLE-5 VERIFIED)
**Title:** Deep Dive #1: Prior-day OHLC Levels
**Status:** Completed (Dual-Year Validated)
**Ruleset:** Symmetric 2.0$\sigma$ barriers. Open = 08:30 CT. MFE Clamped.

## PQ: Empirical Expectation (EV)
> *Note: For symmetric barriers ($\pm 2\sigma$), a random walk baseline expectation is a PF-WR of 0.00.*

### Results for 2024
| Setup | Description | N | PF-WR | Mag (Mode) | EV (Mean $\sigma$) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | PDH Bearish Bounce | 75 | -0.03 | -2.05$\sigma$ | **-0.03** | [-0.45, 0.40] | No |
| 2 | PDL Bullish Bounce | 64 | 0.00 | -2.05$\sigma$ | **0.00** | [-0.50, 0.50] | No |
| 3 | PDC Gap Fill Bounce | 155 | -0.04 | -2.05$\sigma$ | **-0.04** | [-0.35, 0.27] | No |

### Results for 2025
| Setup | Description | N | PF-WR | Mag (Mode) | EV (Mean $\sigma$) | EV 95% CI | Sig? |
|---|---|---|---|---|---|---|---|
| 1 | PDH Bearish Bounce | 64 | -0.22 | -2.05$\sigma$ | **-0.26** | [-0.75, 0.25] | No |
| 2 | PDL Bullish Bounce | 58 | -0.07 | -2.05$\sigma$ | **-0.07** | [-0.55, 0.41] | No |
| 3 | PDC Gap Fill Bounce | 136 | -0.14 | -2.05$\sigma$ | **-0.15** | [-0.47, 0.21] | No |

## Graphical Descriptive Statistics (Aggregate)
![Distribution Plot](./DOC-OHLC-01_distributions.png)