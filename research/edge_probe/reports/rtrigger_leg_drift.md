# Leg-terminated drift (entry -> next R-trigger reversal, ATLAS 1m)
65,234 legs, 520 days. Window = ONE ebb/flow leg (entry to next reversal), not session close.

- median leg length: 17 bars (mean 28.4)
- **within-leg MFE +43.3 pts vs MAE -24.8 pts** -> asymmetry MFE/|MAE| = 1.74
- captured at R-trigger exit: -0.06 pts ($-0.12); that is -0% of within-leg MFE
- median bar@MFE within leg: 7

## Short-horizon drift: MFE vs |MAE| at N bars after entry
| horizon N (bars) | mean MFE | mean |MAE| | MFE/|MAE| | drift? |
|---|---|---|---|---|
| 3 | +8.6 | 6.9 | 1.24 | UP-drift |
| 5 | +13.2 | 11.0 | 1.20 | UP-drift |
| 8 | +18.2 | 14.6 | 1.25 | UP-drift |
| 10 | +20.8 | 16.2 | 1.29 | UP-drift |
| 15 | +25.7 | 18.6 | 1.38 | UP-drift |
| 20 | +29.0 | 20.0 | 1.45 | UP-drift |
| 30 | +33.4 | 21.5 | 1.55 | UP-drift |

Read: MFE/|MAE| > 1 within the leg (and at short horizons) => the entered leg drifts favorably and a ride exists (exit is the lever). ~1 => symmetric even leg-by-leg => the entry has no directional drift and no exit can extract $ from it.
