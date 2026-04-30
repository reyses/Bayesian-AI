# v1.0.6-RC full-window Python backtest

Generated: 2026-04-28 02:56

Atlas: `DATA/ATLAS` (391,514 1m bars, 288 days)
Params: R=45.0 SL=90.0pt MFE-cut=17/$2.0 Trail=21.0pt/5%

**CAVEAT**: Python sim has known 2× trade-count bias vs NT8 SA. Numbers are directional, not authoritative.

## Aggregate

| Metric | Value |
|---|---:|
| Trades | 5,825 |
| Trades/day | 20.2 |
| Win rate | 55.4% |
| Total Net PnL | $-21,715.50 |
| Total Gross PnL | $-10,648.00 |
| $/day (net) | $-75.40 |
| Best day | $+1,274.80 |
| Worst day | $-1,712.20 |
| Positive days | 129/288 (44.8%) |
| Buy-and-hold MNQ | $+5,887.50 ($+17.02/day) |
| Strategy - BH | $-27,603.00 |

## Exit reasons

| Reason | Count | % |
|---|---:|---:|
| TrailExitLong | 1697 | 29.1% |
| TrailExitShort | 1671 | 28.7% |
| PivotExitLong | 949 | 16.3% |
| PivotExitShort | 932 | 16.0% |
| MfeCutShort | 261 | 4.5% |
| MfeCutLong | 216 | 3.7% |
| EodExitShort | 31 | 0.5% |
| EodExitLong | 25 | 0.4% |
| HardStopLong | 23 | 0.4% |
| HardStopShort | 20 | 0.3% |

## Per-month

| Month | Trades | Net PnL | $/day est |
|---|---:|---:|---:|
| 2025-01 | 371 | $-725.90 | $-34.57 |
| 2025-02 | 297 | $-1,361.80 | $-64.85 |
| 2025-03 | 459 | $-2,368.60 | $-112.79 |
| 2025-04 | 1042 | $-209.80 | $-9.99 |
| 2025-05 | 363 | $-2,245.20 | $-106.91 |
| 2025-06 | 192 | $+268.20 | $+12.77 |
| 2025-07 | 192 | $+1,133.70 | $+53.99 |
| 2025-08 | 247 | $-647.80 | $-30.85 |
| 2025-09 | 118 | $-617.20 | $-29.39 |
| 2025-10 | 366 | $-300.40 | $-14.30 |
| 2025-11 | 559 | $-7,239.10 | $-344.72 |
| 2025-12 | 263 | $+920.30 | $+43.82 |
| 2026-01 | 354 | $-3,061.10 | $-145.77 |
| 2026-02 | 532 | $-2,973.30 | $-141.59 |
| 2026-03 | 470 | $-2,287.50 | $-108.93 |
