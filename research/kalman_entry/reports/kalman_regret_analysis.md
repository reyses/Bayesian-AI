# GA-Kalman REGRET analysis — entry (direction) vs exit (giveback)

Entry regret over a fixed 900s forward window (decoupled from exit). Exit regret on the realized window. DIAGNOSTIC (hindsight), not a signal.

## OOS  (N=4833)
### (A) ENTRY / DIRECTION regret
- picked the WRONG side (flip had larger MFE): **51%** (50% = coin flip; >50% = anti-predictive)
- net price drifted our way over 15m: 49% of entries
- mean MFE chosen 38.5pt vs flipped 37.4pt (edge +1.1pt) | mean regret (oracle−chosen) 22.1pt
### (B) EXIT regret — of right-direction trades (MFE≥5pt), giveback by time-from-peak
- right-direction trades: 4340 (90% of OOS); total giveback **625,198$** (312,599pt)
| time peak→exit | trades | mean giveback (pt) | total giveback ($) |
|---|---|---|---|
| 0-30s | 91 | 63.9 | 11,638 |
| 30-60s | 131 | 70.6 | 18,508 |
| 1-2m | 259 | 73.0 | 37,802 |
| 2-5m | 673 | 73.1 | 98,345 |
| 5m+ | 3186 | 72.0 | 458,906 |

## IS (ref)  (N=630)
### (A) ENTRY / DIRECTION regret
- picked the WRONG side (flip had larger MFE): **51%** (50% = coin flip; >50% = anti-predictive)
- net price drifted our way over 15m: 49% of entries
- mean MFE chosen 20.1pt vs flipped 17.3pt (edge +2.9pt) | mean regret (oracle−chosen) 9.6pt
### (B) EXIT regret — of right-direction trades (MFE≥5pt), giveback by time-from-peak
- right-direction trades: 564 (90% of IS (ref)); total giveback **73,610$** (36,805pt)
| time peak→exit | trades | mean giveback (pt) | total giveback ($) |
|---|---|---|---|
| 0-30s | 6 | 59.2 | 710 |
| 30-60s | 3 | 81.5 | 489 |
| 1-2m | 7 | 66.5 | 931 |
| 2-5m | 24 | 49.4 | 2,374 |
| 5m+ | 524 | 65.9 | 69,107 |
