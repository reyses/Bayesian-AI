# Entry-variant comparison

All variants use same physics exit: reg flip + residual flip → 30s sniper. No SL, no thesis_broken, no adverse_reg.

| Variant | IS $/day | OOS $/day | IS WR | OOS WR | IS $/tr | OOS $/tr | IS N | OOS N | IS hold | Max loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BASELINE | $+288 | $+429 | 55.7% | 57.0% | $+26.24 | $+30.12 | 3,044 | 969 | 2099s | $-1312 |
| A_FILTERS | $+171 | $+271 | 56.6% | 58.2% | $+20.51 | $+25.35 | 2,308 | 726 | 2546s | $-1530 |
| B_TIGHT | $+292 | $+384 | 55.3% | 56.7% | $+26.64 | $+26.78 | 3,032 | 975 | 2100s | $-1310 |
| C_RESID | $+207 | $+267 | 56.0% | 57.9% | $+20.26 | $+19.91 | 2,834 | 913 | 1774s | $-1108 |
