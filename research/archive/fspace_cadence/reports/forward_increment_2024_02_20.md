# Forward-increment OOS predictive test — 2024_02_20

Predict close[t+h]-close[t] from causal F-space at t. Train/test 70/30 time-disjoint, embargo h, RidgeCV. Metric = OOS R^2 (vs train-mean baseline; <0 = worse than mean), (corr/dir-acc).

```
cell                   | h=1   R2(corr/dir) | h=5   R2(corr/dir) | h=30  R2(corr/dir) | h=60  R2(corr/dir)
B2C_continuous/REAL    | -0.0152(+0.01/43%) | -0.0954(-0.01/46%) | -0.4628(-0.05/46%) | -0.6824(-0.06/46%)
B2C_continuous/BROWN   | -0.0077(-0.01/43%) | -0.0786(-0.02/46%) | -0.3052(-0.03/49%) | -0.4434(-0.02/49%)
B2C_continuous/FOUR    | -0.0012(+0.02/46%) | -0.0146(+0.02/49%) | -0.0919(+0.03/49%) | -0.1936(-0.00/48%)
B2T_tiled/REAL         | -0.0097(-0.01/42%) | -0.0583(-0.01/47%) | -0.3137(-0.05/49%) | -0.4124(-0.02/50%)
B2T_tiled/BROWN        | -0.0039(+0.00/43%) | -0.0267(-0.00/47%) | -0.1206(-0.03/48%) | -0.2176(-0.06/49%)
B2T_tiled/FOUR         | -0.0027(-0.00/45%) | -0.0179(-0.01/48%) | -0.0878(-0.03/50%) | -0.1611(-0.05/51%)

=== real - null OOS R^2 ===
B2C_continuous   REAL-BROWN: h1:-0.0075 h5:-0.0168 h30:-0.1577 h60:-0.2390
B2C_continuous   REAL-FOUR : h1:-0.0140 h5:-0.0808 h30:-0.3710 h60:-0.4888
B2T_tiled        REAL-BROWN: h1:-0.0058 h5:-0.0316 h30:-0.1931 h60:-0.1948
B2T_tiled        REAL-FOUR : h1:-0.0070 h5:-0.0404 h30:-0.2258 h60:-0.2513
```

SIGNAL iff real OOS R^2 > 0 AND > both nulls. Nulls expected ~0 (no forecastable structure).
