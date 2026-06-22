# Seed / regime-length calibration — diminishing-returns sweep (2024_02_20)
Tiled B2T, 1s base, corrected break rule. Seed = SEED_BARS = floor on regime length. Real vs Fourier null.
Metric = % of day modelable (pristine coverage) at each seed; the real-MINUS-null gap is the real structure.

```
seed | REAL%  NULL%  | REAL-NULL gap | real nPris medLen
 30  | 55.4   11.2   |   +44.2%      |  840  38s
 25  | 61.3   17.2   |   +44.1%      | 1087  32s
 20  | 68.6   27.1   |   +41.5%      | 1417  27s
 15  | 76.2   39.9   |   +36.3%      | 1941  22s
 10  | 83.8   59.2   |   +24.6%      | 2766  17s
```

VERDICT: real-over-null structure is MAXED & FLAT at seed 25-30 (+44%), then ERODES below 25 (null catches up:
11%->59% as seed drops -> small seeds fit noise trivially). So lowering the floor reveals MORE regimes (hunch
confirmed: 840->2766) BUT the extra ones are increasingly noise-fittable, not real. FUNCTIONAL REGIME FLOOR ~25-30
bars (~25-38s at 1s). seed=30 was already near-optimal. Below ~25 = admitting junk the random walk fits equally.
medLen sits at the floor at every seed -> regimes are "as short as allowed"; real content caps at ~25-30 bars.

Fig: reports/findings/assets/fig_seed_descent.png
