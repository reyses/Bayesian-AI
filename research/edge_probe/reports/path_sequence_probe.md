# Path vs snapshot — leg-direction anticipation (GRU mamba-proxy)
12029 episodes, 81 days, seq feat dim 15. Temporal 70/30 split.

- **GRU over the run-up path: acc 81.8%, AUC 0.896**
- price-sign LEAK baseline sign(px-pivot)@cutoff: 96.9%
- gov_dir baseline (same test): 57.9%
- snapshot GBM (earlier, all ingredients): 62.4%, AUC 0.658
- curvature-flip episodes in test: 2078/4169 (50%); GRU acc on them 81.4%

LEAK CHECK: if the GRU ≈ the price-sign baseline, the 'signal' is just the already-realized move (not anticipation). Real anticipation = GRU >> price-sign.
Verdict: GRU ~= price-sign LEAK -> mostly reading the realized move, NOT anticipation
