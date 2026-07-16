# TMPL0 findings — 2024-frozen K-means pattern-template stream

Built by research/nt8_catalog/tools/template_stream_builder.py. FIT 2024-only; STREAM 2024+2025+2026; features strictly trailing.

## Event counts (detected, causal, RTH-gated)
```
year          2024   2025  2026
ptype                          
BREAKDOWN    26991  25066  6276
COMPRESSION  29132  25686  6366
DOJI         13552  11629  2895
ENGULF_BEAR   9726   7933  1999
ENGULF_BULL   9414   7885  1886
HAMMER        2755   2468   586
WEDGE         5525   4768  1097

TOTAL events: 203635   (1m 159157, 5m 32858, 15m 11620)
```

## Templates
- total templates in codebook: **1020**
- with >=20 2024 members: **977**
- also |long_frac-0.5|>=0.05 (FIRING templates): **768**
- member_count: {1: 12.0, 5: 21.0, 10: 34.0, 25: 55.0, 50: 87.0, 75: 130.25, 90: 168.0, 95: 190.0, 99: 223.0}
- long_frac (templates with labeled members): {1: 0.056, 5: 0.1333, 10: 0.176, 25: 0.3021, 50: 0.4721, 75: 0.5912, 90: 0.7361, 95: 0.8, 99: 0.8943}
- labeled_count per template: {1: 11.19, 5: 21.0, 10: 33.0, 25: 53.0, 50: 85.0, 75: 127.25, 90: 164.0, 95: 185.05, 99: 213.0}

## KILL-POINT 1 — assignment-margin stability (test 2025+26 events)
- margin = d(2nd-nearest) - d(nearest), standardized L2
- ALL events margin pct: {1: 0.0033, 5: 0.0146, 10: 0.0274, 25: 0.0668, 50: 0.1461, 75: 0.2656, 90: 0.4054, 95: 0.5005, 99: 0.7091}
- TEST events margin pct: {1: 0.002, 5: 0.0104, 10: 0.0211, 25: 0.0558, 50: 0.1299, 75: 0.2439, 90: 0.3793, 95: 0.4701, 99: 0.6661}
- fraction of TEST margins < 1e-6 (tie/unstable): 0.0000

## KILL-POINT 2 — do 2024 biases transfer? (textbook-agree per template, 2024 in-sample)
- textbook is_long_raw vs active 2024 label, per-template mean: {1: 0.3663, 5: 0.4416, 10: 0.4767, 25: 0.5443, 50: 0.6269, 75: 0.7468, 90: 0.8333, 95: 0.8776, 99: 0.9469}
  (this is IN-SAMPLE 2024; the real transfer test is the OOS-AUC + test terciles from dsp.evaluate below — a flat 0.50 across terciles = no transfer)

## KILL-POINT 3 — beat the no-clustering baseline
- bar: PTRN-ENGULF OOS-AUC 0.616 / PTRN-HAMMER 0.615 (same harness). TMPL0 OOS-AUC below ~0.616 => clustering adds nothing over raw pattern events.

## dsp.evaluate() output (pasted verbatim by executor)
Note: dsp.evaluate() RE-SAVES signal_rows_TMPL0.parquet as the labeled subset (drops
fires with no active label, adds y/year/inter) — the pipeline's convention for every
league stream. The builder emits ALL 159,498 fires (2024 75,891 / 2025 67,163 / 2026
16,444); the 2,385 unlabeled fires carry no target and are never used by the split.
The on-disk parquet is the all-fires build; running evaluate() trims it to N=157,113.
```
TMPL0      N=157113 OOS-AUC 0.631 base 0.68 || low: 0.56 [0.55,0.57] N=27738 | mid: 0.68 [0.67,0.69] N=27738 | high: 0.79 [0.78,0.80] N=27738

raw evaluate() dict:
{'auc': 0.6312426429378936,
 'base_te': 0.6769654144735261,
 'coefs': {'inter': -0.043,
           'pivot_age_min': 0.025,
           'sig_with_leg': 0.023,
           'tod': -0.003,
           'value': 0.522},
 'det': 'TMPL0',
 'n': 157113,
 'n_te': 83214,
 'n_tr': 73899,
 'ter': {'high': (27738, 0.7918739635157546, 0.7830689375760058, 0.7993932722761969),
         'low':  (27738, 0.5632345518782897, 0.5544799221619127, 0.5720792461444306),
         'mid':  (27738, 0.675787728026534,  0.6655766653391405, 0.6851553078097493)}}
```

## KILL-POINT VERDICTS
1. **Assignment-margin stability — PASS (not degenerate).** TEST margin median 0.130
   (std-L2, 977 live centroids), 25th pct 0.056; fraction of exact ties (<1e-6) = 0.0000.
   Margins are non-zero and well-spread → nearest-centroid routing is stable, not a
   coin-flip. Caveat: a thin low tail (1st pct 0.002, 5th pct 0.010) sits near cluster
   boundaries, but it is a small minority.
2. **2024 biases transfer — PASS.** OOS terciles are monotonic and CI-separated
   (low 0.56 [0.55,0.57] < mid 0.68 < high 0.79 [0.78,0.80]); non-overlapping → the
   frozen 2024 template conviction ranks OOS agreement. NOT the flat-0.50 null.
3. **Beat the no-clustering baseline — MARGINAL PASS.** TMPL0 OOS-AUC 0.631 >
   PTRN-ENGULF 0.616 and PTRN-HAMMER 0.615, so it clears the bar. But the INCREMENTAL
   edge over the best raw pattern stream is +0.015 AUC — BELOW the project's 0.05
   signal-magnitude bar (noise band). The logistic leans almost entirely on `value`
   (coef 0.522 = per-template conviction |long_frac-0.5|, a quantity only the clustering
   produces); the shared zigzag features are ~0. Honest read: the clustering earns a
   real-but-small lift; most of TMPL0's headline agreement (base 0.68) comes from
   FREEZING each template's direction to its 2024 label majority, not from pattern shape.
