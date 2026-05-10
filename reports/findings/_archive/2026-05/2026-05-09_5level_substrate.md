# 5-Level Hierarchical Segmentation Substrate (2026-05-09)

## TL;DR

Built a 5-level hierarchical 2D-shape segmentation of MNQ price action
across all 345 days. Each level fits the same simple-shape primitive
library; the only thing that changes per level is the (TF anchor, min
span, slope lookback) tuple. Three parametric analysis stages — stepped
EDA, stepped surface regression, Bayesian probability tables — sit on
top of the segment CSVs.

This is the **Bayesian-table substrate** the meta-router needs to look
up `P_cascade(t)` at-bar with no lookahead.

## Hierarchy

```
LEVEL       TF       window     min span    slope-lookback    n (345 days)
phrase      15m      3 hr       5  min      1   hr            2,047
motif       5m       45 min     1  min      30  min           6,276
sub_motif   1m       15 min     30 s        7.5 min           21,048
measure     15s      3 min      20 s        2.5 min           70,782
note        5s       30 s       10 s        1   min           203,801
```

## Shape distributions per level

```
phrase  (n=2,047)    NOISE 21.3%  FLATLINE 17.0%  GENTLE_LINEAR_UP 13.8%
motif   (n=6,276)    FLATLINE 16.5%  NOISE 15.5%  GENTLE_CONCAVE_UP 9.4%
sub_mot (n=21,048)   FLATLINE 15.4%  STEEP_LINEAR_DOWN 13.3%  STEEP_LINEAR_UP 12.9%
measure (n=70,782)   STEEP_CONCAVE_UP 15.3%  NOISE 14.8%  STEEP_CONVEX_DOWN 14.8%
note    (n=203,801)  NOISE 31.6%  STEEP_LINEAR_UP 12.1%  STEEP_LINEAR_DOWN 12.0%
```

Two trends across levels:
- NOISE share rises from 21% (phrase) to 32% (note). Smaller scales have
  more spans that don't fit any clean primitive at r >= 0.85.
- FLATLINE share drops from 17% (phrase) to 7% (note). At 5s/30s scale
  almost everything has measurable directional intent.

## Top Bayesian findings — P(fwd_return > 0 | shape, parent_shape)

Beta(1,1) Jeffreys posterior. CIs are 95% credible intervals from the
Beta distribution. Filtered to n>=50 and |p−0.5|>=0.10.

```
LEVEL     SHAPE              | PARENT_SHAPE       n     P_up   [CI]
note      NOISE              | STEEP_LINEAR_DOWN 9539  0.355  [0.35, 0.36]   ★ TIGHT
note      NOISE              | STEEP_LINEAR_UP   9793  0.622  [0.61, 0.63]   ★ TIGHT
note      NOISE              | STEEP_CONVEX_UP   1008  0.643  [0.61, 0.67]
note      NOISE              | STEEP_CONCAVE_DOWN1076  0.338  [0.31, 0.37]
sub_mot   STEEP_CONCAVE_UP   | STEEP_LINEAR_UP   321   0.684  [0.63, 0.73]
sub_mot   FLATLINE           | STEEP_LINEAR_UP   306   0.646  [0.59, 0.70]
sub_mot   FLATLINE           | GENTLE_LINEAR_UP  279   0.644  [0.59, 0.70]
sub_mot   FLATLINE           | STEEP_LINEAR_DOWN 284   0.367  [0.31, 0.42]
measure   NOISE              | STEEP_CONVEX_UP   134   0.728  [0.65, 0.80]
measure   NOISE              | STEEP_LINEAR_DOWN1530   0.377  [0.35, 0.40]
motif     FLATLINE           | STEEP_LINEAR_UP    57   0.746  [0.63, 0.85]
motif     FLATLINE           | FLATLINE           92   0.372  [0.28, 0.47]
```

Full 60-finding table: `reports/findings/segments/bayes_tables/STRONG_findings_n50_e10.csv`.

## Structural insights

1. **Continuation dominates immediately after directional moves** — at
   the note level, NOISE following STEEP_LINEAR_DOWN runs 35.5% UP
   (n=9,539). NOISE following STEEP_LINEAR_UP runs 62.2% UP (n=9,793).
   Symmetric, tight CIs. Trend continuation in the immediate aftermath is
   real and large-n at the smallest scale.

2. **Pause-after-rally is bullish** — FLATLINE following STEEP_LINEAR_UP
   at motif level runs 74.6% UP (n=57). At sub_motif: 64.6% (n=306). At
   note: 64.3% (n=1,008 for NOISE-after-STEEP_CONVEX_UP, similar
   pattern). Compression-precedes-expansion in shape form.

3. **Acceleration-during-trend is bullish** — STEEP_CONCAVE_UP within
   STEEP_LINEAR_UP at sub_motif: 68.4% UP (n=321). The classic
   "accelerating uptrend within larger uptrend" pattern.

4. **Counter-trend curves fail** — STEEP_CONVEX_UP inside
   GENTLE_CONCAVE_UP at sub_motif: 24.5% UP (i.e. 75.5% DOWN). A small
   accelerating up inside a larger decelerating up means the larger
   pattern is exhausting.

5. **CIs collapse at deeper levels** — phrase n=2k spreads CI ±0.15;
   note n=10k tightens to ±0.005. The substrate is statistically dense
   at the bottom; sparse at the top.

## Output tree

```
reports/findings/segments/simple_bulk_v2/
    themes.csv                       345 day-summary rows
    all_phrases.csv                  2,047 rows
    all_motifs.csv                   6,276 rows
    all_sub_motifs.csv              21,048 rows
    all_measures.csv                70,782 rows
    all_notes.csv                  203,801 rows
    per_day/<day>.json              345 full-hierarchy JSONs

reports/findings/segments/stepped_eda/
    <level>_shape_dist.csv           (5 levels)
    <level>_length_dist.csv
    <level>_skew_dist.csv
    <level>_r_dist.csv
    <level>_split_dist.csv
    <level>_markov.csv
    <child>_given_<parent>.csv
    <level>_overview.png

reports/findings/segments/stepped_surface_reg/
    <level>_marginal_by_shape.csv
    <level>_surface_shape_length.csv  + .png
    <level>_surface_shape_sigma.csv   + .png
    <level>_surface_parent_self.csv   + .png
    <level>_ols_coefs.csv
    <level>_mfe_mae.csv

reports/findings/segments/bayes_tables/
    <level>_priors.csv  + priors.png
    <level>_p_up_given_shape.csv  + p_up.png
    <level>_p_up_given_shape_skew.csv
    <level>_p_up_given_shape_sigma.csv
    <level>_p_up_given_shape_parent.csv  + p_up_parent_extremes.png
    STRONG_findings_n50_e10.csv      60 cross-level discoveries
```

## Tools (parametric, all reusable)

```
tools/segment_simple_shapes.py            (5-level recursion + max_depth param)
tools/segment_simple_bulk_v2.py           (parametric bulk)
tools/segment_stepped_eda.py              (per-level shape/length/skew/markov + parent-child)
tools/segment_stepped_surface_regression.py  (forward returns + 2D heatmaps)
tools/segment_bayes_tables.py             (Beta-Binomial probability tables)
tools/segment_chart_multilevel.py         (5-panel hierarchy visualization)
```

## Re-run pipeline

```
python tools/segment_simple_bulk_v2.py --threshold 0.85 --max-depth 5
python tools/segment_stepped_eda.py
python tools/segment_stepped_surface_regression.py
python tools/segment_bayes_tables.py
python tools/segment_chart_multilevel.py --extremes --random 5
```

Each stage is independently re-runnable. Downstream tools load the bulk
CSVs/JSONs directly so adding a new analysis doesn't require re-segmenting.

## Pending (next session)

1. Add **TOD-bucket** axis to every Bayesian cell (locked structural
   requirement from earlier today).
2. Add **calendar-event** axis (FOMC/NFP/CPI flags).
3. **OOS sign-stability check** per cell (drop cells whose IS sign
   doesn't match OOS sign — same overfit pattern documented in
   `memory/feedback_quantile_selection_overfit.md`).
4. **Per-tier oracle PnL per primitive chord** (failure-mode
   identification — the goal articulated mid-session).
5. **V0 meta-router prototype** using `P_cascade(t)` from this substrate.

## Caveats / open questions

- Tables currently pool IS+OOS. Splits computed in `<level>_split_dist.csv`
  but not yet stratified into separate Bayesian tables.
- "fwd_return > 0" is a coarse target. Magnitude / MFE / MAE are stored
  separately but not yet joined into multi-target conditioning.
- Forward horizons are FIXED per level (note=30s, ..., phrase=60min).
  No reason these are right; experiment with horizon as another axis.
- The 60 strong findings include some where the n=50 cell may collapse
  in OOS-only. **Do not trade off these without OOS validation.**
