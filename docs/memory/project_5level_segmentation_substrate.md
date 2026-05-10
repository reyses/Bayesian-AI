---
name: 5-level hierarchical segmentation substrate
description: 2D-shape recursive segmenter at 15m/5m/1m/15s/5s with stepped EDA, hyper-volume regression, and Bayesian probability tables — the lookup substrate for the meta-router
type: project
---

# 5-LEVEL HIERARCHICAL SEGMENTATION SUBSTRATE (built 2026-05-09 evening)

## What it is

A recursive 2D-shape segmenter that decomposes every trading day into a
5-deep hierarchy of simple-shape primitives. Same library at every level
(13 primitives + NOISE); only the (TF anchor, min span, slope lookback)
tuple changes per level.

```
LEVEL       TF anchor       window           min span        slope lookback
phrase      15m M_close     12 bars (3hr)    60 5s (5min)    720 5s (1h)
motif       5m  M_close     9  bars (45min)  12 5s (1min)    360 5s (30min)
sub_motif   1m  M_close     15 bars (15min)  6  5s (30s)     90  5s (7.5min)
measure     15s M_close     12 bars (3min)   4  5s (20s)     30  5s (2.5min)
note        5s  M_close     6  bars (30s)    2  5s (10s)     12  5s (1min)
```

## Why it matters

This is the **Bayesian-table substrate** the meta-router needs to look
up `P_cascade(t)` at-bar with no lookahead. The 13-primitive simple-shape
library at every TF gives a small-cardinality categorical vector per
hierarchy node — perfect Bayesian-table key.

**Why:** the 2026-05-09 architectural lock established that the meta-router
needs a chord-keyed lookup (replacing the deterministic CRM detector v2
that over-suppressed slow-buildup macro pivots). The 5-level substrate
provides exactly that: at any 5s bar, the hierarchy gives `(phrase_shape,
motif_shape, sub_motif_shape, measure_shape, note_shape)` — a 5D
categorical chord that conditions all downstream probability lookups.

**How to apply:**
1. To compute P(fwd_up) at any bar, look up the (level, shape, parent_shape)
   row in the matching Bayesian table and apply Beta(1,1) Jeffreys posterior.
2. To detect "cascade in progress", combine note-level + measure-level
   shapes — large-n strong-signal cells like NOISE-after-STEEP_LINEAR_DOWN
   (35.5% UP, n=9,539, CI [0.345, 0.365]) are the structural cascade
   continuation markers.
3. To EXTEND the substrate, add an axis to the cell key — TOD bucket,
   calendar event, regime label. Cell granularity grows multiplicatively;
   pool sparse cells via the Beta hierarchy.

## Bulk-run results (345 days)

```
LEVEL          n         top-3 shapes
phrase     2,047    NOISE 21.3%  FLATLINE 17.0%  GENTLE_LINEAR_UP 13.8%
motif      6,276    FLATLINE 16.5%  NOISE 15.5%  GENTLE_CONCAVE_UP 9.4%
sub_motif 21,048    FLATLINE 15.4%  STEEP_LINEAR_DOWN 13.3%  STEEP_LINEAR_UP 12.9%
measure   70,782    STEEP_CONCAVE_UP 15.3%  NOISE 14.8%  STEEP_CONVEX_DOWN 14.8%
note     203,801    NOISE 31.6%  STEEP_LINEAR_UP 12.1%  STEEP_LINEAR_DOWN 12.0%
```

## Strongest cross-level findings (n>=50, |p−0.5|>=0.10)

```
LEVEL     SHAPE              | PARENT               n     P_up   [CI]
note      NOISE              | STEEP_LINEAR_DOWN  9539  0.355  [0.35, 0.36]   ★ tight
note      NOISE              | STEEP_LINEAR_UP    9793  0.622  [0.61, 0.63]   ★ tight
sub_mot   STEEP_CONCAVE_UP   | STEEP_LINEAR_UP    321   0.684  [0.63, 0.73]
sub_mot   FLATLINE           | STEEP_LINEAR_UP    306   0.646  [0.59, 0.70]
motif     FLATLINE           | STEEP_LINEAR_UP    57    0.746  [0.63, 0.85]
measure   NOISE              | STEEP_CONVEX_UP    134   0.728  [0.65, 0.80]
sub_mot   STEEP_CONVEX_UP    | GENTLE_CONCAVE_UP  51    0.245  [0.14, 0.37]
```

Full 60-finding table: `reports/findings/segments/bayes_tables/STRONG_findings_n50_e10.csv`.

Three structural patterns:
1. **Continuation dominates immediately after directional moves** (note level)
2. **Pause-after-rally is bullish** (motif/sub_motif FLATLINE-after-UP)
3. **Counter-trend curves fail** (small-shape inside larger-opposing)

## Tools (all parametric, all reusable)

```
tools/segment_simple_shapes.py            (5-level recursion + max_depth param)
tools/segment_simple_bulk_v2.py           (bulk runner; --threshold --max-depth)
tools/segment_stepped_eda.py              (per-level EDA: shape/length/skew/markov/parent-child)
tools/segment_stepped_surface_regression.py  (forward returns + 2D heatmaps + OLS)
tools/segment_bayes_tables.py             (Beta(1,1) Jeffreys posteriors at every conditioning)
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

## Output tree

```
reports/findings/segments/simple_bulk_v2/
    themes.csv  +  all_<level>s.csv (5)  +  per_day/<day>.json (345)

reports/findings/segments/stepped_eda/
    <level>_shape_dist.csv / length_dist.csv / skew_dist.csv / r_dist.csv
    <level>_split_dist.csv / markov.csv
    <level>_given_<parent>.csv
    <level>_overview.png
    overall_summary.csv

reports/findings/segments/stepped_surface_reg/
    <level>_marginal_by_shape.csv
    <level>_surface_shape_length.csv  + .png
    <level>_surface_shape_sigma.csv   + .png
    <level>_surface_parent_self.csv   + .png
    <level>_ols_coefs.csv  +  <level>_mfe_mae.csv

reports/findings/segments/bayes_tables/
    <level>_priors.csv  +  priors.png
    <level>_p_up_given_shape.csv  +  p_up.png
    <level>_p_up_given_shape_skew.csv
    <level>_p_up_given_shape_sigma.csv
    <level>_p_up_given_shape_parent.csv  +  p_up_parent_extremes.png
    STRONG_findings_n50_e10.csv      ★ 60 cross-level discoveries
```

## Caveats / open questions

- Tables currently pool IS+OOS (split CSVs exist but separate tables not yet built).
- "fwd_return > 0" is coarse; magnitude/MFE/MAE stored separately not joined.
- Forward horizons fixed per level (note=30s, ..., phrase=60min). No reason these are right.
- Some n=50 cells in the strong-findings list may collapse on OOS-only.
  **Do not trade off these without OOS validation.**

## Pending (next session)

1. Add **TOD-bucket axis** to every Bayesian cell (locked structural req).
2. Add **calendar-event axis** (FOMC/NFP/CPI flags).
3. **OOS sign-stability check** per cell — drop cells whose IS sign != OOS sign.
4. **Per-tier oracle PnL per primitive chord** (failure-mode identifier).
5. **V0 meta-router prototype** using `P_cascade(t)` from this substrate.

## Key journal cross-refs

- `docs/daily/2026-05-09.md` (evening section: "5-LEVEL HIERARCHICAL SEGMENTATION SUBSTRATE BUILT")
- `reports/findings/2026-05-09_5level_substrate.md` (focused finding doc)
- `memory/project_original_bayesian_brain_architecture.md` (original 8-step pipeline this substrate replaces step 3 of)
- `memory/feedback_no_human_regime_terms.md` (vocabulary translation table)
- `memory/feedback_quantile_selection_overfit.md` (the OOS-validation requirement that drives "pending #3")
