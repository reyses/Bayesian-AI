# RESULTS — Fast Reference

Two-minute read. Numbers, no fluff. See `HONEST_CAVEATS.md` for what's
NOT robust.

## Headline OOS performance (32 NT8 days, ATR×4 zigzag)

| Pipeline stage                          | $/day mean | 95% CI                | Days >$0 | Days >$200 |
|-----------------------------------------|------------|----------------------|----------|------------|
| Naked R-trigger (no GBM sizing)         | $475       | [$237, $741]          | 24/31    | —          |
| **+ B7 GBM sizing (gbm_ev)**            | **$927**   | **[$534, $1,372]**    | **26/31**| **24/31**  |
| + B8 hour-risk gate (linear)            | $1,505     | —                     | 26/31    | 25/31      |

**Per-leg (gbm_ev hardened):**
- Mean +$15.74, Median **-$11.00**, KDE mode -$21
- 42.2% legs positive
- Asymmetric: small losers + occasional big winners drive the mean

**Per-day distribution (gbm_ev hardened):**
- Mean $927, Median $627, **KDE mode $468**
- Min $-598, Max $+4,575
- Top 3 days = 40% of total P&L
- Heavy right tail; mean is not the typical day

## B-model OOS metrics (the stack)

| Model | Target | OOS AUC / metric | Best operating point | Coverage |
|-------|--------|------------------|----------------------|----------|
| B1 | P(pivot in K min) | AUC 0.716 at K=10 | thr=0.85 → 78% prec | 4% |
| B2 | P(fakeout) per pivot | AUC 0.695 at K=10 | thr=0.70 → 66% prec | 19% |
| B4 | P(in ±W of pivot) | AUC 0.738 at W=300s | thr=0.85 → 79% prec | 10% |
| B5 | 3-class leg phase | Accuracy 38.6% | P(MID)>0.60 → 71% prec | 0.25% |
| B6 | 3-class directional pivot | per-dir 53-57% prec at K=10 thr=0.70 | — | 5% per dir |
| B7 | E[leg amp / R] | Pearson 0.22 | monotonic ranking | n/a |
| B8 | E[forward 60min P&L] | Pearson 0.22 | monotonic ranking | n/a |

**Critical pattern**: every forward-looking model lands at Pearson ~0.22.
This is the structural info content of V2 features. Adding LSTM/
transformer/RL with same features → same ceiling.

## Sizing schemes compared (gbm_ev wins, hardened OOS)

| Scheme | Mean/day | Δ vs flat | Wins | CI vs flat |
|--------|----------|-----------|------|------------|
| flat (1.0× always) | $475 | baseline | 24/31 | — |
| hand_aggressive | $857 | +$382 | 25/31 | [+$218, +$569] |
| gbm_quantile | $940 | +$465 | 26/31 | [+$276, +$688] |
| **gbm_ev** | **$927** | **+$452** | **27/31** | **[+$278, +$654] ⭐** |

`gbm_ev` rule: `size = max(predicted_R - 1, 0)` clipped to [0, 3].
Translates B7's expected-amplitude prediction directly into size.

## What FAILED (don't repeat)

| Experiment | Result | Why |
|------------|--------|-----|
| Trail tightening (naive) | -$44/leg, CI strict negative | NEAR_PIVOT zone over-fires on pullbacks |
| Trail tightening (B6 + hysteresis sweep 18 cfgs) | Best -$0.29/leg | Strict gate fires too rarely; loose gate same as naive |
| Target placement (12 configs) | All CI strict negative | Target before peak = early exit; target above = never hits |
| Binary entry-skip filters | All reduce $/day | Skipping any positive-EV leg costs more than quality lift adds |
| Hardened-exits sweep (480 configs, SL+TP+cap) | None reach 32/32 days >$200 | Bad days look like good days at decision time |
| Day-level circuit breakers | Net negative (early stops cut winners too) | Same: in-day bad signal not separable from in-day noise |

## Structural lessons

1. **Direction is solved** by zigzag construction (~97% accuracy). No
   model adds to this; it's mechanical from the indicator.
2. **R-trigger exit is structurally optimal** at current signal
   precision. No tighter exit (trail/target) beats it.
3. **Sizing is the actual lever.** B7's monotonic amplitude ranking
   drives +$452/day in hardened OOS.
4. **Bad days aren't filterable from V2 features alone.** Would need
   cross-day features (overnight gap, calendar events, intermarket)
   that we don't have.

## What an outside reviewer should sanity-check first

1. **Reproduce hardened forward pass**: 
   ```
   python tools/composite_forward_pass_hardened.py
   ```
   Should print mean $927/day matching `reports/forward_pass_outputs/`.

2. **Cross-check B7 monotonic calibration on OOS** — load
   `caches/b7_leg_sizer_OOS.parquet`, bucket `pred_amp_R`, verify each
   bucket's actual median is monotonically increasing.

3. **Run sensitivity on friction** — current sim assumes $6/leg
   (commission $4 + slippage $2). Try $4 and $10 to bracket range:
   ```
   python tools/composite_forward_pass_hardened.py  # default $6
   # edit COMMISSION_PER_LEG + SLIPPAGE_PER_LEG to test others
   ```

4. **Inspect the 5 losing days** in
   `caches/composite_forward_pass_hardened.csv` — group by day, look
   at leg sequence. Were they news-driven? Choppy? What did B8 predict?

## What we'd build next (in order of expected payoff)

1. **Day-level regime classifier** using overnight + calendar features.
   Address the bad-day filtering problem. Highest expected payoff.
2. **Retrain B7/B8 on R-trigger-bar features** (currently trained on
   pivot-bar features — small distribution shift). Removes last clear
   peek. Expected impact: 10-30% reduction.
3. **Live data collection** during babysit deployment. Real distribution
   may differ from NT8 OOS.
4. **Run through `training_iso_v2/` streaming engine** for full
   end-to-end validation. Catches any remaining bar-by-bar issues.

## What we'd NOT build

- Transformer/LSTM on same V2 features — won't break Pearson 0.22 ceiling
- RL agent on same V2 features — same ceiling, ×10 cost
- More B-models on same data — diminishing returns

## File map for fast access

| Need | File |
|------|------|
| Replay the headline result | `tools/composite_forward_pass_hardened.py` |
| See per-leg data | `caches/composite_forward_pass_hardened.csv` |
| See per-day data | `reports/forward_pass_outputs/composite_forward_pass_hardened.txt` |
| Understand B7 (sizing) | `tools/train_b7_leg_sizer.py` + `models/b7_leg_sizer.pkl` |
| Understand B8 (hour risk) | `tools/train_b8_hour_risk.py` + `models/b8_hour_risk.pkl` |
| Trade-management findings | `reports/2026-05-17_composite_trade_management.md` |
| Why exit-hardening failed | `reports/forward_pass_outputs/composite_forward_pass_hardened_exits.txt` |
| Mode + Pareto analysis | `reports/forward_pass_outputs/composite_forward_pass_hardened_pareto.txt` |
| Sizing sweep result | `reports/forward_pass_outputs/composite_gbm_sizing_sim.txt` |
