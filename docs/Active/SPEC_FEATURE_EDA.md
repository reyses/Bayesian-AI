# SPEC: Multi-TF Feature Exploratory Data Analysis

**Status:** Specification — build next
**Foundation:** Clean ATLAS data, 15 months of hand-drawn levels, 13D features per TF

---

## Purpose

Understand how each of the 13 features behaves:
1. Within each level zone (between control lines)
2. At level boundaries (what changes when price hits a level?)
3. Across timeframes (how does 1h DMI relate to 1m DMI?)
4. Before price reversals (what predicts the turn?)

This is pure exploration — no models, no predictions. Measure and visualize.

---

## The 13 Features

```
Directional (7D):
  [0] dmi_diff     — who's winning (buyers vs sellers)
  [1] dmi_gap      — how decisive (trend strength)
  [2] vol_rel      — participation (volume vs average)
  [3] dir_vol      — directional participation (signed volume)
  [4] velocity     — speed (rate of price change)
  [5] z_se         — position in range (z-score)
  [6] price_accel  — acceleration (change in velocity)

Regime (4D):
  [7] std_price    — volatility (rolling std of price)
  [8] variance_ratio — short vs long volatility
  [9] bar_range    — bar size (high-low in ticks)
  [10] wick_ratio  — rejection (how much of bar is wick)

Context (2D):
  [11] vwap_distance — distance from VWAP
  [12] time_of_day  — session phase (0-1)
```

---

## Module E1: Feature Distributions Per Zone

For each month × each zone between adjacent levels:
- Histogram of each feature
- Mean, median, std, skew, kurtosis
- Compare: does DMI look different in the 24503-24970 zone vs 24970-25278?

**Question answered:** Do features have different distributions in different zones?

---

## Module E2: Features at Level Boundaries

For each level, compare features when price is:
- Within 20 ticks of the level (AT boundary)
- 100+ ticks away from the level (MID zone)

**Metrics:**
- T-test: is each feature statistically different at boundary vs mid-zone?
- Which features change the most at boundaries? (these are the level detectors)

**Question answered:** Which features signal "we're at a level"?

---

## Module E3: Features Before Reversals

For each reversal at a level (price touches level and turns):
- What did each feature look like 1, 2, 3, 4 bars BEFORE the reversal?
- Compare to non-reversal touches (price passed through the level)

**Metrics:**
- Feature values at t-4, t-3, t-2, t-1, t=reversal
- Which features diverge between reversal and breakout?

**Question answered:** Which features predict "price will reverse here" vs "price will break through"?

---

## Module E4: Cross-TF Feature Correlation

The key module — how do features at different TFs relate?

For each pair of TFs (1h-1m, 1h-15m, 15m-1m, 1m-15s, 1m-1s):
- Pearson/Spearman correlation of each feature across TFs
- Lead/lag: does 1m DMI change BEFORE or AFTER 1h DMI?
- Alignment: when features agree across TFs, what happens to price?

**Feature pairs to measure:**
```
1h dmi_diff  ↔  1m dmi_diff   (do they agree on direction?)
1h velocity  ↔  1m velocity   (do they agree on speed?)
1h z_se      ↔  1m z_se       (do they agree on position?)
1h vol_rel   ↔  1m vol_rel    (do they agree on participation?)

Same for all TF pairs × all 13 features = 13 × 10 pairs = 130 correlations
```

**Question answered:** Which features are redundant across TFs? Which add unique info?

---

## Module E5: Cross-TF Lead/Lag at Levels

When price reverses at a level:
- Which TF's features change FIRST? (the leader)
- Which follow? (the laggard)
- How many bars of lead time does each TF provide?

This is the resonance question: do the TFs cascade (1h first, then 15m, then 1m) or do they all flip simultaneously?

**Question answered:** Can we use higher-TF features as early warning for lower-TF reversals?

---

## Module E6: Daily Range Within Zones

Per zone per month:
- Average daily range (high-low) in ticks
- Range as % of zone size (is the daily range smaller than the zone?)
- Distribution of daily ranges (normal? fat-tailed?)
- Time of day: when is the range largest? (session open? close?)

**Question answered:** How much of the zone does price use per day? Is the oscillation bigger/smaller than expected?

---

## Module E7: Feature-Level Interaction Matrix

For each feature × each level:
- Correlation between feature value and distance to level
- Does DMI increase as price approaches resistance?
- Does vol_rel spike at level touches?

Heatmap: features on Y, levels on X, color = correlation strength.

**Question answered:** Which features are "level-aware"? Which are oblivious?

---

## Output

All results saved to `reports/findings/eda/`:
- CSV per module (E1-E7)
- PNG charts per module
- Summary report: `reports/findings/eda/summary.txt`

Levels loaded from `DATA/levels/`. Features computed from ATLAS 1h (primary) with cross-TF from 15m, 1m.

---

## Build Order

Each module is independent — can run in any order:

```
python -m tools.feature_eda --module E1   # distributions per zone
python -m tools.feature_eda --module E2   # features at boundaries
python -m tools.feature_eda --module E3   # features before reversals
python -m tools.feature_eda --module E4   # cross-TF correlation
python -m tools.feature_eda --module E5   # cross-TF lead/lag at levels
python -m tools.feature_eda --module E6   # daily range within zones
python -m tools.feature_eda --module E7   # feature-level interaction
python -m tools.feature_eda --module all  # everything
```

---

## What This Tells Us

After running all modules, we'll know:
1. Which features matter at which TF (drop the rest)
2. Which features detect levels (input to the CNN)
3. Which features predict reversals (the trading signal)
4. How TFs relate (which ones to use, which are redundant)
5. How big the oscillation is within each zone (position sizing)

This is the foundation for the CNN feature selection. No guessing — measured from data.

---

## Files

| File | Purpose |
|------|---------|
| `tools/feature_eda.py` | Main EDA tool with E1-E7 modules |
| `tools/zone_analysis.py` | Basic zone stats (already built) |
| `reports/findings/eda/` | All output CSV + PNG |
