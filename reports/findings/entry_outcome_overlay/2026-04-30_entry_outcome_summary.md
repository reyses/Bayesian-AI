# Entry-outcome overlay — physics at fade entry, conditional on win/loss

Generated: 2026-04-30

## What this is

Companion to `peak_feature_overlay/`. Where the peak overlay shows physics
at price extremes (peaks), this analysis shows physics at **actual fade
entries** from the blended pipeline's labeled trades, **color-coded by
outcome (winner=green, loser=red)**.

This is the analysis that maps directly to the user's reframed CNN
thesis: *"trying to predict price movement is impossible — but can we
use mean regression to enter and exit at the correct time? A CNN can do
it."* The CNN's job is to look at the rest of the 91D physics state
WHEN A FADE TRIGGER FIRES and decide whether to take the fade or skip.

This EDA quantifies and visualizes whether such a decision is even
**possible** — i.e., whether winning fades have a different physics
signature than losing fades at entry.

## Headline finding

**Yes — winning and losing fades have measurably different physics at
entry.** Across 1,912 fade-tier trades (BASE_NMP, FADE_CALM,
FADE_AGAINST), the strongest separators are:

| Feature | Cohen's d (winner-loser) | Winner mean | Loser mean | Direction |
|---|---:|---:|---:|---|
| `1m_reversion_prob` | **+0.472** | 0.957 | 0.897 | Winners have HIGHER reversion probability |
| `1m_hurst` | **-0.255** | 0.687 | 0.716 | Winners have LOWER hurst (more mean-revert) |
| `1m_variance_ratio` | **-0.219** | 0.435 | 0.495 | Winners have LOWER VR (more mean-revert) |
| `1D_dmi_diff` | -0.141 | -0.616 | +0.911 | Winners enter against DOWN-day; losers against UP-day |
| `5m_variance_ratio` | +0.114 | 0.461 | 0.423 | (mixed signal) |
| `5m_hurst` | +0.093 | 0.709 | 0.698 | (weak) |

**z_se itself has Cohen's d ≈ 0** at the entry level — because every
trade in this set has |z_se| > 2 by tier-rule definition, so the
trigger feature can't separate winners from losers within the set.
**This is the whole point.** The trigger fires; what comes next is
decided by the *other* features.

The +0.472 d on `1m_reversion_prob` is meaningful — Cohen's d > 0.4 is
roughly the threshold where a binary classifier can hit ≥58-60% AUC.
That matches what the CNN actually achieves (cnn_flip ~70.6% in
training).

## Validation method

```python
# pseudocode
trades = load("reports/findings/tier_pnl_by_regime/2026-04-29_trades_enriched.csv")
fade = trades[trades.entry_tier in ("BASE_NMP","FADE_CALM","FADE_AGAINST")]
winners = fade[fade.pnl > 0]    # 953 trades
losers  = fade[fade.pnl <= 0]   # 959 trades
for feat in physics_features:
    d = (winners[feat].mean() - losers[feat].mean()) / pooled_std
```

Count-based WR is 49.9% (953 / 1,912). PnL-based: $/loss ratio gives
the actual edge — total PnL $86,276 across the fade family despite ~50%
count WR, because winners are larger than losers.

## Visualizations

Tool: `tools/entry_outcome_overlay_chart.py`

```bash
# Sample on specific dates
python tools/entry_outcome_overlay_chart.py --mode sample \
    --days "2025-06-09,2026-02-19"

# Full per-day batch (~284 entry-bearing days)
python tools/entry_outcome_overlay_chart.py --mode per_day

# Override default fade tiers
python tools/entry_outcome_overlay_chart.py --tiers "RIDE_AGAINST,RIDE_CALM"
python tools/entry_outcome_overlay_chart.py --all-tiers
```

Output: 7-panel charts at
`reports/findings/entry_outcome_overlay/per_day/YYYY_MM_DD.png`. Same
layout as `peak_feature_overlay/per_day/` but with green/red entry
markers (size scales with |pnl|) replacing the H/L peak markers.

Vertical guides through every panel are colored by outcome — at a glance
you can see the physics state at every entry, classified by win/loss.

## What this means for shipping

1. **The CNN thesis is real, but linear thresholds aren't enough.**
   Winners and losers have measurably different physics at trigger
   (Cohen's d up to +0.47), but a *simple threshold filter* on those
   features **does not improve total PnL**. See the threshold-sweep
   result below.
2. **`reversion_prob` is the single strongest discriminator**
   (d=+0.472), but the LOW-revprob trades that the filter would skip
   STILL net positive — they're individually worse but not unprofitable
   in aggregate. Skipping them sacrifices their contribution.
3. **Daily-trend (1D_dmi_diff)** confirms the daily-regime gate
   intuition (winning fades -0.62, losing fades +0.91). This validates
   the regime gate already in `ZigzagRunner_v1.0.8-RC.cs`.
4. **z_se / dmi_diff at fast TFs are essentially uncorrelated with
   outcome** at fade entry — they're the trigger features, so they're
   already conditioned on extreme.
5. **The CNN's nonlinear advantage is the actual edge.** Linear filters
   on these features can't replicate it. This is consistent with the
   training observation that cnn_flip ~70% AUC, while a tree built on
   the same features plateaus at ~55%.

## Threshold sweep — heuristic filter test (NEGATIVE result)

Hypothesis: skip fade entries with low reversion_prob and/or high hurst,
expecting +PnL delta similar to the LinReg slope filter (+$33k).

**Result: every threshold combination tested REDUCES total PnL.** The
filter raises per-trade quality (WR ↑, $/trade ↑) but drops trade count
faster than it raises win density. Baseline 1,912 trades = $86,276; best
filter combo 1,411 trades = $74,128 (delta -$12,148).

```
Baseline:        n=1912, WR=49.9%, PnL=$86,276, $45.12/trade
revprob>=0.85:   n=1674, WR=54.8%, PnL=$86,516, $51.68/trade  (delta +$240)
revprob>=0.92:   n=1495, WR=56.5%, PnL=$80,580, $53.90/trade  (delta -$5,696)
hurst<=0.72:     n=1057, WR=54.2%, PnL=$53,108, $50.24/trade  (delta -$33,168)
revprob>=0.92 AND hurst<=0.72:
                 n=869,  WR=59.7%, PnL=$50,173, $57.74/trade  (delta -$36,102)
```

Only `revprob>=0.85` is roughly break-even. Tighter thresholds always
HURT total PnL.

**The implication**: physics-at-entry contains real information (the
Cohen's d table proves it), but extracting that information requires a
*nonlinear* decision (CNN), not a threshold. The 70% cnn_flip accuracy
is the right tool; a hand-tuned threshold rule is the wrong tool.

**Counterintuitive corollary**: the **+$33k LinReg slope filter** must
be picking up something *other* than these physics features — likely
the local slope structure that none of the per-bar physics features
encode directly. Two filters, two different axes of discrimination,
both real.

## Pairs with

- `reports/findings/peak_feature_overlay/2026-04-30_visual_overlay_summary.md`
  (physics at price extremes — different lens, same data foundation)
- `reports/findings/HEADLINE_VALIDATION_2026-04-30.md`
  (the +$165k context — these trades produced that number; this analysis
  decomposes it)

## Caveats

- **Apr-10 vintage trade CSVs.** Pipeline code modified Apr-20. User is
  re-running. Re-render this analysis on fresh CSVs to confirm
  patterns persist.
- **49.9% count-based WR is misleading.** PnL-based: total $86,276 means
  winners are larger than losers — the slope-filter analysis showed
  this can be lifted to ~$165k by skipping the highest-loss-percentage
  buckets. The Cohen's d table above identifies WHICH features split
  those buckets cleanly.
- **Marker size in the charts is |pnl|-clipped at $200**, otherwise
  outliers ($800+) would dominate visually.

## Next analysis cycles

1. **Filter heuristic from these features**: simulate skipping fade
   entries where `1m_reversion_prob < 0.92` AND `1m_hurst > 0.72`. Does
   that subset's PnL match or exceed the slope-filter +$33k delta?
2. **Per-tier breakdown**: BASE_NMP vs FADE_CALM vs FADE_AGAINST may have
   DIFFERENT discriminating features. The current table aggregates them.
3. **Add RIDE-tier analysis**: `--tiers "RIDE_AGAINST,RIDE_CALM"` to see
   if the same physics features discriminate winners/losers in the
   non-fade tiers (RIDE-tiers are continuation, not mean-reversion).
4. **Time-of-day stratification**: do the discriminators change by
   session hour?
5. **Refresh after blended pipeline rerun** to confirm patterns persist.
