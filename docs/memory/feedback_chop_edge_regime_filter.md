---
name: Chop-Edge Discovery & Regime Filter (v1.5-RC)
description: The zigzag counter-trend strategy has +$89/day on chop days and -$95/day on trend days. Forward-available rule (prior_range + range_compression) discriminates with d_OOS=+0.77/+0.78. Filter converts -$552/95-day net into +$5,000-6,000.
type: feedback
date: 2026-04-27
originSessionId: bb5b3851-d849-49aa-9f93-bcd7b0dc113f
---
## Insight

**The strategy is a CHOP SPECIALIST, not a trend follower.**

User framing (2026-04-27): *"trend days are the easy ones — chop is what
everyone avoids and we thrive."* This insight inverted the entire optimization
direction:

- Stop trying to make zigzag work in all regimes
- Start identifying the regimes where it DOES work and skipping the rest

## Empirical evidence

NT8 backtest of v1.0.x counter-trend over 1/2-4/24/2026 (1,678 trades, 95 days):
- Net: −$552 (looks broken)
- **Working window 1/2-2/26**: 46 days, +$4,096 (+$89/day, +$5.74/trade)
- **Bleed window 2/27-4/24**: 49 days, −$4,648 (−$95/day, −$4.82/trade)
- Inflection at exactly 2026-02-26 (= MNQ regime change from chop to strong
  bull rally)

## Forward-available regime classifier

Two features discriminate BLEED vs HARVEST days with walk-forward stability:

| Feature | d_IS | d_OOS | Source |
|---|---|---|---|
| `prior_range` | +0.576 | +0.774 | yesterday's daily H-L |
| `range_compression` | +0.475 | +0.782 | prior_range / 20d_mean_range |

Both POSITIVE: BLEED days follow LARGE-range yesterdays AND yesterdays whose
range was BIG vs the 20-day baseline. The chop edge is in the QUIET aftermath
of quiet days, not after volatile events.

## Rule

```
bleed_score = z(prior_range) + z(range_compression)
trade_today = (bleed_score <= -0.34)   // skip top-50% bleed-scored days
```

IS-calibrated (1/2-3/1/2026, N=48 days):
- MEAN_PRIOR_RANGE = 385.32, STD_PRIOR_RANGE = 219.83
- MEAN_RANGE_COMPRESSION = 1.0315, STD_RANGE_COMPRESSION = 0.5502

## Validation results

Threshold sweep on 1,678-trade ledger:
- z=−0.5: 39 days kept, +$5,021 net, $129/day on kept
- **z=−0.34 (MVP default): 50% skip, OOS-validated, $6,202 OOS lift**, 82% bleed catch
- z=+0.75: 67 days kept, +$5,214 net (biggest aggregate)
- z=0.0: AVOID — empirical local minimum on the threshold curve

**Strategy goes from −$552 to +$3,977-$5,214 across all reasonable thresholds.**

## Methodology lessons

1. **Start with daily features**, not multi-TF, when building day-level
   regime classifiers. Multi-TF feature extension (ATLAS 1m → 5m, 1h
   variance ratios) added zero OOS lift over the simple 2-feature rule.
   Adding 3 more features dropped OOS lift from +$5,084 to +$2,094 — pure
   overfitting.

2. **Cohen-d walk-forward shortlist**: only keep features with sign(d_IS) ==
   sign(d_OOS) AND min(|d_IS|, |d_OOS|) >= 0.30. Two features met this bar
   with strict-forward-only data.

3. **z=0.0 is NOT the sweet spot**. The threshold curve has a literal local
   minimum at zero. Either tighten (z=-0.5, conviction) or loosen (z=+0.75,
   aggregate) — both beat the middle.

4. **Tier-day-classifier methodology generalizes**: same Cohen-d + IS/OOS
   walk-forward approach worked on the 79D ML pipeline (RIDE_AGAINST,
   2026-04-18) AND on the NT8 trade-export simple-feature space. Different
   domain, same statistical structure.

## Implementation

Spec: `docs/JULES_v15_chop_specialist.md`
Validation tool: `tools/v15_filter_apply.py`
Classifier tool: `tools/nt8_bleed_harvest_classifier.py`

NT8 implementation: copy v1.4-RC, replace `MaxMeanRange5dPts` with the
combined-z bleed-score logic. ~50 LOC change. Inherits all v1.4-RC risk
machinery (DRM trail, StagnationMonitor, missed-breach handler).

## Open questions

1. Does the rule generalize across MNQ contract rolls? Tested only inside
   the 06-26 contract.
2. Is the chop-edge specific to MNQ or does it apply to ES/NQ/YM?
3. R sensitivity: tonight's analysis was at R=50. Re-run at R=30 to confirm.
4. Hour-of-day mask: secondary in-sample filter shows additional +$2,194 lift
   on filter-pass days. Validate on hold-out before deploying.
5. The **bull regime drift** (+$247/day passive long over the 32-day NT8 dump
   window) is FORFEITED on filter-skip days. An optional `OnFilterSkipDay
   = AlwaysLong` mode could harvest the drift. Defer.
