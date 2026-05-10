---
name: High-vol "harness as profit" angle failed
description: 2026-05-05 — tested two ways to convert high-vol bleed into profit (direction flip, vol-adaptive exits). Both rejected. Reveals state-exit leak as the actual mechanism.
type: feedback
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## The rule

When loser autopsy shows a "bleed zone" tied to entry volatility, the
intuitive levers (flip direction in high-vol, scale exit thresholds with
vol) DON'T work. Both fail because:

1. **Peaks are symmetric across direction in high-vol.** Both fade and
   flip directions reach similar peak amounts in Q5 vol. Flipping doesn't
   capture more.
2. **Peak distributions are fat-tailed.** Mean peak in Q5 was $144, but
   the typical (median) trade doesn't reach $144. Threshold formulas
   based on means or q_30 quantiles overshoot — wider TPs miss most
   trades; later giveback arming never fires.

## Evidence (2026-05-05 V2-native NMP)

Tested via re-simulation on 19,106 IS + 4,495 OOS NMP trades binned by
`L2_1m_vol_mean_15` quintile.

**Vol-flip hypothesis (rejected):**
- Q5 fade_peak $144 vs flip_peak $151 — 5% advantage, not real
- Counterfactual flip on Q5 only: total IS $3,419 → $866 (worse)

**Vol-adaptive exit thresholds (rejected):**
- Per-bin Bayesian-derived TP/SL/gb_min/gb_keep
- Q5 thresholds: tp=$51, gb_min=$70 (vs prod tp=$26, gb_min=$41)
- IS delta vs prod: -$75.51/day (CI [-$101, -$50])
- OOS delta vs prod: -$112.41/day (CI [-$170, -$59])
- Q5 specifically: prod +$4.68/t, vol-adapt -$0.71/t (delta -$5.39/t)

## Reveal: the actual mechanism is state-exit leak

Production thresholds via re-simulation show **+$4.68/trade in Q5**.
The actual engine produced **~$0.33/trade in Q5**. The gap (~$4.35/trade)
is from state-driven exits (`ZSeReversal`, `SwingNoiseSpike`) firing in
high-vol periods and trimming profit before the re-sim policy would.

## How to apply

1. **Don't try to flip or rescale exits to harness high-vol bleed.** Both
   approaches have been tested and rejected.
2. **Two remaining levers:**
   - Filter (skip high-vol entries) — straightforward but doesn't HARNESS
   - Surgical state-exit modification (disable/loosen ZSeReversal in
     high-vol bins) — promising, recovers re-sim/engine gap
3. **Mean-based threshold derivation overshoots fat-tail distributions.**
   When the peak distribution is fat-tailed (a few big peaks pulling up
   mean), use modal/median quantiles (q_05 to q_15) for TP rather than
   q_30 default. Skews the formula toward typical, not headline peaks.

## Anti-patterns

```
# Bad: scale TP up in high-vol because "peaks are bigger"
tp_pts(vol_bin) = peak_mu(vol_bin) * tp_quantile_q30   # OVERSHOOTS

# Bad: flip direction in high-vol because "fades fail there"
if vol_at_entry > threshold:
    direction = opposite(direction)                     # peaks are symmetric

# OK: investigate state exits instead of widening primary exits
# OK: filter (skip) high-vol entries entirely if EV doesn't justify
```
