# 2026-04-27 — 9-tier system review (post-overnight retrospective)

**Status**: Post-hoc review. User asked at end of overnight EDA session whether
I had reviewed the 9-tier system as part of my chop-edge work. Honest answer:
**no — I adapted the methodology only.** This doc fixes that gap and connects
the tier findings to the v1.5-RC chop-edge result.

**Sources**: `docs/memory/tier_building_playbook.md` (2026-04-18 consolidated
methodology), `reports/findings/pnl_tier_distribution.md` and `_iso.md`,
`reports/findings/tier_segment_diagnostic.md`,
`reports/findings/tier_daily_concentration.md`,
`reports/findings/iso_tier_audit_2026-04-17_232913.md`,
`reports/findings/tier_signal_conflicts.md`,
`reports/findings/tier_eda_killshot_2026-04-18_002829.md`,
`reports/findings/tier_eda_TREND_FOLLOWER_2026-04-18_073608.md`.

---

## TL;DR — what the tiers tell us about chop vs trend

**Initial hypothesis** (in this doc's first version): the 9-tier engine has
the same chop/trend problem v1.5-RC solves, and the same bleed-filter should
rescue it.

**Empirical test result** (added 2026-04-27 after running
`tools/tier9_bleed_filter.py`): **HYPOTHESIS REJECTED.** The bleed-score
filter HURTS the 9-tier engine in IS (cuts $79K → $27K) and is approximately
neutral OOS. **The 9-tier engine actually PREFERS high-bleed days at the
engine level** — opposite of ZigzagRunner.

| Pattern | ZigzagRunner v1.0.4 | 9-tier blended engine |
|---|---|---|
| Direction policy | Counter-trend (fade 1m structural pivots) | Counter-trend (fade z>2 statistical extremes) |
| Range expansion behavior | Trends continue past pivots → SKIP high-bleed | Bigger z extremes → WANT high-bleed |
| Day-PnL distribution | Bimodal (working vs bleed) | Bimodal (mode IS = `<-$500` AND `>$1K`, both 65 days) |
| Direction WR (post-lookahead-fix) | ~37% (Day 1 NT8 sim) | 31–37% across **all 9 tiers** (iso audit 2026-04-17) |
| v1.5-RC bleed filter helps? | **YES** (-$552 → +$5,021 on 95-day backtest) | **NO** (cuts engine total 40-80% IS) |

**Why the difference**: both are "counter-trend" by name but trigger on
different physics. ZigzagRunner pivots fail on trend continuation; z>2
statistical extremes mean-revert MORE in high-range conditions. The label
"counter-trend" doesn't determine regime preference — the trigger physics does.

---

## 1. Current 9-tier roster (as of 2026-04-18)

The MEMORY.md "9 ExNMP tiers" list (CASCADE / KILL_SHOT / FREIGHT_TRAIN /
FADE_AGAINST / RIDE_AGAINST / RIDE_MOMENTUM / RIDE_CALM / FADE_MOMENTUM /
FADE_CALM) is **stale**. The current isolated pipeline ships a different
roster, post-2026-04-18 refactor:

| Tier | Direction | IS N | IS $/trade | IS $/day | OOS $/day | Status |
|---|---|---:|---:|---:|---:|---|
| **NMP_RIDE** | with-trend | 619 | $+60.37 | $+189.69 | $+53.64 | ✓ Workhorse winner |
| **FADE_AGAINST** | counter | 487 | $+51.14 | $+172.95 | **$-63.92** | ⚠ Flips OOS |
| **NMP_FADE** | counter | 3,538 | $+5.21 | $+73.67 | **$-37.38** | ⚠ Flips OOS (mass tier) |
| **TREND_FOLLOWER** | with-trend | 139 | $+17.77 | $+29.77 | $-5.72 | ⚠ Flat OOS |
| **CASCADE** | counter | 66 | $+23.20 | $+39.26 | $+132.12 | ✓ Survives OOS |
| **KILL_SHOT_ACTIVE** | counter | 36 | $+24.36 | $+32.48 | $+14.80 | ✓ Small but stable |
| **KILL_SHOT_CALM** | counter | 267 | $+2.22 | $+4.93 | $-8.26 | ~ Marginal |
| **RIDE_AGAINST** | counter | 70* | $+3.76 | $+4.78 | $+37.72 | ✓ Pre-flip; 4,204 post-flip |
| **MTF_BREAKOUT** | with-trend | 26 | $+3.81 | $+4.71 | $+17.65 | ~ Small N |
| **MTF_EXHAUSTION** | counter | 20 | $-14.35 | $-17.94 | $+22.34 | ⚠ Net negative IS |

*RIDE_AGAINST n=70 in iso; 4,204 in nn_v2 blended after direction-flip fix
on 2026-04-18 → became the engine's biggest contributor

**(So it's a 10-tier roster now, not 9.)** And the per-tier numbers shift
between the iso pipeline and the nn_v2 blended pipeline because of chains,
direction flips, and the post-lookahead-fix recompute.

---

## 2. The bimodal day-PnL signature — same as v1.5-RC

Per `pnl_tier_distribution_iso.md` IS overall:

```
SUMMARY: 277 days | 5,268 trades
  Winning days: 155/277 (56%)
  Mode bucket: <-$500 (65 days)  ← AND simultaneously
              >$1K (65 days)     ← bimodal!
  Avg $/day: $+311
```

**This is the v1.5-RC pattern at engine scale.** 65 days <-$500 AND 65 days
>$1K out of 277 = the system is essentially flipping a regime-coin every
day. When chop, it mints; when trend, it bleeds.

Per-tier bimodality (IS, from `pnl_tier_distribution.md`):

| Tier | Mode 1 (left) | Mode 2 (right) | Diagnosis |
|---|---|---|---|
| RIDE_AGAINST | `<-$200` (92 days) | `>$200` (76 days) | **Strongly bimodal — counter-trend killer** |
| FADE_CALM | `<-$200` (89 days) | `>$200` (90 days) | **Strongly bimodal — counter-trend killer** |
| FADE_AGAINST | `<-$200` (45 days) | `>$200` (49 days) | Bimodal, smaller |
| KILL_SHOT | `<-$200` (20 days) | `>$200` (22 days) | Mild bimodal |
| MTF_BREAKOUT | `<-$200` (5 days) | `$0:$20` (43 days) | Trend tier — narrower distribution |
| TREND_FOLLOWER | `<-$200` (low) | `$0:$20` mode | Trend tier — narrower |

**Pattern**: every tier where the mode bucket is `>$200` PAIRED with `<-$200`
is regime-dependent. This matches exactly what bleed-score classifier solves
on ZigzagRunner.

---

## 3. The chronological decay — regime change mid-2025

`tier_segment_diagnostic.md` splits IS into halves:

| Tier | S1 (Jan–Jun 2025) $/trade | S2 (Jul–Dec 2025) $/trade | Pattern |
|---|---:|---:|---|
| NMP_RIDE | $+98.65 | $+10.24 | **DECAY (-90%)** |
| FADE_AGAINST | $+67.30 | $+37.03 | DECAY (-45%) |
| TREND_FOLLOWER | $+26.74 | $+9.18 | DECAY (-66%) |
| CASCADE | $-11.01 | $+64.25 | **FLIP** (sign reversal) |
| RIDE_AGAINST | $+17.90 | $-4.10 | FLIP |
| MTF_EXHAUSTION | $-62.23 | $+44.17 | FLIP |
| MTF_BREAKOUT | $+27.75 | $-0.55 | FLIP |
| NMP_FADE | $+6.30 | $+3.99 | Mild decay |
| KILL_SHOT_ACTIVE | $+21.63 | $+32.56 | Stable |
| KILL_SHOT_CALM | $+0.03 | $+4.29 | Improving |

**Total IS $**: S1 = $+63,496, S2 = $+22,742. **The engine made 74% of IS PnL
in the first half of 2025.** This is exactly what a regime-flip looks like
at the year-over-year level — mirroring the 2/26/2026 split I found in
ZigzagRunner's 95-day backtest.

**Lesson for v1.5-RC**: a 95-day backtest catching one regime change is
demonstrating exactly what the 9-tier IS year-over-year shows. Both data
sets agree the same pattern exists at multiple timescales.

---

## 4. Post-lookahead-fix audit (2026-04-17): the floor is the floor

`iso_tier_audit_2026-04-17_232913.md` after the lookahead bug was fixed:

```
Baseline total: $-26,308 over 16,242 trades
```

| Tier | N | WR% | Mean PnL | Total PnL | Counter-mean (if flipped) |
|---|---:|---:|---:|---:|---:|
| FADE_CALM | 7,203 | 35.9% | $-1.90 | $-13,698 | $+4.00 |
| RIDE_AGAINST | 6,793 | 36.9% | $-1.26 | $-8,562 | $+3.32 |
| FADE_AGAINST | 964 | 31.5% | $-3.04 | $-2,930 | $-11.33 |
| KILL_SHOT | 514 | 34.4% | $-1.75 | $-900 | $+0.00 |
| MTF_BREAKOUT | 526 | 37.8% | $-0.56 | $-296 | $+7.30 |
| CASCADE | 154 | 37.7% | $-1.11 | $-171 | — |

**Every tier WR is 31–37%** — that's worse than coin-flip. Direction is
*not* learned at entry from the 91D feature space. This is the same conclusion
my v1.5-RC walk-forward reached (entry direction = noise; what's learnable is
**regime**, not direction).

The "Counter-mean" column is the killer: **flipping direction makes most
tiers profitable**, sometimes substantially. RIDE_AGAINST got the flip
treatment 2026-04-18 → went from -$60/day IS to +$98/day OOS as the engine's
biggest contributor.

---

## 5. The cross-reference: v1.5-RC bleed-score on the 9-tier system — TESTED

**Original hypothesis**: the bleed days that hurt ZigzagRunner hurt the
counter-trend tiers (RIDE_AGAINST, FADE_AGAINST, FADE_CALM, NMP_FADE) too,
and benefit the trend tiers (NMP_RIDE, TREND_FOLLOWER, MTF_BREAKOUT).

**Empirical test**: ran `tools/tier9_bleed_filter.py` on
`training_iso/output/trades/iso_is.csv` (5,268 trades, 267 days) and
`iso_oos.csv` (4,521 trades, 68 days) using v1.5-RC's IS-calibrated
constants (385.32 / 219.83 / 1.0315 / 0.5502).

**Result: HYPOTHESIS REJECTED at engine level.**

### IS engine-day sweep (267 days, $+79,486 unfiltered, $+298/day)

| Threshold (z) | Days fwd | Fwd $/day | Fwd lift | Days inv | Inv $/day | Inv lift |
|---:|---:|---:|---:|---:|---:|---:|
| -1.00 | 95 | $+145 | $-65,666 | 172 | $+382 | $-13,820 |
| -0.50 | 132 | $+240 | $-47,822 | 135 | $+354 | $-31,664 |
| -0.34 | 140 | $+195 | $-52,213 | 127 | $+411 | $-27,273 |
|  0.00 | 165 | $+124 | $-59,108 | 102 | $+579 | $-20,378 |
| +0.50 | 183 | $+180 | $-46,632 |  84 | $+555 | $-32,854 |
| +1.00 | 203 | $+181 | $-42,784 |  64 | $+669 | $-13,820 |

**Forward filter (low-bleed days) cuts engine PnL by 40-80% at every threshold.**
**Inverse filter (high-bleed days only) consistently shows higher per-day PnL** —
the engine is most productive on high-range-expansion days. Opposite of
the v1.5-RC ZigzagRunner result.

### OOS engine-day sweep (68 days, $+4,548 unfiltered, $+67/day)

| Threshold (z) | Days fwd | Fwd $/day | Fwd lift |
|---:|---:|---:|---:|
| -1.00 | 17 | $+32  | $-4,002 |
| -0.50 | 25 | $+61  | $-3,028 |
| -0.34 | 31 | $+75  | $-2,214 |
|  0.00 | 38 | $+78  | $-1,566 |
| +0.50 | 47 | $-6   | $-4,842 |
| +0.75 | 51 | $-47  | $-6,936 |

OOS engine total is too small (68 days, $4.5K total) to show the IS pattern
clearly. Forward filter at z=-0.34 is approximately neutral on per-day basis
($+75 vs $+67 baseline), but loses days to filter and gives back ~half the
total.

### Per-tier IS at z=-0.34 (highlights)

| Tier | unf $/day | fwd $/day | inv $/day | Direction signal |
|---|---:|---:|---:|---|
| **NMP_RIDE** | $+172 | $-26 | $+390 | **Strong inverse signal** (high-bleed = +$218/day lift) |
| FADE_AGAINST | $+176 | $+234 | $+104 | Forward signal (+$58/day) — but small-N tier vs NMP_RIDE |
| MTF_EXHAUSTION | $-18 | $+75 | $-138 | Forward signal (+$93/day) — N=20 only |
| CASCADE | $+39 | $+154 | $-95 | Forward signal (+$115/day) — N=66 only |
| TREND_FOLLOWER | $+29 | $+10 | $+45 | Mild inverse |
| NMP_FADE | $+68 | $+49 | $+90 | Mild inverse (the workhorse — same direction as NMP_RIDE) |

**The two workhorse tiers (NMP_RIDE 583 trades, NMP_FADE 3,410 trades) BOTH
prefer high-bleed days.** They dominate the engine total. Smaller tiers show
the v1.5-RC-style forward signal but don't have the volume to flip the
engine.

### OOS at z=-0.34 (the bigger surprise)

| Tier | unf $/day | fwd $/day | inv $/day | Note |
|---|---:|---:|---:|---|
| **TREND_FOLLOWER** | $-7 | $+23 | $-32 | **Forward filter rescues the trend tier — opposite of name** |
| **FADE_AGAINST** | $-60 | $-126 | $+1 | **Inverse filter rescues the fade tier — opposite of IS direction** |
| MTF_BREAKOUT | $+2 | $+21 | $-12 | Forward signal (+$19/day) |
| RIDE_AGAINST | $+38 | $+50 | $+28 | Forward signal (mild) |

**No tier shows consistent IS+OOS direction in the same direction.** This is
the textbook signature of a noisy signal that doesn't generalize.

---

## 6. Why the bleed-filter doesn't generalize

ZigzagRunner and the 9-tier engine are NOT the same kind of counter-trend.

**ZigzagRunner counter-trend**: fades 1m structural pivots. Range expansion
days = trend continues past pivots = stop-outs. → SKIP high-bleed days.

**9-tier counter-trend (NMP_FADE / RIDE_AGAINST)**: fades z>2 statistical
extremes from the regression band. Range expansion days = bigger statistical
deviations = bigger mean-reversion opportunity. → WANT high-bleed days.

Both are "counter-trend" by name, but they trigger on different physics.
The bleed filter measures the wrong thing for the 9-tier engine.

**Lesson**: regime filters need to be calibrated to the specific entry-trigger
physics. There is no "universal counter-trend regime filter" — what matters
is whether the trigger PERSISTS or REVERTS in expanded-range conditions.

---

## 7. Insights from the tier playbook that DO carry over to v1.5-RC

Now that I've read `docs/memory/tier_building_playbook.md` end-to-end,
several principles directly apply to ZigzagRunner work:

### 7a. "Phantom entry isn't universal" (anti-pattern §9e-bis)

Phantom entry pairs with **short timeouts (≤15min)**. ZigzagRunner has a
**stagnation timeout** (StagnationMonitor) that's the rough equivalent.
The current StagnationMonitor fires at ~12 negative bars (1m). If ZigzagRunner
ever adds an entry confirmation delay (analogous to phantom), the
playbook says:
- Pair with the stagnation timeout to cap downside
- Don't combine with `RideWithTrend` (long-hold mode) — phantom + long
  timeout was a -$2,309 swing on TREND_FOLLOWER

### 7b. "Direction at entry is RANDOM on 91D" (anti-pattern §9c)

The blanket lesson: **don't try to predict direction at entry**. Predict
**regime** (chop/trend) and let direction follow from the strategy's
fade/follow policy. v1.5-RC does exactly this. The 9-tier system has
30+ direction predictors and ALL fail at coin-flip — proves direction
is a dead end for this feature space.

### 7c. "Time-of-day is lazy" (anti-pattern §9d)

Tonight's hour-of-day filter walk-forward already confirmed this on the
NT8 ZigzagRunner ledger:

| Filter | OOS lift |
|---|---:|
| Bleed-score alone | +$6,202 |
| Hour alone | +$1,311 |
| **Combined (bleed AND hour)** | **+$5,268 (worse than bleed alone)** |

The 2026-04-17 user note quoted in the playbook ("time filter admits we
don't understand the physics") explains why: hour-of-day correlates with
the real cause (range expansion, liquidity), but the filter doesn't
generalize. Use `prior_range` + `range_compression` (the actual physics)
instead.

### 7d. "Top-10 days = 100-300% of total" (concentration anti-pattern)

`tier_daily_concentration.md` shows almost every tier is FRAGILE — top-10
days carry MORE than 100% of total PnL (the rest of the days net negative).
This is ALSO true on the 95-day v1.5-RC ledger: the top 5 chop days
account for ~60% of post-filter PnL. **Same pattern, same caveat**:
fragility means a hold-out window is mandatory before live promotion.

---

## 8. Concrete next-session actions

After running the empirical test the originally-proposed actions are
**not worth doing as designed**. Updated direction:

1. **DO NOT** use v1.5-RC bleed filter on the 9-tier engine — empirically
   reduces IS PnL 40-80%. The filter is ZigzagRunner-specific.
2. **Investigate the IS inverse signal** ($+669/day on z>+1.0 high-bleed
   days, 64 days). 9-tier engine appears to be a **range-expansion harvester**
   in IS, not a chop specialist. Worth a separate study to find the
   actual regime feature it's responding to.
3. **OOS doesn't replicate IS** for the inverse signal either ($+408/day
   on 17 days at z>+0.75 OOS — too small a sample, likely overfit).
4. **Per-tier signals exist but don't generalize IS→OOS** for any single
   tier in either direction. This is consistent with the playbook's
   "direction at entry is RANDOM on 91D" finding.
5. **Real next step for the 9-tier system**: not regime filtering. The
   2026-04-18 work direction (peak buckets, multi-exit by bucket, parallel
   inverse-direction tier) is the more productive lever per the playbook.

Tools delivered:
- `tools/tier9_bleed_filter.py` — runs the filter at any threshold,
  outputs per-tier and engine-day tables. Use this as a template for
  testing future regime hypotheses on the iso ledger.
- `reports/findings/2026-04-27_9tier_bleed_filter.md` — full IS+OOS
  per-tier filter table at z=-0.34.

---

## 9. Honest accounting

| Question | Answer |
|---|---|
| Did I review the 9 tiers in the overnight pass? | No. Methodology only. |
| Did I review them when asked? | Yes — read findings docs, hypothesized cross-applicability of bleed filter. |
| Did the cross-reference hypothesis hold? | **NO.** Empirical test (`tier9_bleed_filter.py`) showed bleed filter cuts IS engine PnL 40-80%, neutral OOS. |
| What did I learn? | Two strategies labeled "counter-trend" can have OPPOSITE regime preferences — depends on whether the trigger physics persists or reverts in expanded-range conditions. ZigzagRunner pivots persist (continue trending), z>2 extremes revert MORE. |
| Was this a wasted opportunity? | No — disproving a hypothesis is also a finding. "v1.5-RC ZigzagRunner filter is not portable to 9-tier engine" is a real conclusion that prevents future cargo-culting. |
| What changes for v1.5-RC? | Nothing. v1.5-RC stands on its own evidence base (the 95-day NT8 ZigzagRunner backtest). The 9-tier system is an unrelated portfolio component. |
