# Tier EDA Fix Plan — Per-Tier Diagnosis + Remediation

Generated: 2026-04-29 (overnight session)

This plan synthesizes findings from today's analyses into per-tier remediation
recommendations. Each tier is diagnosed against:

1. Aggregate IS+OOS PnL (baseline)
2. PnL when filtered by 1D regime direction
3. PnL when filtered by 5m_dmi_diff sign
4. PnL when filtered by zone behavior
5. PnL when filtered by LinReg slope (period=30)
6. PnL by 1W macro phase (does the tier survive regime shifts?)

---

## EXECUTIVE SUMMARY

**Combined IS+OOS across 9 tiers**: 4,124 trades, $136,635 total PnL ($33.13/trade).
Several tiers are profitable but with MASSIVE LEFT TAIL — winning months get clawed
back by losing months. Three sub-strategies emerge:

| Strategy | Effort | Expected Impact |
|---|---|---|
| **A. Add LinReg slope filter to weak tiers** | 1 hr | +$33,202 (+24% total PnL) |
| **B. Apply 1D regime gate (skip DOWN regimes for some tiers)** | 2 hr | +$5-15k estimated |
| **C. Combine A+B per tier** | 4 hr | +$30-50k estimated |
| **D. Per-1W-phase tier rotation** | weeks | needs proper backtest |

**My recommendation: ship A first** — it's a pure win, low risk, fast to implement.
B should be tested per-tier (some tiers actually want DOWN regimes).

---

## PER-TIER DIAGNOSIS

### Tier 1: FADE_CALM (PROFITABLE, LIMITED-FRAGILE)

**Headline**: 365 trades, $42,641 total ($117/trade), **77.5% WR**. The single
strongest tier in the system.

**Where it works**:
- UP 1D regimes: 214 trades, **84% WR**, $123/trade — sweet spot
- DOWN 1D regimes: 148 trades, 80% WR, $106/trade — also strong
- Both dmi_5m signs work (+ and -): 82-84% WR
- All zone behaviors profitable

**Where it dies**:
- Phases P12_DOWN_12/05 onward — drops out of the winners list. Not by losing
  but by NOT TAKING TRADES. Implies its entry conditions stopped firing in
  late 2025.

**Diagnosis**: FADE_CALM has **excellent edge but it's regime-dependent on
something that changed Dec 2025**. The tier doesn't bleed in adverse conditions
— it just goes dormant.

**Fix recommendations**:
- ⚠️ NO LinReg filter needed — already optimal (only +$239 improvement at T=3).
- ⚠️ NO regime gate needed — works in both UP and DOWN.
- 🔍 INVESTIGATE: why did FADE_CALM stop firing after Dec 2025? Check entry
  feature distributions in Q4 2025 vs prior.
- ✅ Can ship as-is for the regimes where it triggers.

### Tier 2: RIDE_AGAINST (PROFITABLE, SLOPE-FIXABLE)

**Headline**: 1,423 trades, $39,330 ($27/trade), 63% WR. Workhorse tier.

**Where it works**:
- Both UP and DOWN 1D regimes ($25-31/trade, 62-65% WR)
- Both dmi_5m signs ($30-31/trade)
- between_zones zone behavior ($26/trade across 790 trades)

**Where it leaks**:
- Adverse slope: ~24% of its baseline PnL is given back to trades where
  LinReg slope opposes direction by > 0.5 pts/bar.

**LinReg filter result**: at T=0.5, kept 1077 of 1423 trades, PnL $50,807
(**+$11,477 improvement, +29%**). Filter recommendation: **strong yes**.

**Fix recommendations**:
- ✅ ADD LinReg slope filter at T=0.5
- 🤔 Consider whether to also gate by 1D regime (small additional gain only)

### Tier 3: FADE_AGAINST (PROFITABLE, MODERATE FILTER GAIN)

**Headline**: 352 trades, $23,637 ($67/trade), 64% WR.

**Where it works**:
- Both UP and DOWN 1D regimes
- Both dmi_5m signs

**LinReg filter result**: at T=1.5, +$4,195 improvement (+18%).

**Fix recommendations**:
- ✅ ADD LinReg slope filter at T=1.5 (slightly looser than RIDE_AGAINST)
- ✅ Already broadly profitable — minor additional fixes

### Tier 4: BASE_NMP (BREAK-EVEN-ISH, SLOPE-FILTER UNLOCKS)

**Headline**: 1,195 trades, $19,997 ($16.7/trade), 36.6% WR. Low WR
because it's a momentum-fade tier; relies on big winners outweighing many
small losers.

**Where it works**: All 1D regimes equally weakly ($14-17/trade across UP/DOWN).

**Where it leaks**: Adverse slope = 21% of PnL given back.

**LinReg filter result**: at T=0.5, +$4,162 improvement (+21%).

**Fix recommendations**:
- ✅ ADD LinReg slope filter at T=0.5
- ⚠️ Tier has a low-WR profile — accept that or consider tightening entry
  criteria to raise WR at the cost of trade frequency.

### Tier 5: RIDE_CALM (BLEEDING, SLOPE-FILTER REVIVES)

**Headline**: 637 trades, $3,761 ($5.9/trade), 49% WR. Marginal aggregate.

**The shocking finding**: with LinReg slope filter at T=0.5, 451 trades kept,
**$16,891 PnL** ($37.45/trade) — **4.5× improvement over baseline**.

**Diagnosis**: RIDE_CALM has decent edge in benign-slope conditions, but takes
many trades during strong slopes that immediately go against it. The "calm"
detection in its entry doesn't actually filter for benign slope.

**Fix recommendations**:
- ✅ HIGHEST PRIORITY: ADD LinReg slope filter at T=0.5
- 🔍 LONG-TERM: revisit RIDE_CALM's entry conditions to inherently filter for
  the benign-slope state it claims to detect.

### Tier 6: KILL_SHOT (PROFITABLE, ALREADY OPTIMAL)

**Headline**: 106 trades, $2,571 ($24/trade), 87% WR. Small but extremely
high-WR.

**Diagnosis**: Already optimal — LinReg filter at all thresholds gave $0
improvement. The wick-rejection entry logic inherently captures the
slope-favorable setup.

**Fix recommendations**:
- ⚠️ NO changes recommended.
- ✅ Could increase trade frequency by relaxing entry criteria (currently 106
  trades over 14 months = 0.3/day) but would risk degrading WR.

### Tier 7: RIDE_MOMENTUM (TINY SAMPLE)

**Headline**: ~10 trades. Insufficient for reliable analysis.

**Fix recommendations**:
- ⚠️ Too few trades to draw conclusions. Needs longer sample or relaxed entry.

### Tier 8: FADE_MOMENTUM (TINY SAMPLE)

**Headline**: ~10-15 trades. Same issue.

**Fix recommendations**:
- ⚠️ Same as RIDE_MOMENTUM — sample too small.

### Tier 9: FREIGHT_TRAIN (TINY SAMPLE)

**Headline**: ~13 trades. Same issue.

**Fix recommendations**:
- ⚠️ Same — sample too small. Could combine into a meta-tier with the other
  small tiers.

### Tier 10: CASCADE (TINY SAMPLE BUT GOOD WR)

**Headline**: 16 trades, +$172, 87% WR. Tiny sample but 100% WR on UP, 100%
on DOWN.

**Fix recommendations**:
- ⚠️ Sample too small to ship standalone. Logic may be correct; just rare
  triggers.

---

## REMEDIATION PRIORITY ORDER

### Priority 1 — IMMEDIATE: ship LinReg slope filter for 4 tiers

Add `if abs(linreg_slope_30) > T_skip and slope_opposes_direction: skip_trade`
logic to:
- RIDE_CALM (T=0.5) — biggest win: +$13,130
- RIDE_AGAINST (T=0.5) — second biggest: +$11,477
- FADE_AGAINST (T=1.5)
- BASE_NMP (T=0.5)

Total expected improvement: **+$33,202 across 14 months on Python sim** (~$80/day
NT8 equivalent after 2× translation factor).

**Risk**: Low. The filter only SKIPS trades; it doesn't add new ones.

### Priority 2 — INVESTIGATE: FADE_CALM late-2025 dropout

FADE_CALM stopped firing after Dec 2025. Need to determine:
- Did its entry feature distribution shift?
- Did the market regime change in a way that doesn't hit its triggers?
- Is the stop-firing just sample-noise (small phase samples)?

If we can recover FADE_CALM in late 2025, we'd add another +$10-20k expected.

### Priority 3 — PHASE-AWARE TIER ROTATION

The 1W phase analysis showed BASE_NMP wins phases P13-P17 (Dec 2025 onward),
while FADE_CALM and others won earlier. A "phase classifier" + "tier per phase"
could capture this rotation, but requires forward-looking work.

### Priority 4 — STATISTICAL CLEANUP OF TINY TIERS

RIDE_MOMENTUM, FADE_MOMENTUM, FREIGHT_TRAIN, CASCADE all have <30 trades.
Either:
- Combine into a meta-tier
- Relax entry to get more samples
- Drop them and replace with a more frequent tier

---

## CONNECTING TO EARLIER FINDINGS

**Today's main thread**: v1.0.4 zigzag bleeds 101% of losses in DOWN regimes.

**Today's tier finding**: most tiers DON'T have the v1.0.4 problem — they work
in both UP and DOWN regimes, they just leak to adverse-slope trades.

**Why the difference?**: v1.0.4 is single-feature (just zigzag retracement).
The tier classifiers use multi-feature entry conditions that are inherently
more selective. They still leak to slope, but at smaller magnitude.

**This validates the harnessability framework**: the tier system encodes more
of the harnessability check upfront, which is why it's profitable in aggregate
where v1.0.4 isn't.

**Remaining gap**: even the tier system has the LinReg-slope leak. Adding the
slope filter closes that gap. Combined system would be:
- Tier classifier handles entry direction + state filter
- LinReg slope filter handles exit-/skip-on-adverse-slope
- 1D regime gate handles macro skip days

**Expected combined impact**: tier system's $137k baseline + $33k slope filter
+ $5-15k regime gate = $175-185k over 14 months on Python sim.
NT8 equivalent: ~$87-92k = ~$420/day on 209 days. Well above v1.0.4's $50/day.

---

## FILES CONNECTED TO THIS PLAN

- `reports/findings/peak_feature_overlay/2026-04-29_summary.md` — feature → peak direction signal (dmi_diff dominates)
- `reports/findings/strategy_pnl_by_regime/2026-04-29_v104_pnl_by_regime.md` — v1.0.4 baseline (DOWN bleeds)
- `reports/findings/tier_pnl_by_regime/2026-04-29_summary.md` — tier × regime cross-tab
- `reports/findings/tier_pnl_by_regime/2026-04-29_09_phase_winners.md` — best tier per 1W phase
- `reports/findings/tier_pnl_by_regime/2026-04-29_10_tier_linreg_slope_filter.md` — LinReg filter analysis
- `reports/findings/macro_segments/2026-04-29_summary.md` — zone-based segmenter
- `reports/findings/regime_eda/2026-04-29_cross_tf_nesting_full_14mo.md` — UP/DOWN asymmetry confirmed
- `tools/standalone_report.txt` — regression-on-features re-run
