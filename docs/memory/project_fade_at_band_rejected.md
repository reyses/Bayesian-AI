---
name: FadeAtBand v1 rejected (chart-validated framework, IS-validation killed it)
description: 5s-touch-15m-2sigma fade rule with 4 robustness filters; IS net -$17/day disqualifies despite OOS +$29/day driven by 6-day fluke
type: project
---

**STATUS**: REJECTED 2026-05-09. Do not deploy. Do not "fix" without changing the entry rule.

**Hypothesis tested**: at 5s grid, when price touches 15m ±2σ_close band,
fade to 5m M_close. Robustness filters: hurst<0.60, max_counter_trend_vel=25,
require_divergence between 1m and 5m means, confirm_bars=6.

**Result**:
```
                IS (261 days)              OOS (68 days)
total_PnL       -$4,506                    +$1,964
$/day           -$17.27                    +$28.88
$/trade         -$0.18                     +$0.30
PF-WR           negative                   +0.07 (break-even)
```

**Why OOS looked positive but IS killed it**:
- 6 OOS FLAT_SMOOTH days delivered +$1,650 of the +$1,964 OOS total
- IS FLAT_SMOOTH (53 days): only +$0.13/trade — 22× weaker
- 3 of 6 regime cells (UP_CHOPPY, FLAT_CHOPPY, DOWN_CHOPPY) flip sign IS↔OOS
- Textbook overfit fingerprint per `feedback_quantile_selection_overfit.md`

**What we keep from this experiment**:
1. The 4 robustness filters DO suppress macro-event blowups
   - 2026_03_03 (v1 reversion lost -$1,324): FadeAtBand made +$51, n=125
   - 2026_02_12 macro-pivot day: FadeAtBand made -$3.50 (flat)
   - These filters are **reusable for any reversion strategy** that needs
     macro-day suppression
2. The 2D regime split shows real structure but day-aggregate labels are
   not exploitable as an intra-day filter; signs flip across IS/OOS

**What does NOT work**:
- Entry rule "5s touches 15m ±2σ → fade to 5m mean" with k=2.0
- Default exit suite tuned on this rule (TargetMeanReached, ZSeRetracement,
  Z15sOvershoot, MFEPriceTarget, MFEArmedGiveback) — calibration is noise
  on top of a non-edge entry
- Regime-router approach using `regime_2d` daily labels — no regime is
  stably positive across both IS and OOS

**Next-iteration ideas (NOT yet tested)**:
- k_sigma=2.5 or 3.0 (rarer, higher-conviction band touches)
- Use M_close ±3σ_close OUTER WALL trigger (~1.5% of bars, much rarer)
  instead of M_high/M_low ±2σ
- Build empirical first-passage probability table conditioned on
  (state, regime, hurst, sn, tod, dow, cal_event) FIRST, then design
  entry rule against the cells with stable positive structure on IS

**Pickles preserved as overfit research artifact**:
  training_iso_v2/output/is_FADE_AT_BAND.pkl
  training_iso_v2/output/oos_FADE_AT_BAND.pkl
  training_iso_v2/output/{is,oos}_FADE_AT_BAND_regret.pkl

**Lesson reinforced**: small-sample OOS positivity (68 days) does NOT
justify shipping when IS (261 days) is net negative. The CLAUDE.md
anti-doom-cascade rule cuts both ways — don't claim positive edge from
+$28.88/day when PF-WR is +0.07.
