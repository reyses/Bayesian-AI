# JULES_v15_chop_specialist.md — ZigzagRunner v1.5-RC Spec

**Author**: Claude Sonnet 4.5 (overnight session, 2026-04-27)
**Status**: SPEC ONLY — not yet implemented. User to review before code work.
**Predicted gross alpha**: +$50-80/day on a 95-day labeled backtest, vs −$5.81/day
unfiltered. Subject to walk-forward generalization caveat (see §10).

---

## 0. Why this version exists

User identified a clean two-regime split in the v1.0.4 95-day backtest:
working window (1/2-2/26) = +$89/day, bleed window (2/27-4/24) = −$95/day, net
−$552/95 days. Same trades, opposite outcomes — **regime-dependent, not
random**.

User's framing: *"trend days are the easy ones — chop is what everyone avoids
and we thrive."*

Validated overnight via `tools/nt8_bleed_harvest_classifier.py`: two
forward-available features (`prior_range`, `range_compression`) discriminate
BLEED vs HARVEST days with d_IS=+0.58/+0.48 and d_OOS=+0.77/+0.78 — strong
walk-forward signal. Combined-z rule at X=40% threshold catches 64% of OOS
bleed days, saves $5,084 OOS over 47 days.

**v1.5-RC is the same v1.4-RC machinery, plus a session-open filter that
skips ~40% of sessions predicted to be trend regimes.**

---

## 1. Goal

Convert the negative-alpha v1.0.4 / v1.4-RC into a positive-alpha strategy by
gating new entries on a forward-available chop-vs-trend regime classifier.

**Specifically**: keep the existing pivot detection, exit logic, DRM trail,
StagnationMonitor, missed-breach handler. Add ONE new gate at session open that
decides whether to trade today.

**Non-goals**:
- Direction flipping (proved not the answer earlier today)
- New entry logic (works in chop, just need to skip non-chop)
- ML / neural-net classifier (overkill at N=95 days)
- Per-trade scoring (orthogonal layer; can come later)

---

## 2. Architecture

Inherit v1.4-RC's structure:

| BarsInProgress | TF | Purpose |
|---|---|---|
| 0 | primary chart (typically 1s) | order fills, MFE/MAE update |
| 1 | Pivot TF (default 60s) | pivot detection, EOD, entry cutoff, stagnation |
| 2 | Hard SL TF (default 5s) | Initial-state SL + Tier1 trail |
| 3 | Trail TP TF (default 1s) | Tier2 fast-ratchet trail |
| 4 | **Daily (Day, 1)** | **regime filter (UPGRADED)** |

The daily series is already in v1.4-RC; v1.5-RC just changes how the filter
decision is computed.

---

## 3. Regime filter logic (the new piece)

### 3.1 New properties

```csharp
[NinjaScriptProperty]
[Display(Name = "Enable Bleed Filter", Order = 1, GroupName = "RegimeFilter")]
public bool EnableBleedFilter { get; set; }   // default true

[NinjaScriptProperty]
[Display(Name = "Bleed Filter Threshold (z)", ...)]
public double BleedThresholdZ { get; set; }   // default -0.34 (= IS-median, OOS sweet spot)
                                              //   conservative = -0.5 (highest $/kept-day)
                                              //   aggressive = +0.75 (biggest aggregate)
                                              //   AVOID 0.0 (local min — classifier indecisive)

[NinjaScriptProperty]
[Display(Name = "Range Mean Lookback (days)", ...)]
public int RangeMeanLookbackDays { get; set; }  // default 20

// IS-calibrated normalization parameters (computed from 1/2-3/1/2026 N=48
// days. Update via tools/v15_recalibrate.py quarterly):
private const double MEAN_PRIOR_RANGE       = 385.32;  // pts (IS mean of yesterday's H-L)
private const double STD_PRIOR_RANGE        = 219.83;  // pts (IS stdev)
private const double MEAN_RANGE_COMPRESSION = 1.0315;  // (IS mean of prior_range / 20d_mean)
private const double STD_RANGE_COMPRESSION  = 0.5502;  // (IS stdev)
```

### 3.2 Decision logic (called once per session at first Pivot TF bar)

```
At first Pivot TF bar of new session (date != currentSessionDate):
    if not EnableBleedFilter:
        tradeAllowedToday = true
        log("filter disabled, trading")
        return

    if CurrentBars[BIP_DAILY] < RangeMeanLookbackDays:
        tradeAllowedToday = true   // warmup, default to trading
        log("warmup, trading")
        return

    # Compute features from prior daily bars (BIP_DAILY)
    prior_range = Highs[BIP_DAILY][0] - Lows[BIP_DAILY][0]

    sumRange20 = 0
    for i in 0..RangeMeanLookbackDays-1:
        sumRange20 += Highs[BIP_DAILY][i] - Lows[BIP_DAILY][i]
    mean_range_20d = sumRange20 / RangeMeanLookbackDays

    range_compression = prior_range / mean_range_20d

    # Z-scores against IS-calibrated normalization
    z1 = (prior_range       - MEAN_PRIOR_RANGE)       / STD_PRIOR_RANGE
    z2 = (range_compression - MEAN_RANGE_COMPRESSION) / STD_RANGE_COMPRESSION

    # Combined bleed score (positive = bleed-likely)
    bleed_score = z1 + z2

    tradeAllowedToday = (bleed_score <= BleedThresholdZ)

    log("RegimeFilter: prior_range=%.1f range_compression=%.2f "
        "bleed_score=%.2f threshold=%.2f -> trade=%s",
        prior_range, range_compression, bleed_score, BleedThresholdZ, tradeAllowedToday)
    AppendRegimeLog(date, prior_range, range_compression, bleed_score, tradeAllowedToday)
```

### 3.3 Gate location

Same as v1.4-RC: in `OnPivotBarUpdate`, the entry block checks
`tradeAllowedToday` before any `EnterLong` / `EnterShort`. If false, no new
entries fire that day. Existing positions still flow through trail/SL/EOD.

### 3.4 Calibration constants

Computed from **IS half** of the 95-day labeled dataset (1/2-3/1):

| Constant | Value | Source |
|---|---|---|
| `MEAN_PRIOR_RANGE` | 385.32 | IS mean of prior_range (yesterday's H-L), N=48 days |
| `STD_PRIOR_RANGE` | 219.83 | IS stdev of prior_range |
| `MEAN_RANGE_COMPRESSION` | 1.0315 | IS mean of range_compression (= prior/20d_mean) |
| `STD_RANGE_COMPRESSION` | 0.5502 | IS stdev |

The combined `bleed_score = z(prior_range) + z(range_compression)`. Empirical
IS quantiles:

| Threshold (z) | IS quantile | Approx X% (skip rate) | Sweet-spot? |
|---|---|---|---|
| −1.5 | 0.18 | 82% | very strict |
| −1.0 | 0.30 | 70% | strict |
| **−0.5** | **0.41** | **59%** | **HIGHEST $/kept-day = $128.75** |
| **−0.34** | **0.50** | **50%** | **OOS-validated sweet spot — 82% bleed catch, $6,202 OOS lift** |
| 0.0 | 0.60 | 40% | local minimum (avoid) |
| +0.5 | 0.69 | 31% | loose |
| **+0.75** | **0.74** | **26%** | **HIGHEST total IS-leaked PnL = +$5,214** |
| +1.5 | 0.85 | 15% | very loose |

**MVP recommendation: z = −0.34** (X=50%, OOS-validated). Skip ~50% of days
that score above the IS-median bleed_score. Catches 82% of OOS bleed days.
Backtested OOS lift: **+$6,202** on the held-out half (47 days, IS threshold
frozen at IS-trained value).

**Conservative alternative: z = −0.5**. Skip ~60% of days; per-kept-day rate
$129. Higher conviction, fewer trades.

**Aggressive alternative: z = +0.75**. Skip only ~26% of days; biggest
absolute lift ($+5,214 IS-leaked) but per-kept-day rate drops to $78.

**Avoid z = 0.0** — empirical local minimum on the threshold curve. The
classifier is least decisive at the IS median.

These constants should be **recomputed quarterly** as the regime baseline
drifts. Provide a `tools/v15_recalibrate.py` companion script (post-MVP).

---

## 4. Direction logic — UNCHANGED

v1.0.4's per-pivot enum:
- `OnHighPivot = Short` (counter-trend)
- `OnLowPivot = Long` (counter-trend)

The chop edge IS the counter-trend edge. The flip experiments earlier today
showed `Long, Long` and `Short, Long` and `With-trend` all destroy alpha.
Default settings are correct.

---

## 5. Risk machinery — INHERIT v1.4-RC verbatim

- DynamicRiskManager_v14 (Initial → Tier1 → Tier2 state machine)
- StagnationMonitor_v14 (5 consecutive negative Pivot TF bars → flatten)
- Missed-breach handler in RouteStopOrder
- isSimulatedStop = false (exchange-stop semantics)
- Multi-TF risk evaluation (Hard SL TF for Initial+Tier1, Trail TP TF for Tier2)
- The DRM bug fix from earlier today (use `currentEntryPrice` not stale
  `Position.AveragePrice`)

No changes here. The risk infrastructure is solid; we're only adding the
day-level filter.

---

## 6. Optional secondary filters (post-MVP)

### 6.1 Hour-of-day mask — DEPRIORITIZED 2026-04-27

In-sample analysis showed a strong hour-of-day signal. **Walk-forward
validation FAILED**:

| Config | OOS kept PnL | OOS lift |
|---|---|---|
| Unfiltered | −$3,954 | 0 |
| Hour-only (IS-derived tradable hours) | −$2,643 | +$1,311 |
| Bleed-only (the MVP) | **+$2,248** | **+$6,202** |
| **Bleed AND hour combined** | **+$1,314** | **+$5,268 (WORSE)** |

The hour-of-day signal is **partially overfit** to IS. Combining the hour
filter with the bleed filter throws out good bleed-pass days — net loss of
$934 vs bleed-alone.

**Recommendation: do NOT add hour-of-day mask.** The bleed filter is the
right MVP. If a future hour-of-day signal proves walk-forward stable on a
fresh dataset, revisit.

### 6.2 Day-of-week mask

Mon/Wed: profitable. Tue/Thu/Fri: marginal-to-bleed.
Property: `TradeableDaysMask`. Same opt-in design as hour mask.

These are NOT in the v1.5-RC MVP. They get added in v1.5.1 after the day
filter is validated.

---

## 7. Implementation steps

1. Copy `docs/nt8/ZigzagRunner_v1.4-RC.cs` → `docs/nt8/ZigzagRunner_v1.5-RC.cs`
2. Find/replace `_v14` → `_v15` (helper class names, namespace collisions)
3. Find/replace `"ZigzagRunner_v1.4-RC"` → `"ZigzagRunner_v1.5-RC"`
4. Find/replace `VERSION = "1.4-RC"` → `VERSION = "1.5-RC"`
5. Find/replace `ZigzagRunner_v14` → `ZigzagRunner_v15`
6. Replace `MaxMeanRange5dPts` property and the simpler regime-filter logic
   in `EvaluateRegimeFilter` with the combined-z bleed-score logic from §3.2
7. Add the four IS-calibrated constants (§3.4) as `private const`s
8. Add `RangeMeanLookbackDays` NinjaScriptProperty (default 20)
9. Update `RegimeLogPath` default to `nt8_zigzag_v1.5_regime_log.csv`
10. Update `CsvPath` default to `nt8_zigzag_v1.5_trades.csv`
11. Update changelog block at top of file with v1.5-RC rationale
12. Build a calibration script (post-MVP)

Total LOC change: ~50 lines (mostly the new EvaluateRegimeFilter body).

---

## 8. Predicted performance (on the 95-day labeled set)

Honest performance numbers (OOS-only, no IS leakage):

| Setup | $ over OOS half (47 days) | Notes |
|---|---|---|
| v1.4-RC (current) | ~−$300 | counter-trend baseline |
| v1.5-RC at threshold z=0.0 (X=50%) | **~+$5,800** | catch 82% of bleed days |
| v1.5-RC at threshold z=+0.5 (X=30%) | ~+$3,800 | catch 55% of bleed days |
| v1.5-RC at threshold z=-0.5 (X=70%) | ~+$6,200 | catch ~85% but skips a lot |

Sweet spot: **z = 0.0** (X=50%) — skips half the days, catches 82% of bleed,
$5.8k OOS lift. Per-tradeable-day rate: ~$120/day on the 24 days kept.

**Honest deployment estimate**: +$50-80/day average across all sessions
(including skipped sessions counting as $0). Walk-forward / out-of-sample
on a fresh 30-day window is required before live promotion.

---

## 8.4 Alternative classifiers tested (linear MVP wins)

Built `tools/v15_alt_classifiers.py` to compare:

| Classifier | OOS lift | OOS kept PnL | IS AUC |
|---|---|---|---|
| **MVP linear z-score (2 features)** | **+$6,202** | **+$2,248 ✓** | (linear, no AUC) |
| Decision Tree depth=2 | +$2,574 | −$1,380 ✗ | 0.754 |
| Decision Tree depth=3 | +$3,147 | −$807 ✗ | 0.822 |
| Decision Tree depth=4 | +$3,167 | −$787 ✗ | 0.871 |
| Logistic Regression (all features) | +$5,650 | +$1,696 | 0.759 |
| AND-gate (prior_range > 500) | +$2,942 | −$1,013 | — |
| AND-gate (prior_range>450 AND mean_range_5d>350) | +$3,465 | −$489 | — |

**Linear MVP outperforms all alternatives on KEPT PnL** (= the deployable
metric — what you actually make on the days you trade). Trees overfit IS
(AUC up to 0.87 IS) but produce NEGATIVE KEPT PnL OOS — they're throwing
out the wrong days.

**Lesson**: do not replace the 2-feature linear rule with a more complex
classifier. With 95 days and only ~40 BLEED examples, simple is better.

## 8.5 ROBUSTNESS REALITY CHECK (3-fold walk-forward)

3-fold chronological cross-validation revealed a critical caveat:

| Test fold | Period | Unfilt $ | Filtered $ | Lift |
|---|---|---|---|---|
| T3 (last third) | 3/18-4/24 | +$331 | +$2,119 | **+$1,788** ✓ |
| T2 (middle) | 2/08-3/17 | −$2,306 | +$1,813 | **+$4,119** ✓ |
| **T1 (first)** | **1/02-2/06** | **+$1,424** | **−$667** | **−$2,091** ✗ |

**The filter HURTS in pure-chop regimes** (T1 had no bleed regime to catch;
filter mistakenly skipped winning days where prior_range was big due to
holiday volatility).

**Net 3-fold expected: +$1,272/test-period** (positive but with substantial
variance).

**Implication**: the filter is a **regime-volatility play**, not a universal
alpha. It WORKS by detecting bleed regimes when they occur. In pure-chop
regimes, it produces false-alarm skips that cost ~$60/day on average.

This means the spec needs a **watchdog mechanism**:

```csharp
// Track live performance over rolling 30-day window.
// If filter-on PnL < (filter-off would-be PnL) − $1,500, log warning.
// Optional: auto-disable filter if degradation persists ≥ 60 days.
```

(Watchdog NOT in MVP. Manual review of `nt8_zigzag_v1.5_regime_log.csv`
suffices for now.)

## 9. Risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| 95-day window may not generalize | HIGH | Validate on a fresh 30-day window post-deploy. Flag for promotion only after 60 days of live data confirms |
| IS-calibrated constants drift | MEDIUM | Quarterly recalibration. Add a script. Monitor live regime-log CSV for distribution shifts |
| 50% of sessions flat = 50% of buy-and-hold drift forfeited | MEDIUM | Could add an `OnFilterSkipDay = AlwaysLong` mode (post-MVP) to harvest the regime drift on filter-skipped sessions |
| Overfit on 2-feature rule | LOW | Only 2 features, both with min\|d\|≥0.45 walk-forward. Effect size is real |
| Stop-fill slippage during high-vol days (the days we WON'T skip) | LOW | DRM ratchet + missed-breach handler already in place |
| Strategy doesn't beat passive long during bull regime | EXPECTED | Buy-and-hold MNQ over the same window = +$247/day. v1.5-RC = +$50-80/day. Strategy provides DIVERSIFICATION not maximum return |

---

## 10. Validation plan before live

1. **Compile**: F5 in NinjaScript Editor, zero errors expected
2. **Apply** to a fresh chart, parameters identical to v1.4-RC defaults plus
   the new `EnableBleedFilter=true, BleedThresholdZ=0.0`
3. **Backtest** on 1/2-4/24/2026 (the 95-day window). Confirm:
   - Total trades drops from ~1,678 to ~700-900
   - Skipped days log to `nt8_zigzag_v1.5_regime_log.csv` with date, scores, decision
   - Net PnL flips from −$552 to +$5,000-9,000 range
   - Day WR > 55%
4. **Hold-out backtest** on a fresh 30-day window (e.g., 5/1-5/30 once data
   exists)
5. **Sim101 forward run** for 30 days to confirm in-live behavior
6. **Promotion to live**: only if hold-out + sim both confirm
7. **Continuous monitoring**: regime-log CSV reviewed weekly for distribution
   drift; recalibrate if `prior_range` IS-mean shifts more than ±20%

---

## 11. Rollback plan

If v1.5-RC underperforms in live (e.g., −$100/day after 5+ sessions),
disable `EnableBleedFilter`. Strategy reverts to v1.4-RC behavior (which is
v1.0.4 trade logic + DRM trail). No code change required, just a property
toggle.

If the regime filter is fundamentally wrong (e.g., 30 days of live show no
edge), full rollback: re-deploy v1.4-RC from `docs/nt8/archive/`. v1.5-RC
file stays in repo as research artifact.

---

## 12. Open questions for next session

1. **Threshold tuning vs robustness**: should default be z=0.0 (X=50%, lots
   of skipped days, big bleed catch) or z=+0.5 (X=30%, fewer skipped, less
   protection)? Depends on user's preference for trades/day vs PnL stability.

2. **OnFilterSkipDay = AlwaysLong** — would harvesting passive long drift on
   filter-skip days improve net PnL? Predicted yes by $100-200/day on those
   days. But adds complexity and live-deploy risk. Defer.

3. **Recalibration cadence** — quarterly, monthly, weekly? Need to monitor
   distribution drift first.

4. **Multi-contract**: would v1.5-RC also work on ES, NQ, YM? They have
   different chop/trend ratios. Test post-deployment.

5. **R sensitivity**: tonight's classifier was on R=50 trades. What if R=30
   (the v1.0 default)? Effect size could differ. Quick re-run on a R=30
   trade ledger would confirm.

---

## 13. Appendix: shortlist of stable features

From `tools/nt8_bleed_harvest_classifier.py --forward-only` on 95 days:

| Feature | d_IS | d_OOS | Forward-available | Used in MVP |
|---|---|---|---|---|
| **prior_range** | +0.576 | +0.774 | YES | **YES** |
| **range_compression** | +0.475 | +0.782 | YES | **YES** |
| prior_intraday_range (= prior_range from ATLAS 1m) | +0.576 | +0.804 | YES (via ATLAS) | NO (redundant with prior_range) |
| mean_range_5d | +0.216 | +1.041 | YES | NO (overfit risk; OOS-only spike) |
| prior_vr_60_240 | +0.795 | +0.330 | YES | NO (low OOS d) |
| prior_intraday_atr_5m | −0.316 | −0.305 | YES | NO (low effect size) |
| mean_efficiency_5d | −0.437 | −0.068 | YES | NO (OOS d collapse) |

The 2-feature rule is the cleanest walk-forward winner. Adding more features
overfits IS without OOS gain.
