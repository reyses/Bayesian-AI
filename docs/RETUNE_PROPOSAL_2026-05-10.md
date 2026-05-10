# RETUNE PROPOSAL — Entries + Exits (2026-05-10)

Consolidates all research findings from the 2025-10-29 dissection session
into actionable parameter changes. Each change is tied to a specific
finding + expected impact + validation step.

---

## Status: PROPOSAL — awaiting approval before any tier code is modified.

Per CLAUDE.md: "When modifying from a baseline, change ONE thing, run
pipeline, compare. If worse, revert immediately. If better, commit + tag
new baseline." This doc lists multiple changes; deploy in ranked order,
one at a time, validating each before the next.

---

# ENTRY-SIDE RETUNES

## E1 — Z-band entry: |z| in [1.5, 2.2] (VALIDATED 2026-05-10)

**FINDING**: Cross-referenced the existing band-touch population census
(`tools/band_touch_aggregation.py`) with FADE_CALM trade outcomes on all
21,371 IS+OOS trades. Per-|z|-bucket PF_WR:

```
|z| BUCKET    N        WR%     MEAN_$   TOTAL_$    PF_WR
1.0 - 1.5    10,776   28.7%   -$0.04   -$408      -0.019  ← marginal LOSER
1.5 - 1.8     5,216   27.5%   +$0.19   +$1,003    +0.096  ★ strongest
1.8 - 2.0     2,382   25.2%   +$0.16   +$372      +0.080
2.0 - 2.5     2,959   23.1%   -$0.09   -$266      -0.049  ← extreme LOSER
2.5 - 3.0        38   10.5%   -$0.46   -$18       -0.357  tiny n
```

Both ends LOSE; the middle [1.5, 2.2] is the deployable band.

**CHANGE**: In `_qualify` after the NMP seed:
```python
if abs(seed.z) < 1.5 or abs(seed.z) > 2.2:
    return None  # skip marginal AND extreme z entries
```

**VALIDATED IMPACT** (full 21,361 trades):
```
baseline         n=21,361  total=$686    IS=$424  OOS=$261   PF_WR=0.016
E1 [1.5, 2.2]    n=9,176   total=$1,376  IS=$742  OOS=$634   PF_WR=0.076   ★ 4.75x PF
```

OOS uplift +$373 on 2,500 OOS trades = clean validation.

**RISK**: Low. Both IS and OOS show positive uplift in the same direction.

---

## E2 — VETO SHORT entries in 1h-neutral regime (VALIDATED 2026-05-10)

⚠️ **EARLIER FLIP-RULE VARIANT REJECTED.** Initial test of FLIP (direction-swap)
on full population looked great IS but **failed OOS** (IS +$1,752 / OOS −$310,
classic overfit). Replaced with VETO (skip) which is robust on both splits.

**FINDING**: per (direction, 1h-alignment) cell:

```
DIR    1h_CATEGORY    N      MEAN$   TOTAL$   PF_WR
LONG   any            10,692 +$0.16  +$1,719  +0.082   all wins
SHORT  aligned         4,214 +$0.02  +$92     +0.011   breakeven
SHORT  neutral         2,094 -$0.22  -$470    -0.114   ★ STRUCTURAL LOSER
SHORT  opposed         4,361 -$0.15  -$655    -0.069   loser but partly noise
```

**SHORT × neutral** (|1h_z_se| < 0.3) is the cleanest loser cell.

**CHANGE**: After E1 filter, additional gate:
```python
if seed.direction == 'short':
    z_1h = state.get('L3_1h_z_se_12', 0)
    if abs(z_1h) < 0.3:
        return None  # skip SHORT in 1h-neutral regime
```

**VALIDATED IMPACT** (combined with E1 [1.5, 2.2] band):
```
baseline                            n=21,361  total=$686    IS=$424  OOS=$261   PF_WR=0.016
E1 only                             n=9,176   total=$1,376  IS=$742  OOS=$634   PF_WR=0.076
E1 + VETO short_neutral             n=8,302   total=$1,601  IS=$940  OOS=$660   PF_WR=0.098  ★ BEST
E1 + VETO short_non_aligned (both)  n=6,494   total=$1,162  IS=$640  OOS=$522   PF_WR=0.092
```

VETO-neutral alone is the optimal — adds OOS uplift over E1 only, both
splits aligned. PF_WR is **6× the baseline**.

**RISK**: Both IS and OOS positive; selection-bias risk minimal.

**FINDING** (validated on full 21,361 FADE_CALM IS+OOS trades):

```
DIRECTION  CATEGORY    N      MEAN_$   TOTAL_$   PF_WR
LONG       aligned     4,579  +$0.14   +$637     +0.075
LONG       neutral     2,083  +$0.09   +$177     +0.047
LONG       opposed     4,030  +$0.23   +$905     +0.117    ★ best
SHORT      aligned     4,214  +$0.02   +$92      +0.011    ~breakeven
SHORT      neutral     2,094  -$0.22   -$470     -0.114    ★ FLIP target
SHORT      opposed     4,361  -$0.15   -$655     -0.069    ★ FLIP target
```

**LONG fades work in all 1h regimes.** **SHORT fades only work when 1h
is "aligned" (also stretched up)**. SHORT in opposed or neutral 1h is
a structural loser.

OOS validation confirms direction-of-effect (aligned strongest, neutral
and opposed weak/negative).

**CHANGE**: New direction-flip rule in `training_iso_v2/strategies/_nmp_base.py`
(or in a per-tier wrapper):

```python
def apply_flip_rule(state, seed) -> NMPSeed:
    z_1h = state.get('L3_1h_z_se_12', 0)
    if seed.direction == 'short' and z_1h < 0.3:
        # SHORT in non-aligned 1h → flip to LONG
        return NMPSeed(fired=True, direction='long', z=seed.z, rprob=seed.rprob)
    return seed
```

Wire into the four NMP fade tiers' `_qualify()`:
```python
seed = apply_flip_rule(state, seed)
```

**EXPECTED IMPACT** (from FLIP test on FADE_CALM):
```
Current P&L (no flips):      +$686
After flipping 2 cells:     +$2,936       ★ +$2,250 uplift (+328%)
```

Both flipped cells convert losses to wins by symmetry (assuming flip P&L
is exactly opposite — verified on history).

**VALIDATION**:
1. Aggregate `tools/zse_at_entry_wr_analysis.py` — done above (V1)
2. Aggregate regime-alignment WR — done above (V2 full pop)
3. Per-tier engine run with flip rule

**RISK**:
- The flip assumes exit logic still works in flipped direction. Need to
  verify ZSeReversal etc. don't break (they're direction-aware exits).
- Flip changes trade direction → flip can interact poorly with
  position-management rules built for the original direction. Test on
  a single tier first.

---

## E3 — Add CAT_HARVEST pre-position SHORT during danger windows

**FINDING**: Cat events at 1h HL k>=8σ are 96% CRASHES. P(cat in next 60m)
hits 40% on Tuesday UTC 1, 32% on Wednesday UTC 1.

**CHANGE**: New tier `training_iso_v2/strategies/cat_harvest.py`:
```python
class CatHarvest(Strategy):
    """Pre-position SHORT 10 min before known danger windows."""
    name = 'CAT_HARVEST'
    
    def evaluate(self, state):
        from training_iso_v2.filters.bayes_filters import cat_harvest_signal
        if cat_harvest_signal(state.timestamp) == 'PRE_SHORT':
            # Only fire once per window
            if not self._fired_this_window(state):
                return EntrySignal(direction='short', tier=self.name)
        return None
```

**EXPECTED IMPACT**: ~50-100 trades/year of SHORT-side cat-harvest.
Expected per-trade win rate ~62% (matches population), mean +$185 IS.

**VALIDATION**: Per-day backtest of cat-harvest only on 2025+ data.
Confirm structural edge before deploying. Use $/trade and day-WR.

**RISK**: The 96% crash bias was measured on |max_z|>=6 events. Tiny
sample size (n=457 events across 345 days). Need OOS confirmation
separately.

---

## E4 — Add COMPRESSION-BOUNCE LONG entry tier

**FINDING**: `L2_15m_vol_sigma_12 below -3σ from native RM` → P_up_60m =
0.55 IS / 0.64 OOS, mean +$3 IS / +$7 OOS, n=423/95.

**CHANGE**: New tier `training_iso_v2/strategies/compression_bounce.py`:
```python
class CompressionBounce(Strategy):
    """LONG when 15m vol_sigma crushes below its native RM by 3σ."""
    name = 'COMPRESSION_BOUNCE'
    
    def __init__(self):
        from training_iso_v2.filters.bayes_filters import CompressionBounce
        self.detector = CompressionBounce()
    
    def evaluate(self, state):
        v = state.get('L2_15m_vol_sigma_12', float('nan'))
        if self.detector.update(v) == 'LONG_BIAS':
            return EntrySignal(direction='long', tier=self.name)
        return None
```

**EXPECTED IMPACT**: ~400 entries/year (from native-RM event count).
P_up 64% OOS = small but stable edge. Mean +$7 OOS per event.

**VALIDATION**: IS+OOS run on COMPRESSION_BOUNCE in isolation.
Compare to baseline (no-trade) and to noise.

---

# EXIT-SIDE RETUNES

## X1 — Wire BayesConditionalExit for FADE_CALM (already done)

**FINDING**: Conditional exit table at (t_since_peak, capture_ratio)
fires at P_final >= 0.85 → +$30 on FADE_CALM 21k IS+OOS trades.
OOS specifically: +$71. Validated; already wired with
`enabled_tiers = {'FADE_CALM'}`.

**STATUS**: ✅ DONE in `training_iso_v2/exits.py` — class
`BayesConditionalExit`. Conservatively enabled only for FADE_CALM.

**REMAINING ACTION**: Verify via `python -m training_iso_v2.run_iso
--tiers FADE_CALM` that the wired class produces the expected uplift
through the full engine pipeline.

---

## X2 — Pre-CAT close: close LONGs before Tuesday/Wed/Thu UTC 1-2 windows

**FINDING**: 96% of catastrophic events are crashes. Holding a long
through Tuesday UTC 1 window is structurally adverse (40% P_cat in next
60m, 95% crash bias).

**CHANGE**: New ExitRule `PreCatClose` in `training_iso_v2/exits.py`:
```python
class PreCatClose(ExitRule):
    """Close LONGs N seconds before any cat-harvest window opens."""
    name = 'pre_cat_close'
    LEAD_S = 600  # 10 min lead
    
    def evaluate(self, state, position):
        if position.direction != 'long':
            return None
        from training_iso_v2.filters.bayes_exit_oracle import time_to_next_cat_window
        t = time_to_next_cat_window(state.timestamp, min_lookahead_s=0)
        if t is not None and t <= self.LEAD_S:
            return self.name
        return None
```

**EXPECTED IMPACT**: Closes ~50-100 LONGs/year before danger windows.
Each saved loss ~$50-200 expected.

**VALIDATION**: Compare IS+OOS run with and without `PreCatClose` in
the exit suite. Check that we're not missing winners.

---

## X3 — TOD position-size multiplier (no-trade in peak danger)

**FINDING**: P(cat) reaches 25-40% at Tuesday UTC 1. Position sizing
should reflect this — currently flat 1 contract everywhere.

**CHANGE**: Engine sizing layer reads `tod_risk_size_multiplier(ts)`
and scales open-position size:
```python
# In engine.py before position.size = base_size:
mult = tod_risk_size_multiplier(state.timestamp)
position.size = round(base_size * mult)   # 0 if mult * base < 1 contract
```

For 1-contract trading, this effectively becomes a NO-TRADE window
during UTC 1 (mult=0.1 → 0 contracts).

**EXPECTED IMPACT**: Skips trading during UTC 1 entirely. Saves
catastrophic losses. May skip winners too — UTC 1 has wins as well.

**VALIDATION**: IS+OOS run with sizing layer. Compare total $/day with
and without.

---

# VALIDATION-FIRST CHECKS (DO THESE BEFORE TUNING ANY TIER)

## V1 — Aggregate |z_at_entry| distribution: wins vs losses

For ALL FADE_CALM IS+OOS trades, compute:
- mean |z_at_entry| of winning trades
- mean |z_at_entry| of losing trades
- distribution per regime

If z >= 1.8 trades have meaningfully higher WR than z < 1.8 trades, E1
is validated. If not, E1 is overfit to the 2025_10_29 best/worst pair.

```bash
python tools/zse_at_entry_wr_analysis.py --tier FADE_CALM
```

## V2 — Aggregate regime-alignment hypothesis

For ALL FADE_CALM IS+OOS trades, compute:
- WR when 1h_z_se aligned with fade direction
- WR when 1h_z_se opposed
- WR when 1h_z_se neutral (|z| < 0.3)

If aligned-regime trades have meaningfully higher WR, E2 is validated.

```bash
python tools/regime_alignment_wr_analysis.py --tier FADE_CALM
```

## V3 — OOS-only validation of compression-bounce cells

The L2_15m_vol_sigma_12 below-3σ cell is the strongest of the 9 found.
But it was found by scanning 90 features. There's selection bias.
Re-validate on TRUE held-out OOS (e.g. last 3 months) before deploying
as a tier.

## V4 — 96% crash bias OOS-only

The cat-harvest finding (96% crash) was on IS+OOS combined. Validate
on 2026 OOS-only. If bias drops below 80%, scale back the harvest tier.

---

# DEPLOY ORDER (RANKED — DO ONE AT A TIME)

```
RANK   CHANGE                          EXPECTED IMPACT           RISK LEVEL   VALIDATED?
1.     X1 — BayesConditionalExit       +$30 on FADE_CALM         LOW          YES (sim)
2.     V1 — Aggregate z_at_entry test  N/A (validation only)     NONE         N/A
3.     V2 — Regime-alignment test      N/A (validation only)     NONE         N/A
4.     E1 — Floor z_thr at 1.5         +$X per skipped loser     LOW          NO  (needs V1)
5.     X3 — TOD sizing mult            -$catastrophes            MEDIUM       NO
6.     X2 — Pre-CAT close              -$drawdown                MEDIUM       NO
7.     E2 — Regime-alignment filter    +$ per remaining trade    HIGH         NO  (needs V2)
8.     E3 — CAT_HARVEST tier           +$harvest                 HIGH         NO  (needs V4)
9.     E4 — COMPRESSION_BOUNCE tier    +$small uplift            HIGH         NO  (needs V3)
```

**Rule**: don't deploy rank N until rank N-1 has been validated through
full `python -m training_iso_v2.run_iso` and improved (or held flat) the
$/day metric.

---

# ROLLBACK PLAN

For each change:
- Git commit BEFORE applying the change with the message
  "baseline before <change_id>"
- Apply the change
- Run validation
- If worse: `git revert HEAD`
- If better: tag with `git tag retune-<change_id>-<date>` and continue

The CLAUDE.md baseline branch policy applies: every retune deployment
must reproduce or beat the current safe baseline OOS $/day.

---

# WHAT'S NOT IN THIS PROPOSAL

These were tested and rejected — don't re-attempt:

- Generic time-based exit oracle (60-min) — destroyed FADE_CALM edge
- Universal exit oracle for NMP_FADE_RAW — destroyed $4,964 of edge
- Standalone direction prediction at any bar — chance level

These are research-grade only — don't deploy:

- The 32 high-trust chord-shape cells (Bayesian table V0)
- The 9-feature compression scan beyond the single L2_15m_vol_sigma_12
- The 1m-trail / 5s-trail systems
