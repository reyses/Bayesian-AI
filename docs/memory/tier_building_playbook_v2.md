---
name: V2-native tier development playbook
description: Step-by-step methodology for deriving tiers in the V2-native training_v2 pipeline using multi-axis regret analysis. Companion to tier_building_playbook.md (V1-shape iso pipeline).
type: project
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
# V2-Native Tier Development Playbook

**Established 2026-05-05 from the regret-based discovery cycle on 19,106 NMP IS trades.**

This playbook is the working methodology for deriving, validating, and
rejecting candidate tiers in the V2-native `training_v2/` pipeline.
Supersedes `feedback_tier_three_questions.md` for V2 work; companion to
`tier_building_playbook.md` (which covers the V1-shape iso pipeline).

The core insight: **multi-axis regret analysis is the LABEL GENERATOR
that drives tier discovery.** Without it you're guessing; with it, every
sub-tier is a falsifiable hypothesis with a measured baseline.

---

## 0. Pre-requisites

Before running any tier work, verify:

- **V2 features built**: `DATA/ATLAS/FEATURES_5s_v2/{L0, L1_<TF>, L2_<TF>, L3_<TF>}/YYYY_MM_DD.parquet` for all IS+OOS days
- **2D regime labels**: `DATA/ATLAS/regime_labels_2d.csv` from `tools/atlas_regime_labeler_2d.py`
- **No lookahead in V2 SFE**: `core_v2/statistical_field_engine.py` has the lookahead audit baked in
- **Engine sanity**: `python -m training_v2.run --smoke` (single-day no-thresholds, no-CNN)

---

## 1. The 6-Step Discovery Cycle

### Step 1 — Run the seed entry alone

Base NMP / REVERSION trigger only. No CNN, no thresholds, no other strategies.

```
python -m training_v2.run --is  --strategies REVERSION
python -m training_v2.run --oos --strategies REVERSION
```

This gives the **honest baseline** the rest of the cycle is measured against.
Save the trade pickles — they're the input to every subsequent step.

**Sanity check**: count WR should be near 50-55%, $/trade near $0. If WR is
much higher, the trigger is too restrictive (not enough trades to discover
sub-tiers from). If WR is much lower, your trigger is broken.

### Step 2 — Run multi-axis regret on the trades

```
python -m training_v2.regret_full --trades training_v2/output/nmp_only.pkl \
    --out training_v2/output/regret_full_nmp.pkl
```

`regret_full.py` produces `FullRegretLabel` per trade with:

| label | meaning |
|---|---|
| `same_early_best` | best PnL strictly before actual exit |
| `same_at_exit` | actual exit (baseline) |
| `same_extended_best` | best PnL after the actual exit |
| `counter_early_best` | best counter PnL in bars 0-3 (flip-and-go) |
| `counter_at_exit` | counter PnL at the actual exit bar |
| `counter_extended_best` | best counter PnL after the actual exit |
| `best_action` | argmax over the 6 options above |
| `regret` | best_pnl − actual_pnl |
| `early_entry_gain` | extra $ if we'd entered at the best earlier bar |

These labels are what tells you *what the trade SHOULD have been*. The
distribution of `best_action` is the first signal: if 50% of trades are
`counter_early` or `counter_extended`, you have a direction-flip splitter.
If 40% are `same_extended`, your exit is firing too early.

### Step 3 — Identify splitter axes

Three discovery angles, in order of OOS-survival likelihood:

#### 3a. Categorical splitters (highest survival rate)

Cross-tab pivot the trades by **categorical** entry features × the
counterfactual best-direction:

```python
df.groupby(['regime_idx', 'direction']).agg(
    n=('actual_pnl', 'size'),
    actual=('actual_pnl', 'mean'),
    fade_peak=('peak_pnl', 'mean'),
    flip_peak=lambda l: (-l).mean()  # = -mae_pnl
)
```

Look for cells where `flip_peak >> fade_peak` (or vice versa) — those are
flip-rule candidates. **The 2026-05-04 V2 discovery: 3 of 12
(regime × direction) cells qualified for flip:**
- `(UP_SMOOTH, short)` → flip to long
- `(UP_CHOPPY, short)` → flip to long
- `(DOWN_SMOOTH, long)` → flip to short

Categorical splitters survive walk-forward + true OOS far more reliably
than continuous-feature filters because the cells have hundreds-thousands
of trades each.

#### 3b. Continuous-feature filters (medium survival rate, requires care)

For each (regime, direction) cell, run within-cell winner-vs-loser EDA
on all 185 V2 columns (`training_v2/within_cell_eda.py`). Look for
features with Cohen's d ≥ 0.2 that survive 70/30 walk-forward.

**Confirmed pattern (2026-05-05): 9 of 12 cells had walk-forward-surviving
top features inside IS, but they FAILED on true OOS hold-out.** The
`FilteredRegimeAwareReversion` strategy lost -$19.85/day OOS. Continuous
quantile thresholds remain a known overfit trap (see
`feedback_quantile_selection_overfit.md`, `feedback_high_vol_harness_failed.md`).

Use continuous filters only as second-pass refinements, never as primary
splitters, and demand both walk-forward IS *and* date-disjoint OOS lift
before shipping.

#### 3c. Time / day-of-week splitters

Group by hour-of-day and weekday. Bleed concentration in specific hours
indicates a **structural condition** (volatility expansion at NY open, etc.)
that should be diagnosed BEFORE acting on the time bucket.

**Important: DO NOT filter on time directly.** Time is correlated with
volatility — find the underlying volatility/structural feature and condition
on that. The 2026-05-05 NY-mid-session bleed turned out to be entry vol
expansion, not "time of day".

### Step 4 — Build the candidate as a strategy variant

For each splitter discovered, write a thin strategy class that wraps
the seed and applies the rule. Examples in `training_v2/strategies/`:

- `regime_aware.py` — flip rule (categorical splitter)
- `filtered_nmp.py` — per-cell quality filter (continuous splitter — REJECTED)

Each variant gets its own name (e.g., `NMP_REGIME`, `NMP_FILTERED`) so
the trade pickle's `entry_tier` field carries the experiment identity for
downstream analysis.

### Step 5 — Validate at THREE levels

A candidate must clear all three to ship:

#### 5a. IS apples-to-apples re-simulation

Use `simulate_exit` from `training_v2/regret.py` to apply the same threshold
policy to baseline vs candidate. This isolates the *signal* contribution
from threshold-tuning effects.

#### 5b. IS walk-forward

Train on first 70% of IS days, validate on last 30%. Compute bootstrap CI
on per-day delta. Required: CI lower bound > 0.

**Caution: walk-forward inside IS is NECESSARY but NOT SUFFICIENT.** Many
overfit rules survive 70/30 IS but break on date-disjoint OOS. Continuous
filters in particular often pass IS-WF and fail OOS.

#### 5c. True OOS engine run

Run the actual engine on the OOS days. Compute bootstrap CI on
(candidate $/day − baseline $/day). Required: CI not catastrophically
negative; ideally CI > 0 (significant) but at minimum CI lower bound
not far below 0.

```
python -m training_v2.run --oos --strategies NMP_VARIANT \
    --thresholds training_v2/output/thresholds_prod.json
```

### Step 6 — Ship, hold, or reject

| IS-WF CI | OOS CI | verdict |
|---|---|---|
| significant > 0 | significant > 0 | **SHIP** as production |
| significant > 0 | positive but CI includes 0 | hold; collect more OOS data |
| significant > 0 | clearly negative | **REJECT — overfit** |
| not significant | any | **REJECT — no signal** |

Ship adds a tier to production; reject removes the experimental code.
**"Hold" should never be production**; if a tier needs more data, run the
engine more before shipping.

---

## 2. The Bleed-Cause Causal Pattern

When a tier underperforms in a specific zone (hour bucket, regime, etc.),
the analysis MUST distinguish symptom from cause. The 2026-05-05 NY-bleed
investigation showed:

```
Symptom: hours 14-17 UTC lose money
Apparent cause: "NY mid-session is bad"
Actual cause: 4-8x volume/range expansion at entry, fades fail when
              extremes extend further
```

Workflow:

1. Identify the bleed cluster (autopsy by hour/regime/tier)
2. Compare V2 feature distributions between bleed-zone and profit-zone
   trades (`training_v2/bleed_cause_analysis.py`)
3. Look for STRUCTURAL feature differences with large Cohen's d
4. **The structural feature, not the cluster label, is the lever**

For NY mid-session: filter on `L2_1m_vol_mean_15` or `L1_5m_bar_range`,
NOT on `hour_utc`. The vol features generalize to other volatility-spike
periods (Fed days, news, etc.) that hour buckets miss.

---

## 3. Anti-Patterns (Confirmed Failures)

### 3.1 Re-simulation overestimates engine impact

`simulate_exit` walks the regret pnl_path and ignores state-driven exits
(`ZSeReversal`, `SwingNoiseSpike`). The 2026-05-04 flip-rule estimate
was +$68/day OOS via re-sim; actual engine produced +$1.66/day. **A 40×
discrepancy.**

**Always validate by running the engine, not just by re-simulating.**

### 3.2 Walk-forward IS is not OOS

Continuous-feature thresholds that pass 70/30 IS-WF (9 of 12 cells in
the 2026-05-05 within-cell EDA) failed on true 2026 OOS hold-out
(-$19.85/day). The 70/30 IS-WF data has temporal structure that lets
overfit rules sneak through.

**Demand date-disjoint OOS, not just walk-forward inside IS.**

### 3.3 Mean-based thresholds overshoot fat-tail distributions

Vol-adaptive exits (2026-05-05) used Bayesian-derived TPs based on the
*mean* peak per vol bin. Q5 high-vol mean peak was $144 — but the typical
trade peak was much lower (fat-tailed). TP set at $51 missed most trades.

**For fat-tail distributions, use lower quantiles (q_05 to q_15) for TP
or stick to median-based formulas.**

### 3.4 Direction flip in volatility-expansion zones

Hypothesis: "if fades fail in high vol, flip to ride direction". The data
rejected this — peaks are symmetric across directions in high-vol bins
(d ≈ 0.05-0.10 between fade_peak and flip_peak). High-vol just means
**big peaks for both directions** — not a directional bias.

**A direction-flip rule needs the regime/state to favor one side
asymmetrically. Symmetric volatility expansion is not such a state.**

### 3.5 Tier-aware exit rules are non-negotiable for ride trades

`ZSeReversal` (mean-reversion exit) fired bar 1 on every flipped trade,
killing the flip rule entirely until fixed (`ZSeReversal.RIDE_TIERS`
guard, 2026-05-04). Any new fade/ride distinction MUST audit the existing
exit rules for direction-asymmetric assumptions.

---

## 4. Reference: V2-Native Discovery Cycle Artifacts

| Tool | Output | Purpose |
|---|---|---|
| `training_v2/run.py --is/--oos` | `is.pkl`, `oos.pkl` | Raw trade lists |
| `training_v2/regret_full.py` | `FullRegretLabel` pickles | Multi-axis regret labels |
| `training_v2/regret.py` | `RegretLabel` pickles | Simple peak/MAE labels (legacy compat) |
| `training_v2/tier_discovery.py` | flip/winner/peak EDA | Categorical splitter discovery |
| `training_v2/within_cell_eda.py` | per-cell feature ranking | Continuous filter discovery |
| `training_v2/full_feature_eda.py` | global feature ranking + Spearman | First-pass scan |
| `training_v2/cell_filters.py` | filter JSON | Filter learner (REJECTED 2026-05-05) |
| `training_v2/flip_rule_validation.py` | walk-forward + OOS CI | Categorical splitter validator |
| `training_v2/loser_autopsy.py` | per-zone bleed report | Loser-pattern analysis |
| `training_v2/bleed_cause_analysis.py` | Cohen's d zone-vs-zone | Causal-mechanism identifier |
| `training_v2/threshold_bayesian.py` | thresholds JSON | Per-cell exit derivation |
| `training_v2/strategies/regime_aware.py` | NMP_REGIME class | Production flip rule |

---

## 5. Tier Lineage & Status (2026-05-05)

| tier | source | status | notes |
|---|---|---|---|
| REVERSION | V2 NMP seed | live | base entry condition, all sub-tiers descend from here |
| MA_ALIGN | EDA (vwap alignment) | live | 7-of-8 vwap_w alignment |
| VEL_BODY_CHORD | EDA (chord) | **killed** | 2026-05-04: lottery-day artifact |
| **NMP_REGIME** | regret + flip rule | **production** | RegimeAwareReversion |
| NMP_FILTERED | within-cell filter | **rejected** | 2026-05-05: OOS overfit |
| (vol-adaptive exits) | exit-side variant | **rejected** | 2026-05-05: fat-tail overshoot |

The discovery cycle on REVERSION found one categorical sub-tier (NMP_REGIME)
in 1 day. Continuous filters on top failed; exit-side variants failed.
**Next discovery candidates** worth running the cycle on:
- 4h-context entry (RIDE if 4h vel aligned with z direction)
- Multi-TF velocity exhaustion (legacy MTF_EXHAUSTION analog)
- Wick-rejection quality filter (legacy KILL_SHOT analog) — needs OHLCV math, not in V2 entry vector

Keep the 6-step cycle disciplined: each candidate is a 4-6 hour effort,
most reject, but the ones that survive are real.
