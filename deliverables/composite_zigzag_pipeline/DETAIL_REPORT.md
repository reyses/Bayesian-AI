# Composite Zigzag Pipeline — Detail Report for Outside Contributors

**Single-document brief.** Read this front to back if you want full
context before contributing. Estimated read time: 30-45 minutes.

Each section is self-contained but builds on the previous. Skim
section 1-3 for context, section 5 for work items, section 9 for
acceptance criteria.

---

## 1. Executive Summary

### What we have
A backtest-validated trading pipeline for MNQ (Micro Nasdaq) futures
that combines a **streaming zigzag indicator** (for entry/exit timing)
with a **stack of 7 GBM models** (for sizing and risk management).

**OOS performance (hardened, no peeks, 32 NT8 days, $6/leg friction):**
- Mean: **+$927/day** on 1 contract baseline
- 95% CI: [$534, $1,372]
- Pos days: 26/31 (84%)
- Days >$200: 24/31 (77%)
- Worst day: -$598
- Best day: +$4,575
- Median per-day: $627
- KDE mode: $468

### What we want help with
Three priorities (any one is valuable):

**1. Day-level regime classifier** — predict at session start whether
the day is going to be net positive. The 5/31 losing days in OOS are
the bottleneck on user goals; we can't filter them with current
features. Need cross-day signals (overnight gap, calendar events,
intermarket: ES/VIX/DXY).

**2. Hardened B7/B8 retraining** — current models trained on
pivot-bar features but applied at R-trigger fire bar (slight
distribution shift). Rebuilding IS datasets at R-trigger fire bar
removes the last clear peek. Expected impact: 10-30% reduction in
absolute OOS, but more deployment-ready.

**3. End-to-end streaming engine validation** — run the full pipeline
through the project's `training_iso_v2/` streaming ticker (not
included here, lives in parent project). Catches any remaining
bar-by-bar timing issues that the leg-level sim might miss.

### What we'd specifically NOT pursue
- Transformer/LSTM on same V2 features — earlier session evidence
  showed +1.7pp on direction, marginal on others. Same feature
  ceiling, much higher cost.
- RL agent on same features — same ceiling, ×10 cost, plus
  sample-inefficiency given 277 IS days.
- More GBM models on the same data — diminishing returns.

The pattern across all 7 models is **OOS Pearson ~0.22 on each
forward-looking target.** That's the structural information content
of V2 features. Adding architecture without adding features won't
move it.

---

## 2. Business Context

### Asset: MNQ futures
- Micro E-mini Nasdaq-100 futures
- $2/point multiplier (tick value $0.50, tick size 0.25)
- 24/5 trading, primarily during US RTH 9:30-16:00 ET
- Round-trip commission ~$1.20-$4.00 retail
- 5s and 1m OHLC available; we use 5s for execution decisions and
  1m for ATR / regime classification

### Strategy class: Zigzag swing trading
- Detect price swings using zigzag algorithm with reversal threshold
  R = 4 × ATR(14)
- On confirmed pivot (R-trigger fire), enter in new leg direction
- Exit when next R-trigger fires (= next pivot confirmed)
- This is a well-known indicator-based strategy; the V2+GBM stack
  is layered on top

### Why this approach exists
- Direction by zigzag alternation is ~97% correct by construction
- The leg's eventual amplitude is variable: median ~$95, but with
  heavy right tail (top deciles >$200)
- Position sizing by predicted amplitude captures the right-tail
  asymmetry — the actual edge source
- R-trigger entry/exit "tax" is ~2R per round-trip (~$80-100)
  giving ~50% of legs negative P&L — but the asymmetric winners
  cover this and more when sized correctly

### The user is solo / small operation
Not an institutional fund. Single trader looking to deploy on retail
account. Therefore:
- Friction matters (commission + slippage = $6/leg in our sim)
- Risk management matters (single bad day can wipe out a week)
- Babysitting is expected for first 1-2 months of live trading
- $200-$1000/day target range, scaled to position size

---

## 3. Architecture in Detail

### Data substrate

```
DATA/ATLAS_NT8/
├── 5s/{day}.parquet           ← raw 5s OHLC bars
├── 1m/{day}.parquet           ← raw 1m OHLC bars
└── FEATURES_5s_v2/            ← V2 184-feature set at 5s cadence
    ├── L1_{TF}/{day}.parquet  ← 8 TFs × 1 layer
    ├── L2_{TF}/{day}.parquet  ← 8 TFs × 1 layer
    └── L3_{TF}/{day}.parquet  ← 8 TFs × 1 layer (derived)
```

**V2 features** (184 total): 8 timeframes (5s, 15s, 1m, 5m, 15m, 1h,
4h, 1D) × 3 layers (L1=current-bar, L2=rolling-window, L3=derived
stats). Examples:
- `L1_5s_price_velocity_1b` — 1-bar price velocity at 5s
- `L2_15s_price_mean_12` — 12-bar mean of 15s closes (= 3-min mean)
- `L3_1m_z_se_15` — z-score of close vs 15-bar 1m mean

Features are computed in `core_v2/features.py` (lives in parent
project, NOT in this standalone — but you don't need it to retrain
the GBMs since labels + features are pre-computed in
`caches/zigzag_pivot_dataset_*.parquet`).

### Truth labels: zigzag pivot detection

`tools/build_zigzag_pivot_dataset.py` detects swing pivots:
- ATR(14) computed on 1m bars (median of true ranges)
- `min_reversal_ticks = max(4, round(ATR_pts / TICK_SIZE × 4.0))`
- Run `detect_swings()` on 5s closes (from `_viz/auto_swing_marker.py`)
- Each pivot becomes a row marker in the per-1m-bar dataset
- `leg_direction` (LONG/SHORT) propagated to every bar between pivots
- `trend_class` adds NEUTRAL zone within ±120s of any pivot

Output (one per 1m close):
```
timestamp, day, is_pivot, pivot_dir, pivot_price,
leg_direction, trend_class, atr_pts, min_rev_ticks,
target_split (IS/OOS), L1_*, L2_*, L3_* (184 V2 features)
```

### The 7-model stack

#### B1 — Pivot-imminent (binary × 4 K)
- **Task**: P(pivot within next K minutes), K ∈ {1, 3, 5, 10}
- **Label**: For each 1m bar t, label = 1 if any pivot in (t, t+K×60s]
- **Training**: HistGradientBoosting, class_weight='balanced'
- **OOS**: K=10 thr=0.85 → 78.1% precision at 4% coverage, AUC 0.716
- **Use**: pivot-imminent warning, "prep for reversal"
- **File**: `tools/train_b1_pivot_imminent.py`, `models/b1_pivot_imminent.pkl`

#### B2 — Fakeout (binary × 3 K)
- **Task**: P(this just-confirmed pivot is a fakeout)
- **Definition**: Fakeout = next pivot in opposite direction within K min
- **Sample**: One row per pivot event (not per bar)
- **OOS**: K=10 thr=0.70 → 66% precision at 19% coverage, AUC 0.695
- **Use**: post-pivot quality filter, suppress entries on flagged pivots
- **File**: `tools/train_b2_fakeout.py`, `models/b2_fakeout.pkl`

#### B4 — Pivot region (symmetric, binary × 4 W)
- **Task**: P(this bar is within ±W seconds of any pivot)
- **Symmetric**: captures pre-pivot AND post-pivot bars as positive
- **W**: {30, 60, 120, 300} seconds
- **OOS**: W=300s thr=0.85 → 79% precision at 10% coverage
- **Use**: wide pivot-region detector for composite zones
- **File**: `tools/train_b4_pivot_region.py`, `models/b4_pivot_region.pkl`

#### B5 — Leg phase (3-class)
- **Task**: EARLY / MID / LATE phase of current leg
- **Label**: leg_age_ratio = time_since_last_pivot / leg_duration
  - EARLY [0, 0.25), MID [0.25, 0.75), LATE [0.75, 1.0]
- **OOS**: accuracy 38.6%, below MID-baseline 46.4% (weak standalone)
- **BUT**: P(MID) > 0.60 → 71% precision at 0.25% coverage
- **Use**: diversity in composite, NOT standalone
- **File**: `tools/train_b5_leg_phase.py`, `models/b5_leg_phase.pkl`

#### B6 — Directional pivot (3-class × 4 K)
- **Task**: NO_PIVOT / PIVOT_TO_LONG / PIVOT_TO_SHORT in next K min
- **Definition**: PIVOT_TO_LONG = next pivot is a LOW (kicks off long leg)
- **OOS**: K=10 thr=0.70 → 53-57% precision per direction at 5% coverage
- **Use**: directional pre-position for upcoming flip
- **File**: `tools/train_b6_directional_pivot.py`, `models/b6_directional_pivot.pkl`

#### B7 — Leg amplitude regressor (continuous) ⭐ THE WORKHORSE
- **Task**: E[leg_amplitude / R] at entry time
- **Label**: leg_amp_pts / r_price (continuous, ranges 0.5 to 8+)
- **Median target**: 2.11 R on IS
- **OOS**: Pearson 0.22, MAE 1.03 (0.7% better than baseline)
- **BUT**: calibration is MONOTONIC — predicted-3.0+ → actual median 2.76
- **Use**: drives `gbm_ev` sizing: `size = max(pred_R - 1, 0)` clipped [0, 3]
- **Impact**: this single model adds +$452/day OOS (from $475 flat to $927)
- **File**: `tools/train_b7_leg_sizer.py`, `models/b7_leg_sizer.pkl`

#### B8 — Hour-level risk regressor (continuous)
- **Task**: E[forward 60-min total leg P&L in $]
- **Label**: sum of leg P&Ls (leg_amp - 2R - friction) for legs starting
  in [t, t+3600]
- **OOS**: Pearson 0.22, MAE $101
- **Use**: hour-level sizing modifier on top of B7 per-leg sizing
- **Impact**: linear hour gate adds +$580/day on top of B7 (from $927 to $1,505)
- **File**: `tools/train_b8_hour_risk.py`, `models/b8_hour_risk.pkl`

### The composite zone classifier

`tools/pivot_probability_cloud.py` reads B1+B4(+B5) predictions per
bar and assigns a discrete zone:

```python
if   B4.W=60s P >= 0.85:    zone = AT_PIVOT      # 0.14% bars
elif B4.W=120s P >= 0.70:   zone = NEAR_PIVOT    # 33% bars
elif B1.K=1 P >= 0.70:      zone = IMMINENT      # 1.9%
elif B1.K=3 P >= 0.70:      zone = NEAR_3m       # 1.9%
elif B1.K=5 P >= 0.70:      zone = NEAR_5m       # 5.4%
elif B4.W=300s P >= 0.85:   zone = WIDE_ZONE     # 0.9%
elif B1.K=10 or B4.W=300 >= 0.70: zone = WATCH   # 15.6%
else:                       zone = CLEAR         # 57.7%
```

Plus a trajectory state (RISING/FLAT/DECAYING) based on the slope of
expected-TTP over the trailing 10 bars.

**Used for**: interpretation, rule-based filtering, multi-axis state
representation. The hand-coded `aggressive` sizing scheme uses zone +
B6 directional. The GBM-based `gbm_ev` scheme bypasses zones and uses
B7 directly.

### How the models combine in the forward pass

```python
# tools/composite_forward_pass_hardened.py (the canonical OOS sim)
for each leg in OOS:
    # Honest entry detection (streaming)
    detect R-trigger fire on 5s closes after each pivot
    entry_ts = first 5s close at or past pivot ± R
    entry_price = 5s close at entry_ts

    # Look up GBM predictions at the 1m close at-or-before entry_ts
    pred_amp_R = B7.predict(V2_features[bar_at_entry])
    pred_hour_pnl = B8.predict(V2_features[bar_at_entry])
    zone = pivot_probability_cloud.zone[bar_at_entry]
    b6_match = B6.proba_LONG if leg_dir == LONG else proba_SHORT

    # Sizing
    leg_size = max(pred_amp_R - 1, 0) clipped to [0, 3]      # B7 / gbm_ev
    hour_mult = piecewise_linear(pred_hour_pnl, ...)         # B8 (optional)
    total_size = leg_size × hour_mult

    # Honest exit
    detect next R-trigger fire after entry_ts
    exit_price = 5s close at next R-trigger fire

    # P&L
    pnl_pts = (exit_price - entry_price) × leg_dir
    net_pnl = pnl_pts × $2/point - $6 friction
    weighted_pnl = net_pnl × total_size
```

This is the canonical evaluation loop. It's the source of all
hardened OOS numbers.

---

## 4. Current State & Known Issues

### Validated results (high confidence)

| Metric | Value | Source |
|--------|-------|--------|
| Hardened OOS mean (gbm_ev) | $927/day | `reports/forward_pass_outputs/composite_forward_pass_hardened.txt` |
| 95% CI | [$534, $1,372] | paired bootstrap on per-day P&L |
| Direction accuracy (oracle) | ~97% | zigzag construction |
| B7 monotonic calibration | confirmed | `reports/training_outputs/b7_leg_sizer.txt` |
| Sizing edge vs flat | +$452/day | paired bootstrap, CI [+$278, +$654] strict positive |

### Known peeks (in order of severity)

1. **B7/B8 train at pivot bar, infer at R-trigger bar.** Distribution
   shift, not mathematical leakage. Expected impact: 10-30% reduction.
   Fix: retrain on R-trigger-bar features.

2. **B8 labels use formula `leg_amp - 2R - friction`** (theoretical
   R-trigger P&L). In real R-trigger detection, actual P&L differs
   slightly due to 5s bar slippage. Small impact.

3. **R-trigger detection on 5s closes** is optimistic vs intra-bar
   detection that NT8 would do live. ±$1-2/leg in either direction.

4. **Friction estimate ($6/leg)** may be too generous or too strict
   depending on broker. Real range $2-$10/leg. Sensitivity: ±$340/day.

### Untested

1. **Live deployment data** — entire backtest only, no live ground truth.
2. **Day-level regime gating** — bad days remain unfiltered, ~5/31 are
   net negative.
3. **Streaming engine** — leg-level sim only; bar-by-bar engine in
   `training_iso_v2/` would catch any temporal issues.
4. **Partial exits** — full position entry/exit, no scaling.
5. **Cross-day autocorrelation** — per-day P&L is correlated (regime
   persistence). Bootstrap CIs mildly optimistic.

### Failed experiments (DON'T repeat)

| Attempted | Result | Reason |
|-----------|--------|--------|
| Trail tightening | -$44/leg | NEAR_PIVOT zone fires on pullback noise |
| B6-gated trail (18 cfgs) | Best -$0.29/leg | Strict gates fire too rarely |
| Target placement (12 cfgs) | All CIs negative | Target before peak = early; above = never hits |
| Binary entry-skip filters | All hurt per-day | Skipping positive-EV legs costs more than quality lift |
| Hardened exits sweep (480 cfgs) | None reach 32/32 >$200 | Bad days indistinguishable at decision time |
| Daily loss circuit breakers | Net negative | Stops cut winners too |

---

## 5. Specific Work Items (Pick Any)

In rough order of expected impact.

### W1. Day-level regime classifier ⭐ HIGHEST IMPACT
**Goal**: skip predicted-bad days entirely.

**Features needed** (collect externally):
- Overnight gap: `(today_open - prior_day_close) / atr`
- Pre-session VIX, ES, DXY levels and 30-min changes
- Calendar events: FOMC/CPI/NFP/earnings scheduled for the day
- Prior-day's regime label (we have a generator: `tools/atlas_regime_labeler_2d.py`)
- Day-of-week, week-of-month
- Prior N days' actual P&L (autocorrelation feature)

**Target**: `day_pnl < 0` (binary) or `day_pnl` (continuous)

**Training**: IS days for which we have all features. ~277 days
minus days where features missing.

**Acceptance**: precision >70% on "bad day" predictions at recall >40%.
Should improve worst-day filter from "5 of 31 lose" to "≤1 of 31 lose"
in held-out OOS.

**Build effort**: 2-5 days, mostly data acquisition.

**Why it's the biggest win**: bad-day variance is the user's main
complaint. Currently $-598 worst day. Filtering bad days would push
worst-day toward $0 or small loss and dramatically tighten the
distribution.

### W2. B7/B8 retrain on R-trigger-bar features
**Goal**: remove last distribution-shift peek.

**Procedure**:
1. For each IS day, detect R-trigger fires on 5s closes (same as
   `composite_forward_pass_hardened.py` does for OOS).
2. At each R-trigger fire timestamp, look up V2 features at the
   1m close at-or-before that timestamp.
3. Compute the leg's actual amplitude (peak/trough until next R-trigger)
   as label.
4. Train B7 on (R-trigger-bar features → leg amplitude).
5. Same for B8: aggregate forward 60-min P&L using R-trigger-detected
   legs.

**Acceptance**: hardened OOS mean drops by 10-30% but remains strictly
positive in 95% CI. Pipeline becomes fully honest.

**Build effort**: 1-2 days.

### W3. Streaming engine end-to-end run
**Goal**: validate through the project's streaming ticker.

**Path**: `training_iso_v2/ticker.py` provides a `MultiDayV2Ticker` that
yields BarState per 5s bar. Build a `Strategy` subclass (interface in
`training_iso_v2/strategies/base.py`) that:
- Tracks current leg direction + running extreme
- On R-trigger detection, fires EntrySignal
- Engine handles position + exits (via existing exit suite)
- Externally apply B7 sizing per trade

This requires interfacing with the parent project's `training_iso_v2/`
module (NOT included in this standalone). The current standalone
provides only the data flow + the B7/B8 sizing logic in pure-Python
leg-level sims.

**Acceptance**: P&L from streaming engine matches leg-level sim within
±$50/day after accounting for bar-by-bar timing differences. Find any
divergence and explain.

**Build effort**: 3-7 days.

### W4. Partial exits / scaled positions
**Goal**: take some profit at +1R, ride rest to R-trigger.

**Approach**: at each leg, after price reaches +1R favorable, close
50% of position at limit; let remaining 50% ride to R-trigger.

**Why it might help**: locks in some certain gains; remaining
position has higher Sharpe (no downside on the closed portion). May
flatten the per-day distribution.

**Acceptance**: per-day distribution narrower (smaller std) with same
or higher mean.

**Build effort**: 1 day to simulate, 2-3 days to validate variants.

### W5. Live deployment babysit + data logging
**Goal**: collect actual live distribution.

**Recipe**:
- Week 1-2: 0.25× full sizing
- Week 3-4: 0.50× if normal
- Month 2: 0.75×
- Month 3+: 1.0× full

**Log per leg**:
- entry_ts, entry_price, leg_dir, direction signal at entry
- B7 prediction, B8 prediction, zone, B6 confidence
- Exit reason, exit price, exit ts
- Manual interventions (pause/resume/size override)
- Subjective day quality notes

After 30 live days: compare live distribution to NT8 OOS. Retrain
models on combined data if distribution shift detected.

**Build effort**: deployment infrastructure first; data collection
is ongoing.

### W6. Multi-output GBM (combine B1+B2+B6 into one)
**Goal**: smaller, faster, more stable prediction pipeline.

A single multi-output GBM predicting all the pivot-related tasks
(timing, direction, fakeout) may have shared representations.

**Acceptance**: same or better per-task OOS metrics with single model.
**Likely impact**: marginal. The Pearson 0.22 ceiling won't move.

---

## 6. Codebase Tour

### Tools (35 .py files in `tools/`)

```
tools/
├── build_zigzag_pivot_dataset.py    ← truth label generator
├── train_b1_pivot_imminent.py       ← train B1
├── train_b2_fakeout.py              ← train B2
├── train_b4_pivot_region.py         ← train B4
├── train_b5_leg_phase.py            ← train B5
├── train_b6_directional_pivot.py    ← train B6
├── train_b7_leg_sizer.py            ← train B7 (the workhorse)
├── train_b8_hour_risk.py            ← train B8
├── precompute_b1_b2_oos.py          ← generate per-bar prediction caches
├── live_zigzag_baseline.py          ← causal indicator baseline
├── pivot_probability_cloud.py       ← composite zone classifier
│
├── composite_entry_analyzer.py      ← entry-time features → leg quality
├── composite_entry_accuracy.py      ← pure-entry trajectory analysis
├── composite_entry_filter_sweep.py  ← entry-skip filter sweep
├── composite_sizing_simulator.py    ← hand-coded sizing schemes
├── composite_gbm_sizing_simulator.py ← B7-driven sizing sweep
│
├── composite_trail_simulator.py     ← FAILED (naive trail tightening)
├── composite_trail_simulator_v2.py  ← FAILED (B6 + hysteresis)
├── composite_target_simulator.py    ← FAILED (target placement)
│
├── composite_forward_pass.py            ← peeky (theoretical 2R cost)
├── composite_forward_pass_hardened.py   ← HONEST forward pass ⭐
├── composite_forward_pass_hardened_exits.py ← exit-hardening sweep (FAILED)
├── composite_forward_pass_hour_gated.py ← B8 hour gate sweep
├── composite_forward_pass_mode_analysis.py ← mode/median/mean
├── composite_forward_pass_pareto_mode.py   ← GPD + KDE mode fit
│
├── b1_trajectory_bridge.py          ← B1 temporal decay analysis
├── b1_fire_pivot_proximity.py       ← where do B1 fires land?
├── b1_b2_per_day_ci.py              ← per-day CI on B1/B2 metrics
├── b1_b2_feature_audit.py           ← feature importance audit
├── b6_visual_diagnostic.py          ← chart B6 directional predictions
│
├── direction_signal_accuracy.py     ← direction-signal validation
├── imminent_exit_advantage.py       ← failed exit-at-IMMINENT test
├── blended_signal_forward_pass.py   ← combined-signal sim
└── _viz/
    └── auto_swing_marker.py         ← zigzag swing detector (dependency)
```

### Models (`models/`)

```
models/
├── b1_pivot_imminent.pkl       ← dict[K -> {model, v2_cols, ...}]
├── b2_fakeout.pkl              ← dict[K -> {model, v2_cols, ...}]
├── b4_pivot_region.pkl         ← dict[W -> {model, v2_cols, ...}]
├── b5_leg_phase.pkl            ← {model, classes, v2_cols, label_encoder}
├── b6_directional_pivot.pkl    ← {models: dict[K -> bundle], label_encoder}
├── b7_leg_sizer.pkl            ← {model, v2_cols, target_mean_R, ...}
└── b8_hour_risk.pkl            ← {model, feat_cols, is_median, is_mean}
```

Load with:
```python
import pickle
with open('models/b7_leg_sizer.pkl', 'rb') as f:
    bundle = pickle.load(f)
model = bundle['model']
v2_cols = bundle['v2_cols']
```

### Caches (`caches/`)

Per-bar prediction parquets for OOS:
- `b1_proba_OOS_NT8.parquet` — per-bar `p_pivot_{1,3,5,10}m`
- `b2_proba_OOS_NT8.parquet` — per-pivot `p_fakeout_{3,5,10}m`
- `b4_proba_OOS_NT8.parquet` — per-bar `p_region_{30,60,120,300}s`
- `b5_leg_phase_OOS_NT8.parquet` — per-bar phase probabilities
- `b6_proba_OOS_NT8.parquet` — per-bar `p_PIVOT_TO_{LONG,SHORT}_{K}m`
- `b7_leg_sizer_OOS.parquet` — per-leg predicted amp_R + truth
- `b8_hour_risk_OOS.parquet` — per-bar predicted forward 60-min P&L
- `pivot_probability_cloud.parquet` — per-bar composite zone + state

Plus the truth datasets:
- `zigzag_pivot_dataset_IS_atr4.parquet` — 277 IS days
- `zigzag_pivot_dataset_NT8_OOS_atr4.parquet` — 32 OOS days

Plus leg-level output of forward pass:
- `composite_entry_analyzer.csv` — per-leg entry features
- `composite_entry_accuracy.csv` — per-leg trajectories
- `composite_forward_pass_hardened.csv` — per-leg R-trigger P&L

---

## 7. Reproducibility / Quickstart

### Verify the headline number
```bash
cd composite_zigzag_pipeline
pip install -r requirements.txt
python tools/composite_forward_pass_hardened.py
```

Should print:
- `Days: 31   Legs: 1,826`
- `Mean per-leg P&L (NET):   $+8.07`
- For each scheme, the breakdown
- Total $/day per scheme

Compare against `reports/forward_pass_outputs/composite_forward_pass_hardened.txt`.

### Reproduce a specific finding
The flagship finding is the GBM sizing improvement. Run:
```bash
python tools/composite_gbm_sizing_simulator.py
```
Expected: gbm_ev beats flat by ~+$1,155/day on oracle entries
(paired delta CI [+$851, +$1,474]).

### Retrain a B-model
```bash
python tools/train_b7_leg_sizer.py
```
Takes 1-2 min. Updates `caches/b7_leg_sizer_OOS.parquet`.

To retrain ALL:
```bash
python tools/train_b1_pivot_imminent.py
python tools/train_b2_fakeout.py
python tools/train_b4_pivot_region.py
python tools/train_b5_leg_phase.py
python tools/train_b6_directional_pivot.py
python tools/train_b7_leg_sizer.py
python tools/train_b8_hour_risk.py
```

Each is 1-5 min. Total ~15-20 min for the full stack on CPU.

### Build new truth labels (requires DATA/ATLAS_NT8/)
```bash
python tools/build_zigzag_pivot_dataset.py \
    --root DATA/ATLAS_NT8 --target oos --atr-mult 4.0 \
    --out caches/zigzag_pivot_dataset_NT8_OOS_atr4.parquet
```

---

## 8. Performance Targets / Acceptance Criteria

### For W1 (regime classifier)
- ≥70% precision on "bad day" predictions at ≥40% recall
- OOS worst-day improves from -$598 to ≤-$100
- Days >$200 improves from 24/31 to ≥28/31

### For W2 (R-trigger retraining)
- Hardened OOS mean within 10-30% of current ($650-$835/day)
- CI strictly positive
- No more pivot-bar feature peek

### For W3 (streaming engine)
- P&L matches leg-level sim within ±$50/day
- Any divergence identified and explained
- Code merged into `training_iso_v2/` strategies directory

### For W4 (partial exits)
- Per-day std reduced ≥10% without mean reduction
- Or per-day mean improved ≥10% with same std
- Trade-off explicit in the report

### General code-style requirements (per parent project's CLAUDE.md)

- All P&L claims must include 95% CI + significance statement
- Trade WR = profit-factor-based, NOT count-based
  - PF Trade WR = (∑ profit_winners / |∑ loss_losers|) - 1
- $/trade and $/day reported as MODE + MEAN with bootstrap CI
- Day WR = winning_days / active_days (count-based)
- No `# TODO` or planning docs unless explicitly requested
- Magic numbers must be named constants

---

## 9. Communication / Deliverable Format

### Code deliverables
- Python ≥3.10, sklearn ≥1.3 (see requirements.txt)
- Pure-Python preferred; numba/cuda OK if necessary and documented
- Reusable scripts under `tools/`; each must be runnable standalone
- Each new tool MUST write its results to a file in
  `reports/findings/` (don't print to stdout only)
- Add new tools to a TOOLS_INDEX.md or this DETAIL_REPORT
- Include unit tests for any numerical logic (zigzag detection,
  R-trigger detection, sizing math)

### Findings deliverables
- One markdown report per finding, dated `YYYY-MM-DD_topic.md`
- Always include:
  - Headline numbers (mean + CI)
  - Comparison to baseline
  - Failed alternative attempts
  - Honest caveats
- Plus the underlying data (CSV/parquet)

### What to flag as risky
- Any change that alters R-trigger detection logic
- Any change that bypasses the friction model
- Any new "peek" feature (always document)
- Anything that changes the truth label generator
  (`build_zigzag_pivot_dataset.py`)

These touch the validation reliability; changes need a paired-day
comparison to demonstrate they don't inflate results.

---

## 10. Anti-Patterns / DON'T

1. **Don't train on OOS data.** IS = 2025 days, OOS = 2026 NT8 days
   (post-March-20). Keep them strictly separated.
2. **Don't quote any P&L without CI.** Per CLAUDE.md, every claim
   needs significance.
3. **Don't bypass R-trigger entries for "convenience."** Oracle pivot
   entries inflate everything by 2-3x.
4. **Don't assume the model is right when it disagrees with the
   indicator.** Indicator + sizing is the right mental model;
   model alone is brittle.
5. **Don't add architectural complexity to fix a feature problem.**
   If the Pearson 0.22 ceiling isn't budging with simpler models,
   bigger ones won't break it either. Find new features instead.
6. **Don't skip bad-day analysis.** The 5 OOS losing days are the
   most important diagnostic data we have. Always inspect them
   per-leg when proposing changes.
7. **Don't commit to a regime classifier that uses lookahead.** The
   project's existing `regime_labels_2d.csv` is end-of-day computed;
   any classifier using it as a feature has lookahead.

---

## 11. Contact & Handoff

The author of this pipeline can be reached through the project
repository. Pull requests welcome on:
- New B-models with documented OOS validation
- Failed experiments (we want to know what doesn't work too)
- Bug fixes in existing tools
- Documentation improvements

For substantial work items (W1-W5), please:
1. Open an issue describing the approach
2. Propose acceptance criteria in advance
3. Run the baseline (current `composite_forward_pass_hardened.py`)
   before changes
4. Submit results with paired-day comparison

The goal is to push from $927/day current OOS to a robust live
deployment. Every contribution that brings us closer or eliminates
a risk is valuable.

---

End of report. See `RESULTS.md` for the fast-reference numbers.
See `HONEST_CAVEATS.md` for what we know is broken.
See `SESSION_LOG.md` for chronological context of how decisions were made.
