# Architecture — Composite Zigzag Pipeline

## The 7-model stack

Each model is a HistGradientBoosting GBM (sklearn) trained on the 184
V2 feature set computed at 5s/15s/1m/5m/15m/1h/4h/1D timeframes. Each
predicts a different aspect of the next leg / next hour / next pivot.

The truth labels come from **zigzag swing detection at ATR(14)×4** on 5s
closes (see `tools/build_zigzag_pivot_dataset.py`). A "leg" is the
trajectory between two consecutive zigzag pivots.

| Model | Target | Type | OOS metric | Use in composite |
|-------|--------|------|------------|------------------|
| B1 | P(pivot within K minutes) | Binary × 4 K (1/3/5/10) | K=10 thr=0.85 → 78.1% prec at 4% cov, AUC 0.716 | Pivot-imminent warning |
| B2 | P(fakeout) per pivot | Binary × 3 K (3/5/10) | K=10 thr=0.70 → 66% prec at 19% cov, AUC 0.695 | Pivot-quality filter |
| B4 | P(in ±W window of any pivot) | Binary × 4 W (30/60/120/300s) | W=300 thr=0.85 → 79% prec at 10% cov, AUC 0.738 | Wide pivot-region detector |
| B5 | P(EARLY/MID/LATE leg phase) | 3-class | OOS acc 38.6%; P(MID)>0.60 → 71% prec | Phase-aware sizing |
| B6 | P(NO/LONG-pivot/SHORT-pivot next K) | 3-class × 4 K | K=10 thr=0.70 → 53-57% prec at 5% cov | Direction-aware sizing |
| B7 | E[leg amplitude / R] | Regression | OOS Pearson 0.22, MAE 1.03 | **Drives per-leg sizing** |
| B8 | E[forward 60-min total P&L] | Regression | OOS Pearson 0.22, MAE $101 | **Hour-level risk gate** |

**Critical pattern**: every model's Pearson correlation is ~0.22 on its target.
This is the structural information content of V2 features for
forward-looking targets. **Adding model capacity (LSTM, transformer)
doesn't break this ceiling — features do.**

## Why we built each one

### B1 — Pivot-imminent classifier
> "When will the next pivot happen?"

For each 1m bar, predicts whether a zigzag pivot occurs within the next
K minutes. Multiple K values trained simultaneously. Used as an early
warning that the current leg is ending.

Operating point: K=10 thr=0.85 → 78% precision, 4% coverage. ~42
warnings/day where the model is 78% confident a pivot is coming within
10 minutes.

**Why it works (or doesn't)**: V2 features have moderate signal for
pivot imminence (AUC 0.69-0.72). High thresholds isolate the sparse
high-signal bars. The trajectory of B1 across bars (rising/decaying)
adds another dimension — see B1_trajectory_bridge.

### B2 — Fakeout classifier
> "Will the JUST-CONFIRMED pivot turn out to be a fakeout?"

For each pivot event, predicts if the new leg starting at this pivot
will die within K minutes (i.e., we'll see a reverse pivot quickly).
Defines "fakeout" as `time_to_next_pivot ≤ K`.

Operating point: K=10 thr=0.70 → 66% precision, 19% coverage. ~11
fakeout warnings/day at 66% reliability.

**Why we built it**: a fresh leg that's about to reverse is a costly
trap. Filter or de-size on flagged pivots.

### B4 — Pivot-region (symmetric)
> "Am I currently inside a ±W window of any pivot?"

Symmetric version of B1. Counts both pre-pivot AND post-pivot bars as
positive. Captures the structural neighborhood of pivots.

Operating point: W=300s thr=0.85 → 79% precision, 10% coverage. Beats
B1 K=10 thr=0.85 at the same precision but 2.5× more coverage.

**Why it's better than B1 for some purposes**: the post-pivot signal
(volume residuals, mean-reversion artifacts) is information B1 misses.
B4 captures the full "pivot neighborhood" not just the lead-in.

### B5 — Leg-phase 3-class
> "Where in the current leg are we — early, middle, or late?"

For each 1m bar inside a leg, predicts the phase based on
`leg_age_ratio = time_since_last_pivot / total_leg_duration`:
- EARLY: ratio in [0, 0.25)
- MID: ratio in [0.25, 0.75) — deepest "in trend"
- LATE: ratio in [0.75, 1.0]

Standalone is weak (OOS accuracy 38.6%, below MID-baseline 46.4%). But
P(MID) > 0.60 → 71% precision at 0.25% coverage. **The diversity vs B4
is what matters for the composite.**

### B6 — Directional pivot classifier
> "If a pivot happens next, which direction will it flip to?"

3-class per K: NO_PIVOT / PIVOT_TO_LONG (low pivot incoming) /
PIVOT_TO_SHORT (high pivot incoming).

Operating point: K=10 thr=0.70 → 53-57% precision per direction at
5% coverage. **First model that gives directional information**, not
just timing.

**Why critical**: legs alternate by construction (zigzag), so direction
of next leg is the opposite of current. But knowing the direction the
PIVOT will form lets us pre-place the right exit/reverse order.

### B7 — Leg amplitude regressor
> "How big will this leg be (in R units)?"

For each pivot/entry, predicts `leg_amplitude / R` where R = 4×ATR.
Continuous regression with MAE objective. Trained on 17,789 IS legs.

OOS MAE 1.03 vs baseline 1.04 — only 0.7% reduction. But the calibration
is **monotonic**: predicted 3.0+ → actual median 2.76; predicted 1.5-2.0
→ actual median 1.78.

**Why monotonic > MAE matters for sizing**: position sizing only needs
correct *ranking*, not precise prediction. B7's ranking is usable even
though its MAE is barely above baseline.

**Sizing rule (`gbm_ev`)**: `size = max(predicted_R - 1, 0)` clipped to
[0, 3]. Predicted-tiny legs get size 0; predicted-3R get size 2.0;
predicted-4R+ get the cap at 3.0. The size scales linearly with
expected gain in R units.

This is **the single highest-leverage model in the stack**. The +44.5%
oracle / +95% hardened improvement over flat sizing comes from B7 alone.

### B8 — Intraday hour-risk
> "What's the next 60 minutes of total leg P&L look like?"

For each 1m bar, regresses forward 60-min total leg P&L (in $) based
on V2 features at that bar.

OOS MAE $101, Pearson 0.22. Same calibration pattern as B7 — monotonic
but noisy.

Used as a sizing modifier:
- Predicted hour P&L < -200: scale current leg size by 0.7×
- Predicted hour P&L in [-200, 300]: linear interp 0.7× → 2.0×
- Predicted hour P&L > 300: scale by 2.0×

When B8 says "the next hour looks bad," reduce exposure. Combined
sizing: `total_size = b7_ev_size × b8_hour_multiplier`.

Hardened forward pass with B8 gate: +$1,505/day vs $927 without.

## The composite cloud (pivot_probability_cloud.parquet)

Per-bar composite zone classifier combining B1 + B4 (+ optional B5):

```
zone = AT_PIVOT     if  B4_W=60s P >= 0.85          (we're at a pivot)
zone = NEAR_PIVOT   elif B4_W=120s P >= 0.70        (within 2m of pivot)
zone = IMMINENT     elif B1 K=1 P >= 0.70           (<1m forward)
zone = NEAR_3m      elif B1 K=3 P >= 0.70           (1-3m forward)
zone = NEAR_5m      elif B1 K=5 P >= 0.70           (3-5m forward)
zone = WIDE_ZONE    elif B4_W=300s P >= 0.85        (5min pivot zone)
zone = WATCH        elif B1 K=10 P >= 0.70 or B4_W=300 P >= 0.70
zone = CLEAR        else                              (no pivot risk)
```

Plus a "trajectory state" axis: RISING / FLAT / DECAYING based on the
slope of expected-TTP over the trailing 10 bars.

Use for interpretation and rule-based filtering, not core sizing.

## How the models combine in the forward pass

```
For each leg in OOS:
    Detect R-trigger fire (streaming, on 5s closes)         # honest entry
    entry_ts = R-trigger timestamp
    Look up B7 prediction at entry_ts 1m bar                # leg-amplitude prediction
    Look up B8 prediction at entry_ts 1m bar                # hour-risk prediction
    Look up zone + B6 directional at entry_ts                # context

    leg_size = gbm_ev(B7 prediction)                         # 0-3x based on amp_R
    hour_mult = piecewise_linear(B8 prediction)              # 0.7-2.0x based on hour P&L
    total_size = leg_size × hour_mult

    Hold position until NEXT R-trigger fires                 # honest exit
    raw_pnl = (exit_5s_close - entry_5s_close) × leg_dir × $2/pt
    net_pnl = raw_pnl - $6 friction (commission + slippage)
    weighted_pnl = net_pnl × total_size
```

This is what's in `composite_forward_pass_hardened.py` (without B8 gate)
and `composite_forward_pass_hour_gated.py` (with B8 gate).

## Key design decisions

### Why GBM not transformer/LSTM
- Tabular data: GBM is state-of-the-art benchmark
- Earlier LSTM experiment showed +1.7pp on direction, similar on others
- Information ceiling is in V2 features, not model class

### Why R-trigger entries not pivot entries
- Pivot entries are oracle (we can't enter at the actual extreme live)
- R-trigger fires when 5s close moves R away from running extreme
- This matches what a live indicator-based strategy would do

### Why per-leg sizing not per-bar sizing
- Each leg is a discrete trading event with R-trigger entry/exit
- Per-bar sizing within a leg adds complexity without clear benefit
- B7 predicts at entry-time and applies for the whole leg

### Why we kept R-trigger as exit (didn't tighten/target)
- All exit-modification experiments (trail tightening, target placement)
  produced statistically negative results
- R-trigger waits for actual reversal — structurally optimal at this
  signal precision
- See HONEST_CAVEATS for the failed-exit-mods detail

### Why we DIDN'T build a daily regime classifier
- Bad days look like good days in V2 features at session start
- Would need cross-day features (overnight, calendar, intermarket)
- Out of scope for this build; flagged as next step

## What's next (architecturally)

1. Day-level regime classifier on overnight + calendar features
2. Retrain B7/B8 on R-trigger-bar features (remove last feature peek)
3. Run end-to-end through streaming engine (training_iso_v2/) for full validation
4. Add partial-exit logic (take some off at +1R, ride remainder)
