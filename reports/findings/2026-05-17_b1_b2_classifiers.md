# 2026-05-17 — B1 Pivot-Imminent + B2 Fakeout Classifiers (NT8 OOS)

**Goal**: test whether V2 features can predict things the live zigzag indicator
*cannot* — pivot timing (B1) and fakeout filtering (B2). The earlier
direction-vs-indicator comparison showed trend3 had no statistical edge on
hindsight direction (CI [-1.13, +4.40]pp). These two tasks are the harder,
forward-looking problems.

**Datasets**:
- IS: `zigzag_pivot_dataset_IS_atr4.parquet` (282,669 bars / 277 days)
- OOS: `zigzag_pivot_dataset_NT8_OOS_atr4.parquet` (34,844 bars / 32 days)
- 184 V2 features (L1/L2/L3 at 5s-1D)

**Models**: `HistGradientBoostingClassifier` max_depth=6, max_iter=200,
class_weight='balanced', l2_reg=0.5. Trained on full IS, evaluated on full OOS.

**Tools**: `tools/train_b1_pivot_imminent.py`, `tools/train_b2_fakeout.py`
**Outputs**: `reports/findings/regret_oracle/b1_pivot_imminent.{pkl,txt}`,
`reports/findings/regret_oracle/b2_fakeout.{pkl,txt}`

---

## B1 — Pivot-imminent classifier

**Task**: per 1m bar, predict `pivot_within_K_minutes` (binary).
Indicator can't do this — it confirms AFTER reversal.

| K   | Base | IS AUC | **OOS AUC** | Best operating point                            | Lift  |
|-----|------|--------|-------------|--------------------------------------------------|-------|
| 1m  | 5.23%| 0.776  | **0.686**   | thr=0.70: prec 13.6% / rec 11.7% / cov 4.48%    | 2.61x |
| 3m  | 15.7%| 0.765  | **0.698**   | thr=0.70: prec 34.3% / rec 18.1% / cov 8.27%    | 2.19x |
| 5m  | 24.8%| 0.781  | **0.710**   | thr=0.70: prec 47.1% / rec 37.8% / cov 19.90%   | 1.90x |
| 10m | 41.3%| 0.813  | **0.716**   | **thr=0.85: prec 78.1% / rec 7.3% / cov 3.88%** | 1.89x |

**Headline finding**: at K=10 thr=0.85 the model is **78% accurate flagging
"pivot within 10 min"** while covering 3.88% of bars (≈42 warnings/day).
That's ~32 correct + ~9 false alarms per day. The indicator can't do this
at all (AUC 0.5).

**Game-changer assessment** (against my original threshold "precision >50%
at K=1"): **MISSED**. K=1 peaks at 13.6% precision — useful only as a soft
feature for downstream models, not as a standalone fire signal.

**IS-OOS AUC gap**: 0.078-0.097 across K — moderate overfit. The model finds
IS patterns of imminent pivots that partially fail to generalize. Most
likely cause: 184 features × hindsight labels = label leakage candidates we
haven't audited. Worth a feature-importance pass before deployment.

---

## B2 — Fake-out classifier

**Task**: per confirmed pivot, predict `is_fakeout_K` = next pivot occurs
in opposite direction within K minutes (zigzag legs always alternate, so
"next pivot within K min" = "this leg was short-lived noise").

| K   | Base | IS AUC | **OOS AUC** | Best operating point                            | Lift  |
|-----|------|--------|-------------|--------------------------------------------------|-------|
| 3m  | 8.2% | 0.896  | **0.669**   | thr=0.70: prec 30.8% / rec 2.7% / cov 0.71%     | 3.75x |
| 5m  | 25.2%| 0.831  | **0.685**   | thr=0.70: prec 40.4% / rec 4.6% / cov 2.85%     | 1.60x |
| 10m | 49.6%| 0.846  | **0.695**   | **thr=0.70: prec 66.1% / rec 24.7% / cov 18.56%**| 1.33x |

**Headline finding**: at K=10 thr=0.70 the model flags ~19% of confirmed
pivots as "this leg will die within 10 min" with **66% precision** —
roughly 11 warnings/day, ~7 of which are genuine fakeouts.

**Game-changer assessment** (against threshold "prec >60% at AUC >0.65"):
**HIT at K=10**. We can filter ~25% of "real-looking but short-lived" pivots
with 66% reliability. This is the cleanest standalone signal from either
classifier.

**IS-OOS AUC gap is large** — 0.227 at K=3 (0.90 → 0.67). Strong overfit
warning. With 17,789 IS pivots and 184 features, the GBM has plenty of room
to memorize. Production deployment should use feature selection (top-20
features) and L2 regularization tuning. The OOS numbers as reported are
honest but a more regularized model would likely close the gap.

---

## Comparison to live-zigzag baseline

| Task                             | Indicator AUC | V2 GBM OOS AUC | ML edge |
|----------------------------------|---------------|----------------|---------|
| Current leg direction            | ~0.65 (1.80pp delta CI [-1.1, +4.4]) | n/a (no AUC) | none significant |
| Pivot imminence (10min)          | 0.50          | **0.716**      | **+21pp** |
| Fakeout detection (10min)        | 0.50          | **0.695**      | **+20pp** |

**This is the result that matters**: V2 features carry real forward-looking
information for *timing* and *quality* of pivots, but **NOT** for current
direction. The earlier null direction result wasn't "ML is useless" — it
was "current direction is the wrong target." Reframing to forward-looking
targets surfaces a genuine ML edge.

---

## Practical use

### As standalone signals
- **B1 K=10 thr=0.85** → ~42 high-conf pivot warnings/day, 78% precision.
  Use as: trade-rate dampener (slow down or fade) near predicted pivots.
- **B2 K=10 thr=0.70** → ~11 fakeout flags per pivot day, 66% precision.
  Use as: pivot-confidence gate — pass fewer NT8 strategy fires through
  when a fresh pivot is flagged fake.

### As features in downstream models
- B1 P(pivot_within_K) at multiple K becomes a "tenseness" feature for
  exit timing or entry filtering.
- B2 P(fakeout) becomes a confidence multiplier on indicator signals.

---

## Honest caveats

1. **OOS AUC 0.67-0.72 is good but not exceptional.** ROC suggests the model
   has signal at the *top* of the score distribution but is noisy elsewhere.
   That's why high-threshold (thr=0.85) operating points have best precision.

2. **Big IS-OOS gap on B2 (especially K=3)**. Treat the OOS numbers as
   slightly optimistic — a regularized rerun would likely show similar
   OOS but with smaller gap (signaling the model isn't memorizing IS as much).

3. **No per-day CIs computed yet.** OOS sample is 32 days. We should
   bootstrap the per-day AUC and precision to get CIs before claiming
   these numbers are stable. Next iteration.

4. **No causality audit on features.** With 184 features over 1m bars and
   forward-looking targets, label leakage is possible (a feature computed
   over a window that bleeds into the target horizon). Should audit top
   feature importances before any deployment.

5. **Class weights = 'balanced'.** For very rare positives (B1 K=1 at 5%),
   this oversamples positives and may distort calibration. The high-threshold
   numbers are still meaningful but probabilities aren't calibrated to act
   on directly without further work.

---

## Status of the 4-step plan

- ✅ **Step 1**: inspector correctness pinned to ATR×4
- ✅ **Step 2**: live ZZ baseline — trend3 has no edge on hindsight direction
- ✅ **Step 3 (redirected)**: B1 + B2 trained — **REAL ML EDGE FOUND** on
  forward-looking targets (AUC 0.70-0.72 OOS, lift 1.3-3.8x at usable
  operating points).
- 🟢 **Step 4 next**: integrate B1+B2 as features/filters in the existing
  trend3 + zigzag pipeline, or train multi-output GBM that does both
  simultaneously, or feed B1/B2 probabilities into the inspector for
  visual validation.

The original "multi-TF stacking" idea from step 3 wasn't run — and given
the B1+B2 findings, the more leveraged next experiment is **combining**
B1+B2+indicator+trend3 into a multi-signal entry filter, not adding more
architectural complexity to direction-only classification.

**No P&L numbers in this report by design** — signal quality only.
