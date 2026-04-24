# JULES_PIVOT_LGBM_V2 — LightGBM pivot filter on v2 (185D) features

**Status**: DRAFT — pending review / counterbalance critique
**Date**: 2026-04-24
**Author**: Claude (with user oversight)
**Depends on**: `core_v2/*`, `training_RM_physics_v2/*`, feature_spec_v2.md

---

## 1. Problem statement

The pivot-direction CNN on v2 features collapsed to random (AUC 0.51) on 9,407 v2 engine trades. A diagnostic run of the same CNN on v1's 357 curated trades got AUC 0.60 — so the v2 features carry SOME signal, but it's not surviving the noise-flooded v2 trade set through a CNN architecture.

Hypothesis we're testing:
> **CNN was the wrong architecture for this task.** The 8×23 feature grid has no spatial locality, TF ordering is arbitrary, and deep learning is known to under-perform gradient boosting on tabular finance data at N < 10k samples.

We want a fast, honest answer to: "Given v2 features at an engine-generated pivot, can ANY learnable model separate winners from losers?" LightGBM is the standard benchmark for this and will answer cleanly.

## 2. Why LightGBM (not CNN / MLP / Transformer)

| | Why it's right here |
|---|---|
| **Tabular native** | Our 185D feature vector IS tabular. GBM was designed for this; DL is a mismatch. |
| **Small-data regime** | 5-10k train samples is GBM sweet spot. DL needs 100k+ for deep models to outperform. |
| **Non-spatial features** | No adjacency assumption. Each feature is split-tested independently. |
| **Mixed scales** | `z_se ~ 0-3` and `L2_price_mean ~ 20000` coexist fine — trees only care about split thresholds, not magnitudes. |
| **NaN native** | GBM handles missing values by sending them down a specific branch during training. No warmup-skip hack. |
| **Interpretable** | Feature importance + SHAP tell us WHICH features matter — validates whole L1/L2/L3 spec. |
| **Fast** | 30-sec training vs 5-min CNN. 10× iteration speed. |
| **Empirically strong on finance tabular** | Kaggle, Numerai, JPM QuEST benchmarks all show GBM ≥ DL for this class of problem. |

Not chosen (for explicit justification):
- **MLP**: still deep learning. Same data-hungry regime. Loses GBM's structural advantages (NaN handling, split-based selection).
- **Transformer / TabNet**: overkill at this data size. Would overfit.
- **Bayesian NN**: solves calibration, not discrimination. Useful AFTER we have a discriminating base model.
- **Random Forest**: usually dominated by GBM. Worth keeping as baseline.

## 3. Data pipeline

### Inputs
- **Trade labels**: `training_RM_physics_v2/output/trades/rm_is.pkl`
  - 9,407 IS trades at MIN_RES=1.5 (current v2 engine output)
  - Each trade: `{day, timestamp, pnl, direction, ...}`
- **Features**: `DATA/ATLAS/FEATURES_5s_v2/` (layer-family parquets)
  - 185 cols joined via `core_v2.features.load_features()`
  - 5s-aligned, zero lookahead (verified by 28 poisoning tests)

### Lookup
For each trade: snap `entry_timestamp → (ts // 5) * 5`, read that row from the joined features df. Drop trade if feature row missing or contains NaN in required columns.

### Label (OPEN DECISION — see §9)
Proposal: **binary `is_winner = pnl > 0`**. Base rate ~60% in v2 set.

Alternative: **threshold label `is_real_winner = pnl > $5`**. Lower base rate but filters out micro-noise wins. Worth an A/B.

### Splits (matching CNN v2 script)
| Set | Range | ~N trades |
|---|---|---:|
| Train | 2025-02-01 → 2025-08-31 | ~5,300 |
| Val | 2025-09-01 → 2025-09-30 | ~500 |
| Test | 2025-10-01 → 2025-12-31 | ~2,300 |
| OOS | 2026-01-01 → 2026-03-20 | NOT LOADED BY TRAINING |

OOS strictly reserved for final engine-in-the-loop evaluation (Phase 4 of RM engine work).

## 4. Model configuration

### LightGBM hyperparameters (initial)
```python
params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'is_unbalance': False,  # 60% base rate — not heavily imbalanced
}
num_boost_round = 1000
early_stopping_rounds = 50  # on val AUC
```

Defaults are intentionally conservative — we pick best-val round via early stopping rather than grid-searching upfront. If AUC is promising, Optuna sweep of num_leaves / learning_rate / reg comes after.

### Input feature matrix
- Shape: `(N_trades, 185)`
- Column order = `core_v2.features.FEATURE_NAMES`
- NaN preserved (GBM native handling)
- No feature-space normalization (Principle 6 of spec; GBM doesn't need it anyway)

## 5. Validation protocol

### Primary metric
**Test AUC** on held-out 2025-10-01 → 2025-12-31 days.

Benchmarks to beat:
- v1 baseline on 91D: AUC **0.63**
- CNN on v2 trades: AUC **0.51** (current ceiling)
- v2 features on v1 curated trades (diagnostic): AUC **0.60**

Success tiers:
- **Green**: Test AUC ≥ 0.62 → features carry signal, CNN was the bottleneck. Ship as engine filter.
- **Yellow**: 0.55 ≤ AUC < 0.62 → weak signal. Possibly worth tuning or feature engineering.
- **Red**: AUC < 0.55 → features don't discriminate. Need deeper rethink (different labels, different features, or different engine).

### Secondary diagnostics
1. **Feature importance** (top 20 by gain) — sanity check against spec expectations. If v2 features we handcrafted (SE_high, swing_noise, VWAP) don't show up in top-20, something's wrong.
2. **SHAP summary plot** — direction of feature influence. E.g. z_se should push P(win) DOWN at extreme values.
3. **Calibration curve** (binned P(win) vs actual WR) — for position sizing later.
4. **Confusion matrix at optimal threshold** — for the engine filter deployment.
5. **$/trade by predicted-prob quantile** — does the top quintile of predicted winners actually earn more? This is the real economic test.

## 6. Deliverables

| Artifact | Path |
|---|---|
| Training script | `training/train_pivot_lgbm_v2.py` |
| Saved model | `training_RM_physics_v2/output/pivot_lgbm_v2.txt` (LightGBM native format) |
| Report | `reports/findings/pivot_lgbm_v2.md` (AUC, calibration, top-20 features, SHAP, $/trade by quantile) |
| Feature importance CSV | `reports/findings/pivot_lgbm_v2_importance.csv` |
| Inference helper | Add `predict_pwin_lgbm()` to `training_RM_physics_v2/nn_direction.py` so the engine can optionally use it as filter |

## 7. Implementation plan

### Phase A — Build & smoke (~1 hour)
1. Write `train_pivot_lgbm_v2.py` mirroring structure of `train_pivot_cnn_v2.py`:
   - Same CLI flags (`--trades-pkl`, `--features-root`, date splits)
   - Same data loading path via `core_v2.features.load_features`
   - Swap CNN training loop for `lgb.train` with early stopping
2. Verify `--trades-pkl training_RM_physics_v2/output/trades/rm_is.pkl` runs end-to-end and outputs a report.

### Phase B — Primary evaluation (~10 min runtime)
Run on the 9,407 v2 trade set. Report Test AUC.

### Phase C — Diagnostics (~30 min)
If AUC ≥ 0.55:
- Generate SHAP summary, feature importance CSV, calibration plot
- Re-run with threshold labels (pnl > $5) as A/B
- Check $/trade by prediction-quantile (economic relevance)

If AUC < 0.55:
- Run on v1's 357-trade pkl as diagnostic (mirror CNN's H1/H2 test)
- Both failing → features genuinely don't discriminate → halt and reconsider spec
- v1 diagnostic succeeds → engine trade set is the bottleneck (same conclusion as CNN case)

### Phase D — Engine filter integration (only if Phase B is green)
- Write `predict_pwin_lgbm()` helper in `training_RM_physics_v2/nn_direction.py`
- Wire it into `rm_physics_engine.py` as alternative to CNN filter (config flag)
- Re-run `run_rm.py --with-oos` with LGBM filter at threshold 0.65 (same as CNN)
- Report OOS $/day with/without filter

## 8. Risks and how we address them

| Risk | Mitigation |
|---|---|
| **Overfitting on 5K train trades with 185 features** | Early stopping on val, `min_data_in_leaf=50`, `reg_alpha/lambda=0.1` |
| **Leakage through feature lookup** | Already audited: `build_dataset_v2` uses `searchsorted(ts - period)`. 28 lookahead poisoning tests pass. |
| **Regime mismatch train (2025) vs OOS (2026)** | OOS evaluation is engine-in-the-loop on real bars, not just AUC. Tests ACTUAL generalization. |
| **Class imbalance on `pnl > 5` threshold label** | Keep `is_unbalance=True` as a flag; test both balanced and unbalanced. |
| **GBM beats CNN but still too weak for ship** | That's a FINDING, not a failure. Tells us the problem is upstream (engine, labels, or trade structure). |
| **Feature importance dominated by one TF or one layer** | Useful signal. If 1D features dominate, we know horizon matters. If L1 primitives dominate, L2/L3 adds nothing. Either way — actionable. |

## 9. OPEN DECISIONS (want counterbalance input)

Each of these I have an opinion on but want another LLM to challenge:

1. **Label formulation**: binary `pnl > 0` vs threshold `pnl > 5` vs regression. My vote: binary first (matches prior CNN experiment for apples-to-apples).

2. **Temporal delta features**: should we add `z_se[n] - z_se[n-5]` etc. as explicit engineered features, or trust GBM to find interactions from statics? My vote: skip for first pass — GBM with interaction_constraints can learn ratios. Add deltas only if static AUC is weak.

3. **CV scheme**: single train/val/test (matches CNN script) vs 5-fold walk-forward CV for noise-robust AUC estimate. My vote: single split for first run (fast), fold CV if AUC is borderline (0.55-0.62).

4. **Feature selection**: use all 185 features, or pre-select top-K by univariate AUC? My vote: all 185 — GBM does internal selection via `feature_fraction` and gain-based splits.

5. **Class weight on label imbalance**: v2 set is 60% winners — mild imbalance. Should we use `scale_pos_weight=0.67` (inverse base rate) or leave at 1.0? My vote: leave at 1.0 for first run; revisit if recall on losers is pathological.

6. **Compute separate models for long-direction trades vs short-direction trades?** The feature importance might differ (asymmetric regimes). My vote: single model initially, split by direction as a diagnostic if symmetry is suspicious.

7. **What's the AUC ship threshold for engine filter deployment?** We know CNN at 0.63 produced +$2.98/day IS, -$0.81/day OOS (barely breakeven). Need AUC ≥ 0.68 to realistically expect +$400/day post-filter. My vote: formalize 0.68 as ship criterion, anything below triggers further rethink.

## 10. What this does NOT commit to

- Not replacing the CNN code — keeping it as reference/alt architecture
- Not changing v2 feature spec — 185D stays
- Not changing engine behavior — LGBM initially runs as alternate filter, doesn't replace pivot logic
- Not committing to multi-model ensembles — that's Phase 2 if we have a winner

## 11. Approval to proceed

If this doc passes the counterbalance check:
1. User confirms
2. Claude implements Phase A (builds `train_pivot_lgbm_v2.py`)
3. User runs it (per "USER runs training" convention)
4. We triage based on Phase B result

If doc fails counterbalance check:
- Address the critique
- Update doc, repeat review

---

**Key numbers to anchor the conversation**:
- CNN on v2: AUC 0.51 (random)
- CNN on v1: AUC 0.63 (v1 baseline)
- CNN (v2 features, v1 trades): AUC 0.60 — diagnostic showing features DO carry some signal
- v2 engine unfiltered: -$20/day IS, noisy trade set at MIN_RES=1.5
- Target for ship: +$400/day OOS MNQ after filter

**Core question this experiment answers**: does the signal-present-but-CNN-can't-find-it state (AUC 0.60 on v1, 0.51 on v2) resolve when we swap CNN → LightGBM?
