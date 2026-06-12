---
name: V2-native training pipeline (training_v2/)
description: Clean-slate V2-native training built 2026-05-04 — reads core_v2.features directly, no V1 conversion shims. Production state, components, and key files.
type: project
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## What this is

`training_v2/` is the V2-native training pipeline rebuilt from scratch on
2026-05-04. The previous `training_v2/` (V1-shape engine reading a V2-derived
compat cache) was archived to `training_v2_archive/` to preserve the +$78/OOS
calibration work.

**Why:** The user pushed back hard on the V1-shape engine + compat cache. The
new training_v2/ consumes V2 layered features (185D = L0 + 8 TFs × 23 from
`core_v2.features.load_features()`) directly. No conversion. No `v1_compat`.
No 91D V1 vector. No `_1M_OFFSET` indices.

## Architecture

```
training_v2/
├── ticker.py              V2Ticker / MultiDayV2Ticker — yields BarState per 5s bar
├── state.py               BarState dataclass + REGIME_VOCAB
├── v2_cols.py             canonical V2 column-name helpers (z_se_w, vwap_w, body...)
├── ledger.py              Position + ClosedTrade dataclasses
├── engine.py              first-signal-wins entry, first-match exit, threshold injection
├── exits.py               HardStop, TakeProfit, Giveback, ZSeReversal, SwingNoiseSpike,
│                          RegimeFlip, TimeStop. Per-tier thresholds via position.extras.
├── regime_router.py       day-level eligibility (placeholder, all-allowed default)
├── strategies/
│   ├── base.py            Strategy ABC + EntrySignal
│   ├── ma_align.py        MAAlignTrendFollow (7-of-8 vwap_w alignment, 5m close)
│   ├── reversion.py       ReversionFromExtreme (V2 NMP: |z_se_w|≥1.8 + reversion_prob≥0.55)
│   └── velocity_body.py   VelocityBodyChord — KILLED (lottery-day artifact)
├── regret.py              replays each trade's price path → peak/MAE/capture/optimal-exit labels
├── bayesian_table.py      hierarchical posteriors per (regime, tier) — for inspection
├── threshold_optimizer.py grid-search exit thresholds (legacy approach — use sparingly)
├── threshold_bayesian.py  PRINCIPLED — derives thresholds from regret distributions via
│                          formulas (q_tp, q_sl, ttp_factor knobs). Preferred over grid search.
├── tier_discovery.py      flip-signal discovery (FADE_BETTER vs FLIP_BETTER classification + Cohen's d)
├── cnn/                   V2DirectionCNN (8×23 grid + L0 + regime embed → 3-class softmax)
└── run.py                 CLI: --is/--oos --strategies --thresholds --cnn ...
```

## Key principles

1. **No V1 dependencies.** Engine reads V2 columns by canonical name (e.g.,
   `state.get('L3_1m_z_se_15')` via `v2_cols.z_se_w('1m')`). No `_1M_OFFSET`
   indexing.
2. **First signal wins** (user's spec). Strategies evaluated in declaration
   order; first non-None EntrySignal opens the trade.
3. **Per-tier thresholds via `position.extras['thresholds']`.** Engine looks
   up `(regime, tier)` at trade open and copies the threshold dict into the
   position's extras. Exit rules read from there at runtime.
4. **Bayesian thresholds derived, not searched.** `threshold_bayesian.py`
   computes thresholds as quantile/mean functions of cell-grouped regret
   distributions. No grid search overfitting.

## Production state (2026-05-04)

**Strategies (active):**
- MA_ALIGN — 7-of-8 vwap_w alignment, fires on 5m close
- REVERSION — V2-native NMP, fires on 1m close

**Strategies (killed):**
- VEL_BODY_CHORD — see `feedback_outlier_day_optimizer.md`. Negative on 67/68
  OOS days, positive only on 2026-03-20 ($1,333/contract range day).

**Production thresholds:** `training_v2/output/thresholds_prod.json`
(per-tier Bayesian-derived; see `threshold_bayesian.py --group-by tier`).

| tier | TP | SL (capped) | giveback arms | gb_keep | max hold |
|---|---:|---:|---:|---:|---:|
| MA_ALIGN | +$26 | -$50 | $41 | 30% | 41.7m |
| REVERSION | +$27 | -$50 | $41 | 39% | 39.8m |

**Last measured OOS performance** (68 OOS days, MA+REV only, prod thresholds):
- $54.82/day (+$3,728 total)
- Day-WR 57% (vs 51% baseline)
- 95% CI on delta vs baseline: [-$5.29, +$62.89] — NOT statistically significant

## Pipeline (full run sequence)

```
# 1. Generate IS trades (no CNN, no thresholds — baseline)
python -m training_v2.run --is --strategies MA_ALIGN,REVERSION

# 2. Build regret labels
python -m training_v2.regret --trades training_v2/output/is.pkl \
    --out training_v2/output/regret_is.pkl

# 3. Derive Bayesian per-tier exit thresholds
python -m training_v2.threshold_bayesian --regret training_v2/output/regret_is.pkl \
    --out training_v2/output/thresholds_prod.json --group-by tier

# 4. Build Bayesian table (inspection only)
python -m training_v2.bayesian_table --regret training_v2/output/regret_is.pkl \
    --out training_v2/output/bayesian_is.pkl

# 5. Train V2 direction CNN (deferred until used as filter+entry)
python -m training_v2.cnn.train

# 6. Forward pass IS with thresholds + CNN
python -m training_v2.run --is --strategies MA_ALIGN,REVERSION \
    --thresholds training_v2/output/thresholds_prod.json \
    --cnn training_v2/output/cnn/direction_cnn.pt

# 7. OOS validation
python -m training_v2.run --oos --strategies MA_ALIGN,REVERSION \
    --thresholds training_v2/output/thresholds_prod.json \
    --cnn training_v2/output/cnn/direction_cnn.pt
```

## Open questions / known gaps

1. **Directional wicks not in entry feature vector.** They CAN be computed
   per-bar from 1m/5m/15m OHLCV (legacy `directional_wicks_batch` math from
   `core_v2/v1_compat.py` is still valid — pure OHLCV math, not V1-specific).
   Currently the entry_v2 vector is the 185D V2 layered features only. If we
   want to use directional wicks as discriminators, we need to add them.

2. **No sub-tier discrimination.** REVERSION fires for any `|z_se_w|≥1.8`
   entry — no further sub-classification (legacy had 9 sub-tiers under NMP).
   See `project_9tier_discovery_v2.md` for what we tried and what failed.

3. **CNN not yet trained.** The V2DirectionCNN (8×23 grid + L0 + regime
   embed → 3-class softmax) is built but not trained. Expected to push past
   the +$55/day OOS ceiling once trained.

## Why we trust this architecture

- VEL_BODY_CHORD failure was *detected* by the bootstrap CI doom-cascade
  check, not silently shipped (anti-doom rule worked)
- Threshold-tuning ceiling is honestly reported with CIs, not inflated
- Median-day vs total objective comparison made the IS-overfit pattern
  concrete (see `feedback_outlier_day_optimizer.md`)
- Cell-granularity sweep (regime / tier / tier × regime) showed all three
  give within $2/day of each other — robust framework
