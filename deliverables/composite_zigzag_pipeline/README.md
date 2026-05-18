# Composite Zigzag Pipeline

**Standalone snapshot — 2026-05-17.** Self-contained directory with all
trained models, caches, tools, and reports for the V2-feature + GBM
composite trading pipeline built for MNQ futures.

## What this pipeline does

Takes an indicator-based zigzag entry/exit strategy on MNQ 5s futures
data and adds a **stack of 7 GBM models** that predict different aspects
of the next leg / next hour. The GBM predictions drive position sizing
and (optionally) hour-level risk gating.

Architecture:
```
                  V2 features (184 cols, 5s-1D timeframes)
                              │
        ┌─────┬─────┬─────────┼─────┬─────┬─────┐
        ↓     ↓     ↓         ↓     ↓     ↓     ↓
       B1    B2    B4        B5    B6    B7    B8
     pivot  fake  region    leg   dir   amp   hour
     imm    out             phase pivot         risk
        │     │     │         │     │     │     │
        └─────┴─────┴────┬────┴─────┴─────┴─────┘
                         ↓
              Composite zone + sizing
                         ↓
              Live ZZ entry/exit + position sized leg
```

Headline OOS performance (hardened forward pass, no peeks):
- **+$927/day mean** (gbm_ev sizing, 32 NT8 OOS days)
- 95% CI [+$534, +$1,372], wins 26/31 days
- **84% days positive, 77% days >$200**
- Median per-leg P&L: -$11 (most legs small losers, winners cover)

## Quickstart — replicate the forward pass

```bash
# From this directory:
python tools/composite_forward_pass_hardened.py \
    --truth caches/zigzag_pivot_dataset_NT8_OOS_atr4.parquet \
    --b7-pkl models/b7_leg_sizer.pkl \
    --cloud caches/pivot_probability_cloud.parquet \
    --b6 caches/b6_proba_OOS_NT8.parquet \
    --out caches/replay_forward_pass.csv

# View the canonical OOS result:
cat reports/forward_pass_outputs/composite_forward_pass_hardened.txt
```

## Quickstart — train fresh from data

Requires `DATA/ATLAS_NT8/{5s,1m}/...parquet` and `DATA/ATLAS_NT8/FEATURES_5s_v2/...`
(see DATA_REQUIREMENTS.md).

```bash
# 1. Build zigzag truth labels (IS + OOS) at ATR×4
python tools/build_zigzag_pivot_dataset.py --target is --atr-mult 4.0 \
    --out caches/zigzag_pivot_dataset_IS_atr4.parquet

python tools/build_zigzag_pivot_dataset.py --root DATA/ATLAS_NT8 --target oos --atr-mult 4.0 \
    --out caches/zigzag_pivot_dataset_NT8_OOS_atr4.parquet

# 2. Train each B-model
python tools/train_b1_pivot_imminent.py
python tools/train_b2_fakeout.py
python tools/train_b4_pivot_region.py
python tools/train_b5_leg_phase.py
python tools/train_b6_directional_pivot.py
python tools/train_b7_leg_sizer.py
python tools/train_b8_hour_risk.py

# 3. Precompute per-bar probabilities for OOS
python tools/precompute_b1_b2_oos.py

# 4. Build the composite cloud
python tools/pivot_probability_cloud.py

# 5. Run the hardened forward pass
python tools/composite_forward_pass_hardened.py
```

Each training step takes 1-5 minutes on CPU.

## File map

```
composite_zigzag_pipeline/
├── README.md                ← you are here
├── ARCHITECTURE.md          ← what each B-model does + why
├── DATA_REQUIREMENTS.md     ← data needed to retrain from scratch
├── HONEST_CAVEATS.md        ← known leaks, limitations, what's NOT honest
├── SESSION_LOG.md           ← chronological build order + experiments
├── requirements.txt         ← Python deps (sklearn, pandas, numpy, scipy, tqdm)
│
├── tools/                   ← 35 scripts: training, simulation, diagnostics
│   ├── build_zigzag_pivot_dataset.py   ← truth labels
│   ├── train_b{1,2,4,5,6,7,8}_*.py     ← each B-model
│   ├── precompute_b1_b2_oos.py         ← per-bar prediction caches
│   ├── pivot_probability_cloud.py      ← composite zone classifier
│   ├── composite_entry_analyzer.py     ← entry-time signal → leg quality
│   ├── composite_entry_accuracy.py     ← pure-entry-accuracy diagnostic
│   ├── composite_*_simulator.py        ← strategy sims (trail, target, sizing)
│   ├── composite_forward_pass*.py      ← end-to-end OOS evaluation
│   ├── live_zigzag_baseline.py         ← causal indicator baseline
│   ├── direction_signal_accuracy.py    ← directional signal eval
│   ├── b1_trajectory_bridge.py         ← trajectory-of-B1 analysis
│   ├── b6_visual_diagnostic.py         ← chart B6 per-day predictions
│   └── _viz/auto_swing_marker.py       ← zigzag swing detector (dependency)
│
├── models/                  ← trained pickles (7 GBMs, ~30 MB total)
│   ├── b1_pivot_imminent.pkl       ← P(pivot in next K min), K=1,3,5,10
│   ├── b2_fakeout.pkl              ← P(fakeout) per pivot
│   ├── b4_pivot_region.pkl         ← P(in ±W window of pivot), W=30,60,120,300s
│   ├── b5_leg_phase.pkl            ← P(EARLY/MID/LATE) of current leg
│   ├── b6_directional_pivot.pkl    ← P(LONG/SHORT pivot next K min)
│   ├── b7_leg_sizer.pkl            ← E[leg amplitude] in R units (drives sizing)
│   └── b8_hour_risk.pkl            ← E[next 60-min total P&L in $]
│
├── caches/                  ← per-bar predictions + truth datasets (~150 MB)
│   ├── zigzag_pivot_dataset_IS_atr4.parquet   ← IS truth (277 days)
│   ├── zigzag_pivot_dataset_NT8_OOS_atr4.parquet ← OOS truth (32 days NT8)
│   ├── b{1,2,4,5,6,8}_*_OOS*.parquet          ← per-bar predictions
│   ├── b7_leg_sizer_{IS,OOS}.parquet          ← per-leg predictions
│   ├── pivot_probability_cloud.parquet        ← composite zone per bar
│   ├── composite_entry_analyzer.csv           ← per-leg entry features
│   ├── composite_entry_accuracy.csv           ← per-leg trajectories
│   └── composite_forward_pass_hardened.csv    ← per-leg P&L (R-trigger sim)
│
└── reports/                 ← analysis findings + training output
    ├── 2026-05-17_*.md                    ← detailed findings docs
    ├── training_outputs/                  ← per-model training reports
    └── forward_pass_outputs/              ← simulation results
```

## Run order (the canonical sequence)

To reproduce the project from scratch:

1. **Data prep** — `build_zigzag_pivot_dataset.py` (×2 for IS + OOS)
2. **Train B1-B8** — one script each (model-per-task)
3. **Precompute caches** — `precompute_b1_b2_oos.py`, etc.
4. **Composite cloud** — `pivot_probability_cloud.py`
5. **Entry diagnostics** — `composite_entry_analyzer.py`, `composite_entry_accuracy.py`
6. **Forward passes** — `composite_forward_pass_hardened.py` (the main result)
7. **Optional**: `composite_forward_pass_hour_gated.py` for B8 risk gating

## What to read in what order (for outside help)

1. **README.md** (this file) — overview
2. **ARCHITECTURE.md** — what each B-model does, target framing, key results
3. **HONEST_CAVEATS.md** — what's leaky, what's optimistic, what's untested
4. **DATA_REQUIREMENTS.md** — data dependencies
5. **reports/2026-05-17_composite_trade_management.md** — the headline findings doc
6. **SESSION_LOG.md** — chronological context of how decisions were made

## Performance summary at a glance

| Stage of pipeline | OOS $/day mean | 95% CI | Days >$0 | Days >$200 |
|-------------------|----------------|--------|----------|------------|
| Naked R-trigger (no sizing) | $475 | [$237, $741] | 24/31 | — |
| + B7 GBM sizing (gbm_ev) | $927 | [$534, $1,372] | 26/31 | 24/31 |
| + B8 hour-risk gate (linear) | $1,505 | — | 26/31 | 25/31 |

All numbers are **honest** — no oracle-entry peeks. R-trigger entries with $6 friction/leg.

## Known limitations (TL;DR — see HONEST_CAVEATS.md for full list)

- N=32 OOS days is small; CIs are wide
- Strategy has heavy right-tail dependence (mean $927, KDE mode $468)
- ~5/31 days are net negative; no current model can filter them ex-ante
- B7 sizing model trained on pivot-bar features, applied at R-trigger bar (small distribution shift)
- Day-level regime classifier NOT built (would address bad-day filtering)
- No live deployment data — backtest only

## Next steps if continuing

1. **Cross-day regime features** — overnight gap, calendar events, intermarket (ES/VIX/DXY).
   Train a "bad day" GBM on these and skip predicted-bad days.
2. **Live data collection** — babysit deployment at 0.25× size, log per-leg actuals
3. **Retrain B7 on R-trigger-bar features** — removes the last feature peek
4. **Run through training_iso_v2 streaming engine** — full end-to-end validation
