# Data Requirements

## What's in this folder vs what you need externally

**Included** (in `caches/`):
- `zigzag_pivot_dataset_IS_atr4.parquet` — 282,669 IS bars (2025) with V2
  features + zigzag pivot labels. Sufficient for training every B-model.
- `zigzag_pivot_dataset_NT8_OOS_atr4.parquet` — 34,844 OOS bars
  (2026-03-20 to 2026-05-something) with same structure. For OOS eval.

**Required externally** (NOT included due to size, ~10 GB):
- `DATA/ATLAS/5s/{day}.parquet` — IS 5s OHLC bars (for zigzag detection)
- `DATA/ATLAS/1m/{day}.parquet` — IS 1m OHLC (for ATR computation)
- `DATA/ATLAS/FEATURES_5s_v2/{layer}/{day}.parquet` — V2 features (8 TFs × 3 layers)
- `DATA/ATLAS_NT8/5s/{day}.parquet` — NT8 OOS 5s OHLC
- `DATA/ATLAS_NT8/1m/{day}.parquet` — NT8 OOS 1m OHLC
- `DATA/ATLAS_NT8/FEATURES_5s_v2/...` — NT8 OOS V2 features

If you have the IS+OOS parquets in `caches/`, you can:
- Train all B-models WITHOUT external data
- Run forward passes WITHOUT external data (they read OOS truth + caches)

You only need external data to:
- Build truth labels from raw OHLC (`build_zigzag_pivot_dataset.py`)
- Run hardened forward pass with realistic R-trigger detection (needs 5s closes)
- Run the V2-iso streaming ticker

## Data schema reference

### Truth parquet (`zigzag_pivot_dataset_*.parquet`)

Each row = one 1m close timestamp on a day. Columns:

```
timestamp        : Unix seconds (int64)
day              : 'YYYY_MM_DD' string (matches data file naming)
is_pivot         : 1 if this bar is in the ±60s window of a zigzag pivot, else 0
pivot_dir        : 'LONG' or 'SHORT' — direction of leg STARTING at this pivot
                   (LONG = leg up = pivot is a low)
pivot_price      : price at the pivot extreme (set only on is_pivot==1 bars)
leg_direction    : direction of the leg this bar sits in (always LONG or SHORT
                   in interior bars; '' at session-edge gaps)
trend_class      : LONG/SHORT/NEUTRAL (NEUTRAL within ±120s of any pivot)
atr_pts          : day's ATR(14) on 1m bars in price points
min_rev_ticks    : day's R threshold in ticks (4 × ATR_ticks)
target_split     : 'IS' or 'OOS' label
L1_5s_price_velocity_1b  : … (184 V2 feature columns total)
L1_5s_price_accel_1b     :
L2_5s_price_velocity_9   :
…
L3_1D_swing_noise_5      :
```

V2 features are computed in `core_v2/features.py` (not included here —
the project root has it). The 184 features span 8 timeframes (5s, 15s,
1m, 5m, 15m, 1h, 4h, 1D) × 3 layers (L1=1-bar, L2=rolling, L3=derived).

### Per-bar prediction caches

After training, `precompute_b1_b2_oos.py` etc. write per-bar prediction
parquets like `b1_proba_OOS_NT8.parquet`:

```
timestamp        : matches truth bar timestamp
day              : matches truth day
p_pivot_1m       : B1 P(pivot in next 1 min)
p_pivot_3m       : B1 P(pivot in next 3 min)
p_pivot_5m       : B1 P(pivot in next 5 min)
p_pivot_10m      : B1 P(pivot in next 10 min)
```

Same pattern for B4 (p_region_30s, _60s, _120s, _300s), B5 (p_phase_EARLY,
_MID, _LATE), B6 (p_PIVOT_TO_LONG_Km, p_PIVOT_TO_SHORT_Km), etc.

### Per-leg parquets/CSVs

`composite_entry_analyzer.csv` and `b7_leg_sizer_OOS.parquet`:
one row per leg, with entry features + truth amplitude.

`composite_forward_pass_hardened.csv`:
one row per leg with R-trigger entry/exit prices, realistic P&L,
B7 prediction, composite zone, B6 directional confidence.

## Data acquisition (if you don't have ATLAS_NT8)

The project's data pipeline (not included here) ingests NT8
NinjaTrader-recorded OHLC parquets and computes V2 features.

If you have raw NT8 CSV exports, you'd need:
1. The `core_v2/features.py` feature computation pipeline (not in this standalone)
2. The original `tools/build_zigzag_pivot_dataset.py` to generate truth labels
3. Sufficient memory for 184-feature, 5s-cadence parquets

The standalone provides everything needed to **consume** the data, but
not to **regenerate** the V2 features themselves. That requires the
parent project's `core_v2/` module.

## File sizes (for outside team capacity planning)

| File | Size |
|------|------|
| zigzag_pivot_dataset_IS_atr4.parquet | 108 MB |
| zigzag_pivot_dataset_NT8_OOS_atr4.parquet | 15 MB |
| pivot_probability_cloud.parquet | 6 MB |
| b6_proba_OOS_NT8.parquet | 4.4 MB |
| All other caches combined | ~25 MB |
| All 7 models combined | ~30 MB |
| **Total this folder** | ~200 MB |

Easily portable. Fits on email-attachment size limits in most cases.

## Key path conventions

The tools default to absolute paths like
`reports/findings/regret_oracle/...` (the project layout). To run from
this standalone folder, override with `--out` and `--in` args, e.g.:

```bash
python tools/composite_forward_pass_hardened.py \
    --truth caches/zigzag_pivot_dataset_NT8_OOS_atr4.parquet \
    --b7-pkl models/b7_leg_sizer.pkl \
    --cloud caches/pivot_probability_cloud.parquet \
    --b6 caches/b6_proba_OOS_NT8.parquet \
    --out caches/replay_forward_pass.csv
```

Each tool accepts CLI overrides for all file paths. See `--help` per tool.
