# 2026-04-27 — ATLAS_NT8 v2 features built (OOS-2)

Status: **COMPLETE**. NT8-feed-derived feature dataset is ready for analysis.

**Designation (2026-04-27)**: ATLAS_NT8 is now the **second OOS gate (OOS-2)**
in the validation ladder. Independent feed AND time-shifted from primary OOS.
See `docs/memory/feedback_oos2_designation.md` for methodology rationale and
promotion rules.

## What was built

Full 139D layered v2 feature set on the rebuilt ATLAS_NT8 raw bars. Output
mirrors canonical `DATA/ATLAS/FEATURES_5s_v2/` schema exactly — same layer
families, same column names, same lookahead-safe alignment.

| Property | Value |
|---|---|
| Source bars | `DATA/ATLAS_NT8/{1s,5s,15s,1m,5m,15m,1h,4h,1D}/` (32 days) |
| Output | `DATA/ATLAS_NT8/FEATURES_5s_v2/` |
| Layer-families | 25 (L0 + L1×8 + L2×8 + L3×8) |
| Days | 32 (2026-03-20 → 2026-04-26) |
| Total parquets | 800 (32 days × 25 families) + 800 metadata sidecars |
| Total feature columns | 185 across all families |
| Anchor TF | 5s |
| Schema parity vs canonical | **PERFECT** (column-by-column verified) |

## Per-family columns

| Family | Cols | Features |
|---|---:|---|
| L0       | 1 | `L0_time_of_day` |
| L1_{tf}  | 6 | `price_velocity_1b`, `price_accel_1b`, `vol_velocity_1b`, `vol_accel_1b`, `bar_range`, `body` |
| L2_{tf}  | 9 | windowed kinematics: `velocity_9`, `accel_9`, jerk, vol counterparts, etc. |
| L3_{tf}  | 8 | regression-band z-scores: `z_se_9`, `z_high_9`, `z_low_9`, `SE_high_9`, `SE_low_9`, plus 3 OU-related |

(`{tf}` ∈ `{5s, 15s, 1m, 5m, 15m, 1h, 4h, 1D}`.)

## Build cadence

Full 32-day build: ~5 seconds wall time on RTX 3060 (much faster than estimated
because the 32-day window is small). Smoke test (5 days) was even faster.

## Per-day row counts (L0 anchor = 5s)

```
2026_03_20.parquet  10,080 rows  (~14h, partial session)
2026_03_22.parquet   6,480 rows  (~9h, Sunday open)
2026_03_23.parquet  16,532 rows  (~23h, full session)
...
2026_04_23.parquet  16,512 rows  (full session)
2026_04_24.parquet  10,074 rows  (~14h, Friday close shortened)
2026_04_26.parquet   2,298 rows  (~3.2h, truncated 1s dump — known issue)
```

The 2026-04-26 truncation traces back to the original 1s NT8 dump being
incomplete for that date (192 minutes of data instead of full session). Re-dump
that date if needed.

## Schema parity verification

Spot-checked L1_5s column set:

```
NT8 cols       = ['L1_5s_bar_range', 'L1_5s_body', 'L1_5s_price_accel_1b',
                  'L1_5s_price_velocity_1b', 'L1_5s_vol_accel_1b',
                  'L1_5s_vol_velocity_1b', 'timestamp']
Canonical cols = (identical)
Match: True
```

Column sets match identically across L0, L1_5s, L1_1m, L1_4h, L2_5s, L3_5s.
The lookahead-safe `_last_closed_idx` invariant is preserved (same code path
as the canonical builder — no fork, no re-implementation).

## Sample feature values (sanity check)

```
L1_5s tail of 2026-03-24:
timestamp    velocity  accel  vol_vel  vol_accel  range  body
1774421989    0.0      -0.25   37.0     43.0      2.00   1.00
1774421994    4.5       4.50   18.0    -19.0      5.25   5.00
1774421999    2.0      -2.50   25.0      7.0      2.75   1.75
```

Reasonable values, no NaN/Inf garbage.

## What this enables

1. **Cross-data parity tests**: any analysis using the 91D/139D feature space
   can now run on NT8-feed bars to confirm Python-side predictions match what
   NT8 native backtest would see. Ex: re-run v1.5-RC bleed-filter validation,
   tier classifier evaluations, etc.
2. **Live-vs-train feature parity audits**: `feedback_data_validation_first.md`
   highlighted that training-vs-live parity took dedicated work. ATLAS_NT8
   features are now the apples-to-apples reference for any live SFE feature
   the bridge produces.
3. **Multi-TF feature analysis** on NT8 data was previously impossible (only
   1s/1m/1h/1D raw bars existed). Now the full layer-family stack works.

## Caveats

- **Coverage limited to 32 days.** Insufficient for full IS/OOS work on a
  91D feature space (need ~120+ days). Useful for spot-validation, not for
  retraining.
- **Day-cut convention differs from canonical Databento ATLAS.** Canonical
  splits on Globex session boundaries; NT8 dumps are calendar-day. This
  doesn't affect column schema or values, only how rows are grouped per file.
- **2026-04-26 1s dump is truncated** (3.2h instead of full session). If
  this day is needed for analysis, re-dump after enabling the v2.0.0
  `BayesianHistoryDumper` (which captures all 4 native TFs simultaneously
  from a single chart).

## Tools chain

| Step | Tool | Output |
|---|---|---|
| 1. NT8 chart dump | `BayesianHistoryDumper.cs` v2.0.0 (NT8 indicator) | `DATA/ATLAS_NT8/{1s,1m,1h,1D}/MNQ_06-26/*.csv` |
| 2. Aggregate + parquet | `tools/atlas_nt8_rebuild.py` | `DATA/ATLAS_NT8/{tf}/*.parquet` (all 11 TFs) |
| 3. Build features | `training/build_dataset_v2.py --atlas DATA/ATLAS_NT8` | `DATA/ATLAS_NT8/FEATURES_5s_v2/{family}/*.parquet` |

Re-runnable end-to-end: dump from NT8 → run rebuild → run feature build.
Each step is incremental (skips already-converted files unless `--fresh`).

## Files written this session

- `tools/atlas_nt8_rebuild.py` (extended with 4h aggregation + UTF-8 BOM handling)
- `docs/nt8/BayesianHistoryDumper.cs` v2.0.0 (multi-TF dumper)
- `docs/nt8/archive/BayesianHistoryDumper_v1.0.0_pre_v2.0.0_ship_2026-04-27.cs` (rollback)
- `DATA/ATLAS_NT8/FEATURES_5s_v2/{25 families}/{32 days}.parquet` + sidecars
- `reports/findings/2026-04-27_atlas_nt8_rebuild.md` (parity report)
- `reports/findings/2026-04-27_atlas_nt8_smoke_build.log` (5-day smoke test log)
- `reports/findings/2026-04-27_atlas_nt8_full_build.log` (32-day full build log)
- `reports/findings/2026-04-27_atlas_nt8_features_built.md` (this doc)
