# training_v2 — V2 features migration notes

**Created**: 2026-05-04
**Sibling of**: `training/` (the V1-features version, KEPT INTACT for fallback/comparison)
**Status**: Phase 1 complete (path swap + threshold calibration + V2→V1 compat cache builder). End-to-end pipeline runnable but un-validated.

---

## What this folder is

A copy of `training/` rewired to consume V2 features (`DATA/ATLAS/FEATURES_5s_v2/`) instead of V1 features (`DATA/ATLAS/FEATURES_5s/`). The legacy rule-based 9-tier engine is preserved — V2 features are adapted to the V1 91D-shape vector at cache build time so the engine code path is identical.

The cleaner long-term architecture is a V2-native engine (read V2 columns directly, no compat cache). That's a follow-up refactor.

## Architecture

```
DATA/ATLAS/FEATURES_5s_v2/        ← V2 layered cache (L0/L1/L2/L3 × 8 TFs)
                ↓
tools/build_v2_to_v1_compat_cache.py
                ↓
DATA/ATLAS/FEATURES_5s_v2_as_v1/  ← V1-shape per-day parquets (compat cache)
                ↓
training_v2/run.py blended --fresh
                ↓
training_v2/output/...             ← results
```

## How to use

### 1. Build the V1-shape compat cache (one-time, ~30 min full sweep)

```bash
python tools/build_v2_to_v1_compat_cache.py --fresh
```

Reads V2 layered features + raw OHLCV history → writes 91D V1-shape parquets to `DATA/ATLAS/FEATURES_5s_v2_as_v1/`. One file per day.

The shim (`core_v2/v1_compat.py`) derives V1 concepts the V2 cache doesn't store:
- `wick_ratio` from V2 `body / bar_range`
- `p_at_center` from V2 `z_se_w` (3-class softmax)
- `variance_ratio` from raw close history at TF cadence (V1 windows: 10 / 60)
- `vol_rel` from raw volume history (V1 window: 30)
- `dir_vol` = sign(`L2_velocity_w`) * vol_rel
- `dmi_substitute` = sign(`L2_velocity_w`) * 5 (V2 has no DMI)

Unit conversion: V1 stores `bar_range = (high-low)/TICK` in ticks; V2 stores in price units. Compat cache multiplies V2 bar_range by `1/TICK = 4` to match.

### 2. Run training on V2 features

```bash
python training_v2/run.py blended --fresh
```

Same 7-phase pipeline as V1 (NMP physics → regret → labels → CNN flip → CNN hold/risk → forward pass IS+OOS+NT8). Outputs go to `training_v2/output/...` so V1 baseline is preserved.

## What's calibrated

The V2 SFE produces slightly different `z_se` and `velocity` distributions than V1's cuda SFE. Same concepts, different implementations. V1 thresholds applied literally to V2 would gut the engine (5m NMP fire rate drops 88% if ROCHE=2.0 stays literal).

`training_v2/nightmare_blended.py` thresholds calibrated to match V1 bar-firing fractions:

| Constant | V1 value | V2 calibrated value |
|---|---:|---:|
| `ROCHE` | 2.0 | **1.87** |
| `H1_Z_MIN` | 1.0 | **0.87** |
| `H1_AGAINST_Z_MIN` | 1.5 | **1.40** |
| `EXHAUST_Z_MIN` | 1.4 | **1.32** |
| `MTF_Z_MIN` | 1.4 | **1.32** |
| `VELOCITY_THRESHOLD` | 50.0 | **9.95** (5× smaller) |
| `FREIGHT_TRAIN_THRESHOLD` | 100.0 | **20.0** (5× smaller) |
| `MTF_1M_VEL_ALIVE` | 10.0 | **2.47** |
| `MTF_5M_VEL_MIN` | 30.0 | **8.44** |

UNCHANGED (machine-precision parity via shim):
- All VR thresholds (`VR_ENTRY=1.0`, `FREIGHT_TRAIN_VR_MAX=0.85`, etc.)
- `WICK_5M_MIN=0.83`, `WICK_15M_MIN=0.77`
- `MTF_VOL_MIN=2.0`, `ABSORB_VOL_PERSIST_MAX=1.5`

Full calibration table: `reports/findings/v1_v2_threshold_calibration/calibration_table.md`.

## What's parity-checked vs not

| V1 concept | Parity vs V1 cache |
|---|---|
| wick_ratio | EXACT (machine precision, Pearson 1.0000) |
| vol_rel | EXACT (machine precision) |
| variance_ratio at 1m/5m/15m | EXACT |
| variance_ratio at 1h | TIGHT (Pearson 0.97, one-day max abs err 0.32 — needs more history) |
| bar_range (after tick conversion) | EXACT |
| **z_se / z_high / z_low** | **DRIFT** — V1 cuda vs V2 numpy SFE differ. Pearson ~0.87, median 0.47σ shift. Same concept, different implementation. |
| **velocity / acceleration** | **DRIFT** — V1 OLS slope vs V2 MA-derivative. Different formula entirely. |
| p_at_center | DRIFT (inherits z_se drift; formula identical) |
| dir_vol | ~19% sign disagreement (inherits velocity sign drift) |
| dmi_diff | SUBSTITUTE only (V2 has no DMI) |
| reversion_prob | Direct V2 mapping (formula similar) |
| hurst | Direct V2 mapping |

## Expected outcome

The −$164/day baseline (V1, post-lookahead-fix) and tonight's −$19/day (V1 with cnn_hold/risk skipped) **will shift** when running on V2. Could be ±$30/day. Direction unknown until run.

If V2 baseline is meaningfully **better** than V1, V2 is a clear win.
If V2 baseline is **comparable** (±$30), proceed to V2-native engine refactor.
If V2 baseline is meaningfully **worse**, investigate which calibration was wrong (likely a velocity threshold) before declaring V2 inferior.

## What's NOT done

1. **CNN retraining**: the `training_v2/cnn_*.py` modules write to `training_v2/output/nn/`. They'll retrain on V1-shape data (read from compat cache), so this part is unchanged from V1. CNN training will use the calibrated thresholds at inference time.
2. **V2-native engine** (read V2 columns directly without compat cache): deferred. The compat cache approach gets us a working V2 pipeline; a clean refactor is Phase 4 of the migration plan in `docs/JULES_v1_to_v2_migration.md`.
3. **Live engine** (`live/engine_v2.py`): NOT migrated. We're not trading live. Phase 2 of the migration plan is deferred.
4. **NT8 features path**: `NT8_FEATURES_5S = 'DATA/ATLAS_NT8/FEATURES_5s'` still points at V1 NT8 features. NT8 doesn't have V2 features yet. The OOS-NT8 step in Phase 7 will use V1.
5. **Bootstrap CI on the post-migration baseline delta** vs V1: needed before declaring V2 superior/inferior.

## Run plan

```bash
# Step 1 (one-time, ~30 min): build compat cache for all 345 days
python tools/build_v2_to_v1_compat_cache.py --fresh

# Step 2: train + forward pass on V2 features
python training_v2/run.py blended --fresh

# Step 3: compare to V1 baseline
diff <(cat training/output/blended/oos_daily.csv) <(cat training_v2/output/blended/oos_daily.csv)
# Look at $/day, day_WR, daily_pnl distribution
```

## Rollback

If anything goes sideways: `training/` is untouched. V1 path still works:
```bash
python training/run.py blended --fresh
```

V2 cache (`DATA/ATLAS/FEATURES_5s_v2/`) is unmodified — only the compat cache (`DATA/ATLAS/FEATURES_5s_v2_as_v1/`) is generated, can be deleted at any time.
