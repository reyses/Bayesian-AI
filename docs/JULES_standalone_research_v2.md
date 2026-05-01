# JULES — Standalone Research v2 Features Integration

**Status:** Spec doc, pre-implementation
**Date:** 2026-04-30
**Author:** Claude (instructed by reyses)
**Scope:** `tools/standalone_research.py` + `tools/research/data.py` (and possibly a new `tools/research/features_v2.py`)

## Goal

Make `tools/standalone_research.py` consume the v2 feature layout (185D / 8 TFs / per-day parquets in `DATA/ATLAS/FEATURES_5s_v2/`) instead of recomputing v1 SFE physics from raw OHLC every run. Primary path = read precomputed parquets; fallback = compute live via `core_v2.StatisticalFieldEngine`.

## Why

Every run currently calls `core/statistical_field_engine.StatisticalFieldEngine.batch_compute_states()` for each of 12 TFs. On the full ATLAS that's the dominant cost. The v2 features are already on disk in `DATA/ATLAS/FEATURES_5s_v2/`, built by `training/build_dataset_v2.py` using `core_v2.StatisticalFieldEngine`. The script should consume them directly.

## Design

### `--cache` becomes polymorphic

| Path shape | Behavior |
|---|---|
| Directory containing `L0/`, `L1_*/`, `L2_*/`, `L3_*/` | **v2 features mode** (this PR) |
| File ending in `.npz` (existing behavior) | Existing assembled-matrix cache |
| Anything else | Error with actionable message |

Detection function: `_detect_cache_type(path) -> Literal['v2_features', 'npz', 'unknown']`.

### Pipeline in v2 mode

```
1. base_tf default = '1m' (was '15m'); user can override via --base-tf
2. Load raw OHLC for base_tf from DATA/ATLAS/{base_tf}/  (still needed for I-MR, MFE/MAE, plots)
3. Steps 2–6 unchanged (price I-MR, regime detect, oracle MFE/MAE)
4. Step 7 REPLACED:
   a. For each day in [data_start, data_end]:
        - Try load FEATURES_5s_v2/{L0, L1_*, L2_*, L3_*}/YYYY_MM_DD.parquet
        - If any v2 dir is missing for that day → fall back to live compute via core_v2 SFE
        - Inner-join across all 25 dirs on `timestamp` → one 5s-cadence DataFrame per day
   b. Concat across days into single 5s feature DataFrame
   c. Reindex onto base_tf timestamps via searchsorted(side='right') - 1
5. Step 8: build (N_base_bars, 8, 23) stacked matrix → flatten to X (N, 185)
6. Cache assembled matrices to .npz on first run when --cache-out is given
```

### TF mapping

```python
TF_HIERARCHY_V2 = ['1D', '4h', '1h', '15m', '5m', '1m', '15s', '5s']
TF_LABELS_V2    = ['d0_1D', 'd1_4h', 'd2_1h', 'd3_15m', 'd4_5m', 'd5_1m', 'd6_15s', 'd7_5s']
```

Order: macro → micro (matches existing convention).

### Per-TF feature names (23 features × 8 TFs = 184 + 1 L0 = 185)

```python
FEATURE_NAMES_V2 = [
    # L1 — bar-level primitives (6)
    'price_velocity_1b', 'price_accel_1b',
    'vol_velocity_1b',   'vol_accel_1b',
    'bar_range', 'body',
    # L2 — rolling-window stats (9), TF-specific window suffix STRIPPED
    'price_velocity_w', 'price_accel_w',
    'vol_velocity_w',   'vol_accel_w',
    'price_mean_w', 'price_sigma_w',
    'vol_mean_w',   'vol_sigma_w',
    'vwap_w',
    # L3 — approved exceptions (8)
    'z_se',  'z_high', 'z_low',
    'SE_high', 'SE_low',
    'hurst', 'reversion_prob', 'swing_noise',
]
# L0 is a single global feature 'L0_time_of_day' — NOT in the per-TF stack;
# appended as the (N_TFS*N_FEATURES_PER_TF + 0)-th flat-column AFTER the stack.
```

The window suffixes (`_12`, `_15`, `_9`, `_5`, `_18`) are stripped at load time so cross-TF column names line up. The actual window per TF is documented in the spec but invisible to the consumer.

### Constants

```python
N_TFS_V2 = 8
N_FEATURES_PER_TF_V2 = 23
N_FLAT_FEATURES_V2 = N_TFS_V2 * N_FEATURES_PER_TF_V2   # 184
N_FLAT_WITH_MR_V2  = N_FLAT_FEATURES_V2 + 1            # 185 (current_MR appended)
```

Note: `L0_time_of_day` could be appended too. **Decision: include it.** Final flat shape: `(N, 186)` = 184 stack + 1 current_MR + 1 L0_time_of_day.

(Updated `N_FLAT_WITH_MR_V2 = 186`.)

### New helpers

In `tools/research/features_v2.py` (new file):

```python
def detect_v2_cache(path: str) -> bool:
    """True iff path is a v2 features directory (has L0/, L1_*/, L2_*/, L3_*/)."""

def load_v2_features_for_day(v2_dir: str, day_str: str) -> pd.DataFrame | None:
    """Load and inner-join all 25 layer-dir parquets for one day.
    Returns None if any required dir/file is missing."""

def compute_v2_features_live_for_day(atlas_root: str, day_str: str) -> pd.DataFrame:
    """Fallback: load raw OHLC for all 8 TFs for the day, run core_v2 SFE,
    return joined feature DataFrame matching precomputed schema."""

def load_v2_features(v2_dir: str, atlas_root: str, day_strs: list[str]) -> pd.DataFrame:
    """Top-level loader. For each day, try precomputed; fall back to live compute.
    Concat into single DataFrame at 5s cadence."""

def align_v2_to_base_tf(features_5s: pd.DataFrame, base_ts: np.ndarray) -> np.ndarray:
    """For each base_ts (e.g., 1m bar timestamps), find the most recent
    feature row at or before that timestamp (searchsorted side='right' - 1).
    Returns (N_base, K_features) array."""

def reshape_v2_to_stack(flat_features: np.ndarray) -> np.ndarray:
    """Reshape flat 184-feature row into (N, 8, 23) stacked matrix.
    Drops L0 (handled separately as a global)."""
```

### Changes to `tools/standalone_research.py`

1. **`--cache` argument help text** — document the polymorphic behavior.
2. **Cache load branch** ([standalone_research.py:135](tools/standalone_research.py#L135)) — call `_detect_cache_type()` first, dispatch to v2 path or npz path.
3. **Default `--base-tf`** — when v2 mode active and user didn't pass `--base-tf` explicitly, set to `'1m'`.
4. **`col_names` construction** ([standalone_research.py:144-156](tools/standalone_research.py#L144-L156) and [358-372](tools/standalone_research.py#L358-L372)) — use `TF_HIERARCHY_V2` / `TF_LABELS_V2` / `FEATURE_NAMES_V2` when in v2 mode, append `L0_time_of_day`.
5. **Hardcoded `range(12)`, `range(16)`, `(12, 16)`, `192`, `193`** — replace with constants. Sites:
   - [299](tools/standalone_research.py#L299): `np.zeros((12, 16))` → `np.zeros((N_TFS, N_FEATURES_PER_TF))`
   - [339](tools/standalone_research.py#L339): `# 193 features` → `N_FLAT_WITH_MR`
   - [354-355](tools/standalone_research.py#L354-L355): print strings — update
   - [416](tools/standalone_research.py#L416), [499](tools/standalone_research.py#L499): `for d in range(12)` → `range(N_TFS)`
   - [474](tools/standalone_research.py#L474), [654-656](tools/standalone_research.py#L654-L656): print strings — update
   - [2543](tools/standalone_research.py#L2543), [2547](tools/standalone_research.py#L2547), [2550](tools/standalone_research.py#L2550): feature-importance loop — `range(192)` → `range(N_FLAT_FEATURES)`
   - [2686](tools/standalone_research.py#L2686), [2706](tools/standalone_research.py#L2706): comments — update
6. **`FEATURE_NAMES.index('self_pid')` at line [5459](tools/standalone_research.py#L5459)** — swap to v2 equivalent. `'reversion_prob'` is the closest analog (probability-like signal). Rationale: `self_pid` was a PID-controller integral term measuring deviation persistence; `reversion_prob` is OU-first-passage probability of reverting from current z. Different math, similar signal: "how persistently is price away from equilibrium".
7. **Step 7 (load all TFs + compute SFE)** — gated. If v2 mode: skip SFE compute, call `load_v2_features()` instead.
8. **Step 8 (build context matrix)** — gated. If v2 mode: use `align_v2_to_base_tf()` + `reshape_v2_to_stack()` + flatten + concat current_MR + L0_time_of_day.

### What stays unchanged

- Steps 2–6 (price I-MR, regime detect, oracle MFE/MAE) — pure OHLC, no SFE.
- Plot functions (`plot_price_imr`, `plot_regime_summary`).
- Analyses A–S that consume features by `col_names` lookup. Most should keep working with renamed features; semantic interpretation will differ.
- `--full` flag and the SFE compute path (kept as fallback when `--cache` is unset).
- Existing `.npz` cache load/save semantics.

### What may break (triage list)

Analyses that hard-reference v1 feature names:
- **R** (CNN pattern detection): already broken (`tools.cnn_pattern_model` missing). Skipped.
- **Anywhere referencing `self_pid`**: only line 5459 found. Fixed via `reversion_prob` swap.
- **Print strings referencing "192"/"193"/"16"/"12"**: cosmetic; update to constants.
- **Hardcoded reshape `(12, 16)`**: line 2550 in feature-importance grouping. Update.

I expect 2–4 additional breakages to surface during the smoke test. Plan: run, capture stack trace, fix, re-run.

### Backward compatibility

- Running without `--cache` → unchanged v1 SFE path.
- Running with `--cache foo.npz` → unchanged.
- Running with `--cache DATA/ATLAS/FEATURES_5s_v2` → new v2 path.

The v1 constants (`TF_HIERARCHY`, `FEATURE_NAMES`) keep their old values; v2 constants live alongside. The script chooses which set to use at runtime based on cache type.

## Implementation order

1. Write `tools/research/features_v2.py` with the 6 helpers + unit-style smoke test (`if __name__ == '__main__': ...`).
2. Add v2 constants to `tools/research/data.py`.
3. Update `tools/standalone_research.py`:
   - Cache detection + dispatch
   - Default base_tf logic
   - col_names construction (both branches)
   - Hardcoded literal replacements
   - `self_pid` swap
   - Step 7/8 v2 branch
4. Smoke test on `DATA/ATLAS/FEATURES_5s_v2` over a few days.
5. Triage breakages; re-run.
6. Final smoke + screenshot of report tail.

## Rollback plan

Single git commit. To revert: `git revert HEAD`. v1 path stays intact, so even if v2 is broken, plain `python tools/standalone_research.py --data DATA/ATLAS --base-tf 15m` keeps working.

## Out of scope

- Updating the feature interpretations inside individual analyses (e.g., re-deriving what `swing_noise` "means" in regime classification). The script will produce numbers; their interpretation is a separate research question.
- Replacing the SFE compute fallback path entirely. Kept as safety hatch.
- Live compute via `core_v2` SFE for arbitrary date ranges that span unbuilt months. Falls back per-day; if a day is unbuilt, live-compute that day. No batched build.
- Updating `tools/standalone_research.py` to consume v2 OOS data (`DATA/ATLAS_OOS/FEATURES_5s_v2/` if it exists). User passes whatever directory they want.
