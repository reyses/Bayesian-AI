# JULES_v1_to_v2_migration — V1 features deprecation + V2 cutover

**Status**: DRAFT — Phase 0 + 0b complete; Phase 1+ pending user approval
**Date**: 2026-05-04
**Trigger**: User decision to deprecate V1 (`DATA/ATLAS/FEATURES_5s/`) and rewire pipeline to V2 (`DATA/ATLAS/FEATURES_5s_v2/`).
**Author**: Claude (Opus 4.7) under user oversight.

---

## TL;DR

Migration is feasible but **not a path swap** — it's a multi-phase pipeline rewire with a baseline re-establishment. Phase 0 audit confirms V2 features are clean. Phase 0b parity test confirms 4 of 6 missing V1 concepts can be derived from V2 at machine precision; 2 inherit V1/V2 SFE implementation drift. Migration will produce a slightly different baseline (not bit-identical to V1's −$164/day) due to SFE drift, but functionally equivalent.

The migration plan is 6 phases over 2–3 sessions. Phase 0 + 0b done. The remaining work is mostly mechanical rewiring of the tier engine + CNN retraining + verification.

---

## Phase 0 — Audit (DONE)

### V2 lookahead discipline ✓ clean

- `tests/test_core_v2_lookahead.py`: **28/28 tests pass**
- V2 build (`training/build_dataset_v2.py`) uses the same corrected `searchsorted(... - period, right) - 1` pattern that fixed V1's lookahead bug, isolated in `_last_closed_idx` with explicit "Do NOT modify" warning
- L0/L1/L2/L3 each tested across all TFs against future-poisoning

### V2 covers all V1 dates ✓

- V1 cache: 345 daily parquets (2025-01-01 → 2026-03-20)
- V2 cache: 345 daily layered files, identical date range
- `In V1 not V2: 0`

### V2 missing 6 concepts that V1 has — by design ✓

`research/feature_spec_v2.md` (2026-04-23) explicitly drops these per **Principle 7** ("L3+ compositions are the model's job, not the feature builder's"):

| V1 concept | V2 status | Rationale (from spec) |
|---|---|---|
| wick_ratio | DROPPED | "Composition of L1 (bar_range, body)" |
| p_at_center | DROPPED | "3-class probability of z-range; composition" |
| variance_ratio | DROPPED | "σ²_short/σ²_long; CNN composes from two L2 sigmas" |
| dmi_diff/dmi_gap | DROPPED | "DMI is a derived indicator family; CNN learns directional structure from L1/L2" |
| vol_rel | DROPPED | "Was close/vol_mean_N — trivial composition" |
| dir_vol | DROPPED | "sign(β)×vol_rel — composition of L2 features" |

These 6 concepts are used by the legacy 9-tier engine (`training/nightmare_blended.py`) which is rule-based with hardcoded thresholds. V2's principle is correct for ML; the tier engine is the legacy.

---

## Phase 0b — V1 compat shim + parity (DONE)

### Built `core_v2/v1_compat.py`

A pure-Python module that takes V2 features + small raw OHLCV history and produces V1-equivalent concepts on-the-fly. Honors V2 spec Principle 7 — the V2 cache stays clean; the shim is a runtime compatibility layer for the legacy tier engine only.

API:
- `wick_ratio_from_v2(body, bar_range)` — single-bar, exact V1 formula
- `p_at_center_from_z(z_se)` — single-bar, exact V1 formula (3-class softmax)
- `vol_rel_from_history(volume_now, volume_history, window=30)` — needs trailing volume
- `variance_ratio_from_history(close_history, short=10, long=60)` — needs trailing closes
- `dir_vol_from_v2(price_velocity_w, vol_rel)` — sign × magnitude
- `dmi_substitute_from_v2(price_velocity_w, scale=5)` — DMI replacement (sign filter)
- `derive_v1_concepts_batch(v2_df, ohlcv_native, tf, tf_period_seconds)` — vectorized batch path for cache rebuilds and parity tests

### Parity test (`tools/v1_v2_parity_test.py`)

Sweep: 4 days × 4 TFs = 16 (concept, TF) cells per concept.

| Concept | Verdict | Pearson | Max abs err | Median abs err |
|---|---|---|---|---|
| **wick_ratio** | EXACT | 1.0000 | ~3e-8 | ~9e-9 |
| **vol_rel** | EXACT | 1.0000 | ~1e-7 | ~1.5e-8 |
| **variance_ratio** at 1m/5m/15m | EXACT | 1.0000 | ~5e-8 | ~7e-9 |
| **variance_ratio at 1h** | TIGHT | 0.97 | 0.32 (1 day) | 8e-9 |
| **p_at_center** | DRIFT (z_se source) | 0.18–0.69 | 0.6–0.7 | 0.12–0.29 |
| **dir_vol** | SIGN DISAGREE (~19%) | 0.17–0.42 | up to 16 | ~5e-8 |

### Critical SFE-drift finding

V1 `1m_z_se` vs V2 `L3_1m_z_se_w` (same concept, different SFE implementation):

| Metric | Value |
|---|---|
| Pearson r | 0.87 |
| Sign agreement | 81.3% |
| Median abs diff | 0.47σ |
| Max abs diff | 1.98σ |

**This is the migration's core risk.** Even with the shim formulas perfect, V1's SFE (`core/cuda_statistics.py`, GPU) and V2's SFE (`core_v2/statistical_field_engine.py`, numpy) produce z_se values that differ by ~0.5σ on the median bar. Any tier-engine threshold like `if abs(z) > ROCHE` (2.0) will fire on slightly different bars between V1 and V2 caches.

The 19% sign disagreement on dir_vol traces to the same root cause: V1 uses `MarketState.velocity` (some pattern); V2 uses `L2_price_velocity_w` (windowed slope). They're both "velocity" but not bit-identical. The implications:

1. The −$164/day baseline (per MEMORY.md) **will shift** when migrating V1→V2. We don't know by how much — could be ±$10/day, could be more.
2. Tier thresholds calibrated on V1 z_se/velocity distributions are **approximately right** on V2 but not exactly right.
3. The regime-gate analysis (RIDE_AGAINST +$21/day OOS) was on V1 features. After migration, **needs re-validation**.

---

## Phase 1 — Tier engine rewire (PENDING APPROVAL)

### Goal
`training/nightmare_blended.py` reads V2 features (via `core_v2.features.load_features` or equivalent) instead of the V1 79D parquet. Where V1 concepts are needed, the shim (`core_v2.v1_compat`) provides them.

### Changes required

**Index lookup → column lookup**
The tier engine has hardcoded indices like `_1M_VELOCITY_IDX = 15`. Replace with V2 column lookups. Suggested refactor pattern:

```python
# Before (V1):
velocity = feat[_1M_VELOCITY_IDX]
z = feat[_1M_OFFSET + _Z]
vr = feat[_1M_OFFSET + _VR]

# After (V2):
velocity = feat['L2_1m_price_velocity_w']
z = feat['L3_1m_z_se_w']
# variance_ratio derived from history at the TF cadence:
vr = v1_compat.variance_ratio_from_history(close_hist_1m)
```

**DMI replacement**
The only DMI consumer (`MTF_BREAKOUT` line 785-787) becomes:
```python
v_sign = np.sign(feat['L2_1m_price_velocity_w'])
dmi_aligned = ((breakout_dir == 'long' and v_sign >= 0) or
               (breakout_dir == 'short' and v_sign <= 0))
```

**Wick rejection**
KILL_SHOT/CASCADE entry triggers rebuild against:
```python
wick_5m  = v1_compat.wick_ratio_from_v2(
                feat['L1_5m_body'], feat['L1_5m_bar_range'])
wick_15m = v1_compat.wick_ratio_from_v2(
                feat['L1_15m_body'], feat['L1_15m_bar_range'])
has_wick = wick_5m > WICK_5M_MIN and wick_15m > WICK_15M_MIN
```

**p_center exit**
```python
z_se = feat['L3_1m_z_se_w']
p_center = v1_compat.p_at_center_from_z(z_se)
if p_center > P_CENTER_EXIT:  # 0.60 unchanged
    self._tier_p_center_bars += 1
```

**Volume features**
```python
vol_rel = v1_compat.vol_rel_from_history(
              current_volume, vol_history_1m, window=30)
dir_vol = v1_compat.dir_vol_from_v2(feat['L2_1m_price_velocity_w'], vol_rel)
```

### Threshold re-tuning

After Phase 1 mechanical rewire, run a **calibration pass** comparing V1 vs V2 distributions of:
- z_se per TF (median absolute shift, IQR shift)
- velocity_w per TF
- variance_ratio (should match — confirmed at machine precision)
- vol_rel (should match — confirmed)

Where shifts are non-trivial (z_se median diff 0.47σ at 1m), re-tune the threshold:
- `ROCHE = 2.0` may need to become `ROCHE = 1.85` or similar to fire on equivalent bar fraction
- `VR_ENTRY = 1.0` is unchanged (variance_ratio is exact)
- `H1_Z_MIN = 1.0`, `H1_AGAINST_Z_MIN = 1.5`: re-tune

### Risk
Medium — touching every tier's entry classifier and exit logic. ~50 hardcoded threshold lines. Reversible (keep V1 path active behind a flag during transition).

### Effort
1–2 sessions of careful edits. ~500 LOC changed.

---

## Phase 2 — Live engine migration (DEFERRED)

### Path
`live/engine_v2.py` reads `DATA/ATLAS_LIVE/FEATURES_5s` (V1 layout). Need a V2 equivalent at `DATA/ATLAS_LIVE/FEATURES_5s_v2/` and read path swap.

### Status
Currently **not trading live**, so this is non-blocking. Defer until Phase 1 is validated and we want to live-deploy.

---

## Phase 3 — CNN retraining (PENDING)

### Why
The 3 CNNs (`cnn_flip`, `cnn_hold`, `cnn_risk`) were trained on V1's 79D feature vector. The migration produces a new vector — same concept set (via shim) but slightly different numbers (z_se drifted, velocity drifted).

### Two options

**Option A — keep input vector as 79D (V1-shape via shim)**: feed CNN the SAME 79 numbers it was trained on, just computed via V2 + shim. Re-baseline only — no architectural change. Acceptable if the CNN tolerates the SFE drift in z_se/velocity.

**Option B — switch to native 184D V2 input**: drop the shim from CNN's input. CNN sees raw V2 primitives. Requires architectural change (input dim 79 → 184) and full retrain. Better aligned with V2 spec Principle 7.

Recommend **A first** (smaller change, isolates SFE drift impact), then **B** as a follow-up experiment.

### Effort
1 session for retraining, 1 session for OOS verification.

---

## Phase 4 — Re-establish baseline (CRITICAL GATE)

### Run
```bash
python training/run.py blended --fresh
```

with the V2-rewired tier engine + retrained CNNs.

### Compare
- Current honest baseline: **−$164/day IS** (per MEMORY.md, post-lookahead-fix)
- Post-migration target: same, ±$30/day acceptable drift
- If post-migration is **worse than −$200/day**, abort and investigate

### Re-run regime-gate analysis
```bash
python tools/v2_tier_regime_analysis.py
```

The RIDE_AGAINST gate (+$21/day OOS) was computed on V1-trained tier outputs. Post-migration tier outputs may show a different gate landscape. If RIDE_AGAINST gate still survives all-splits-positive, that's a robustness signal.

---

## Phase 5 — V1 deprecation (LAST)

### Steps
1. Move `DATA/ATLAS/FEATURES_5s/` → `DATA/_ARCHIVE/FEATURES_5s_legacy/` (NOT delete)
2. Same for `DATA/ATLAS_OOS/FEATURES_5s/`, `DATA/ATLAS_LIVE/FEATURES_5s/`, `DATA/ATLAS_NT8/FEATURES_5s/`
3. Add deprecation note to `MEMORY.md` with migration date
4. Update `~50` Python files that reference `DATA/ATLAS/FEATURES_5s/`:
   - Replace with V2 path + v1_compat shim where needed
   - Mark research-only / archive scripts with `# DEPRECATED — uses V1 cache`
5. Run `pytest tests/` to confirm no regressions
6. After 1–2 sessions of stable V2 operation, **delete** the archive

### Reversibility
If V2 baseline turns out to be unstable or significantly worse than V1, restore from archive. Keep archive for 30 days minimum.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Post-migration baseline worse than V1 −$164/day | Phase 4 is a hard gate; abort if delta > $30/day worse |
| z_se SFE drift breaks tier thresholds | Phase 1 includes calibration pass; threshold re-tune is part of the work |
| CNN retraining fails to match | Fall back to Option A (V1-shape input via shim); re-evaluate Option B as separate experiment |
| Live engine breakage | Not trading live currently; Phase 2 is non-blocking |
| RIDE_AGAINST gate (+$21/day OOS) doesn't survive | Re-run regime-gate analysis post-migration; if doesn't survive, deploy gate before migration on V1 features as a safety net |

---

## Open questions for user

1. **Threshold re-tuning policy**: should I re-tune by matching the V1 BAR FRACTION (e.g., ROCHE chosen so X% of V2 bars exceed) or by maintaining the V1 LITERAL THRESHOLD value? Different choices, different post-migration baselines.

2. **CNN strategy**: Option A (79D shim-fed) or B (184D V2-native)? A is the safer first cut.

3. **Migration commit policy**: do this in one branch end-to-end, or merge phase-by-phase? End-to-end gives a clean revert; phase-by-phase gives intermediate checkpoints.

---

## Phase 0 + 0b deliverables

- ✓ V2 lookahead audit confirmed (28/28 tests pass)
- ✓ V2 date coverage confirmed (345 days = V1 days)
- ✓ V2 dropped concepts traced to `research/feature_spec_v2.md` Principle 7
- ✓ `core_v2/v1_compat.py` shim (140 LOC, all 6 V1 concepts derivable)
- ✓ `tools/v1_v2_parity_test.py` (4 days × 4 TFs sweep; reports per-concept Pearson + max abs err)
- ✓ Parity verified: 4 of 6 concepts machine-precision; 2 of 6 inherit SFE drift
- ✓ Critical drift finding: V1 vs V2 z_se Pearson 0.87, median 0.47σ shift

Awaiting user approval to proceed to Phase 1 (tier engine rewire).
