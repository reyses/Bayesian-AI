# SPEC — Schema Change: One-Sided Wicks + TF-Hierarchy Anti-Scramble

**Audience:** IDE AI (implementer).
**Trigger:** Adds two L1 primitives (`upper_wick`, `lower_wick`) AND hardens grid assembly against TF/layer scrambling. Both require a feature rebuild, so do them in ONE pass.
**Authoritative sources:** `features_v2.py` (`TF_HIERARCHY_V2`, `FEATURE_NAMES_V2`, `reshape_v2_to_stack`). Storage is per-(layer,TF) parquet folders (confirmed): `DATA/ATLAS/FEATURES_5s_v2/{L0, L1_<TF>, L2_<TF>, L3_<TF>}/YYYY_MM_DD.parquet`.

---

## 0. Push-Back Protocol — BEFORE CODING
Confirm or dispute in writing; if false, stop and report.
1. **Find where `ticker._v2_matrix` is assembled** (not in the files reviewed). State the exact construction. The grid `(N,8,23)` is only correct if columns are placed by NAME into `(tf_idx, feat_idx)` per `TF_HIERARCHY_V2` × `FEATURE_NAMES_V2` (as `reshape_v2_to_stack` does). If `_v2_matrix` is instead built by globbing the layer-family folders or `pd.concat` in directory order, the grid is **scrambled** (layer-major AND lexicographic-TF order, not TF-major hierarchy order) and all prior CNN training is on meaningless channel geometry. Report which it is.
2. On-disk folders sort lexicographically (`1D,1h,1m,4h,5m,5s,15m,15s`), which is NOT the hierarchy order (`1D,4h,1h,15m,5m,1m,15s,5s`). Confirm the assembler does not rely on folder/glob order.
3. The CNN ingests raw features and learns order-agnostically, so this is silent — there is no error to catch at runtime. Confirm you will add the §1 assertion rather than trust the pipeline.

---

## 1. TF-Hierarchy Anti-Scramble (CRITICAL — do first)
The grid's channel axis MUST be `TF_HIERARCHY_V2 = ['1D','4h','1h','15m','5m','1m','15s','5s']` (index 0→7) and the feature axis MUST be `FEATURE_NAMES_V2` order. This ordering is semantic: the CNN's channels are timeframes and its conv windows assume that structure.

**Requirement:**
- Assemble `_v2_matrix` / the grid **exclusively by name-keyed placement** into `stack[:, tf_idx, feat_idx]`, using the canonical lists — i.e. via `reshape_v2_to_stack` or identical logic. Never by `glob`, `concat`, or any directory/column ordering.
- Add a **provenance assertion** at assembly time: build the expected column-name → `(tf_idx, feat_idx)` map from the canonical lists, and assert every grid cell was filled from the correctly-named source column (e.g. assert the `col_map` covers all `8 × N_FEATURES_PER_TF` slots and that no slot is silently zero-filled because a name was missing). Fail loudly on any missing/extra column.
- Persist `TF_HIERARCHY_V2` and `FEATURE_NAMES_V2` (with the new wick entries) into a schema-version stamp checked at load (the spec already mandates `schema_version` per family file — extend the check to the channel/feature order).

**Why:** the same name-keyed discipline also protects the future delta layer (Option B §1) and the convergence projection axes — both select features by name and break if the grid is reordered.

---

## 2. Add One-Sided Wick Primitives (L1)
Two new **raw** L1 features per TF, computed from a single bar's OHLC (window-free, like `bar_range`/`body`):
- `upper_wick = high − max(open, close)`  (≥ 0)
- `lower_wick = min(open, close) − low`   (≥ 0)

**Justification (clears Principle 7):** one-sided wicks are NOT derivable from existing primitives. `bar_range` + `body` fix the range and body size but not where O/C sit within the range — three different wick splits share identical `bar_range`/`body`. This is a missing primitive, not a re-derivable composition (unlike the dropped two-sided `wick_ratio`).

**Raw, not ratio (Principle 6):** store as raw price differences, same as `bar_range`/`body`. Do NOT pre-divide by `bar_range` — the CNN can form `upper_wick / bar_range` itself from co-located features, and pre-dividing would bake in the denominator anchor (Principle 7 violation) and normalize (Principle 6 violation).

**Orthogonality:** instantaneous directional bar shape — distinct axis from windowed `SE_high`/`SE_low` (regime dispersion) and `swing_noise` (windowed giveback). Directly useful for exit (long upper-wick while long = rejection) and for the delta layer (wick appearing since entry = exhaustion).

---

## 3. Schema Constant Updates (`features_v2.py`)
Insert the two features at the END of the L1 block in `FEATURE_NAMES_V2` (keeps layer grouping intact; all downstream is name-keyed so position shift is safe):
```
# L1 (now 8) — bar primitives
'price_velocity_1b','price_accel_1b','vol_velocity_1b','vol_accel_1b',
'bar_range','body','upper_wick','lower_wick',
# L2 (9) ... L3 (8) ... unchanged
```
- `N_FEATURES_PER_TF_V2`: 23 → **25**
- `N_FLAT_FEATURES_V2`: 184 → **200**
- Total feature count: 185 → **201** (200 grid + 1 L0)
- Update `extract_per_tf_block` / `reshape_v2_to_stack` column-name logic: `upper_wick`/`lower_wick` are L1, bare-name (like `bar_range`/`body`) → `col = f'L1_{tf}_{fname}'`.

---

## 4. Builder Updates
- In the SFE L1 computation, add `upper_wick`, `lower_wick` to each `L1_<TF>` parquet (alongside `bar_range`, `body`).
- Regenerate all `L1_<TF>/*.parquet` (L2/L3/L0 unchanged — the layer-family decoupling means only L1 rebuilds).
- Bump `schema_version` on the L1 family files.

---

## 5. Network Updates (`network_research_A.py`)
Input feature axis 23 → 25:
- CNN width: `25 → 23 → 21` after the two 3×3 convs ⇒ `cnn_out_dim = 64 * 21 = 1344`
- `lstm_input_dim = 1344 + 3 = 1347`
- Update the `features=23` default → `features=25`.
- Old checkpoints are shape-incompatible — fail loudly on load mismatch (no silent `strict=False`).

---

## 6. Rebuild & Verification
1. Rebuild L1 parquets; confirm 8 new-schema L1 columns present per TF.
2. Run `validate_sfe_parity` (or equivalent) to confirm live-compute vs parquet match for the new features.
3. Assemble one day's grid and assert: shape `(N,8,25)`, channel 0 = 1D, feature index of `z_se` = 17 (was 15; +2 from the two L1 inserts), `upper_wick` = 6, `lower_wick` = 7. Paste the realized index map.
4. Confirm the §1 provenance assertion is in the assembly path and fires on a deliberately renamed column (quick negative test).

---

## 7. Acceptance — report all
1. `_v2_matrix` assembly is name-keyed TF-major (per §0.1); paste the construction and the provenance assertion.
2. `FEATURE_NAMES_V2` updated; counts 25 / 200 / 201; `upper_wick`/`lower_wick` raw, no normalization.
3. L1 parquets rebuilt + `schema_version` bumped; parity check passes.
4. Network dims 1344 / 1347 / features=25; old checkpoints rejected on shape mismatch.
5. Realized grid index map pasted; `z_se` now at feature index 17; channel order = `TF_HIERARCHY_V2`.
