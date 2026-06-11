# SPEC: core_v2 FPS / Ledger / Engine — Correctness Fixes

**Scope:** `core_v2/FPS/forward_pass_system.py`, `core_v2/FPS/forward_pass_system_vram.py`, `core_v2/ledger.py`, `core_v2/exits.py`, `core_v2/strategy_engine.py`, docstrings across `core_v2/`
**Priority order:** Fixes 1–3 are P0 and invalidate any training/eval runs that touched the affected paths. Fix 4–5 are P1. Fix 6 is hygiene batch.
**Hard invariant (repo-wide):** Segment labels (`status`, `volatility_tier`, segment membership, betas) from `artifacts/stage*_segments_*.json` are NON-CAUSAL and must never enter a live feature vector, network input, or training target. This is already documented in `research/Regression segments/project.md` and the stage1/stage2 module banners. Fix 1 enforces it in code.

---

## Fix 1 (P0) — Remove non-causal segment risk from ForwardPassSystem

**Problem:** `ForwardPassSystem.__init__` loads `artifacts/stage2_segments_{day}.json` and builds `self._segment_risk` (N×4: is_pristine, is_chaos, vol_norm, err_norm) per bar. Segment tiers/statuses/boundaries are in-sample fits over each segment's own future bars. If this array reaches any network input or decision logic, it is lookahead leakage of the exact class the project banner forbids. Three additional bugs exist inside the block, confirming it was never validated:

1. Reads `seg.get('error_band_width', 0.0)` — the actual key is `error_band_used`, so `err_norm` is always 0.
2. Indexes `self._segment_risk` (feature-row space, post-NaN-drop on the segment side vs raw on the FPS side) with `raw_start_idx`/`raw_end_idx` — raw OHLCV coordinates. The FPS feature matrix is row-aligned to the features parquet, not to the segment pipeline's NaN-dropped rows; neither `start_idx` nor `raw_idx` is guaranteed to align with FPS rows.
3. Fills `self._segment_risk[s_idx:e_idx_clamp+1]` — inclusive end, violating the half-open `[start, end)` convention standardized in SPEC_SEGMENT_ANALYTICS_FIXES.md.

**Change (default path — DELETE):**
1. Remove the entire `_segment_risk` block from `ForwardPassSystem.__init__` (the `seg_file` load through the fill loop).
2. Grep the repo for `_segment_risk` consumers. If any exist (network input assembly, BarState extension, train_gpu_research_A), remove those reads too and list them in the PR description — every training run that consumed them is contaminated and must be flagged.
3. Add a comment at the deletion site: `# Segment labels are non-causal (see research/Regression segments/project.md). Never load stage*_segments_* into the forward pass.`

**Alternative path (ONLY if user explicitly wants a diagnostic overlay):** gate behind `__init__(..., load_diagnostic_segments: bool = False)`, store under `self._diagnostic_segment_risk`, never expose it on BarState, and fix the three sub-bugs (key name, coordinate space via raw→feature-row remap, half-open fill). Default to the delete path unless instructed otherwise.

**Acceptance:** `grep -rn "_segment_risk\|stage2_segments" core_v2/` returns nothing (or only the gated diagnostic variant). FPS instantiates and iterates a reference day identically to before in all yielded BarState fields.

---

## Fix 2 (P0) — VRAM ticker must use assemble_v2_grid (anti-scramble)

**Problem:** `forward_pass_system_vram.py` builds its CNN grid as:

```python
grid_all = v2_matrix[:, 1:185].reshape(-1, 8, 23)
```

Three independent corruptions: (a) N_FEATURES is 201, so columns 185–200 are silently dropped; (b) 184 columns reshaped to 8×23 means the feature axis is 23 wide instead of the contract's 25 (`N_FEATURES_PER_TF_V2`), shearing features across TF channels; (c) the flat layout is ascending TF order (5s first) while the CNN channel contract is `TF_HIERARCHY_V2` descending (1D first). This is precisely the TF-channel-scrambling bug class that `assemble_v2_grid` in `core_v2/features.py` exists to prevent — and which the non-VRAM FPS correctly calls. Any model trained or evaluated through the VRAM ticker consumed scrambled inputs.

**Change:**
1. Replace the manual slice/reshape with:

```python
from core_v2.features import assemble_v2_grid
grid_all = assemble_v2_grid(self._v2_matrix)          # (N, 8, 25), 1D-first
self._grid_tensor = torch.nan_to_num(
    torch.tensor(grid_all, dtype=torch.float32, device=device), nan=0.0)
```

2. The L0 tensor: keep `[L0_time_of_day, tod_norm, day_norm]` only if a consumer (network_research_A) actually expects 3 L0 channels — verify against the network's input spec before changing shape. If the network was trained against the scrambled grid, shapes downstream of this fix WILL break loudly; that is desired. Do not paper over shape errors with reshapes.
3. **Deduplicate the class.** Two files each defining `ForwardPassSystem` with different yield signatures is the divergent-engine antipattern. Restructure: `forward_pass_system_vram.py` defines `VRAMForwardPassSystem(ForwardPassSystem)` that (a) calls `super().__init__`, (b) adds the tensor precompute, (c) overrides `__iter__` to yield `(state, l0_slice, grid_slice)`. Same for the MultiDay wrapper. Delete the ~200 duplicated lines.
4. Fix `day_norms`: `dt_series.weekday / 4.0` exceeds 1.0 for Saturday(5)/Sunday(6) bars (futures Sunday session exists). Use `/ 6.0`, or clamp, and document which.

**Contamination flag:** Identify all checkpoints trained through the VRAM path (grep training scripts for `forward_pass_system_vram` imports). Those runs consumed scrambled grids; mark them suspect in a findings note (`reports/findings/<date>_vram_scramble_contamination.md`) listing checkpoint paths and run dates.

**Acceptance:** A parity test (`core_v2/FPS/test_vram_parity.py`): for one reference day, iterate both tickers; assert BarState fields identical, and assert `grid_slice[0, 0, :]` (channel 0) equals the 1D-TF features from `assemble_v2_grid` applied to the same matrix rows. Channel 0 must be '1D', not '5s'.

---

## Fix 3 (P0) — Ledger hardcoded index 12 is wrong in V2 layout

**Problem:** `ledger.py` reads `entry_features[12]` / `features[12]` as "1m z_se" in `add_position` (z_sign/z_peak/z_trough/amplitude init) and `update_bar` (oscillation tracker). Index 12 was the V1 91-D position. In the V2 201-feature canonical order, index 12 is `L2_5s_vol_accel_9`. The entire oscillation tracker — zero_crossings, amplitudes, FADE-tier exit inputs — has been tracking 5s volume acceleration sign flips.

**Change:**
1. At module top of `ledger.py`:

```python
from core_v2.features import FEATURE_NAMES
_1M_Z_IDX = FEATURE_NAMES.index('L3_1m_z_se_15')
```

2. Replace every `features[12]` / `entry_features[12]` / `len(...) > 12` guard with `_1M_Z_IDX` and `len(...) > _1M_Z_IDX`.
3. `sim_executor.py` already defines the same constant — both must resolve from `FEATURE_NAMES.index(...)`, never a literal. Optionally hoist shared indices into `core_v2/features.py` (e.g. `IDX_1M_Z_SE`) so there is one definition; if so, update sim_executor to import it.
4. Remove the stale comment "z_se for 1m lives at index 12 in the canonical 91-D layout."

**Contamination flag:** all sim runs through `Ledger.update_bar` had corrupted oscillation state. Any analysis keyed on zero_crossings/amplitude (FADE exit studies) is suspect — note it in the same findings file as Fix 2 or a sibling.

**Acceptance:** `grep -n "\[12\]" core_v2/ledger.py` returns nothing. Unit test: build a feature vector with a known value at `FEATURE_NAMES.index('L3_1m_z_se_15')`, open a position, assert `z_sign` matches that value's sign.

---

## Fix 4 (P1) — bars_held unit mismatch vs TimeStop

**Problem:** `Ledger.update_bar` sets `bars_held = (ts - entry_ts) // 60` — units are MINUTES. `exits.py` declares `MAX_HOLD_BARS = 360  # 30 min (360 * 5s)` — assumes 5s-bar counts. TimeStop therefore fires at 360 minutes (6 hours), not 30. The research suite's `TimeStop(max_bars=720)` intends 60 min but gets 12 hours.

**Change (pick ONE unit; recommended: minutes, since the ledger formula is cadence-independent):**
1. Rename the Position field to `minutes_held` OR keep `bars_held` but document on the dataclass: `# UNITS: whole minutes since entry, NOT bar counts.` Renaming is safer — it breaks all stale consumers loudly. Grep consumers (`bars_held` appears in PositionView, exits, sim metrics) and update.
2. exits.py: `MAX_HOLD_BARS = 30  # minutes` (was 360 intending 30 min); research suite `TimeStop(max_bars=60)` (was 720 intending 60 min). Rename the param `max_bars` → `max_minutes` for the same loud-break reason.
3. Audit `position.extras['thresholds']['time_stop_bars']` producers (threshold_optimizer) — if the optimizer fit values in 5s-bar units against the minutes counter, those fitted values are also 12x off; flag for refit rather than silently rescaling.

**Acceptance:** Unit test: position opened at ts=0, bar at ts=1800 (30 min) → TimeStop with default fires; bar at ts=1740 does not.

---

## Fix 5 (P1) — strategy_engine.py is dead code against current Ledger

**Problem:** `Engine` calls `ledger.update / open / close / position / closed`; the current Ledger API is `update_bar / add_position / remove_position / primary / closed_trades`. The file cannot have executed since the ledger refactor. It also imports `BarState` from `training.utils.state` while `core_v2/FPS/state.py` defines its own BarState — two competing dataclasses — and is a third engine implementation alongside the decoupled engine/sim_executor pair (the exact competing-engines failure mode from the prior audit).

**Change (decision required — default to DELETE):**
- **Delete path (default):** remove `core_v2/strategy_engine.py`. Grep for importers first; if `exits.py`'s `training.utils.state` import is also part of this dead lineage, fix exits.py to import BarState from `core_v2.FPS.state` (verify the field contract matches — it does: price/timestamp/regime_idx/v2 lookups).
- **Port path (only if the Strategy/ExitRule walk loop is still wanted):** rewrite `_tick` against the real API: `update_bar(feat, price, ts)`, `add_position(...)`, `remove_position(...)`, `primary`, `closed_trades`; single BarState from `core_v2.FPS.state`; and route force-close through `remove_position` per contract.

Either way, `exits.py`'s imports must be checked: it imports from `training.utils.state` and `training.utils.v2_cols` — confirm those modules exist and re-export the V2-correct symbols; if they're V1 remnants, repoint to core_v2 equivalents.

**Acceptance:** repo imports resolve (`python -c "import core_v2.exits"` succeeds); no module imports the deleted file; exactly one BarState class is importable from core_v2.

---

## Fix 6 (P2 batch) — hygiene

1. **Silent day skips:** `MultiDayForwardPassSystem.__iter__` swallows `FileNotFoundError`. Log a warning with the day string; keep skipping (don't raise).
2. **price=0.0 poisoning:** in `ForwardPassSystem.__iter__`, when the 1m lookup fails, `price=0.0` flows into PnL math silently. Change: log once per day and `continue` (skip the bar) rather than yielding a zero-price state. If the first bars of day are legitimately pre-1m-warmup, skipping is correct behavior anyway.
3. **_REGIME_CACHE keyed wrong:** the module-global cache ignores `labels_csv` — a second FPS with a different labels path gets the first file's labels. Key the cache by path: `_REGIME_CACHE: dict[str, dict]`.
4. **Dead `_v2_grid`:** `ForwardPassSystem.__init__` computes `self._v2_grid = assemble_v2_grid(...)` but nothing reads it — N×8×25 float32 per day of dead RAM. Delete it from the base class; after Fix 2, the VRAM subclass owns grid assembly.
5. **Docstring drift:** `features.py` header says "139D", `describe_feature_count` says 139, FPS docstrings say "185 cols" — actual is 201. Sweep core_v2 for `139|185` literals in comments/docstrings and correct to 201 (and 8 TFs × 25). Do NOT touch the `N_FEATURES` assertion logic — it's already correct.
6. **`run_week.py` stale days:** (carry-over note) the hardcoded March 2026 day list in the research runner predates current data; not in this spec's file set but flag to the user.

---

## Execution order & gates

1. **Fix 3** (ledger index) — isolated, unit-testable, unblocks everything else.
2. **Fix 1** (segment risk removal) + grep sweep for consumers.
3. **Fix 2** (VRAM grid + dedup) + parity test + contamination findings note.
4. **Fix 4** (units) — coordinate with threshold_optimizer refit flag.
5. **Fix 5** (dead engine decision) — ask user delete-vs-port if any importer is found; otherwise delete.
6. **Fix 6** batch.

**Global gate:** after Fixes 1–3, run one reference-day sim through sim_executor and through the VRAM ticker parity test. Expect trade outputs to CHANGE (oscillation tracker now reads real z; grid is now correctly ordered) — record before/after deltas in `reports/findings/<date>_core_v2_fix_deltas.md`. Any checkpoint or analysis listed in the contamination notes must not be cited until re-run.
