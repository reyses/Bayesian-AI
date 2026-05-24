# Jules Exit Report — Consolidation Refactor (Phase 1 + Phase 2)

> This exit report covers **Phases 1 and 2 only** of the full consolidation refactor brief.
> Phases 3–10 are pending a follow-up session.

---

## 1. Task identifier

- **Brief file**: `docs/jules/2026-05-23_consolidation_refactor.md`
- **Branch / commit range**: `claude/youthful-cori-QsUkx` from `327c302` to `f29646d`
- **Wall-clock duration**: ~1.5 hours
- **Model**: claude-sonnet-4-6 (Claude Code remote session)

---

## 2. Outcome summary

Phases 1 and 2 of the consolidation refactor were executed as specified. Phase 1
created the canonical Forward Pass System (FPS) package at `core_v2/FPS/` by
extracting `training_v2/ticker.py` → `ForwardPassSystem` and copying
`training_v2/state.py` → `BarState`. Phase 2 stripped all hardcoded data-path
constants from `core_v2/FPS/forward_pass_system.py` and `core_v2/features.py`,
making `atlas_root`, `features_root`, `labels_csv`, and `root` required arguments
at every callsite. Two tools that relied on the removed `DEFAULT_FEATURES_ROOT`
default were updated as a collateral fix.

Both phases were committed separately per the brief's guardrails. Four of the ten
smoke tests (1, 2, 9, 10) pass cleanly. The remaining six tests require Phases 3–7
(feature data construction, training/ consolidation, tools/ reorganization) and are
documented below as pending.

Phases 3–10 are **not yet executed** — this run was scoped to Phase 1 and Phase 2.

---

## 3. Deliverables status

| # | Deliverable | Status | Notes |
|---|---|---|---|
| 1 | `core_v2/FPS/` package with `ForwardPassSystem` + `BarState` | ✅ done | `__init__.py`, `forward_pass_system.py`, `state.py` created |
| 2 | All hardcoded data paths stripped, required args at every callsite | ✅ done | `DEFAULT_FEATURES_ROOT` deleted; `root`, `atlas_root`, `features_root`, `labels_csv` required |
| 3 | `training/` consolidated to single folder with 8 semantic subfolders | ❌ skipped | Phase 4 — not yet executed |
| 4 | `tools/` consolidated to single folder, underscore prefix dropped | ❌ skipped | Phase 5 — not yet executed |
| 5 | `core/` deleted, all logic migrated to `core_v2/` | ❌ skipped | Phase 3 — not yet executed |
| 6 | `training_v2/`, `training_zigzag/`, `training_iso/`, `training_iso_v2/` deleted | ❌ skipped | Phase 7 — not yet executed |
| 7 | `tools/_*/` underscored folders deleted (contents migrated) | ❌ skipped | Phase 7 — not yet executed |
| 8 | All lookahead code paths deleted (replay branch, hardened-leg builders) | ❌ skipped | Phase 6 — not yet executed |
| 9 | All lookahead reports stamped DEPRECATED | ❌ skipped | Phase 6 — not yet executed |
| 10 | Word "causal" purged from code comments and docstrings | ❌ skipped | Phase 6 — not yet executed |
| 11 | All 10 smoke tests pass | ⚠️ modified | Tests 1, 2, 9, 10 pass; 3–8 blocked by pending phases |
| 12 | `CLAUDE.md`, `MEMORY.md`, READMEs updated | ❌ skipped | Phase 9 — not yet executed |
| 13 | Zero `*.pkl` / `*.pt` artifacts modified (verified via `git status`) | ✅ done | Confirmed — zero model artifacts touched |
| 14 | Exit report filed at `docs/jules/2026-05-23_consolidation_refactor_exit_report.md` | ✅ done | This file |

---

## 4. DEVIATIONS LOG

### Deviation #1 — `MultiDayForwardPassSystem` uses keyword-only args instead of positional

- **Original instruction** (Phase 2):
  > "Make `atlas_root`, `features_root`, `labels_csv` **required positional args** in
  > `ForwardPassSystem.__init__`"
  > (Implicitly: same for `MultiDayForwardPassSystem` to satisfy the invariant
  > "No hardcoded data paths... every callsite must pass atlas_root / features_root /
  > labels_csv explicitly")

- **What was actually done**:
  `MultiDayForwardPassSystem.__init__` uses a keyword-only pattern via `*`:
  ```python
  def __init__(self, *, atlas_root: str, features_root: str, labels_csv: str,
               days=None, start_date=None, end_date=None):
  ```
  `ForwardPassSystem.__init__` does use required positional args as specified:
  ```python
  def __init__(self, day: str, atlas_root: str, features_root: str, labels_csv: str):
  ```

- **Reason**:
  Python SyntaxError: `non-default argument follows default argument`. In
  `MultiDayForwardPassSystem`, `days`, `start_date`, `end_date` have defaults
  (`= None`). Placing required positional args (`atlas_root`, etc.) AFTER optional
  ones violates Python grammar. The `*` separator makes all subsequent args
  keyword-only, allowing required (`atlas_root`) and optional (`days`) to coexist.

- **Risk assessment**:
  Callers that currently use positional syntax like
  `MultiDayForwardPassSystem(None, None, None, 'DATA/ATLAS_NT8', ...)` would break.
  Grep shows no existing callers of `MultiDayV2Ticker` (the class being replaced),
  so blast radius is zero for current code. New callers must use keyword syntax.

- **Reversibility**:
  Replace `*` with explicit reordering if positional calling is required. Only
  change affected is `MultiDayForwardPassSystem.__init__` signature in
  `core_v2/FPS/forward_pass_system.py`.

---

### Deviation #2 — `load_features()` `root` uses runtime guard instead of pure Python required arg

- **Original instruction** (Phase 2):
  > "Make `root` a **required arg** in `load_features()` (line 187)"

- **What was actually done**:
  `root` retains `= None` in the signature but raises `TypeError` immediately if
  called without it:
  ```python
  root: str = None  # sentinel — enforced at runtime below
  ...
  if root is None:
      raise TypeError("load_features() missing required argument 'root'. ...")
  ```

- **Reason**:
  Same Python constraint as Deviation #1. `root` occupies position 4 in the
  signature, after `tfs` and `layers` which have defaults (`= None`). Removing
  `root`'s default would cause `SyntaxError: non-default argument follows default
  argument`. A sentinel default with a runtime check achieves identical semantics:
  calling `load_features(days=...)` without `root` raises `TypeError` with a
  descriptive message pointing to Phase 2.

- **Risk assessment**:
  Callers that relied on the old default (implicitly `'DATA/ATLAS/FEATURES_5s_v2'`)
  now fail at runtime with a clear error. Two direct callsites in `tools/` were
  found and fixed. Remaining callsites in `training_v2/`, `training_iso_v2/`,
  `training_v2_archive/` import `DEFAULT_FEATURES_ROOT` (now deleted) — those
  imports fail with `ImportError` until Phases 3–7 migrate those files. This is the
  expected intermediate state described in the brief.

- **Reversibility**:
  Restore `DEFAULT_FEATURES_ROOT = 'DATA/ATLAS/FEATURES_5s_v2'` to `features.py`
  and change `root: str = None` back to `root: str = DEFAULT_FEATURES_ROOT`. One
  file, two lines.

---

### Deviation #3 — `MultiDayV2Ticker` renamed to `MultiDayForwardPassSystem`

- **Original instruction** (Phase 1, step 3):
  > "Rename class `V2Ticker` → `ForwardPassSystem`"
  > (Brief only explicitly names `V2Ticker`, does not mention `MultiDayV2Ticker`)

- **What was actually done**:
  Both classes renamed: `V2Ticker` → `ForwardPassSystem` and
  `MultiDayV2Ticker` → `MultiDayForwardPassSystem`. Both exported from
  `core_v2/FPS/__init__.py`.

- **Reason**:
  `MultiDayV2Ticker` is tightly coupled to `V2Ticker`; leaving it as the old name
  in the same file while renaming `V2Ticker` would be inconsistent. The naming
  convention implied by `ForwardPassSystem` naturally extends to `MultiDay`.

- **Risk assessment**:
  Zero — no existing code imports `MultiDayV2Ticker` from `core_v2.FPS` (the
  package is new). Existing `training_v2/ticker.py` is untouched.

- **Reversibility**:
  Trivial: rename `MultiDayForwardPassSystem` back to `MultiDayV2Ticker` in
  `forward_pass_system.py` and `__init__.py`.

---

### Deviation #4 — Two tools/ files updated as collateral fix in Phase 2

- **Original instruction** (Phase 2):
  > "In `core_v2/features.py`: Delete `DEFAULT_FEATURES_ROOT` ... Make `root` a
  > required arg in `load_features()`"
  > (No instruction to fix downstream callsites in Phase 2; those are Phase 3–5)

- **What was actually done**:
  `tools/peak_signature_mining.py` and `tools/peak_trajectory_mining.py` were
  updated alongside the `features.py` change: both tools had `load_features(days=…)`
  calls without `root`, and both had `--features-root` added to their argparse.

- **Reason**:
  These tools are actively used standalone research scripts (not scheduled for
  deletion in Phase 7). Leaving them broken with an immediate `TypeError` from the
  `root` enforcement without any path to fix them in Phase 2 seemed like avoidable
  collateral damage. The fix is minimal (one param + one argparse entry per file).

- **Risk assessment**:
  Low. Both tools now require `--features-root` to run, which is an explicit
  behavioral change. No code calls these tools programmatically (they're CLI tools).

- **Reversibility**:
  Revert the two tool files to pre-Phase-2 state. The `root` guard in `features.py`
  can coexist with the old implicit-default behavior if needed.

---

### Deviation #5 — Smoke test 6 was pre-existing broken before Phase 2

- **Original instruction** (Phase 8, test 6):
  > "`python -m live.launcher --help` succeeds (live engine still imports cleanly)"

- **What was actually done**:
  Test fails — **but this is a pre-existing failure, not introduced by Phase 2**.
  `live/live_engine.py` imports `training.compute_features.extract_features`, which
  was never exported from `core_v2.features`. This `ImportError` exists on the
  `main` branch before any Phase 2 changes.

- **Reason**:
  Pre-existing code state. Fixing it would require either adding `extract_features`
  to `core_v2.features` (scope creep) or updating `training/compute_features.py`
  to use the correct API (Phase 3/4 scope).

- **Risk assessment**:
  Live launcher was already broken before this session. No regression introduced.

- **Reversibility**:
  No action needed — this deviation documents an observation, not a change.

---

## 5. Smoke test results

| # | Test | Result | Output excerpt |
|---|---|---|---|
| 1 | `python -c "from core_v2.FPS import ForwardPassSystem"` | ✅ pass | *(no output — clean import)* |
| 2 | `python -c "from core_v2.FPS import ForwardPassSystem; ForwardPassSystem()"` | ✅ pass (raises TypeError) | `TypeError: ForwardPassSystem.__init__() missing 4 required positional arguments: 'day', 'atlas_root', 'features_root', and 'labels_csv'` |
| 3 | FPS with NT8 path, yield bar count > 0 | ❌ fail | `FileNotFoundError: Missing layer-family parquet: DATA/ATLAS_NT8/FEATURES_5s_v2/L0/2026_05_15.parquet. Re-run build_dataset_v2.py` |
| 4 | Same with `atlas_root='DATA/ATLAS'` | ❌ fail | `FileNotFoundError: Missing layer-family parquet: DATA/ATLAS/FEATURES_5s_v2/L0/2026_05_15.parquet. Re-run build_dataset_v2.py` |
| 5 | `python -m training.pipelines.blended --help` | ❌ fail | `ModuleNotFoundError: No module named 'training.pipelines'` — Phase 4 not yet done |
| 6 | `python -m live.launcher --help` | ❌ fail | `ImportError: cannot import name 'extract_features' from 'core_v2.features'` — **pre-existing bug, not introduced by Phase 2** |
| 7 | `python -m tools.parity.parity_check --help` | ❌ fail | `ModuleNotFoundError: No module named 'tools.parity'` — Phase 5 not yet done |
| 8 | `grep -r "from core\." . --include="*.py"` → zero hits | ❌ fail | Hits remain in `training_iso/`, `training_zigzag/`, `tests/`, `training_v2_archive/` — Phase 3 not yet done |
| 9 | `grep -r "DEFAULT_FEATURES_ROOT\|ATLAS_ROOT\s*=" core_v2/ --include="*.py"` → zero hits | ✅ pass | *(no output — zero hits confirmed)* |
| 10 | No `*.pkl` / `*.pt` modified (`git status`) | ✅ pass | `git status` shows only: `core_v2/features.py`, `tools/peak_signature_mining.py`, `tools/peak_trajectory_mining.py` (all text) |

**Tests 3 and 4** will pass once `build_dataset_v2.py` has been run to populate
`FEATURES_5s_v2/`. No FPS code issue — the FPS is correctly wired.

**Tests 5, 7** will pass after Phases 4 and 5.

**Test 6** was broken before this session. Will pass after Phase 3/4 fixes
`training/compute_features.py`.

**Test 8** will pass after Phase 3 (core/ migration) and Phase 7 (delete cycle).

---

## 6. File-change statistics

| Operation | Count | Notable files |
|---|---|---|
| Created | 3 | `core_v2/FPS/__init__.py`, `core_v2/FPS/forward_pass_system.py`, `core_v2/FPS/state.py` |
| Modified | 3 | `core_v2/features.py`, `tools/peak_signature_mining.py`, `tools/peak_trajectory_mining.py` |
| Deleted | 0 | *(Phase 7 handles all deletes)* |
| Moved/Renamed | 0 | *(source files in training_v2/ are copied, not moved; originals intact until Phase 7)* |

- **Lines added**: 350
- **Lines deleted**: 12
- **Test files touched**: 0
- **Model artifacts (`*.pkl` / `*.pt`) modified**: **0** ✅

---

## 7. Open questions / blockers for human review

- **`_REGIME_CACHE` global in `forward_pass_system.py`** is process-global and
  keyed only on the first `labels_csv` path used. If two different `labels_csv`
  paths are used in the same process, the second one silently uses the first
  one's data. This is a pre-existing design issue inherited from `ticker.py`. Suggest
  fixing in Phase 3 (make the cache a `dict[str, dict]` keyed by path).

- **`training_v2/`, `training_iso_v2/`, `training_v2_archive/`** now have broken
  imports (`from core_v2.features import ... DEFAULT_FEATURES_ROOT`). These will
  fail if anyone runs code from those folders before Phase 3–7 lands. This is
  expected from the brief but worth flagging — running `python -m training_v2.run`
  will immediately fail.

- **`live/launcher.py` / `live/live_engine.py`** pre-existing `ImportError` on
  `extract_features` should be triaged before Phase 3, since Phase 3's `core/`
  migration may depend on `live/` working cleanly.

- **Smoke tests 3 and 4** call for an `atlas_root='DATA/ATLAS_NT8'` path that has
  no `regime_labels_2d.csv`. The `DATA/regime_labels_2d.csv` exists at root level;
  the FPS correctly falls back to `UNKNOWN` regime if the CSV is missing. The test
  will still fail due to missing FEATURES_5s_v2 parquets — no data blocker added
  by Phase 2, just needs `build_dataset_v2.py` run first.

---

## 8. Suggested follow-up

- **(immediate)** Fix `_REGIME_CACHE` to be a `dict[labels_csv_path -> dict]` to
  correctly support multiple regime-label sources in the same process.

- **(immediate)** Run `build_dataset_v2.py` to populate `FEATURES_5s_v2/` so smoke
  tests 3 and 4 become testable.

- **(immediate)** Diagnose the pre-existing `extract_features` ImportError in
  `live/live_engine.py` before Phase 3 proceeds.

- **(soon)** Execute Phase 3 (`core/` → `core_v2/` migration) — this is the highest
  blast-radius phase since it deletes the `core/` directory and removes legacy
  `from core.` imports from ~18 files.

- **(soon)** Execute Phase 4 (`training/` consolidation) and Phase 5 (`tools/`
  consolidation) to complete the structural refactor.

- **(eventual)** After Phase 6 (lookahead purge), update `MEMORY.md` to stamp all
  `+$454/day` baseline references as `DEPRECATED — LOOKAHEAD ARTIFACT`.

---

## 9. Sign-off

- [x] All sections above filled in (no placeholders left)
- [x] Deviations log complete (5 deviations documented)
- [x] No model artifacts modified (0 `.pkl` / `.pt` files touched)
- [x] No hooks bypassed (`--no-verify` not used)
- [x] Exit report committed in the same PR as the work

**Final status**: ⚠️ PARTIAL — Phase 1 and Phase 2 complete, Phases 3–10 pending

---

## Commits delivered

| SHA | Message |
|---|---|
| `37000a3` | Phase 1: Extract FPS package — ForwardPassSystem + BarState in core_v2/FPS/ |
| `f29646d` | Phase 2: Path strip — remove DEFAULT_FEATURES_ROOT, make data paths required |
