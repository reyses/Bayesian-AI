# Jules Brief — Bayesian-AI Consolidation Refactor

> **Mission**: collapse parallel training folders, extract canonical Forward Pass System (FPS), consolidate tools/, migrate `core/` → `core_v2/`, purge lookahead artifacts. End state: clean engine in `core_v2/` (with FPS subpackage), single `training/` folder with semantic subfolders, single `tools/` folder with semantic subfolders, no `core/`, no `training_v2/training_zigzag/training_iso*/`, no lookahead-contaminated code paths.

> **Exit report mandatory.** When done, create `Jules_instructions/2026-05-23_consolidation_refactor_exit_report.md` following `Jules_instructions/EXIT_REPORT_TEMPLATE.md`. Deviations from this brief — instructions skipped or modified due to circumstances — must be documented in §4 of the exit report.

---

## Phase 0 — Read first (load context before touching anything)

1. `CLAUDE.md` — project rules, esp. metric defs, anti-doom rule, NT8 versioning rule
2. `~/.claude/projects/c--Users-reyse-OneDrive-Desktop-Bayesian-AI/memory/MEMORY.md` — full context, esp. 2026-05-22-LATE lookahead finding
3. `Jules_instructions/README.md` and `Jules_instructions/EXIT_REPORT_TEMPLATE.md`

---

## Invariants (do not violate)

- **No magic numbers.** Every numeric constant must be a named config field or module-level constant with origin comment.
- **No hardcoded data paths.** After this refactor, every callsite must pass `atlas_root` / `features_root` / `labels_csv` explicitly.
- **No lookahead paths.** Any code reading offline `is_pivot` labels or hardened-leg CSVs derived from whole-day zigzag → delete.
- **The word "causal" is dropped.** Just "forward pass" — there's only one path now.
- **CUDA-only.** No CPU fallback (existing project rule).
- **Strategies folder is read-only.** `Documents/NinjaTrader 8/bin/Custom/Strategies/` is the live-deploy boundary — do not touch.
- **Preserve model artifacts** (`*.pkl`, `*.pt`) — only move/rename references, never overwrite a trained model file. Modifying any artifact requires explicit user authorization, otherwise documented as a deviation.

---

## Final architecture (target state)

```
core_v2/
  __init__.py
  features.py            # was: core_v2/features.py + core/feature_extraction.py merged
  ledger.py              # was: core_v2/ledger.py + training_v2/ledger.py merged
  sim_executor.py
  cuda_statistics.py
  strategy_engine.py     # was: training_v2/engine.py
  execution.py           # was: core/execution_engine.py (if still used)
  exits.py               # was: core/exit_engine.py + training_v2/exits.py
  bayesian.py            # was: core/bayesian_brain.py (if still used)
  belief_network.py      # was: core/timeframe_belief_network.py (if still used)
  fractal.py             # was: core/fractal_clustering.py (if still used)
  dmi.py                 # was: core/dmi_flipper.py (if still used)
  models/
    trade_cnn.py         # was: core/trade_cnn.py
  FPS/                   # NEW canonical forward pass package
    __init__.py          # exports: ForwardPassSystem, BarState
    forward_pass_system.py
    state.py             # BarState + regime_to_idx
    README.md            # documents the system + usage pattern

training/                # ONE folder, semantic subfolders
  README.md
  __init__.py
  datasets/
    build_dataset.py, build_dataset_v2.py
    compute_features.py, physics_labels.py
  models/
    cnn/
      entry.py, exit.py, flip.py, hold.py, risk.py
      trade_manager.py, direction.py, pivot_v2.py
      model.py, dataset.py, inference.py
    gbm/                 # B-stack trainers (PULLED FROM tools/)
      b1_pivot_imminent.py … b10_vol_regime.py
  strategies/
    base.py, ma_align.py, reversion.py, velocity_body.py
    regime_aware.py, filtered_nmp.py
  pipelines/
    blended.py, v2_native.py, iso.py
  calibration/
    threshold_optimizer.py, threshold_bayesian.py, threshold_mode_tuned.py
    learn_zband_thresholds.py, tier_discovery.py, cell_filters.py
    velocity_regime.py, seed_per_regime.py   # from tools/
  analysis/
    feature_eda.py, within_cell_eda.py, flip_rule_validation.py
    loser_autopsy.py, bleed_cause_analysis.py, vol_adaptive_test.py
  regret/
    bayesian_table.py, regret.py, regret_full.py, regret_by_regime.py
  utils/
    memory.py, release.py, report.py, level_tracker.py, aggregator.py
  archive/                # deprecated — slated for delete next cycle
    nightmare.py, nightmare_blended.py, forward_blended.py

tools/                   # consolidated, no `_` prefix
  README.md
  TOOLS_INDEX.md         # moved from research/
  __init__.py
  data/                  # was _data + flat data tools
  features/              # feature computation / inspection
  pivot/                 # general pivot detection / analysis (peak prediction, residuals, regression-line pivot)
  zigzag/                # ATR-zigzag-specific (legs, leg-age, cold-start divergence, leg analyses)
  entries/               # entry-filter / B-stack diagnostic tools
  exits/                 # exit-variant analysis, giveback, drawdown studies
  forward_pass/          # FPS-driven sims, parity checks
  regret/                # was _regret + flat regret tools
  tier/                  # was _tier + flat tier tools
  levels/                # was _levels + _peak + level/peak tools
  mtf/                   # was _mtf
  regime/                # regime labeling, regime analyses
  risk/                  # blowout, equity_risk_simulator, l2_risk_budget, saturation_sim
  eda/                   # was _eda + flat *_eda.py
  charts/                # was _charts (general plotting)
  viz/                   # ONLY trade-picking tools + the dynamic viewer
  parity/                # was _parity (methodological, kept separate)
  nn/                    # was _nn (NN-research tools, NOT trainers)
  util/                  # was _util + generic helpers
  suites/                # multi-tool bundles (trade_outcome_suite, etc.)
  archive/               # deprecated tools

live/                    # unchanged — consumes core_v2.FPS
docs/, reports/, DATA/   # unchanged
```

**Folders that no longer exist post-refactor:**
- `core/`
- `training_v2/`, `training_zigzag/`, `training_iso/`, `training_iso_v2/`
- `tools/_*/` (replaced by un-prefixed versions)

---

## Execution phases (do in order — DO NOT REORDER)

### Phase 1 — FPS extraction
1. Create `core_v2/FPS/` package
2. Copy `training_v2/ticker.py` → `core_v2/FPS/forward_pass_system.py`
3. Rename class `V2Ticker` → `ForwardPassSystem`
4. Copy `training_v2/state.py` → `core_v2/FPS/state.py`
5. `core_v2/FPS/__init__.py`: `from .forward_pass_system import ForwardPassSystem; from .state import BarState`

### Phase 2 — Path strip (forces explicit data source at every callsite)
1. In `core_v2/FPS/forward_pass_system.py`:
   - Delete module-level `ATLAS_ROOT`, `LABELS_CSV` constants
   - Make `atlas_root`, `features_root`, `labels_csv` **required positional args** in `ForwardPassSystem.__init__`
2. In `core_v2/features.py`:
   - Delete `DEFAULT_FEATURES_ROOT = 'DATA/ATLAS/FEATURES_5s_v2'` (line 169)
   - Make `root` a **required arg** in `load_features()` (line 187)
3. Smoke check: `from core_v2.FPS import ForwardPassSystem` raises `TypeError` if called with no args.

### Phase 3 — `core/` → `core_v2/` migration
For each file in `core/`:
1. Grep all imports of that module across the codebase
2. If imports exist: port file to `core_v2/`, update all imports
3. If no imports OR only deprecated-pipeline imports: move to `core_v2/archive/` (don't delete yet)
4. After all files processed, run `grep -r "from core\." .` and `grep -r "import core\." .` — must return zero hits (except in archives)
5. Delete `core/` directory

Migration map:
| Old | New | Notes |
|---|---|---|
| `core/feature_extraction.py` | merge into `core_v2/features.py` | |
| `core/statistical_field_engine.py` | `core_v2/sfe.py` (if used) | else archive |
| `core/execution_engine.py` | `core_v2/execution.py` | |
| `core/exit_engine.py` | merge with `training_v2/exits.py` → `core_v2/exits.py` | |
| `core/bayesian_brain.py` | `core_v2/bayesian.py` | |
| `core/timeframe_belief_network.py` | `core_v2/belief_network.py` | |
| `core/fractal_clustering.py` | `core_v2/fractal.py` | |
| `core/dmi_flipper.py` | `core_v2/dmi.py` | |
| `core/trade_cnn.py` | `core_v2/models/trade_cnn.py` | |
| `training_v2/engine.py` | `core_v2/strategy_engine.py` | |
| `training_v2/ledger.py` | merge into `core_v2/ledger.py` (deduplicate) | |

### Phase 4 — `training/` consolidation
1. Create new `training/` skeleton (subfolders + `__init__.py` + `README.md`)
2. Move files per the structure tree above. Run import audit between each subfolder.
3. Move B-stack trainers: `tools/train_b*.py` → `training/models/gbm/b*.py`
4. Move calibration scripts: `tools/calibrate_*.py` → `training/calibration/`
5. Merge ISO pipelines: `training_iso/`, `training_iso_v2/` → `training/pipelines/iso.py` + supporting modules
6. Update all imports across project to new paths
7. Delete `training_v2/`, `training_zigzag/`, `training_iso/`, `training_iso_v2/` (after audit shows zero live imports)

### Phase 5 — `tools/` consolidation
1. Rename `tools/_<category>/` → `tools/<category>/` (drop underscore prefix), 17 folders
2. Create new categories: `entries/`, `exits/`, `forward_pass/`, `zigzag/`, `regime/`, `risk/`, `suites/`, `archive/`
3. Triage every flat `tools/*.py` into a subfolder per the structure tree
4. Move `research/TOOLS_INDEX.md` → `tools/TOOLS_INDEX.md` and update content for new paths
5. Update all imports / shell invocations across project
6. Audit: `grep -r "tools/_" .` must return zero hits in code (docs/memory OK)

### Phase 6 — Lookahead purge
1. Delete `training_zigzag/forward_zigzag.py` `--pivot-source=replay` branch (entire replay code path)
2. Delete `build_is_hardened_legs.py` and any script reading offline `is_pivot` labels
3. Mark these reports with `**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**` header (prepend to file):
   - All under `reports/findings/trade_outcome_table/`
   - All reports quoting the +$454/day baseline
   - Any forward_pass_1contract / forward_pass_full_stack outputs
4. Grep-and-replace: every instance of the word "causal" in code comments and docstrings → "forward pass" (or just drop the qualifier)

### Phase 7 — Delete cycle (safe order)
1. After Phase 3 audit passes: `rm -rf core/`
2. After Phase 4 audit passes: `rm -rf training_v2/ training_zigzag/ training_iso/ training_iso_v2/`
3. After Phase 5 audit passes: `rm -rf tools/_*/` (the old underscored folders, contents already moved)
4. Run full project grep: `from core\.|import core\.|from training_v2|from training_zigzag|from training_iso|tools._` — must return zero hits in `.py` files

### Phase 8 — Smoke tests + audit
Each must pass before declaring refactor complete:
1. `python -c "from core_v2.FPS import ForwardPassSystem"` succeeds
2. `python -c "from core_v2.FPS import ForwardPassSystem; ForwardPassSystem()"` fails with TypeError (path strip working)
3. `python -c "from core_v2.FPS import ForwardPassSystem; t = ForwardPassSystem(day='2026_05_15', atlas_root='DATA/ATLAS_NT8', features_root='DATA/ATLAS_NT8/FEATURES_5s_v2', labels_csv='DATA/ATLAS_NT8/regime_labels_2d.csv'); n = sum(1 for _ in t); print(n)"` yields > 0 bars
4. Same with `atlas_root='DATA/ATLAS'` works (IS path still functional)
5. `python -m training.pipelines.blended --help` succeeds (entrypoint moved correctly)
6. `python -m live.launcher --help` succeeds (live engine still imports cleanly)
7. `python -m tools.parity.parity_check --help` succeeds (tool subpath works)
8. `grep -r "from core\." . --include="*.py"` → zero hits
9. `grep -r "DEFAULT_FEATURES_ROOT\|ATLAS_ROOT\s*=" core_v2/ --include="*.py"` → zero hits (constants stripped)
10. No `*.pkl` / `*.pt` model artifact was modified (`git status` shows none)

### Phase 9 — Documentation updates
1. Update `CLAUDE.md`:
   - Key Files section: new paths for all engine files
   - Remove references to `core/`
   - Add "Forward Pass System" section explaining the FPS rule
2. Update `MEMORY.md`:
   - Add entry under `## HARD RULES`: "FPS-only: any forward pass uses `core_v2.FPS.ForwardPassSystem`. No exceptions."
   - Stamp lookahead-finding entries with confirmed DEPRECATED status on the +$454 baseline
3. Create `core_v2/FPS/README.md` documenting the system and usage pattern
4. Create `training/README.md` documenting subfolder purposes
5. Create `tools/README.md` documenting subfolder purposes + the `viz/` reserved-for-trade-picking rule
6. Append entry to `docs/daily/2026-05-23.md` with refactor summary

### Phase 10 — Exit report
Create `Jules_instructions/2026-05-23_consolidation_refactor_exit_report.md` using `Jules_instructions/EXIT_REPORT_TEMPLATE.md`.
- §4 Deviations Log must list every instruction that was not followed verbatim or was modified due to circumstance.
- §5 Smoke test results must include actual stdout/stderr for each of the 10 tests above.
- §6 file-change statistics with counts.
- §9 sign-off only after all sections complete.

---

## Deliverables checklist

- [ ] `core_v2/FPS/` package created with `ForwardPassSystem` + `BarState`
- [ ] All hardcoded data paths stripped, required args at every callsite
- [ ] `training/` consolidated to single folder with 8 semantic subfolders
- [ ] `tools/` consolidated to single folder, underscore prefix dropped, ~19 semantic subfolders
- [ ] `core/` deleted, all logic migrated to `core_v2/`
- [ ] `training_v2/`, `training_zigzag/`, `training_iso/`, `training_iso_v2/` deleted
- [ ] `tools/_*/` underscored folders deleted (contents migrated)
- [ ] All lookahead code paths deleted (replay branch, hardened-leg builders)
- [ ] All lookahead reports stamped DEPRECATED
- [ ] Word "causal" purged from code comments and docstrings
- [ ] All 10 smoke tests pass
- [ ] `CLAUDE.md`, `MEMORY.md`, READMEs updated
- [ ] Zero `*.pkl` / `*.pt` artifacts modified (verified via `git status`)
- [ ] **Exit report filed at `Jules_instructions/2026-05-23_consolidation_refactor_exit_report.md`**

---

## Pre-commit guardrails

- Commit per phase (10 commits minimum), not one giant commit. Easy to revert if a phase breaks something.
- Run smoke tests after every phase, not just at the end.
- If a smoke test fails: stop, do not proceed to next phase, surface the failure in the exit report.
- Do not skip hooks (`--no-verify`), do not bypass signing.
- If you find a file you can't categorize confidently: place in `archive/` of the appropriate parent folder rather than guessing — document the decision in the exit report.
- **If an instruction here turns out to be wrong or impossible given the actual code state**: deviate, then document the deviation in the exit report. Do NOT silently skip or silently substitute.
