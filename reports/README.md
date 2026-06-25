# reports/ — standalone reports only (NOT research, NOT training)

This folder holds reports tied to **neither a specific research project nor a training run**:
cross-cutting engineering scars / bug records, numerical guards, and live-run telemetry.

## Routing (effective 2026-06-22)
- **Research** reports → `research/<topic>/reports/` (e.g. `research/fspace_cadence/reports/`, `research/nmp_state/reports/`).
- **Training / baseline** reports → `training/reports/`.
- **Here (`reports/`)** → only standalone/general reports tied to neither of the above.

## Current contents
- `2026-06-11_oscillation_state_contamination.md` — `core_v2/ledger.py` hardcoded `feat[12]` index bug (legacy 91D index vs resolved V2 `L3_1m_z_se_15`).
- `2026-06-11_vram_scramble_contamination.md` — `core_v2/FPS/forward_pass_system_vram.py` grid `reshape(-1,8,23)` bypassing `assemble_v2_grid` (scrambled CNN channels).
- `2026_06_11_live_ledger_z_contamination.md` — **live** `engine_v2.py`/`diagnostic_run.py` hardcoded indices (`feat[12/14/15]`) → wrong z/vr/vel in live telemetry after the 91D→185D/201D expansion.
- `2026-06-08_fista_solver_parity.md` — FISTA-CV vs sklearn-CV numerical parity guard for `core_v2/math/fista_gpu.py` (the CV-step bug — see `memory/reference_fista_gpu_cv_step_bug.md`).
- `live/` — live-run telemetry CSVs (NT8 + v2 trades/ledger).

> Legacy: the old 125+ research findings that used to live here were relocated/removed in the 2026-06-22 organize commit;
> the stale master index is archived at `archive/legacy_reports/2026-04_findings_INDEX_stale.md`.
