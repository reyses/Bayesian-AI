# Jules Instruction Folder

All briefs, specs, and exit reports for Google Jules tasks live here.

## Folder layout
```
Jules_instructions/
  README.md                              # this file
  EXIT_REPORT_TEMPLATE.md                # mandatory exit-report format
  YYYY-MM-DD_<task_name>.md              # task brief (one per Jules run)
  YYYY-MM-DD_<task_name>_exit_report.md  # required after every run
```

## Rules

1. **Every Jules task gets its own brief file** here, named `YYYY-MM-DD_<task_name>.md`.
2. **Every Jules run must produce an exit report** using `EXIT_REPORT_TEMPLATE.md`, saved as `YYYY-MM-DD_<task_name>_exit_report.md`. No exceptions.
3. **Deviations from the brief are documented in the exit report**, never silently. If Jules changed or skipped an instruction, the exit report must explain what, why, the risk, and how to reverse.
4. **Briefs are immutable once handed to Jules.** If the spec needs to change mid-run, abort and write a v2 brief with `_v2` suffix.
5. **Briefs reference invariants from `CLAUDE.md` and `MEMORY.md`** rather than restating them, to avoid drift.

## Legacy briefs

Pre-2026-05-23 Jules briefs live at the `docs/` root as `JULES_<TASK_NAME>.md` (e.g. `docs/JULES_ZIGZAG_ONLY_REFACTOR.md`). They stay where they are — this folder applies only to briefs created on or after 2026-05-23.

## Active briefs

| Date | Task | Status |
|---|---|---|
| 2026-05-23 | Consolidation refactor (FPS, training/, tools/, core→core_v2) | Completed (exit report filed) |
| 2026-05-24 | Forward Pass Unification — V2-only, single bar-walker for cache/live | Draft (Phase 0 ready) |
