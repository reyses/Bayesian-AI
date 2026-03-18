---
name: cleanup backlog — architectural debt
description: Accumulated refactoring tasks from the breakthrough sessions (2026-03-16 to 2026-03-18)
type: project
---

## Architecture Cleanups

1. **Make continuous IS→OOS the default** — remove `--continuous` flag. `python training/trainer.py`
   runs IS→checkpoint→OOS by default. `--oos` = resume from checkpoint.

2. **Unify data folders** — one `DATA/ATLAS/` with all data. Oracle coverage boundary determined
   by Phase 1 discovery range, not folder structure. No copy/paste to expand.

3. **Remove `oos_mode` behavioral branches** — 50 references, most are naming. Replace with
   `phase` label ('is'/'oos') that only affects file names and oracle availability.

4. **Delete IS inline forward pass** — 2,000 lines with pattern_map that's now dead code.
   IS uses compressed + peak (same as OOS). The `else` branch after `if True:` is unreachable.

5. **Three-layer separation** — data injector / engine+oracle / reporting. Oracle should
   be fully decoupled from engine (post-processing, not inline).

6. **Remove `_print_oos_comparison`** — OOS2 deleted, this function is dead.

7. **Remove strategy_selection references** — method exists but never called.

8. **Dead imports** — wave_rider, quantum_field_engine references throughout.

9. **Unused config fields** in TradingConfig — audit and remove.

10. **Old gate stats tracking** — nobody reads the gate funnel counters.

## Code Cleanups

11. **`if True:` wrapper** on line ~1566 — remove the wrapper, just use the code directly.

12. **Duplicate peak detection state tracking** — two places update `_peak_prev_pc/fm`.
    Should be one place, updated every bar.

13. **Dashboard chart save** — PostScript files accumulate. Add cleanup or age-out.

14. **Unicode chars** — remaining emoji/unicode in log messages that crash cp1252.

15. **`.gitignore`** — trade_replays JSON files are 60MB+. Should be gitignored.

## Data Pipeline

16. **Resample automation** — when live aggregator saves new 15s/1s, auto-resample
    to other TFs. Currently manual (`tools/` script).

17. **1s bar persistence** in live — bar_aggregator only persists 15s. Need 1s for
    1s worker on restart.

18. **TBN parquet loading** — loads from ATLAS parquets when available, falls back
    to resampling. Verify it's working correctly (was just added).

## Future Avenues (not bugs)

19. **Loose OOS gate** — pattern_type alone (no cascade/struct) → 3x trades but
    lower $/trade. Needs quality filter before enabling.

20. **Phase 1 discovery using peak detection** — rebuild templates around reversal
    patterns instead of z-score threshold crossings.

21. **Maintenance window discovery** — run Phase 1 on daily data during CME halt.

22. **Weekend recalibration** — weekly oracle + brain retrain from live trades.
