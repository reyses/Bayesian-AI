# PLAN — Audit FPS → optimize → rerun ALL strategies through it (Moises' solution)
**Doc:** 043 · **Date:** 2026-07-12 · **Author:** Claude (executor), directive from Moises · **Status:** PLAN (execution follows immediately; journal-first per house rule)

## Why (context for a cold reader / AG continuation)
Doc 042: both Tier-1 candidates died in path-accurate replay because every prior
layer (dossier scripts, stored MFE/MAE, screens) had its own ad-hoc data semantics —
the mismatch class that has burned this catalog five times. Moises' fix: stop
having N replay implementations. **ONE canonical causal engine (core_v2 FPS) is
audited for robustness, optimized for speed, and ALL catalog strategies re-run
through it** for accurate numbers.

## Step 1 — FPS robustness audit (read-only)
Target: `core_v2/FPS/forward_pass_system.py` (+ `_vram` variant awareness).
Checklist:
- Causality: `_last_closed_idx` usage (the load-bearing lookahead fix), no forward
  bar leakage in any tier assembly, is_1m_close semantics.
- Session/RTH alignment: full-session vs RTH indexing (the exact bug class from
  doc 036), session_day boundaries (17:00 CT), DST.
- Warmup/NaN: first-bars behavior, NaN propagation to strategies.
- Roll seams: behavior across contract-roll days (roll_manifest awareness).
- Error paths: missing tier files, short days, empty parquets.
Output: findings appended to next comms doc; any defect = fix + parity test.

## Step 2 — Performance (parity-gated)
- Profile a representative day sweep; find hot paths (parquet loads per tier,
  per-bar Python state assembly, v2 dict builds).
- Optimize ONLY with bitwise/1e-9 parity vs pre-optimization outputs on 3 test
  days (the mamba-perf discipline: every speedup proves loss-parity first).
- Budget: whatever the 5600X/RTX3060 can do; target = all 24 strategies × 2 years
  runnable in one overnight batch or better.

## Step 3 — FPS strategy runner
- `research/nt8_catalog/tools/fps_catalog_runner.py`: drives FPS day-by-day,
  bar-by-bar; each catalog concept = a detector (entry trigger from the
  article-faithful definition) + exits (target/stop/EOD) executed on the SAME
  bar stream — one engine, one data path, no stored-excursion shortcuts.
- Position accounting: stop-first-in-bar (conservative), MNQ 0.25/\$0.50 ticks,
  optional friction line (1-2 ticks RT) in reporting.
- Output per concept/side/config: per-year N, WR, EV raw pts, day-block CI,
  distribution PNG (mode-first). All results = new comms doc + reports/.

## Kill/accept criteria (pre-registered)
- A concept is TRADABLE-CANDIDATE only if FPS-run EV CI_lo > friction (2 ticks)
  in BOTH years at the same config, and the distribution is cluster-shaped
  (|mode| >= 2pts), N >= 100/yr.
- Anything else: recorded, closed.

## Continuation instructions (if Claude stops)
Read this doc + docs/daily/2026-07-12.md; execute steps in order; one numbered
comms doc per step; commit+push each turn; no nulls; day-block CIs everywhere.
