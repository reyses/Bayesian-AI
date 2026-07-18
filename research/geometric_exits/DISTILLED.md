---
name: distilled-geometric_exits
description: Kalman-derived (constant-acceleration) geometric exit rules (AccelFlip, VelDecay) lose money vs baseline hold across every Q tested
metadata: {type: distilled, topic: geometric_exits, status: dead}
---
## Verdict
Asked whether a CA-Kalman-filter-derived geometric exit (exit when filtered
acceleration flips negative, or when filtered velocity decays below 50% of
its running max) beats holding to the end of path. A 5-value Q sweep
(1e-4 → 1e-8) on a trade-paths sample found BOTH candidate exits negative
in USD at every Q, while the naive Base (hold) was strongly positive. No
Define/Measure/Analyze/Improve content was ever filled into `project.md` —
this reads as a single-pass sweep, not a completed DMAIC cycle.

## Key numbers (with CIs where they exist)
- No CIs reported (single point-estimate sweep, no bootstrap). All figures verbatim from `reports/geometric_exit_results.md`:
- Base (hold-to-end): PnL $17,213.72, WR 36.9%, avg 2.36 pts/trade, MFE capture -24.3%
- AccelFlip best: Q=1e-7 → PnL $-7,422.78, WR 47.2%, avg 0.10 pts, MFE capture 4.5%
- AccelFlip worst: Q=1e-5 → PnL $-8,430.78, WR 47.3%, avg 0.01 pts, MFE capture 4.2%
- VelDecay best: Q=1e-4 → PnL $-8,105.28, WR 47.4%, avg 0.04 pts, MFE capture 4.4%
- VelDecay worst: Q=1e-8 → PnL $-12,437.28, WR 47.5%, avg -0.36 pts, MFE capture 4.0%
- Every AccelFlip/VelDecay row across all 5 Q values is negative USD PnL (range $-7,422.78 to $-12,437.28) vs Base's +$17,213.72.

## Graveyard / never-retry (if any)
- Kalman constant-acceleration accel-flip / velocity-decay geometric exits:
  ALL Q values (1e-4..1e-8) lose $7.4k–$12.4k vs a +$17.2k baseline hold on
  the same sample — consistent, not Q-sensitive. Higher count-based win rate
  (~47-48%) than Base (36.9%) but tiny avg pts/trade (0.01-0.10 vs 2.36) and
  low MFE capture (4-4.8% vs Base's -24.3%) — cuts winners far too early.
  (Note: count-based WR shown here per the source file; per project convention
  this is NOT the canonical PF-based Trade WR — treat as directional only.)

## Reusable assets
- `research/geometric_exits/tools/test_geometric_exits.py` — self-contained
  CA-Kalman (`kalman_ca`) implementation + Q-sweep harness + exit-candidate
  eval (`eval_sweep`) + animated demo plot generator. Reusable Kalman code if
  a future geometric-exit idea needs the same pos/vel/acc filter.

## Data locations
- Input: `C:/Users/reyse/.gemini/antigravity/brain/0b405af3-d525-4c87-b71d-cb77ea225a55/reports/findings/trade_paths.parquet`
  — external to this repo (Gemini/antigravity brain path), not reproducible
  from repo-tracked data alone.
- Output: `research/geometric_exits/reports/geometric_exit_results.md` (sweep
  table), `research/geometric_exits/reports/geometric_exit_demo.png` (single
  best-trade animation frame).

## Open threads
- `project.md` DMAIC sections (Define/Measure/Analyze/Improve/Control) were
  never filled in — no documented decision to kill vs iterate; this distill
  treats the sweep's uniform negative result as a de facto kill.
- Source parquet lives outside the repo — if revisited, first locate/rebuild
  `trade_paths.parquet` in-repo.

## Sources
- `research/geometric_exits/README.md`
- `research/geometric_exits/project.md`
- `research/geometric_exits/reports/geometric_exit_results.md`
- `research/geometric_exits/tools/test_geometric_exits.py`

## Archive recommendation
ARCHIVE (uniform negative result across full Q sweep, no open DMAIC
sections, no follow-up work found in the folder — reads as a closed
negative experiment).
