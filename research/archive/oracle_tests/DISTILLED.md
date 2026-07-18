---
name: distilled-oracle_tests
description: KT1 perfect-hindsight oracle ceiling on NMP fade entries is CEILING-FLAT — even the best-selectable regime subset is negative; falsifies recoverable-edge-by-regime-selection.
metadata: {type: distilled, topic: oracle_tests, status: concluded}
---
## Verdict
Asked: with PERFECT hindsight of the true L0 segment geometry (status/tier/
length), does ANY NMP fade-entry selection pay? First pass was INVALID (1s/5s
join bug, 58.6% GAP). Corrected (5s join, IS-only, day-block-bootstrap CI):
STILL FLAT on the true best-selectable fine regime subset. Conclusion: NMP
fade has no recoverable edge via regime-conditioned selection — supported
pivot off NMP-completion toward the RL engine.

## Key numbers (with CIs where they exist)
- Uncorrected IS (invalid join): GAP 58.6%; PRISTINE $-5.19, PURE_CHAOS
  $-5.20, RECOVERED $-5.20 (`research/nmp_strategies/reports/NMP_KT1_Oracle_Ceiling_Test_2026-06-13.md`).
- Corrected 5s join, IS-only: GAP 10.6%; PRISTINE 49.0%, CHAOS 23.0%,
  RECOVERED 17.4%; coarse-status mean $-4.62 to $-5.62, no separation
  (`docs/daily/2026-06-13.md`, Tick 1).
- Coarse oracle peek: top-quartile-day mean $-0.32, only 6% net-positive
  days (`docs/daily/2026-06-13.md`, Tick 1).
- TRUE fine-grained ceiling (best of status×tier×len-quartile / root
  grid-cell / tier×terms, n>=200, day-block CI): **$-3.76, n=277,
  CI[-4.97, -2.45]** -> CEILING-FLAT (`docs/daily/2026-06-13.md`, Tick 2;
  reproduced by `research/oracle_tests/tools/test_kt1_oracle_fine.py`).

## Graveyard / never-retry
- KT1 regime-conditioned selection on NMP fade entries: CEILING-FLAT at
  $-3.76 CI[-4.97,-2.45] w/ perfect hindsight — don't re-attempt regime
  gating this entry type without a new entry source.

## Reusable assets
(all under `research/oracle_tests/tools/`)
- `test_kt1_oracle_ceiling.py` — original join, INVALID (1s-vs-5s bug).
- `test_kt1_oracle_ceiling_fixed.py` — corrected 5s join, coarse stratifier.
- `test_kt1_oracle_fine.py` — valid fine-grained best-subset oracle w/
  day-block CI; the load-bearing script.
## Data locations
- `artifacts/stage2_year_segments.json` — 112,289 L0 segments, 5s-indexed.
- `reports/findings/strategy_runs/nmp_fade_raw_{is,oos}_atr4.csv` — trades
  joined against segments. `DATA/ATLAS/5s/*.parquet` — join-space (IS).
## Open threads
None on the KT1 test itself. A daisy-chain best-trade oracle was flagged
TODO in `_fixed.py`, never built — likely moot given the fine-grained subset
already closed the question.

## Sources
- `research/oracle_tests/README.md`, `project.md` (stub/empty)
- `research/oracle_tests/tools/test_kt1_oracle_ceiling{,_fixed}.py`,
  `test_kt1_oracle_fine.py`
- `research/nmp_strategies/reports/NMP_KT1_Oracle_Ceiling_Test_2026-06-13.md`
  (this topic's report — misfiled under nmp_strategies/reports)
- `docs/daily/2026-06-13.md` (Ticks 0-2, verified numeric trail)

## Archive recommendation
ARCHIVE — question settled NEGATIVE w/ verified CI; the pivot it supported
already happened. Flag for reviewer: the real report lives at
`research/nmp_strategies/reports/NMP_KT1_Oracle_Ceiling_Test_2026-06-13.md`,
outside this folder — decide whether to move it alongside.
