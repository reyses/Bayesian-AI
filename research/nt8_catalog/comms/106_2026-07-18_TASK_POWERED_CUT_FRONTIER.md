# TASK 106 — the POWERED cut frontier (full population, no fleet)
**Doc:** 106 · **Date:** 2026-07-18 · **Author:** Claude (reviewer) · **Status:** TASK (Opus drone)
Finding that motivates this (doc 105): at N=198 even the plain stop's +17.7
t/ep is CI[−12.4,+46.7] — NOT significant. The 198 was sized for LLM-fleet
cost; mechanical policies are free to evaluate. Rerun the ENTIRE frontier on
the FULL test population so the cut question gets statistical power, and on
the NATURAL class mix (the 198 was 1:1 balanced — deployment isn't).

## Design (pure evaluation — NO new tuning; everything stays frozen)
1. POPULATION: every test-split (2025-26) engagement from the wrongdir
   economic universe (the select_wrongdir scan that yielded the 23,378-
   engagement histogram) — INCLUDING dead-band episodes (deployment reality).
   Per-minute drift paths via the same scan machinery. Report N and the
   natural class mix (wrong / good-dipped / good-clean / dead-band).
2. POLICIES (all frozen, no refit): never-bail; plain stop X ∈ {8,16,24,32,48}
   (each X reported — the grid was pre-registered in 103, this is evaluation
   not selection); stop+re-entry frozen (X=48,M=4,B=1 per 103); stop+veto
   frozen (p*=0.45 + coefs per veto_frozen.json — apply the frozen model
   verbatim; its features are path-derivable at trigger).
3. METRICS: net ticks/ep vs never-bail (mean, median, MODE-first) with
   day-block bootstrap CIs (test days ≈ hundreds; 4000 resamples); ABSOLUTE
   net with friction 2.4t/RT; per-class decomposition; the KEY table = each
   policy's delta-vs-never-bail CI and delta-vs-best-stop CI.
4. QUESTIONS to answer plainly: (a) does ANY cut policy beat never-bail with
   CI excluding 0 at scale? (b) does the doc-100 +17.7 stop edge survive the
   natural mix and the power increase? (c) does re-entry's dipped-knife
   repair change sign at natural mix (dipped goods are rarer than 1:1)?
5. Caveats to print: 1m granularity; engagement windows may overlap within a
   day (day-block CI covers dependence); frozen-on-2024 params evaluated on
   2025-26 (transfer risk already demonstrated in 103/105 — this is the
   test of record for it).

## Rules
New files only: tools/powered_cut_frontier.py + reports/wrongdir/
powered_frontier.md. Reuse scan/classification/net helpers by import.
RUN SYNCHRONOUSLY; python3.11; commit NOTHING. Final message: N + mix, the
frontier table with CIs, the three answers, deviations.
