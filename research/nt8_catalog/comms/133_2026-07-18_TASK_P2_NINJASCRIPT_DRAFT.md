# TASK 133 — NT8 port PHASE P2: NinjaScript strategy + native zigzag + tie-rule pin
**Doc:** 133 · **Date:** 2026-07-18 · **Author:** Claude Fable · **Executor: REVIEWER DRONE (Opus) — NOT AG**
**Status:** TASK

Package the P1-passed C# entry engine as a NinjaScript strategy draft, add
the native R-trigger, pin the TMPL0 tie rule. COMPILE-TESTING happens later
in Moises'' NT8 session — your deliverable is reviewable code + a bench
harness rerun, not a compile.
1. **Tie-rule pin (both sides)**: deterministic same-ts resolution for TMPL0
   multi-event bars — RULE: highest-TF event wins; tie -> the event whose
   pattern P (frozen long_frac distance) is larger; still tied -> hold prior
   state. Implement in golden_vector_gen.py AND the C# port; regenerate
   golden; rerun parity — expect TMPL0 -> 100.000% (report the new table).
2. **Native zigzag/R-trigger in C#**: port training/strategies/zigzag.py
   verbatim (ATR14x4 at RTH open, 5s closes, min_bars=36) into the harness;
   parity vs golden zz_* columns (100% pivot agreement bar).
3. **NinjaScript strategy draft**: docs/nt8/7-EnsembleRunner_v0.1-RC.cs —
   class EnsembleRunner_v01 (versioning policy: -RC, own class name, VERSION
   constant, header banner, CHANGELOG). Structure: 5s primary series; the 22
   generators + logistic from the P1 classes adapted to NT8 OnBarUpdate
   streaming (no lookahead: closed-bar semantics per the schema doc);
   z_se + calendar inputs = the two declared external inputs — implement
   z_se natively if the formula is portable from core_v2 (document), else
   file-feed with a clear TODO; entry: long/short at the frozen threshold;
   exit: R-trigger reversal only (ride-only per doc 107); fixed 1 contract;
   catastrophic stop parameter (default OFF in SIM, present for live);
   session guard 15:55 flatten (from the mamba env precedent).
4. Report: reports/p2_report.md — tie-rule parity table, zigzag parity, the
   .cs draft''s structure summary + every TODO that needs the NT8 compile
   loop. DO NOT copy anything to the NinjaTrader folders (deploy gate).
RUN SYNCHRONOUSLY; commit NOTHING; python3.11 for tooling. Final message:
new parity numbers, zigzag parity, .cs summary + TODO list, deviations.
