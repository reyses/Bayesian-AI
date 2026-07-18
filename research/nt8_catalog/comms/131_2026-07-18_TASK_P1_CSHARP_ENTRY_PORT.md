# TASK 131 — NT8 port PHASE P1: the C# entry engine vs golden vectors
**Doc:** 131 · **Date:** 2026-07-18 · **Author:** Claude Fable · **Executor: REVIEWER DRONE (Opus) — NOT AG**
**Status:** TASK

Port the 22 top-K stream generators + the compact entry logistic to C#,
validated bar-by-bar against the P0 golden vectors (research/nt8_port/golden/
+ reports/golden_schema.md — the schema doc is your implementation contract;
research/nt8_port/reports/top_k_streams.txt has the frozen weights,
double-derived and reviewer-verified).

1. DISCOVERY first: is a dotnet SDK available (`dotnet --version`)? If YES:
   build a standalone console parity harness (plain C#, no NT8 references)
   that loads golden parquets (or CSV exports of them) and runs the ported
   logic bar-by-bar. If NO: emit the .cs files + a detailed python-side
   test plan and STOP after static review.
2. PORT: each of the 22 generators as a C# class implementing the exact
   causal conventions (bar stamped at open closes at ts+period; 1m boundary
   evaluation; the quantile-matched thresholds from the frozen params — port
   VALUES from top_k_streams.txt, never re-derive). The compact logistic
   (22 one-hots + base features) with the frozen standardization.
3. PARITY BAR (pre-registered): fire-state agreement ≥ 99.5% of bar-stream
   cells across all 20 golden days; P within 1e-6 of the compact re-fit
   reference; entry decisions 100% agreement at the frozen threshold.
   Report per-day disagreement counts; every disagreement diagnosed.
4. Files: research/nt8_port/csharp/ (the port), tools/parity_check.py
   (drives the comparison), reports/p1_parity.md. NinjaScript packaging is
   P2 — keep this phase platform-neutral C#.
RUN SYNCHRONOUSLY; commit NOTHING; python3.11. Final message: dotnet
availability, per-day parity table, disagreements diagnosed, deviations.
