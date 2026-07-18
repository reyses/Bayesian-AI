# TASK 103 — "Oops, re-enter" — stop + re-entry sim (Moises' counter-proposal)
**Doc:** 103 · **Date:** 2026-07-18 · **Author:** Claude (reviewer) · **Status:** TASK (Opus drone)
Moises (verbatim): "counter proposal the ups I exited early mechanism you
reenter but with a slightly worst position." Context: doc 100 — dumb 24t stop
nets +17.7 t/ep, its cost concentrated in knifed dipped-goods; the knife is
catastrophic only because it's IRREVERSIBLE. Test whether stop+re-entry beats
the plain stop.

## Mechanism to simulate (per-episode path math, 1m drift series)
- Base: adverse-drawdown stop at X ticks (bail when favorable-signed drift ≤ −X).
- RE-ENTRY: after a bail, re-enter SAME direction when the path recovers to
  drift ≥ bail_level + M ticks (confirmation margin), at that recovery price
  (i.e., a slightly WORSE position than original entry by construction —
  quantify the give-up). Cap re-entries per episode at B ∈ {1, 2}.
- Friction: 2.4 ticks (0.6pt MNQ round trip) charged per EVERY entry/exit
  pair including re-entries. Named constant.
- Episode net = sum of realized legs vs never-bail reference (same convention
  as score_wrongdir's net_ticks_vs_neverbail — reuse/extend that code).

## Sealed protocol
1. TUNE on a 2024 population: cut it with tools/select_wrongdir.py machinery
   (same economic truth, BAND=4, same engagement source, split='train'); grid
   X ∈ {8,16,24,32,48}, M ∈ {4,8,16}, B ∈ {1,2}. Pick winner by mean net
   ticks/ep (report the full grid).
2. FREEZE. Evaluate ONCE on the 198 scored test episodes (the doc-100 set,
   reports/wrongdir/truth/). Deliver: net ticks/ep + day-block bootstrap CI,
   vs (a) never-bail 0, (b) plain stop best-X +17.7, (c) blind agents +7.5.
3. Breakdowns: net by class (wrong / good-dipped / good-clean); the dipped-
   good knife cost before/after re-entry; chop-churn cost (episodes with ≥2
   bails); distribution mode-first.
4. PRE-REGISTERED BAR: stop+re-entry retained only if test net > plain-stop
   best-X with the delta's CI excluding 0. State PASS/FAIL plainly.
5. Caveat to print in the report: 1m path resolution understates intrabar
   stop/trigger crossings; results are 1m-granularity estimates.

## Rules
New tool: research/exit_dojo/tools/stop_reenter_sim.py; report
reports/wrongdir/stop_reenter.md. RUN SYNCHRONOUSLY; python3.11 from repo
root; commit NOTHING; touch nothing else under exit_dojo.
