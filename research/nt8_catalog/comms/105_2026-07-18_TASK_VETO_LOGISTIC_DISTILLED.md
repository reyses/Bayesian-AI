# TASK 105 — the DISTILLED VETO: sealed logistic at the stop-trigger moment
**Doc:** 105 · **Date:** 2026-07-18 · **Author:** Claude (reviewer) · **Status:** TASK (Opus drone)
Night charter (Moises): "cut losers fast, let winners ride" — WITHOUT Mamba
(GPU-time expensive; holding back). Both dojos proved the dip-veto information
exists in the path (clean 10% vs dipped 54% false-bail, doc 100). Distill it
NOW: a 2024-sealed logistic that, at the moment the 24t stop triggers, prices
"will this recover?" — the learned veto as a mechanical, CPU-cheap rule.

## Design
1. POPULATION (train 2024 / test = the 198 doc-100 episodes): reuse the
   wrongdir machinery (select_wrongdir engagement cut + per-minute drift
   paths; same BAND=4 economic truth). Simulate the plain stop at X=24t; for
   every episode whose stop TRIGGERS, the trigger minute t* is the decision
   point.
2. FEATURES at t* (path-derivable ONLY — v1 stays cheap; document that aux
   fires/z-streams are v2): loss velocity (1m drift delta) + ACCELERATION
   (2nd diff), giveback dynamics (drift vs path peak; velocity of giveback),
   efficiency ratio over last ≤10m (|net|/Σ|deltas| on the drift series),
   drawdown depth vs trailing path vol, minutes-since-entry, entry P, tod.
   All causal at t* (≤ t* only) — assert it.
3. TARGET: forward economics — sign(terminal − drift[t*]) (does the path from
   the trigger point onward end favorable?). LogisticRegression, standardized
   on 2024.
4. POLICY: VETO the stop iff P(recover) ≥ p*; p* swept on 2024 by mean net
   ticks/ep (friction 2.4t per round trip, same convention as 103); freeze
   (p*, coefs) to reports/wrongdir/veto_frozen.json.
5. TEST ONCE on the 198: net ticks/ep + day-block CI for the frontier:
   never-bail / plain stop 24t / STOP+VETO / (blind agents +7.5 reference).
   Per-class: false-bail on dipped goods (the 54% line is the one to beat),
   catch retention on wrongs, veto precision/recall at p*.
6. PRE-REGISTERED BAR: STOP+VETO beats plain stop on test net with the
   delta's CI excluding 0, AND dipped-good false-bail < 54% at equal-or-
   better wrong-catch. State PASS/FAIL plainly.
7. COMPOSABILITY hook: emit per-episode veto decisions
   (reports/wrongdir/veto_decisions.parquet: eid, t*, P, vetoed) so the
   reviewer can compose with the 103 re-entry sim without re-running you.

## Rules
New files only: tools/veto_logistic.py, reports/wrongdir/veto_{frozen.json,
decisions.parquet}, reports/wrongdir/veto_logistic.md. Do NOT edit
stop_reenter_sim.py or score_wrongdir.py (another drone owns 103's file;
import/copy helpers instead). RUN SYNCHRONOUSLY; python3.11 from repo root;
commit NOTHING; touch nothing else under exit_dojo. Final message: coefs
(which grammar terms carry), 2024 sweep summary, frozen p*, test frontier
table, per-class analysis, PASS/FAIL, deviations.
