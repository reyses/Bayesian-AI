# TASK SPEC — PROP-TURN-P (P-modulated proportional turn) — ready to execute
**Doc:** 094 · **Date:** 2026-07-17 · **Author:** Claude (reviewer) · **Status:** TASK
(execution pending — first launch died at session-limit; relaunch scheduled)

Moises' design (verbatim intent): "use the confidence P() to tune the signal
dynamically — in theory we should have high conviction that it is turning after
the flat and the giveback." Static baseline killed in doc 093; this is the
concept family's last stand under a pre-registered kill rule.

## EXECUTOR SPEC (hand to an Opus worker verbatim; run synchronously; commit nothing)
1. READ FIRST: comms/093 (static kill + ratified escape clause + REVOKED 60/day
   cap), reports/propturn.md, tools/propturn_tune_and_capture.py, the PROP-TURN
   block at the end of tools/dossier_signal_pipeline.py (reuse _propturn_core;
   do not modify existing code — append only).
2. P_turn MODEL (2024 ONLY): at each 1m boundary during a leg, causal features:
   {giveback fraction g, stall_min, A pts, leg_age_min, A/21 ratio, ER(10),
   minutes-since EXITKMDR fire against leg dir (cap 30), same for TURNCLIMAX and
   TURNHA, trail_vol (60×5s std)}. Target: interior label turn within next 3
   minutes. LogisticRegression, standardized on 2024.
3. DYNAMIC TRIGGER: r_eff = r_hi − (r_hi−r_lo)·clip((P_turn−p0)/(p1−p0),0,1);
   fire when g ≥ r_eff (escape clause + A_min floor retained).
   TUNE 2024 ONLY: r_lo∈{.03,.05,.08} × r_hi∈{.15,.25,.35} × (p0,p1)∈{(.2,.6),
   (.3,.7)} × A_min∈{10,15} = 36 cells. OBJECTIVE (corrected per 093): max
   dir-recall@±2m s.t. direction-correctness(near-turn) ≥ 0.80 AND lead-median
   ∈ [−2,+1] min. NO fires/day cap (report rate transparently). Freeze winner.
4. TEST (2025+26, frozen only): full turn scorecard ±1/2/3/5m + standing bar;
   MUST-BEAT deltas vs static (dir-recall@2m 0.042 / precision 0.102); capture
   stop-and-reverse sim (secondary; net-of-friction, ratio vs 0.5-0.8 budget);
   league line 'PROP-TURN-P' via the pipeline.
5. KILL RULE (pre-registered): beat static on BOTH dir-recall@2m AND
   precision@2m with non-overlapping day-block CIs, else the proportional-turn
   family (static+dynamic) is CLOSED — state plainly.
6. OUTPUTS: generator 'PROP-TURN-P' (frozen coefs via json, no pickle);
   tools/propturn_p_tune.py; reports/propturn_p.md; propturn_p_frozen.json;
   propturn_p_run.log; signal_rows_PROPTURNP.parquet.
7. Reviewer then verifies (reproduce scorecard from parquet + frozen json),
   writes doc 095 with the verdict, journals, commits.

## Context for whoever resumes
- Ladder: Fable=spec+verdict; Opus workers=builds; sealed-2024 discipline;
  claim-evidence coupling; skip-rather-than-fabricate.
- Board state: entry solved; brackets dead (091); 46 static detectors + 409-dim
  snapshot can't time turns (089-092); static PROP-TURN dead (093); this task
  is the last pre-sequential candidate; then the Mamba handoff spec.
