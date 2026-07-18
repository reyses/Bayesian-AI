# TASK 122 — the entry-fail RED X: what separates terminal-good from terminal-bad AT ENTRY?
**Doc:** 122 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **Status:** TASK (Opus drone)
Moises' thesis: "the best way not to bite our nails on ride-or-eject is not
getting into fails." Fact: 49.7% of top-decile entries end ≤−4pts — because
entry P was trained on direction-agreement, not terminal economics. This is
the Shainin contrast, sealed.

## Population & truth
- TRAIN: 2024 engagements from the wrongdir/powered scan machinery
  (select_wrongdir engagement cut, split=train), terminal labels at BAND=4
  (good ≥+4 / bad ≤−4 pts; dead-band excluded from FIT, included in volume
  accounting).
- TEST: the 23,378-engagement 2025-26 population (single shot, frozen model).

## Entry-time features (ALL causal at fire ts; document each join)
1. entry P (from econ_drift_rows) — also THE baseline (see bar).
2. det one-hot (which stream fired) + consensus if joinable.
3. Leg geometry at ts: pivot_age_min / sig_with_leg from the fire's own
   signal_rows_<det>.parquet row (join ts+det), plus the _tf_state leg
   block if cheaply computable.
4. λ̂ at ts (z_se store, NMP_K=21 — the NMP-LAMBDA machinery).
5. NMP9 tier at ts (the cached waterfall; 'none' when no tier).
6. tod (session-time), trail_vol (ticks).

## Deliverables
1. **The Shainin contrast table**: good-vs-bad distributions per feature
   (mode/median + day-block CI on the difference) — which variable DOMINATES
   the separation. This is the primary读out even if the filter fails.
2. Logistic P(terminal-good | features), 2024-fit, frozen → single-shot test:
   AUC + calibration; **incremental AUC over the P-only model** (same fit
   protocol) — the honest question is the increment.
3. **Filter frontier** on test: good-rate delta vs base across retained-volume
   levels; judged ONLY at three PRE-REGISTERED operating points frozen from
   the 2024 curve: the thresholds that retained 70% / 50% / 30% of 2024
   volume. At each: good-rate delta vs base AND vs the P-only filter at its
   equal-volume threshold, day-block CIs.
4. Decomposition at each point: what gets sacrificed (goods lost, split
   dipped/clean; dead-band share; fails avoided).

## Pre-registered bar (PASS/FAIL plainly)
At ≥1 pre-registered operating point with retained volume ≥30%: filtered
terminal-good rate beats BOTH (a) unconditional base and (b) the P-only
filter at equal volume, with the delta-vs-P-only CI excluding 0. The P-only
comparison is the whole point — beating base alone just re-discovers P.

## Rules
Files: research/nt8_catalog/tools/entry_fail_redx.py +
reports/entry_fail_redx.md. RUN SYNCHRONOUSLY; python3.11; sealed (nothing
tuned on test); friction irrelevant (no trading sim — rates only);
commit NOTHING; don't touch dojo_forge/exit_dojo gate dirs. Final message:
contrast table top-5, incremental AUC, the three operating points with CIs,
PASS/FAIL, deviations.
