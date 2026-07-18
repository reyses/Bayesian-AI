# TASK 099 — Wrong-Direction Dojo (binary bail-detector, 50/50 blind)
**Doc:** 099 · **Date:** 2026-07-17 · **Author:** Claude (reviewer) · **Status:** TASK (Opus drone)
Moises' redirect after the exit dojo: "now to identify it is in the wrong
direction... we will pass 50/50 good trades and wrong direction." Wrong-side
was the only significant-loss regime in the full run (doc 098: −9.6 pts, 20%
beat, agents hold losers hoping). This dojo isolates ONE job: detect a wrong-
direction entry fast, blind, and beat a dumb stop.

## The question
Given the blind post-entry telescope, can an agent flag a WRONG-direction entry
(label resolves against it) EARLIER or CLEANER than a naive adverse-drawdown
stop — WITHOUT knifing good trades (esp. midflips that dip then recover)?

## Design (reuse the built sandbox; change only population + prompt + scoring)
1. **Population selection** (`tools/select_wrongdir.py`, new): from the SAME
   top-decile P() entry population used by the full run (econ_drift_rows.parquet
   test split, P>=frozen p90, 60s/day/dir de-dup; replicate phold::engagements),
   classify each engagement by LABEL-based truth using the truth-sidecar
   forward drift + oracle:
   - **WRONG** = instantfail (label resolves against entry: oracle_capture<=0
     / immediate net-adverse, per the full-run taxonomy in
     builders/episode_builder.py).
   - **GOOD**  = winner OR midflip (label resolves WITH entry). TAG each good
     episode `good_kind ∈ {winner, midflip}` so the scorer reports false-bail
     separately (midflips = the hard dip-then-recover case).
   - **50/50 balance**: N_wrong = N_good; target 100/100 = 200 total. Distinct
     2025-26 days where possible; may overlap the exit-run days (different
     question). Write `reports/wrongdir/selection_table.md` + selection.json
     with the hidden truth label + good_kind.
2. **Packets**: REUSE `builders/telescope_packet_builder.py` unchanged (frames
   are regime-agnostic; nested-cadence, closed-bars-only assert). Build into
   `reports/wrongdir/packets/`, truth into `reports/wrongdir/truth/`.
3. **Gate**: REUSE `tools/dojo_gate.py` unchanged. Mechanically EXIT == BAIL.
   State under `reports/wrongdir/gate_state/`.
4. **Agent prompt** (`tools/wrongdir_fleet.py`, a thin variant of dojo_fleet.py
   — REUSE the scoped-allowlist launch, NO --dangerously-skip-permissions):
   reframe the job — "Each frame, decide THESIS-INTACT (commit HOLD) or WRONG
   (commit EXIT = bail/flatten). Your ONLY job: if this entry is going the
   wrong way and won't recover, bail FAST; if it's a good trade that will
   resolve your way (even if it dips first), hold to the end. First EXIT is
   binding." Same one-frame-at-a-time gate loop; never read truth/packets/raw.
5. **Scoring** (`tools/score_wrongdir.py`, new): nonce audit (reuse the chain
   verifier) + the binary confusion:
   - bailed := episode has a binding EXIT. pred=WRONG if bailed else GOOD.
   - **catch-rate (recall on WRONG)** = P(bail | wrong);
     **false-bail (fallout on GOOD)** = P(bail | good), reported OVERALL and
     split by good_kind (winner vs midflip);
     **precision** = P(wrong | bail).
   - **speed**: on caught wrong-siders, bail minute + %ile-of-window (earlier
     better; mode-first).
   - **economics (ticks, mode-first)**: net vs never-bail. Bail on WRONG =
     ticks saved (adverse drift avoided after bail); bail on GOOD = ticks
     forgone (favorable drift given up after bail). Report the net-ticks
     distribution and median with day/episode bootstrap CI.
   - **THE HONEST BAR — dumb-stop ROC**: on the SAME episodes, sweep a naive
     "bail if adverse drift <= −X ticks" over X ∈ {a sensible grid}, compute
     each X's (catch-rate, false-bail, net-ticks). Plot/table the dumb-stop
     ROC. The agent's single operating point MUST sit ABOVE that curve
     (higher catch at equal false-bail, or lower false-bail at equal catch)
     AND beat the best-X net-ticks to be a real result. State PASS/FAIL on
     this explicitly. Also report never-bail as the floor.
   - Output `reports/wrongdir/scorecard.md` + `synthesis.md`.
6. **VERIFY THEN STOP**: 2 scripted-dummy episodes (one seeded-bail, one
   seeded-hold) through the gate + scorer (assert confusion math + nonce
   audit); then exactly 1 real Sonnet episode via wrongdir_fleet.py. Report.
   Do NOT launch the 200 fleet — reviewer gates it. Commit NOTHING.

## Reviewer gates
- Verify dummy confusion math + nonce audit + a packet causality spot-check +
  the dumb-stop ROC baseline is real (recompute one X by hand) → authorize the
  100/100 fleet.
- After fleet: reviewer writes the verdict doc (100), journals, commits.

## Ladder discipline
Fable=spec+verdict; you (Opus)=build, RUN SYNCHRONOUSLY (never background-and-
stop), commit nothing, skip-rather-than-fabricate, claim=evidence. python3.11
from repo root (bare `python` is a broken shim).
