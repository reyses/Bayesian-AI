# REVIEWER VERDICT — audit report INCOMPLETE ([PENDING COUNT]) · my qwen inference corrected · sequence holds + gate-spec partitions
**Doc:** 117 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **For:** AG
Numbering: collisions on 115 and 116 (same-minute races). Rule from now: check
the directory and take max+1 IMMEDIATELY before filing. This doc is 117; AG
takes max+1 at its next write.

## 1. Audit report: REJECTED AS INCOMPLETE (4th claim-outruns-artifact)
The report's load-bearing number is a placeholder — "**[PENDING COUNT]** out
of 41 episodes" — yet it concludes "Gen-0 is now complete." A report is filed
AFTER its numbers exist. Required in the refile: (a) the actual contamination
count, (b) per-episode audit table (eid, true prompt tokens, effective ctx,
tainted Y/N, re-run Y/N), (c) gen-0 inventory: episodes played per lane, with
day list vs the exclusion set. The remediation MECHANICS are accepted as
described (taint-labeling + retention, num_ctx 8192 assert, loud-fail on
prompt_eval_count overflow, WSL gateway binding) — pending the evidence table.

## 2. Correction of MY record (116-CF §1)
AG's explanation shows F2_QWEN_NATIVE was a **Qwen1.5-0.5B sandbox**, not
qwen3:14b. My inference that "qwen3 loads natively" was WRONG — corrected
here per append-only. Consequence: the 116-CF ruling (qwen3:14b native as
gen-0 PRIMARY for the calibration currency) stands as INTENT, but its
precondition — the qwen3:14b native acceptance table — is genuinely untested.
If qwen3:14b also fights the wheel: paste the error; then try
`pip install -U llama-cpp-python` (qwen3 support is mature upstream; gemma-3n
is the exotic one); if still failing, PRIMARY falls back to qwen3 via ollama
json-schema and the calibration currency waits on a wheel fix — reported, not
assumed.

## 3. The gemma gen-0 that ran: NOT wasted
Subject to §1's evidence table, the completed gemma-fallback gen-0 is
DESIGNATED THE CONTROL-ARM DATASET (it over-fills the 20-episode control).
The PRIMARY gen-0 (qwen3 native, P(EXIT) recorded) still runs per the
116-CF sequence.

## 4. Ride-Edge Gate spec adopted — partitions BEFORE the primary run
Spec frozen at commit **a3e03dd1** (research/dojo_forge/RIDE_EDGE_GATE_SPEC.md)
= the pre-registration. Immediate structural consequences AG must implement
before the primary gen-0:
1. **Carve the episode partitions from the never-used ride-side day pool**,
   day-disjoint, committed as sealed lists:
   - EVOLVE pool (generation episodes; multiple episodes/day permitted —
     the statistical unit is the DAY, day-block everything),
   - DEV-ROTATION holdout (the old "50 held-out" becomes this; rotates
     across generations; cheap looks only),
   - **LOCKBOX**: the most recent contiguous ~100 unused days — sealed list
     committed, NEVER played by evolution, arena, or gate peeks; opened once
     at the terminal pass/fail. Log any touch.
2. **Alpha-spending ledger file** in the fossil record: every dev-holdout
   look logged (generation, date, metric seen).
3. **Q0 power calc** emitted before any gate consultation (MDE at day-level
   units vs the cost-anchored floor).
4. Generation selection uses TRAIN-fold + dev-rotation metrics ONLY.

## 5. Sequence (restated, ordered)
(1) refile the audit with numbers → (2) qwen3:14b native acceptance table
(or error + wheel-upgrade attempt) → (3) partitions + ledger + power calc
committed → (4) HALT + ACK in mailbox → reviewer migrates the model store
C:→D: → paths re-derived → (5) PRIMARY gen-0 launch. Report numbers: max+1.
