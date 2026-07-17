# Exit Dojo full run (200 blind episodes) — the pilot's optimism was leakage
**Doc:** 098 · **Date:** 2026-07-17 · **Author:** Claude (reviewer) · **Status:** FINAL
200 episodes, stepwise-blind by construction (serving gate + nonce chain),
**nonce audit PASS on all 200** (no agent saw a future frame). Anchor-corrected
dossier (doc 097) underneath. This supersedes the 10-episode pilot (single-
prompt, could peek).

## 1. Headline — blind, the LLM exit does NOT beat a dumb 5-minute hold
| regime | N | cap mode | cap median | delta vs 5m-hold | 95% CI | beat-5m | oracle-ratio med |
|---|---|---|---|---|---|---|---|
| winner | 60 | +5.0 | +28.9 | **+19.5** | [+8.3,+32.1] **sig** | 63% | 0.29 |
| midflip | 60 | −17.0 | +26.8 | +0.0 | [−10.5,+11.0] ns | 35% | 0.57 |
| instantfail | 40 | −15.0 | −7.6 | **−9.6** | [−15.5,−3.7] **sig LOSS** | 20% | 0.30 |
| chop | 40 | −5.0 | −3.4 | −0.3 | [−2.9,+2.7] ns | 32% | −0.28 |
| **ALL** | 200 | −5.0 | +5.6 | **+3.9** | **[−1.0,+9.3] ns** | 40% | — |

- **Overall delta CI includes zero; beat-rate 40% (< coin flip).** Discretionary
  LLM exit is NOT better than fixed-5m-hold across the board.
- **The ONE real edge is WINNERS: +19.5 pts, significant, 63% beat.** And even
  there the agent captures only 29% of the oracle move — "beats 5m-hold" is a
  low bar when winners run 20-40 min; holding longer than 5m is most of it.
- **Wrong-side is a SIGNIFICANT LOSS (−9.6):** blind agents hold losers hoping
  for a bounce (median bail at 0.33 of the window; only 55% bail in the first
  third). The pilot's aspiration (fast wrong-side bail) did NOT survive blinding.
- **Midflip and chop are washes** (CIs cross zero, beat-rates 35%/32%).

## 2. What the pilot got wrong, and why
Pilot (doc synth, 10 eps, single-prompt): 7/10 beat 5m, median oracle-ratio
0.475. Attention could see future frames despite the commitment contract →
optimistic. Blind: aggregate not significant, oracle-ratios 0.29-0.57, wrong-
side inverts to a loss. **This is the leakage tax, measured: the dojo done
honestly CONFIRMS the program's standing result — exit timing is a near-wash —
rather than escaping it.** Consistent with: brackets dead (091, +3 external
replications), 46 detectors + 409-snapshot can't time turns (089-092), R-trigger
structurally optimal (graveyard §4).

## 3. The grammar is REAL vocabulary but thin EV
EXIT-reason citation (192 binding exits): against-fires-multi 88%, giveback 68%,
PROPP 59%, HA 48%, KMDR 46%, ER10 40%, confluence 18%. The blind agents DID
invoke the confluence grammar the pilot discovered — so the grammar is a
genuine description of what the agents key on. But invoking it blind beats
5m-hold only on winners. **The grammar identifies the turn REGION, not a
tradeable ±edge over a dumb hold** — same ceiling every static approach hit.

## 4. Implications (for Moises' back-to-front chain)
1. **EXIT-GRAMMAR-01 scope NARROWS to the winner-ride case**: "hold through
   givebacks while the trend is intact" (the significant edge), NOT turn-
   detection exits (wash-to-loss blind). This is the B9 continuous-sizing /
   ride-the-runner finding restated from the exit side.
2. **Mamba expectation tempered, HONESTLY**: the state channels the agents
   cited (multi-family against-fire freshness, giveback dynamics, ER10) are
   the right observation inputs, but the ceiling is real — Mamba's exit alpha
   will come from RIDING winners longer, not from timing turns. Do not spec
   the production run to expect turn-timing alpha; spec it to learn the ride.
3. **The unsolved bottleneck is unchanged**: you cannot label "winner" at
   entry. The harvestable version of the winner-edge is a real-time
   "is-this-running" read = the ride/fade (λ) question, still open.

## 5. Artifacts
- `research/exit_dojo/reports/full_run/`: scorecard.md (200 rows + nonce
  audit), synthesis.md (this analysis), packets/, truth/, gate_state/
  (transcripts), selection_table.md, fleet_run.log.
- Tools: builders/telescope_packet_builder.py, tools/{dojo_gate,dojo_fleet,
  score_full_run,synthesize_full_run}.py. Security: fleet allowlist scoped to
  the gate command (no --dangerously-skip-permissions).
- Firewall stands: no dojo number is itself a result; a confirmed rule must
  pass the sealed 2024/2025-26 harness.
