# Turn-Perception Probes — does the model see turns / is there signal at all?

**Date**: 2026-07-07 · Follows the 50-epoch verdict (SEQ_TRAINER_50EPOCH_2026-07-07.md).
User question: does the trained agent actually detect turns, and do the
hindsight trade labels contain real signal in the features?

Label (both probes): `turn_imminent` = bar within 125 s BEFORE an ai_cusp_picks
exit_ts (the trainer's BCE target). ~7% of bars. Hindsight labels, causal
features. Days: 5 train (2024_02_20…26) + held-out 2024_02_27. N ≈ 96k bars.

## Probe A — the epoch-25 model's exit head vs turn labels

| day | AUC (all bars) | AUC (in-position = train dist) |
|---|---|---|
| 02_20 | 0.539 | 0.531 |
| 02_21 | 0.478 | 0.483 |
| 02_22 | 0.564 | 0.562 |
| 02_23 | 0.478 | 0.479 |
| 02_26 | 0.489 | 0.498 |
| 02_27 (held out) | 0.454 | 0.476 |
| **mean** | **0.5002** | — |

**Verdict: NOISE — the model learned zero turn perception.** Coin-flip even on
its own training days, despite 50 epochs of a dedicated BCE hazard loss
(pos_weight 10.4, w_aux 0.2). This mechanistically explains the over-holding
regression: the exit signal is noise → policy gradient minimizes fee/noise
exposure by never exiting.

## Probe B — is the signal in the features at all? (model-independent)

Leave-one-day-out on the 682 action-independent obs dims (v2+l0+macro+tod),
single-bar snapshot:

| probe | mean LODO AUC | min day | gap vs 0.5 |
|---|---|---|---|
| logistic | 0.5585 | 0.511 | +0.0585 |
| MLP-64 | 0.5061 | 0.462 | +0.0061 |
| shuffled-label null | 0.520–0.525 | — | (+0.02 null inflation) |

**Null-anchored logistic gap ≈ +0.036 → below the 0.05 house bar. Verdict:
the 125s-before-exit label is essentially NOT predictable from single-bar V2
features.** The MLP is *worse* than logistic (overfits 682 dims on 5 days) —
no hidden nonlinear rescue at this sample size. Per-day spread 0.51–0.60 =
unstable.

## Combined read

The RL failure is now fully explained and NOT an RL-algorithm problem:
**the aux perception target was unlearnable as posed.** The agent was asked
to time an exact hindsight timestamp (±125 s) from bar snapshots — a target
even direct supervision can't hit above noise. Everything downstream
(overtrading → over-holding oscillation) follows from having no perception
signal to organize behavior around.

Consistent with project history: direction/amplitude signal EXISTS on
zigzag-family labels (B9 in-trade remaining-amplitude AUC 0.864; direction
classifier 0.864 with the ~83% info ceiling), but **precise turn TIMING has
always been the unsolved bottleneck**. This probe re-confirms the timing wall
from a new angle, on the ai-picks label family.

## Implications for the perception-first curriculum (stage 0)

Training a perception stage against THIS label would fail — supervised can't
learn it either. The label, not the architecture, must change first.
Candidates to probe next (cheap, same harness):

1. **Zigzag-pivot proximity** labels (the project's canonical turn
   definition) instead of ai-pick exit timestamps.
2. **Wider windows / soft targets** — e.g. "turn within 5–15 min" or
   regression on time-to-next-turn; trade timing precision for learnability,
   then tighten.
3. **Sequence-aware supervised probe** — train the Mamba trunk directly on
   the turn target (the model has recurrence the snapshot probes lack); this
   is the miniature version of stage 0 and doubles as its feasibility gate.

Tool: `tools/probe_turns.py` (--probe model|--probe signal); raw outputs in
`reports/probe_turns_{model,signal}.txt`.
