---
name: project-moises-trade-postmortem
description: "The 4-figure trade postmortem (2026-07-07) that defines the RL curriculum's north star — four failure modes = four system functions; manual method ≡ NMP equation"
metadata: 
  node_type: memory
  type: project
  originSessionId: 580dfdb1-f5da-48a4-b423-63d89c91bbd5
---

2026-07-07: Moises walked through a real MNQ SEP26 trade via examples/Figure_1-4.png
(~29,400-29,600 prices, 2026 sim). Sequence: (1) shorted the micro-oscillation high
(statistically sound — measured 59-60% rejection), (2) panicked out on noise with a
minor loss (no state-based conviction framework), (3) missed that the oscillation sat
on the cusp of the macro level cluster (~29400) which then broke down; entered late,
it worked, (4) forgot the sim trade and gave it back to breakeven. Figure_4 = the
correct strategy: short macro top → add on cluster-retest → cover at cluster return.

**Key insight**: his Figure-4 strategy IS the NMP master equation (|Z|>Z* ∧ λ<0 →
fade; λ>0 → ride) executed visually at two scales; "where in the curvature we are"
(his cubic read) = λ's sign by eye. The system lacks his LAYERED STATE AWARENESS,
not a new signal.

**Four failures → four system functions** (curriculum north star):
1. micro-entry w/o macro context → multi-scale state rep (stage-0 scope)
2. panic exit on noise → conviction must be state-based, not P&L-based (reward design)
3. late breakdown recognition → structure-break detector (fresh-zone stat is the seed)
4. forgot the exit → mechanical execution = free alpha, machine gets it by existing

**Measured context** (research/level_hold/, 63 days): micro-reversion strong but
unconditional (any line "bounces" 60%); level long-memory REAL but weak (+1-2pp on
pivot location, p=.007); touch history whisper (fresh zones break +2-4pp more); ai-picks
125s turn label UNLEARNABLE (model AUC 0.500, supervised null-anchored +0.036). →
Stage-0 detector = calibrated COMBINER of weak signals (macro position, level memory,
touch history, curvature phase), judged by confidence-tier separation, not single AUC.
Related: [[reference-mamba-ssm-wsl-perf]]
