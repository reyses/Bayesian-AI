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
3. late breakdown recognition → structure-break detector (needs a real feature — see below)
4. forgot the exit → mechanical execution = free alpha, machine gets it by existing

**Measured context** (research/level_hold/, 63 days): micro-reversion strong but
unconditional (any line "bounces" 60%); level long-memory REAL but weak (+1-2pp on
pivot location, p=.007); ai-picks 125s turn label UNLEARNABLE (model AUC 0.500,
supervised null-anchored +0.036).

**Methodology lesson (2026-07-07, mid-session correction from Moises)**: touch-history
"prior visit count" was first measured with a FIXED-TICK zone (8 ticks) → showed fresh
zones break +2-4pp more. Moises corrected: his NT8 bands are 2-SIGMA, i.e. the zone must
scale with the band's OWN current sigma, not a constant ("it's not a set amount").
Redone sigma-relative (zone = k*sigma(t)): the +2-4pp claim did NOT survive — flat at
0.5σ/1σ (buckets ~31-35% / ~20-22%, no dose-response); a reversed, opposite-sign,
small-N (217) effect appeared at 2σ but is exploratory only (found on the 3rd width
tried — multiple-comparisons risk — not trusted without a dedicated pre-committed
test). **RETRACTED the +2-4pp finding.** Net: prior-visit count as a scalar heuristic
shows no reliable trend once the zone is properly volatility-relative. General
lesson: any future probe using a "how near" tolerance must scale it by the local band
sigma, never a fixed tick/point constant — Moises will catch it if it doesn't.

Stage-0 detector = calibrated COMBINER of weak signals (macro position, level memory,
curvature phase — NOT touch-count, that one didn't hold up), judged by confidence-tier
separation, not single AUC. Related: [[reference-mamba-ssm-wsl-perf]]

**Actionable-item pivot (same day)**: rather than wait on the full RL stage-0, Moises
asked to try indicators for manual trading FIRST. Built `6-StructureContext_v1.0-RC.cs`
(NT8 indicator, `docs/nt8/` + live Custom/Indicators, commit fa09c6de) — a state
DASHBOARD (not a signal) showing MacroPosPct (session-anchored position in the tracked
band cluster — fixes Figure_3's macro blindness), NearestLevelSigmaDist (volatility-
relative, per the correction above), and CurvaturePhaseCode (cross-referenced from his
own tested 2-CubicRegressionEndpoint indicator, not recomputed). Deliberately omits
touch-count. NOT YET compiled/tested in NT8 — that's the next step, on him.
Also learned: `3-BayesianBridge` (live TCP bridge to the Python engine) and
`4-BayesianHistoryDumper` (feeds ATLAS_NT8) already exist in his NT8 setup — more
NT8<->Python plumbing than assumed; check `Documents/NinjaTrader 8/bin/Custom/
Indicators/` before assuming something needs building from scratch.
