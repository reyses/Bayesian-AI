# Research Project: Mid-Leg Entry (Missed-Signal Late-Join)

Opened 2026-05-20. Autonomous sleep-run.

## Define
The live L5 engine misses R-trigger entries: cold start (zigzag not primed),
or busy in another position (1 contract / 1 position constraint). The user
observed genuine lost signals on the NT8 chart (`examples/example1.png` —
~9 swing legs over 5 h). Question: when the engine is flat and a missed
R-trigger's leg is STILL running, can it late-join the leg and capture
positive EV?

A late-joiner enters at bar K (K bars after the missed R-trigger), rides to
the leg's hardened zigzag-pivot exit, capturing
    `remaining_pnl_usd = exit_pnl_usd - pnl_usd_so_far`.
This is EXACTLY B9's training target, so B9 is in-distribution for the
late-join decision (the K-row feature vector is identical whether or not you
entered at the original trigger).

## Measure
- OOS: `reports/findings/regret_oracle/trade_trajectory_OOS_full.parquet`
  (51 sealed days, 2,926 legs, K ∈ {5,10,30,60,120} in 5s units = 25s..10min).
- IS:  `trade_trajectory_IS.parquet` (threshold calibration only).
- Models: `b9_remaining_amplitude_K{K}.pkl`; B1-B6 pivot-structure models.
- Metrics (CLAUDE.md): $/day mode + mean + 95% bootstrap CI (4000 resamples);
  significance = CI excludes 0. Friction $6/leg primary; sensitivity {0,6,12}.

## Analyze — cycles
- cycle_01 / E1 — Fork 1: B9-gated unconstrained late join. Opportunity
  ceiling. Cheapest test; if even this is non-significant, mid-leg entry dies.
- cycle_02 / E2 — Fork 2: augment B9's feature set with B1-B6 pivot-structure
  model predictions. Does pivot-structure stacking add OOS lift?
- cycle_03 / E3 — Position-constrained 1-contract sim. The operational truth:
  greedily take legs, mark genuinely-missed legs, late-join the still-running
  ones gated by B9. Incremental $/day vs no-late-join.
- cycle_04 / E4 — Other options: pullback-filtered join, B10 day-regime gate,
  b5 leg-phase as a standalone gate.

Per-cycle PDCA (Plan / Predicted / Actual / Verdict) is recorded in the
consolidated findings report rather than separate cycle files.

## Improve
No improvement to deploy. All five experiments converge on: do not build
mid-leg entry. E1 (unconstrained) is +EV but E3 (position-constrained) shows
the engine already catches 2922/2926 legs -- there is no missed-leg
population. E2 (B1-B6 augmentation) is neutral-to-harmful. The one actionable
finding is that the live "lost signals" are a cold-start problem; the fix is
zigzag-state priming -- a separate, live-engine change (awaiting approval).

## Control
Project CLOSED 2026-05-20. Verdict: mid-leg entry shelved -- structural (the
sequential leg partition leaves no parallel-signal population; will not
change with more data or contract scaling). Consolidated report:
`reports/findings/regret_oracle/2026-05-20_midleg_entry_research.md`. If ever
revisited, the premise (overlapping/missable signals) must be re-checked
first -- E4 shows 99.8% of consecutive-leg gaps are exactly 0.

## Artifacts
- Tools: `tools/forward_pass_midleg_entry.py` (E1),
  `tools/midleg_b1to6_augmented.py` (E2),
  `tools/midleg_constrained_sim.py` (E3).
- Consolidated report: `reports/findings/regret_oracle/2026-05-20_midleg_entry_research.md`

## Prior context (paradigm caveat)
The L5 paradigm boundary (MEMORY.md) found binary cut/cap/preempt actions on
the EXIT side fail 6×; only continuous sizing on signed amplitude (B9) wins.
Mid-leg ENTRY is a binary join/skip action (1-contract constraint forbids
continuous sizing). Binary actions have a poor track record here — failure
mechanism is Type-1 cost on marginal predictions. The late-join gate must
therefore be CONSERVATIVE: a joined-bad-leg costs real money; a
missed-good-leg costs only opportunity. Expect the optimal threshold ≥ friction.
