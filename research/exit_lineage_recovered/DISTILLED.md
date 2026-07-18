---
name: distilled-exit_lineage_recovered
description: Git-history recovery of the exit-decay lineage (2026-01→05) — the ancestor of P_hold and the exit/wrong-direction dojos; already a fully-written archive.
metadata: {type: distilled, topic: exit_lineage_recovered, status: concluded}
---
## Verdict
Recovered 2026-07-17 in response to Moises asking what happened to the early
exit-decay work. It survived in git history though deleted from HEAD. The
folder's own README already gives the full verdict: one idea — "confidence
decays as the move turns, exit on the decay" — run through three
progressively-rigorous stages (genesis rule/trail → hand-extracted decay
rules → today's P_hold + dojos), and every rigorous version reached the SAME
conclusion: the decay curve is real but lags the flip (~+3 min), so it never
beats a dumb hold/stop. Nothing here is live code; it is the documented paper
trail, not a reusable asset.

## Key numbers (with CIs where they exist)
- `decay_sim_2026-05/` marked_v4_prob (final version): capture ratio **4%**
  (sim $142 vs oracle $3,308), median **−$30.50/trade**, exits dominated by
  hard_stop(12)/time_stop(7).
- `decay_sim_2026-05/oos_decay_analysis.md` (separate PyTorch-agent OOS test,
  330 untrained ATLAS days, trained on first 15 days only): cumulative bleed
  to **−$1.4M** — cited in the folder as proof overfitting to a short
  training window doesn't survive regime drift; not folded into the README's
  headline verdict, worth flagging if anyone re-reads this folder for numbers.

## Graveyard / never-retry (if any)
- Hand-extracted 2-rule decay strategy (RALLY_LONG / DECAY_SHORT) off the 1h_high
  rail, iterated v2→v3→v3_ext→v3_strict→v4_prob — 4% capture ratio ceiling,
  superseded by P_hold (doc 089, full V2 F-space logistic).
- Trailing high-water-mark exit (`wave_rider.py`, genesis era) — ancestor of
  R-trigger, superseded.

## Reusable assets
None recommended for re-run — README states explicitly "re-running these
specific artifacts has no forward value." Kept as documented origin only.

## Data locations
- `decay_sim_2026-05/outputs/` — 53 recovered run artifacts (trades/daily/summary
  CSVs per version: marked_v2…v6, oos_*, unmarked_*, strongest_cell_*).

## Open threads
None — the question ("what happened to early exit-decay work") is answered;
superseded by P_hold + the exit/wrong-direction dojos (currently EXEMPT/live
per doc 119's assignment list).

## Sources
- research/exit_lineage_recovered/README.md
- research/exit_lineage_recovered/genesis_2026-01-31/bayesian_brain.py
- research/exit_lineage_recovered/genesis_2026-01-31/wave_rider.py
- research/exit_lineage_recovered/regret_2026-02/batch_regret_analyzer.py
- research/exit_lineage_recovered/decay_sim_2026-05/sim_decay_rules.py
- research/exit_lineage_recovered/decay_sim_2026-05/oos_decay_analysis.md

## Archive recommendation
This folder IS already an archive (title says so, README states "nothing here
is live code"). Recommend KEEP under `research/exit_lineage_recovered/` as-is
— it is a self-contained, dated recovery with full provenance, not something
that needs a project.md or subfolder restructure. Only fold under a unified
`research/archive/` if that convention gets adopted project-wide for ALL
concluded topics simultaneously (see nt8_catalog reviewer phase); doing it for
this one folder alone just adds a path-rewrite with no benefit.
