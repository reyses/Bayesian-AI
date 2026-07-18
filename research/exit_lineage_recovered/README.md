# Exit-decay lineage — recovered from git history (2026-07-17)

Moises asked what happened to the early exit-decay work and whether it survived
git. It did — deleted from HEAD but preserved in history. This folder recovers
the full ancestry of today's exit / wrong-direction work, oldest → newest, with
commit provenance. Nothing here is live code; it's the paper trail of the idea.

## The lineage (each stage = an ancestor of P_hold + the exit/wrong-dir dojos)

### 1. `genesis_2026-01-31/` — the origin (root commit abf7beeb, 2026-01-31)
The very first exit + learning models, from the "ProjectX/physics" era:
- **`bayesian_brain.py`** — THE original training model. A HashMap Bayesian
  learner: `StateVector → WinRate`, trained on `TradeOutcome`s tagged by
  `exit_reason ∈ {trail_stop, structure_break, time_exit}`. This is the
  "training model very early on" — the direct ancestor of every learned
  edge since (V1 Bayesian → V2 combiner → P_hold → Mamba).
- **`wave_rider.py`** — "Wave Rider Exit System": trailing exit off a
  high-water-mark (ride the wave, exit on the decay-from-peak). The first
  exit-decay logic in the project. Ancestor of the R-trigger / trail exit.
- **`velocity_gate.py`, `state_vector.py`** — the state substrate they read.

### 2. `regret_2026-02/` — early exit-TIMING optimization (~2026-02)
- **`batch_regret_analyzer.py`** — counterfactual regret on exits; the commits
  around it ("Bias regret analysis to prefer early exits over late exits",
  2026-02-14) are the first formal attempt to LEARN exit timing. Ancestor of
  the current `training/regret/` package and the dojo's early-bail scoring.

### 3. `decay_sim_2026-05/` — the two-rule decay strategy (2026-05-10)
- **`sim_decay_rules.py`** — walk-forward simulator for a two-rule decay
  strategy extracted from human cusp picks, around the 1h_high rail:
  RALLY_LONG (ride to the rail) + DECAY_SHORT (fade the overshoot back into
  the band). Iterated marked_v2 → v3 → v3_ext → v3_strict → v4_prob.
- **`bayes_table_magnitude_and_decay.py`** — Bayesian magnitude+decay table.
- **`outputs/`** — 53 recovered run artifacts (trades/daily/summary per version).
- **VERDICT (marked_v4_prob, the final version):** capture ratio **4%**
  (sim $142 vs oracle $3308), median −$30.50/trade, exits dominated by
  hard_stop(12)/time_stop(7). i.e. the hand-extracted decay rules captured
  almost none of the available move — the SAME near-wash exit conclusion the
  rigorous successors reached.

## Why this is the ancestor of the current work (not a lost opportunity)
The through-line is one idea — "confidence decays as the move turns, exit on
the decay" — attempted with progressively more rigor:
- genesis wave_rider (rule/trail) → decay_sim (hand-extracted 2-rule) → **P_hold
  (doc 089, full V2 F-space logistic, anchor-corrected doc 097)** → the exit &
  wrong-direction dojos (blind, gate-audited, running now).
- Every rigorous version found the same thing: the decay curve is REAL but
  exit-on-decay LAGS the flip (~+3 min), so it doesn't beat a dumb hold/stop.
  The decay_sim's 4% capture is the physics-era version of that verdict.

**Bottom line:** the early exit-decay work was not lost and was not wasted — it
posed the question the whole program is still answering. But re-running these
specific artifacts has no forward value: they're superseded by P_hold and the
dojos, which test the same idea causally and honestly. Kept here as the
documented origin.

## Provenance
Recovered via `git show <commit>:<path>` from the newest commit retaining each
file's content. Genesis ← abf7beeb (2026-01-31). Regret ← 8fe88bfd. decay_sim
code ← 44e1f465; oos_decay_analysis ← 4b658e2a (the 2026-05-28 RL-pivot deletion
commit). See `git log --all -- <path>` for full history of any file.
