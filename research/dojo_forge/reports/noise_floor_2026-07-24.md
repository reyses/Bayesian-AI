# Teacher label noise floor (doc-152 adoption #5) — 2026-07-24
Probe: 6 episodes (187 frames) relabeled with seed 43 vs the banked seed-42
tiered run; identical genome/harness/ctx.

**RESULT: 2/187 decision flips (1.07%) | median |ΔP| = 0.00000 | max |ΔP| = 0.269**

Interpretation: logit readout is deterministic given tokens; seed only perturbs
allocation/graph planning → reduction-order wiggle → flips ONLY at genuine
near-ties (median exactly 0 confirms). Same physics as E2c.

**Standing rule:** any gen-over-gen claim must exceed what a ~1% random flip
rate could produce, on day-block CIs. Near-tie frames (|p-0.5| small) are the
unstable population; consider reporting gen deltas excluding ties as a
robustness check.
