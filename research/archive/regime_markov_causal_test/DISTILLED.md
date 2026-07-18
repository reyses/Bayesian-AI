---
name: distilled-regime_markov_causal_test
description: Never executed — no output artifacts exist; folder is two untested scripts plus empty DMAIC stubs
metadata: {type: distilled, topic: regime_markov_causal_test, status: dead}
---
## Verdict
Folder intends to causally test (1) whether a Markov transition model over
regime-cluster labels beats a marginal baseline at predicting the NEXT regime
segment, and (2) whether a "SMEP" full-feature model beats a vol-only baseline
at early-predicting a segment's volatility tier (`regime_causal_earlypredict.py`).
Neither script has ever produced output: their required inputs
(`artifacts/stage2_year_segments.json`, `artifacts/regime_buckets.json`) and
output paths (`reports/findings/regime_markov_test_summary.txt`,
`reports/findings/regime_earlypredict_summary.txt`) do not exist anywhere in
the repo. `README.md` and `project.md` are unfilled stubs (no Define/Measure/
Analyze/Improve/Control content). No claim can be made either way — this is
an unexecuted design, not a null/negative result.

## Key numbers (with CIs where they exist)
None. No summary files, no printed output, no numbers of any kind exist in
this folder or anywhere it writes to.

## Graveyard / never-retry (if any)
None recorded — nothing was ever run to graveyard.

## Reusable assets
- `research/regime_markov_causal_test/regime_markov_causal_test.py` — Laplace-
  smoothed Markov transition-matrix test vs marginal baseline, day-block
  bootstrap (2000 resamples) + sequence-shuffle null (100 resamples),
  pre-committed KEEP/DEAD decision rule. Design is sound (day-block CI, null
  control) but depends on missing upstream artifacts.
- `research/regime_markov_causal_test/regime_causal_earlypredict.py` —
  GradientBoostingClassifier "SMEP" (L1_5s+L2_5s features) vs vol-only
  baseline for volatility-tier early prediction, day-block bootstrap +
  label-shuffle null. Same missing-input problem.

## Data locations
- Expects `artifacts/stage2_year_segments.json` and `artifacts/regime_buckets.json`
  (repo-root-relative) — NOT FOUND in current repo state.
- Expects `DATA/ATLAS/FEATURES_5s_v2/{L1_5s,L2_5s}/<day>.parquet` — existence
  not verified as part of this distillation (script never ran to reach them).

## Open threads
- Never run: were the upstream artifacts (`stage2_year_segments.json`,
  `regime_buckets.json`) produced by `regime_clustering` and later deleted, or
  never generated? Needs a run to determine if the Markov/SMEP causal tests
  say anything at all.

## Sources
- research/regime_markov_causal_test/README.md
- research/regime_markov_causal_test/project.md
- research/regime_markov_causal_test/regime_markov_causal_test.py
- research/regime_markov_causal_test/regime_causal_earlypredict.py

## Archive recommendation
ARCHIVE (never executed, no findings, no dependencies satisfied — dead weight
until/unless someone regenerates the missing artifacts and reruns; low cost
to archive, trivial to resurrect since the test design itself looks fine).
