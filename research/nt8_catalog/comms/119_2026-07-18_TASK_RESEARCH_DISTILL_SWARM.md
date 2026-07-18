# TASK 119 — research distillation swarm (Sonnet) → DISTILLED cards → archive pass
**Doc:** 119 · **Date:** 2026-07-18 · **Author:** Claude Fable (reviewer) · **Status:** TASK
Moises: "set a sonnet swarm to review all the research docs, extract what's
useful, and put them in research archive." Two phases: SWARM distills (no
moves), REVIEWER gates the archive moves.

## Per-topic drone contract (Sonnet, one topic each)
Read the topic folder's README + reports/*.md (+ project.md/cycle docs if
present). Write `research/<topic>/DISTILLED.md`, ≤60 lines, EXACTLY this
template:
```
---
name: distilled-<topic>
description: <one line — the topic's verdict>
metadata: {type: distilled, topic: <topic>, status: <live|concluded|dead>}
---
## Verdict
<2-4 lines: what was asked, what was found, current status>
## Key numbers (with CIs where they exist)
<bullets — only numbers that appear in the topic's own reports; NO invention>
## Graveyard / never-retry (if any)
<bullets with measured costs>
## Reusable assets
<tools/scripts worth keeping, one line each, with paths>
## Data locations
<parquets/stores the topic depends on or produced>
## Open threads
<unfinished questions, if any — else "none">
## Sources
<the 3-8 most load-bearing files, paths>
## Archive recommendation
<KEEP-LIVE | ARCHIVE (reason) — a recommendation only; reviewer decides>
```
Rules: claims must trace to files you actually read (cite the path); numbers
copied verbatim, never recomputed or embellished; if the folder is thin or
already an archive, say so in 10 lines; write NOTHING outside your topic's
DISTILLED.md; commit nothing.

## Topic assignments
- WAVE 1 (concluded-looking, June era): chaos_precursors, cusp_launch_detector,
  edge_case_triage, geometric_exits, kalman_entry, kalman_tuning_eda,
  l5_distribution, leg_clock, level_hold.
- WAVE 2: llm_capability, nmp_state, nmp_strategies, oracle_tests,
  orange_line_eda, order_flow_ablation, recovery_dynamics, regime_clustering,
  regime_markov_causal_test.
- WAVE 3: response_surfaces, reward_design, wick_absorption_signal,
  ai_auto_labeler, fspace_cadence, misc_archive, exit_lineage_recovered,
  exnmp_lineage_recovered.
- SPECIAL (Opus, scoped): nt8_catalog — read reports/*.md + comms docs whose
  filenames contain VERDICT|FINAL|SYNTHESIS|NIGHT_RESULTS only (skip
  raw_articles*, skip task specs); card may run to 120 lines.
- EXEMPT (live, distill-later): exit_dojo, mamba_zigzag_baseline, dojo_forge.

## Phase 2 (reviewer, after cards land)
1. Extend build_memory_db to index research/*/DISTILLED.md (done in this
   commit) → cards flow into shared FTS recall.
2. Review ARCHIVE recommendations; for each approved: run the CLI-false-orphan
   grep (bare filename across docs/, config/, subprocess callers) BEFORE
   `git mv research/<topic> research/archive/<topic>`. Live-referenced topics
   stay put regardless of deadness.
3. Journal + INDEX line + commit.
