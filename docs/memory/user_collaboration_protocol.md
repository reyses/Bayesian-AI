---
name: user-collaboration-protocol
description: How the user works through research design. Topic-at-a-time, terse directives, critical-collaborator pushback expected, configurable defaults over preemptive engineering, examples-of-prompts that produced breakthroughs in the regret-oracle arc.
metadata:
  type: user
---

The user has a consistent research-design protocol. Match it.

## Working style

- **Terse, action-oriented prompts.** "ok lets do X", "build it",
  "highlights report", "pushback if you find holes". Match the energy in
  responses — don't over-explain when not asked.

- **Topic-at-a-time when designing.** When the user says "one topic at a
  time" or starts laying out a multi-decision protocol, lock decisions
  sequentially. Don't batch.

- **Critical-collaborator role mandatory.** Explicitly requested
  ("pushback if you find holes"). Surface methodological holes BEFORE
  building. Don't be a yes-man. CLAUDE.md persona codifies this.

- **Configurable defaults > preemptive engineering.** When I propose
  "this won't work because of X edge-case", the user often replies "we
  won't know if we don't try — make it configurable." Build tools with
  defaults matching the user's stated values; let empirical results
  drive adjustment.

- **Empirical-first.** "Run it, then we'll talk" rather than "what should
  the threshold be." Lock-by-running.

- **Walk-me-through requests.** When the user says "im not catcking it"
  or "walk me thru your concern", drop jargon and use plain-language
  examples (concrete numbers, analogies). Don't restate the formal version.

- **Sleep-run handoffs.** "going to sleep, run X then Y then Z." User
  delegates multi-step autonomous work; expects findings doc + journal
  + INDEX update ready when they wake up. Override the "don't run training
  via Bash" rule for these explicit handoffs.

## Prompts that produced breakthroughs (regret-oracle arc)

Each of these single-line directives produced a methodological win.
Reference for understanding the user's mental model:

- **"the most important signal we need is direction"** →
  pivoted target from mfe_dollars to signed_mfe; R² jumped 0.187 → 0.262.
  See [[signed-mfe-pivot]].

- **"if no strong signal is found, then we will proceed to cluster and
  regression on bins so smaller like-to-like samples should help separate
  the shaft from the seeds"** → stratified analysis approach; stratified
  k=2 matched unstratified k=5 with far fewer parameters.

- **"we will first cluster all trades per feature kinda like sedementing
  them on each feature"** → sediment 1D approach (per-feature quantile
  bins) before joints.

- **"lets first do a simple 1D regresión on each feature"** → 1D
  regression as the headline before pair/triplet.

- **"ok let's do paired strataficatication of all features x all features
  same process first clustering then regressions"** → pair-level
  escalation pattern: clusters first, then regression, then escalate to
  triplets.

- **"lets do 3 same approach, then 4 and so on"** → k-way escalation
  (with bin reduction at higher k to manage sparsity).

- **"how about we drop those features?"** → pragmatic fix for
  zero-crossing degeneracy in per-feature 5% matching.

- **"the other option is that we go the regresion line route"** → N-D
  trajectory representation as trade signature (PCA line in 190-D space).

- **"we have like 190+ features per bar"** → corrected my narrow ~19-feature
  pool; pointed to the full V2 stack at DATA/ATLAS/FEATURES_5s_v2/.

- **"option will also open the door to trade decay"** → trade-decay
  insight (d(t) trajectory drift from cluster's PCA line) unifies
  entry/exit/duration/Bayesian-update in one framework.

- **"thtats why im asking for configurable we wont know if we dont try
  the 5%"** → configurable defaults > preemptive engineering.

- **"build the protocol so we dont forget"** → write the spec/project.md
  document BEFORE building; capture open questions explicitly.

## Methodological levers the user reaches for

Recognize when these are about to come up:

| User signal | What's coming |
|---|---|
| "we won't know if we don't try" | configurable defaults; build to test empirically |
| "shaft from seeds" / "like-to-like" | stratification before more features |
| "extremes first" | peel-iteratively, don't density-cluster |
| "pushback if you find holes" | critical-collaborator role active |
| "walk me thru your concern" | drop jargon, use plain-language + concrete numbers |
| "lets address one point at a time" | topic-sequential lock, no batching |
| "build it" / "lets begin" | stop planning, start building |
| "going to sleep, run X" | autonomous multi-step handoff; findings doc + journal + INDEX ready |

## Format preferences

- Tables for comparison (especially when surfacing multiple options).
- Mode + mean + 95% CI in every $/trade or $/day report (CLAUDE.md
  mandate; user enforces).
- Caveats explicit in every findings doc (IS-only, multi-comparison,
  OOS-pending).
- Code references as markdown links to file:line.
- Brief topic headers; dense content under.

See also [[regret-research-methodology]] for the technical workflow.
