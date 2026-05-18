---
name: regret-six-layer-architecture
description: The 6-layer architecture of the regret-oracle research arc — L1 oracle (done), L2 direction (done), L3 Bayesian archetypes (pending), L4 selector (missing), L5 execution (missing), L6 validation (missing). Use this frame to organize new work and identify gaps.
metadata:
  type: project
---

The regret-oracle research arc is organized as a 6-layer architecture.
When new work is proposed, identify which layer it touches before building.

**L1 — Trade Seeds via Regret (DONE)**
Tool: `tools/regret_daisy_chain_oracle.py`
Output: `reports/findings/regret_oracle/daisy_chain_IS_full_daisy.csv`
Status: 7,925 sequential non-overlapping trades, 2025 full IS, 5s base TF,
60-min window cap. Sequential ceiling = $1,046,546/yr (~$4,500/day under the
1-hour-budget premise). This is the LABEL UNIVERSE — what's theoretically
capturable. It is NOT a strategy; it requires lookahead (centered-window
extrema detection + best-extreme-in-1h window).

**L2 — Direction Discrimination (DONE)**
Tools: `regret_kway.py`, `regret_stratified.py`, `regret_distribution_eda.py`,
`regret_feature_regression.py`, `regret_pair_clusters.py`,
`regret_triplet_clusters.py`, `regret_pair_regression.py`,
`regret_triplet_regression.py`
Findings: signed_mfe target pivot was decisive ([[signed-mfe-pivot]]).
R² saturates at ~0.35 ([[kway-r2-saturation]]). Stratified k=2 matches
unstratified k=5. Direction-callable cells at 43-59% rate; per-cell
accuracy 82-86% in callable cells; 93% in extreme cells (>90% one side).
Findings doc: `reports/findings/regret_oracle/2026-05-16_direction_signal_kway.md`.

**L3 — Bayesian Table via N-D Trajectory (PROTOCOL LOCKED 2026-05-16, BUILD PENDING)**
Spec: `research/bayesian_archetypes/project.md` (DMAIC frame).
See [[bayesian-archetypes-pending]] for locked decisions + deferred open questions.
The L3 build will add: PCA-line clustering, hierarchical r-ladder,
trade-decay tracking that unifies entry classifier + exit signal +
duration prediction + Bayesian posterior update for direction.

**L4 — Selector / Strategy (MISSING)**
Real-time entry trigger that combines L2 cell-gates, L3 cluster matches,
or an ensemble. Open architectural question: discrete-cell selector (L2-only)
vs trajectory-match selector (L3-driven) vs both.

**L5 — Live Execution Model (MISSING)**
Costs ($0.50/tick × 2-4 ticks slippage + $1-2 commission per round-turn on
MNQ), spread, halt-after-N-losses, intra-day DD caps. Per CLAUDE.md
anti-doom-cascade rule: report under multiple cost/intervention assumptions.

**L6 — Validation (MISSING)**
2026 OOS run of L1+L2+L3 cell-gates and cluster-tables. Day WR + mode $/day
+ 95% bootstrap CI per CLAUDE.md protocol. Per-cell / per-cluster
sign-stability OOS check. Live-vs-sim gap measurement (historical gap ~$680/day
per Day-1 v1.0 evidence per CLAUDE.md).

**Architectural observation**: L2 and L3 answer different questions
(L2 = discrete cell classifier; L3 = trajectory matcher with decay).
They are complementary, not substitutes. The natural test when L3 lands:
does L3 add edge over L2 alone? If yes → ensemble. If no → L2 cell-gate
is good enough and L3 was conceptually rich but practically redundant.
