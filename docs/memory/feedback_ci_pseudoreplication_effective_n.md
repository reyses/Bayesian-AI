---
name: feedback-ci-pseudoreplication-effective-n
description: outcome CIs must count unique market trades / days, not agent-votes or correlated samples (effective-N, block-bootstrap)
metadata:
  type: feedback
---

The unit of independence for an **outcome** confidence interval is the **unique market
trade**, not the number of measurements. Found 2026-06-08 in research_A's
`evaluate_is_mastery_gate` (`training/rl_engine/train_gpu_research_A.py`): it used
`stderr = std/sqrt(raw_N)` while up to 128 agents share one network → deterministic
greedy → byte-identical trades → `raw_N >> effective_N`. The code even computed
`effective_N = len(set(metadata))` then ignored it. Result: CI ~√(raw/eff)× too
narrow → the automated mastery gate passes on noise and the curriculum advances
prematurely.

**Why:** duplicating a measurement adds zero information. A diverse agent fleet helps
a CI only by producing more *distinct* trades (coverage), never by re-voting the same
one. Same-entry/different-exit trades are also NOT independent — they ride one realized
price path (correlated). In a historical backtest there is **one fixed path**, so the
honest independent block is ultimately the **day**.

**How to apply:** for any $/edge significance claim or gate, dedup/cluster to unique
trades (or unique entries), and prefer **block-bootstrap by day** — which is exactly
what the canonical $/day metric already does (4,000 resamples over days). Never divide
by a raw count of correlated/duplicated samples. This is the agentic-RL instance of
the existing metric-definitions CI rule. See [[reference_fista_gpu_cv_step_bug]] for
the other 2026-06-08 shared-math finding. Gate fix flagged but not yet applied.
