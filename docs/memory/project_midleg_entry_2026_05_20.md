---
name: midleg-entry-research
description: Mid-leg / missed-signal late-join research — REJECTED. Zigzag legs are sequential, engine runs at 99.9% utilisation, no parallel-signal population exists.
metadata:
  type: project
---

Mid-leg entry (late-joining an R-trigger the live engine missed) was
researched 2026-05-20 (autonomous sleep-run, 51-day sealed OOS) and
**REJECTED — do not build.** Full report:
`reports/findings/regret_oracle/2026-05-20_midleg_entry_research.md`.

**The structural fact (the durable takeaway):** hardened zigzag legs are a
*sequential partition* of the price path — leg N exits at a pivot, leg N+1
is born from that pivot. 99.8% of consecutive-leg gaps are exactly 0. A
1-contract greedy engine runs at **99.9% utilisation** and catches **2,922
of 2,926** OOS legs; it misses only 4 in 51 days, and all 4 are losers.
There is **no population of legs missed because the engine was busy.**

**Why:** the zigzag produces one leg at a time; legs essentially never run
in parallel. This is structural — it will NOT change with more data or with
contract scaling. Mid-leg / missed-signal / parallel-position / "catch the
signal we lost" ideas are all moot for the same reason.

**How to apply:** if a future session proposes catching missed signals,
late-joining, or running parallel positions on this zigzag pipeline — point
at this finding first. The premise (overlapping signals) is false. The one
legitimate "add to a running leg" action is pyramiding the leg you are
ALREADY in, which B9's continuous sizing already covers.

**Experiment results:**
- E1 Fork 1 (B9-gated, *unconstrained*): +$303/day @ K=5 [CI +160,+457] —
  +EV but over a population that does not exist live. Note this is the value
  of B9-*gating* late entries, not new money (baseline already enters at K=0).
- E2 Fork 2 (B1-B6 pivot-structure augmentation): −$76/day @ K=5 [CI −158,−2]
  significantly negative; neutral-to-harmful at all K. B1-B6 ARE
  in-distribution mid-leg (trained on all 1m bars), but the B9 GBM already
  extracts pivot-structure from the raw V2 features — stacked B1-B6
  predictions add only noise (same as the rejected lead-in PCA, see
  [[feedback_leadin_pca_rejected]]).
- E3 (position-constrained 1c sim): incremental −$1/day, 4 late-joins / 51d.
- E4 (overlap): 99.9% utilisation, 99.8% gaps = 0.

**What the "lost signals" the user sees actually are:** cold-start — ~7
first-of-session legs/day lost to the ~20-min (240-bar) feature-engine +
zigzag warmup (the engine cannot compute during warmup), or deliberate B7
skips. The real fix is **zigzag-state priming** (replay `detect_swings` over
prior-day history at startup + pre-warm the V2 feature engine), NOT late-join.
Priming was recommended to the user, awaiting approval as of 2026-05-20.
