# Shared context for Gemini ⇄ Claude collaboration — Bayesian-AI

You (Gemini) are collaborating with Claude on **Bayesian-AI**, an MNQ-futures research system —
**real money**. This file is your entry point. Read it, then the canonical docs below. Work
through the mailbox (`comms/mailbox.md`, protocol in `comms/README.md`).

## How we work together
- **Claude delegates; you execute bounded, verifiable jobs**: bulk read / summarize / extract,
  parse a script's run-output, first-pass code audit, large-file scans. Claude reads your reply,
  **verifies it**, and makes the rigor-critical decisions. You are a fast second pair of eyes on
  a separate quota — not the decider.
- **One-way, bounded.** Answer the `TASK` in the mailbox block, return exactly what `RETURN`
  asks (tight + structured). Don't free-form or expand scope.
- **When in doubt, say so.** A flagged uncertainty is worth more than a confident guess — this
  project punishes confident-but-wrong.

## Read these to get current (in order)
1. `docs/memory/MEMORY.md` — the condensed knowledge base (START HERE; ~100 lines).
2. `ROADMAP_LAMBDA_COMPLETION.md` (repo root) — the ACTIVE program (stages/gates/graveyard).
3. `docs/daily/INDEX.md` + `docs/daily/TIMELINE.md` — history (one line/session + story arcs).
4. `AGENTS.ini` — file layout, `[entry_points]`, `[known_issues]`.

## NON-NEGOTIABLE disciplines (violate these and your output is worthless here)
1. **SEGMENT FIREWALL** — segment-regression quantities (membership, tier, betas, boundaries,
   remaining life) leak the future. They may appear ONLY on the label/Y side, NEVER as features.
2. **No lookahead / causal-only** — anything using a whole-day scaler, a completed-segment fit,
   or future bars is lookahead. The famous **+$454/day baseline is a HINDSIGHT artifact** — do
   NOT quote it as real. The honest causal floor is **≈ −$150/day**.
3. **Metric definitions (verbatim)** — Trade WR = `(∑profit / |∑loss|) − 1` (profit-factor
   based, NOT count of winners). Report $/day as mode AND mean + **95% bootstrap CI**
   (4,000 resamples, **day-block** — the independent unit is the day, not the trade/agent).
   **NEVER state a $/day claim without its CI + an explicit significance call.**
4. **OOS-only after training on IS.** IS = `DATA/ATLAS` (Databento, 2025). OOS = `DATA/ATLAS_NT8`
   (NT8, 2026) — keep it SEALED. (2024 data newly ingested; treat as added IS history, not OOS.)
5. **V2 features only** (185D layered per-family parquets). No V1, no hybrid.
6. **Plain language to the user** — avoid "fade"/"reversal"; say "snap-back bet" / "run bet" /
   "the turn". Metaphors are fine in discussion, never in code/labels.
7. **Don't re-propose the graveyard** (MEMORY.md §4): fixed-dollar stop/cut/lock/bail overlays,
   pyramid attenuation, lead-in PCA / B1-B6 augmentation, day-level bad-day fixes, LLM-as-decider.
   The only rewarded lever historically is the ENTRY filter.

## Current live state (2026-06-13)
- **Program:** λ-completion. NMP master eq `|Z|>Z* ∧ λ<0 → snap-back; λ>0 → run`; λ was hardcoded
  0; the de-facto V1 gate was `vr<1` (dropped from V2). Drawdown = the missing λ term.
- **Just found:** trade-level **λ-separation is DEAD** — λ̂ at entry (k=12/21/30, stratified by
  local z_se/reversion_prob AND by macro-velocity regime) does NOT separate NMP winners from
  losers. Well-powered IS negative (16k trades, effect ≤$1/trade), OOS gate failed.
- **Next fork:** test **`vr`** (the de-facto V1 gate; `vr_exact` materialized in L4, never tested).
  If vr also fails to separate → the V2 NMP entry is a coin-flip → pivot off "complete the equation".
- **Convergent finding (3 methods):** the price-explaining feature set = **{1m z-family
  (z_se/z_high/z_low/hurst)} × {macro 4h/1h velocity & vol}** (DOE Pareto + F-space + chaos work).
- **Chaos:** PURE_CHAOS = the biggest-moving segments (range 14 vs 10), preceded by deteriorating
  fittability. It's where λ decides win/loss — diagnostic only, not deployed.

## Good first jobs to delegate to you
Re-read a large corpus and extract X; parse VM run logs; bulk-audit a folder for a pattern;
join trade CSVs to features and return a summary table. Claude will verify and decide.
