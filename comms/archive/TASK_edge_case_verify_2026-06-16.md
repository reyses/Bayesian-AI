# TASK → Gemini — Independent verification of the 40 edge-case trades (3-way verify)

This is step 3 of a 3-way verification (Claude ✓ → human ✓ → **you**) that produces the
ground-truth labels which will become **Gemma's few-shot teaching set** for triaging all 6.8k trades.

> **Verify INDEPENDENTLY. Do NOT echo Claude's labels.** The whole point is an independent set of
> eyes to catch bias — Claude has a documented over-conclude bias on this work. Where you disagree
> with Claude's (or the human's) call, say so in a `PUSHBACK:` note. Catching a wrong label is the
> high-value contribution here, not agreement.

## Context (settled — don't re-derive)
- These are the **CLEAN, gap-guarded** GA-Kalman trades (the prior set had session/roll gap artifacts
  — losses defeating the $100 stop, e.g. −$454; those are removed, worst clean loss now −$127).
- The strategy verdict is already established and is **not your task to re-litigate**: on clean data
  it's **break-even-to-negative, not significant** (trade-level). Regret showed entries are
  *right-direction-but-late*. You are verifying **trade-by-trade behavior**, not the edge.
- **Work at TRADE LEVEL** (the user dropped day-stats — don't reintroduce $/day or day-block CI).
- These are non-causal hindsight inspections → **diagnostic only** (firewall-fine).

## Inputs
- 40 trajectory plots: `reports/findings/edge_cases_clean/trade_*.png` (signed PnL path; green=entry, red=exit, ▲=MFE, ▼=MAE).
- Manifest: `reports/findings/edge_cases_clean/edge_case_manifest.csv`
  (columns: tid, arch, dir, net_usd, mfe_pts, mae_pts, dur_min, gap_close, proposed_*, **verify_claude**, verify_human, **verify_gemini**←you).

## Your job — for each of the 40 trades, looking at the PLOT
Fill `verify_gemini` with your independent call on:
1. **Archetype right?** Does the path SHAPE match its label (CLEAN_RIDE / GAVE_BACK / CHOP / STOPPED / SMALL_WIN/LOSS)? If not, give the correct one.
2. **Entry quality** — did it fire on a real developing move, or a false-start / mistimed entry (price went against it fast)?
3. **Exit quality** — did the 79pt trail give back the peak (GAVE_BACK), was the stop reasonable, or did it exit well?
4. **The 8 FLAGGED trades** (Claude flagged): tids **2218, 774, 208** = ~whole-session (13–22h) holds — *confirm they aren't spanning a session boundary*; tids **5846, 971, 263, 4229, 6037** = near-instant trades — *confirm they're real, not stitching artifacts.* These are the ones that most need your eyes.

## Return
- The filled `verify_gemini` column (or a tid→label list Claude will merge).
- A `PUSHBACK:` block listing any tids where you disagree with Claude's `verify_claude` (with why).
- A 2–4 line **SUMMARY** + the **LOCATION** of anything you wrote (per `comms/CONTEXT_FOR_GEMINI.md`).

## Disciplines
Standard `comms/CONTEXT_FOR_GEMINI.md` rules apply (independent, flag uncertainty, no scope creep,
no secrets in comms, statistical language). Don't re-run the backtest — this is pure trade inspection.
