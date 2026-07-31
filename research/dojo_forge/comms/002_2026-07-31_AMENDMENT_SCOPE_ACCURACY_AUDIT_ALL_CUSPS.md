# 002 — AMENDMENT: scope change — accuracy audit of EVERY cusp label

**Doc:** 002 · **Date:** 2026-07-31 · **Author:** Claude (reviewer) · **For:** AG (executor)
**Supersedes the SCOPE of doc 001** (docs are finalized-on-write; this is the
correction, not an edit). Protocol unchanged: `comms/CLAUDE_AG_REVIEW_PROTOCOL.md`.

## Owner correction (verbatim, 2026-07-31, on waking)

> "B) go to every cusp ground thruth label and review that they are accurate"

The owner's intent for AG is an **accuracy audit of ALL ground-truth labels**,
not a 150-sample owner-tradeability curation. Doc 001's tradeability grading is
**OUT of scope for this loop** (may return as a later loop).

## Revised task

Audit **all 10,682** mechanical cusp labels (R=30, 120 days) for accuracy.

**Inputs (ready, mtimes 2026-07-31):**
- `research/dojo_forge/gate_state/cusp_review/cusps_all.csv` — the COMPLETE
  label set. Columns: day, idx5s, ts, price, kind (T/B), prev_idx, next_idx,
  leg_in_pt, leg_out_pt.
- Labeler source (the thing under audit):
  `research/dojo_forge/tools/cusp_ground_truth.py::hindsight_zigzag`
- Frame renderer (deterministic, re-runnable at any sample size):
  `research/dojo_forge/tools/export_cusp_frames.py`
- The 150 frames from doc 001 remain valid as a visual starter set.

## Required next actions

1. Write your **PLAN as doc `003`**, ending `*(Awaiting Reviewer Verdict)*`.
   No execution before APPROVED.
2. The plan must achieve **100% coverage** of `cusps_all.csv` with an accuracy
   check per label. Recommended (not mandated) funnel — propose your own if
   better:
   a. **Programmatic pass over ALL labels**: for each cusp, verify
      (i) bar-extreme match: `high[idx5s] == price` for T / `low[idx5s] == price`
      for B; (ii) **extremum dominance**: the pivot price is the max (T) / min
      (B) of the ENTIRE span `(prev_idx, next_idx)` — a label that fails this is
      a labeler BUG to report, never to silently fix; (iii) alternation and
      ≥R reversal on both legs; (iv) tie detection: bars elsewhere in the span
      within 1 tick of the pivot price (flag as AMBIGUOUS, with indices).
   b. **Visual review** of every flagged/AMBIGUOUS case **plus a random
      control sample of ≥300 passing labels** (state your seed), using rendered
      frames. 100% visual review is acceptable instead, if you prefer —
      throughput estimate required either way.
3. Output artifacts:
   - `research/dojo_forge/gate_state/cusp_review/cusp_accuracy_audit.jsonl` —
     one row per cusp: `{day, idx5s, kind, checks: {extreme_match, dominance,
     alternation, tie_flag}, visual: PASS/FAIL/AMBIGUOUS/null, note}`
   - `research/dojo_forge/comms/00N_*_EXECUTION_REPORT.md` with pasted raw
     output (counts per check, every failure listed with indices) per the
     claim-evidence rule.
4. **Pre-registered auto-fail tell**: 10,682/10,682 all-pass with zero
   AMBIGUOUS flags is a red flag (tie cases exist in real data — e.g. equal
   highs). If your audit finds literally nothing, audit the audit.
5. Judging note kept from doc 001: cusps are R=30 5s-scale swings; judge at the
   5s panel — some are micro-swings inside larger 1m-context moves and are
   still CORRECT labels at this scale.

**Status: AWAITING AG PLAN (doc 003)**
