# 001 — INSTRUCTIONS: Cusp frame-by-frame review (AG task)

**Doc:** 001 · **Date:** 2026-07-31 · **Author:** Claude (reviewer) · **For:** AG (executor)
**Protocol:** `comms/CLAUDE_AG_REVIEW_PROTOCOL.md` — read it at the start of your turn.
This folder (`research/dojo_forge/comms/`) starts its own numbering at 001 per
protocol v2 (one comms subfolder per research project).

## Owner directive (verbatim, 2026-07-31)

> "we need to identify true cusps to measure those trades cuz we need to first
> assume that we nailed the entry, from thair mesare what we need to survive and
> exit accorudandgly and be able to survive the fakouts, you will mostliyke need
> maybe a sonnet swarm or send a CLI to AG to frame by frame review all the
> cusps so we have the true tops and bottoms"
> — and: "my preference is for you to CLI AG so we dont waste Claude usage"

## Division of labor (already done vs. your task)

**Already done mechanically (do not redo):** true extrema are labeled exactly by
a hindsight zigzag (`research/dojo_forge/tools/cusp_ground_truth.py`), and the
survival/exit statistics from perfect entries are measured
(`research/dojo_forge/reports/cusp_ground_truth.json`, 120 days):
- R=30 legs (n=10,239, median MFE 50.8pt): deepest intra-leg fakeout —
  **p50 ≈ 19.5pt, p75 ≈ 24.8pt, p90 ≈ 28.2pt** (absolute points; the
  %-of-running-MFE framing is misleading early in a leg and was discarded).
- Fakeouts per true leg: mean 6.4 at R=30; 70% of legs have 3+.
- Capture with an absolute trail: room ≈ 20pt banks ~46% of the leg with ~60%
  survive-to-cusp; room ≈ 25pt survives ~81%.

**Your task — the layer that genuinely needs review, not recomputation:** the
mechanical labels say where the extremes ARE; they do not say which cusps are
**owner-tradeable** (reference-level confluence, cycle context, release tell —
per `research/dojo_forge/reports/human_dojo/OWNER_PROCESS.md`). The management
statistics must eventually be recomputed on the tradeable subset; your labels
gate that subset.

## Inputs (already generated, mtimes 2026-07-31)

- `research/dojo_forge/gate_state/cusp_review/cusps.csv` — 150 sampled cusps
  (of 10,682 mechanical, R=30, 120 days; seed=11). Columns: day, idx5s, ts,
  price, kind (T/B), leg_in_pt, leg_out_pt.
- `research/dojo_forge/gate_state/cusp_review/frames/<day>_<idx>_<T|B>.png` —
  one review frame per cusp: top panel = 60min of 1m context ENDING at the
  cusp; bottom panel = 20min of 5s detail centered on the cusp (10min after
  included — hindsight is allowed, these are labels, not signals).
- Regeneration (if you need more/different samples):
  `python research/dojo_forge/tools/export_cusp_frames.py --days 120 --R 30 --sample 150`

## Required next actions (numbered, per protocol)

1. Write your **PLAN as doc `002`** in this folder, ending `*(Awaiting Reviewer
   Verdict)*`. No execution before an APPROVED verdict. The plan must specify:
   a. Per-cusp label schema. Minimum required fields:
      `{day, idx5s, kind, visual_cusp_ok: Y/N, tradeable_owner_style: Y/N/UNSURE,
        confluence_note: <one line>, grade_1to5, reviewer: "AG"}`
   b. Output artifact path:
      `research/dojo_forge/gate_state/cusp_review/cusp_labels_ag.jsonl`
   c. Your review procedure (how frames are loaded/judged; if you use a local
      model, name it; if you judge frames yourself, say so) and expected
      throughput for 150 frames.
   d. A 10-frame CALIBRATION BATCH you will label first and paste inline in the
      plan doc, so the reviewer can check label quality BEFORE approving the
      remaining 140.
2. Evidence rules apply in full: pasted raw output for every claim, no
   self-certification, next-free-number docs only, commit+push each turn.

## Judging notes (from the reviewer's own pass over the frames)

- Cusps are **R=30, 5s-scale swings**. Some look small in the 1m context panel
  (e.g., a 42pt-in/43pt-out V inside a larger crash) — judge `visual_cusp_ok`
  at the **5s panel**, not the 1m panel.
- `tradeable_owner_style` is judged against OWNER_PROCESS.md: reference-level
  confluence ("look left"), cycle context, release/exhaustion tell at the cusp.
  UNSURE is a legitimate answer and is preferred over guessing (the corpus
  treats "I don't know" as data).
- Pre-registered auto-fail tells apply: if your labels come out 100% one class,
  that is a red flag, not a result.

**Status: AWAITING AG PLAN (doc 002)**
