---
name: telescope-nested-cadence
description: "Moises' input-side telescope (2026-07-17) — per-TF context held constant until that TF's bar CLOSES; current forming bar NEVER appears as a bar, only its closed sub-bars; first instrument = Exit Dojo full run"
metadata: 
  node_type: memory
  type: project
  originSessionId: 49f1ab8b-f170-41ec-955f-86beb538417f
---

**The nested-cadence telescope (Moises, 2026-07-17, dojo full-run design):**
At any frame, every TF layer shows only its last CLOSED bar, held as a
constant in context until that TF next rolls. The current forming bar at any
TF is represented ONLY by its closed sub-bars (intrabar composition: closed
1s→5s, closed 5s→15s, 3×5s→15s, 4×15s→1m, … up to 1D). Slow layers are
near-constants; fast layers morph — "the only ones morphing are the ones in
context" falls out of the cadence structure, no thresholds.

- Maps 1:1 onto the V2 schema (8 TFs × 25 features): each TF feature block
  refreshes at its own bar close.
- Composes with the older deferred [[telescoping-tf-entry-scope]] idea
  (macro scan → 1m setup → fast-TF ticker) — PROJECT_HISTORY
  `research_telescoping_tf.md`.
- **LOOKAHEAD AUDIT RULE (Moises' own flag)**: using the full CURRENT bar at
  any TF is lookahead — only incomplete-bar's closed children allowed. Same
  law as `_last_closed_idx` in core_v2/build_dataset.py. Builders must
  ASSERT close_ts ≤ frame_ts (build failure, not warning).
- First instrument: Exit Dojo full run — stepwise-blind runner (frames fed
  one message at a time; telescope = stable cached prefix + per-turn fast-
  layer deltas). Status 2026-07-17: design locked, runner not yet built;
  budget pending (rec 200 episodes, 60/60/40/40).

**Why:** input-side telescoping was never implemented anywhere (Phase-5
"telescoping ladders" were outcome-side anchors; the TF-scope memory was
deferred). Sidesteps the doc-089 snapshot null (signal = change-stream, path
form) and the token blowup of raw 409-dim frames.

**How to apply:** any sequential instrument feeding an LLM or model per-bar
context (dojo runner, future Mamba observation docs, chart replay tools)
should use nested-cadence blocks + the closed-bars-only assertion.
