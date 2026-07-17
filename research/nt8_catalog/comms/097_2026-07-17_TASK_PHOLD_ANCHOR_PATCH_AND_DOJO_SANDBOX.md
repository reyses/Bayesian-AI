# TASK 097 — phold anchor patch + rerun (Drone A) · causal Dojo sandbox build (Drone B)
**Doc:** 097 · **Date:** 2026-07-17 · **Author:** Claude (reviewer) · **Status:** TASKS DISPATCHED
Moises: "send the drones to fix it, re-run to make sure, and build the dojo as
causal no lookahead; if we need to build a sandbox so they interact with it
then go ahead."

## Background (both drones read this)
- BAR CONVENTION (core_v2/build_dataset.py:96-111, "Do NOT modify"): a bar
  labeled B covers [B, B+period) and CLOSES at B+period. Causal read at wall
  time t = `searchsorted(ts, t - period, 'right') - 1`. The V2 feature store
  enforces this per higher-TF (`_last_closed_idx`) — higher TFs are CLEAN.
- AUDIT FINDING (2026-07-17): phold_exit_model.py τ-anchors omit the −period
  shift at the 5s base layer → features/drift up to 5s fresher than the
  wall-clock label check. Doc-089 verdicts were NULLS and both models shared
  anchors → conclusions expected to SURVIVE; the rerun proves it.
- TELESCOPE LAW (new, memory `telescope-nested-cadence`): per-TF context is
  constant until that TF's bar CLOSES; the current forming bar at ANY TF
  never appears as a bar — only its closed sub-bars. Builders ASSERT
  `row_ts + period <= frame_ts` (violation = build failure).

## DRONE A (Sonnet) — phold anchor patch + verification rerun
1. `research/nt8_catalog/tools/phold_exit_model.py`: add module constant
   `BAR_S = 5  # 5s base bar; row B closes at B+BAR_S (build_dataset.py:96)`.
   Patch ONLY the two τ-anchor sites (lines ~206 and ~402):
   `ai = int(np.searchsorted(ts_grid, t - BAR_S, side='right') - 1)`.
   Do NOT touch `ei` (entry ref = the trigger close the generator acted on —
   consistent) or `oi` (oracle measurement — hindsight-allowed reference).
2. BEFORE rerun: copy reports/phold_exit_model.md → phold_exit_model_prepatch.md
   and phold_run.log → phold_run_prepatch.log.
3. Rerun from repo root: `python3.11 research/nt8_catalog/tools/phold_exit_model.py`
   (long; RUN SYNCHRONOUSLY, do not background-and-stop).
4. Write `reports/phold_anchor_patch_report.md`: old-vs-new table for FULL/
   BASE AUC + delta + day-block CI, kill-point A/B statuses, flip lead-time
   mode/median, policy capture medians. Expected: tiny drifts, verdicts
   UNCHANGED. If ANY verdict flips: STOP, report, conclude nothing.
5. Commit NOTHING.

## DRONE B (Opus) — the causal Dojo sandbox (stepwise-blind by construction)
Goal: agents can NEVER see a future frame because frames are SERVED one at a
time by a gate that requires a committed decision before the next serve.
All in `research/exit_dojo/` (follow the existing folder conventions;
`builders/episode_builder.py` + `tools/score_decisions.py` are references).

1. `builders/telescope_packet_builder.py` — per-episode frame packets,
   1-min cadence, NESTED-CADENCE layout:
   - Frame 0 wide field: per-TF last-CLOSED V2 feature blocks (named, grouped
     by TF) + OHLC context + entry info (dir, entry P, anchored price=0).
   - Frames k≥1: a layer's block is re-emitted ONLY when that TF's bar closed
     since the previous frame; the current forming bar at any TF appears only
     as its closed sub-bars (e.g. closed 1m bars inside the current 15m).
   - Pilot block kept (drift, leg age/amp/giveback, ER10, vol(5m)+delta,
     KMDR/CLIMAX/HA/PROPP fires with AGE and with/against, close-in-range).
   - CAUSALITY ASSERT everywhere: `row_ts + period <= frame_ts` (incl. the
     5s layer — use the −period shift; the phold bug must not recur here).
   - Output: per-episode JSON for the gate (NOT human-pretty md on disk).
     Truth sidecar (label end, oracle, post-window path) in a SEPARATE file
     the gate never serves.
2. `tools/dojo_gate.py` — the sandbox:
   - `next --episode E`: serves frame k+1 ONLY if frame k has a valid commit;
     each serve includes a fresh random NONCE.
   - `commit --episode E --decision HOLD|EXIT --nonce <n> [--reason "..."]`:
     accepted only with the CURRENT frame's nonce; append-only log
     (ts, frame, nonce, decision, reason). First EXIT is binding (gate stops
     serving; remaining frames never revealed).
   - `finish --episode E --summary "..."`: closes the transcript.
   - No skip-ahead, no re-serve, truth never served. State under
     `reports/full_run/gate_state/`.
   - Nonce chain = the audit: scorer verifies every commit carries the
     serve-time nonce in order → proves sequential play for the served path.
     (Residual risk: an agent could analyze raw ATLAS itself; instructions
     forbid it, the access pattern would show in its transcript, and the
     graduation firewall means no dojo number is ever a result.)
3. `tools/dojo_fleet.py` — drives episodes via headless CLI sessions
   (`claude -p <prompt> --model sonnet`), N-parallel arg, resume-safe (skip
   episodes with a finished gate transcript). The embedded agent prompt:
   play via the gate loop (next→decide→commit), one frame at a time, never
   read episode/truth files or raw data, finish with a summary.
4. Sampling: 200 episodes, 60/60/40/40 winner/midflip/instantfail/chop (the
   pilot taxonomy from post-entry label geometry), one DISTINCT 2025-26 day
   each, EXCLUDING the 10 pilot days. Write `reports/full_run/selection_table.md`.
5. Scoring: extend the scorer to read gate transcripts + verify nonce chains
   (audit PASS required per episode); wrong-side episodes ALSO scored on
   exit-minute percentile (speed is the skill there).
6. VERIFY THEN STOP: 2 scripted-dummy episodes end-to-end (assert causality +
   nonce audit + scoring), then exactly 1 real Sonnet episode via the fleet
   runner. Report results; do NOT launch the 200-episode fleet — the
   reviewer gates that.
7. Commit NOTHING. RUN SYNCHRONOUSLY.

## Reviewer verdict gates
- A: verdict-parity table verified against doc 089 → then commit.
- B: dummy+single-episode evidence verified (causality asserts, nonce audit,
  packet spot-check vs raw parquet) → fleet launch (200 eps, ~4 parallel).
