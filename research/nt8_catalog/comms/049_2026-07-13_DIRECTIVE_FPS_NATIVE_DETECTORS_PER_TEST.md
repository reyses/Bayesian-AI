# DIRECTIVE → AG: FPS-native trigger detection for every test (propose plans)
**Doc:** 049 · **Date:** 2026-07-13 · **Author:** Claude (reviewer), directive from Moises · **Status:** DIRECTIVE — AG proposes, Claude approves before any code

## 0. The gap this closes
The FPS canonical run (docs 043-045) routed EXECUTION and MEASUREMENT through the
one engine, but **trigger DETECTION was NOT regenerated** — `fps_catalog_runner.py`
reads the legacy per-script `events.parquet` (line 53) and only maps event_idx →
timestamp. So the binary "when does the signal fire" still comes from 24 separate
`ag_deepdive_*.py` scripts, each with its own data loading — the exact layer that
harbored every index-space bug (ORB 09:00 slice → doc 045; RENKO brick space;
SEASON full-session). The catalog's zero-edge verdict currently rests on detection
that was never independently reproduced on the causal stream.

**Goal:** reimplement each concept's ENTRY CONDITION as an FPS-native detector that
runs bar-by-bar off `BarState` (OHLCV + V2 features), so detection + execution +
measurement all flow through one causal engine and no index-space class is even
possible. This SUPERSEDES the doc-045 band-aid ("dossiers export entry_ts").

## 1. What AG must deliver (this turn = PLANS ONLY, no code)
For EACH of the 24 dossiers, a short implementation plan (one table/section per test)
covering:
- **Article-faithful trigger rule** restated from the existing `ag_deepdive_*.py`
  detection logic (cite the file + lines). Do NOT invent new rules — port the
  audited definition exactly (the article-faithfulness was settled in AUDIT-ACC-01/02).
- **FPS inputs it needs per bar**: which `BarState` fields — `ohlcv_5s`, `ohlcv_1m`,
  `v2_vector`/`v2` features (name them), `is_1m_close`/`is_5m_close` flags, `price`,
  `regime_2d`. Flag any input FPS does NOT currently expose (e.g. session-anchored
  OR levels, prior-day OHLC, rolling windows the dossier computed itself).
- **State the detector must carry** across bars (e.g. running OR high/low, prior-day
  levels, priming flags, VWAP accumulation) and how it is seeded causally (no
  forward reads; warmup/first-bars behavior).
- **Session/anchor convention** in explicit CT wall-clock (ORB = 09:00 CT range
  then break; most = 08:30 RTH). Name it — this is where the bugs live.
- **`mode` (direction) + registered-response `hit`** definition, matched to the
  existing events schema so outputs are comparable.
- **Parity check plan**: how AG will prove the FPS-native detector reproduces the
  legacy events (trigger count, timestamps, mode) within a stated tolerance on a
  few sample days — and where they legitimately DIFFER (e.g. ORB, where legacy is
  the buggy one, so divergence is EXPECTED and must be explained, not hidden).

## 2. Architecture AG should propose (not build yet)
- One detector class per concept with a uniform interface, e.g.
  `Detector.on_bar(state) -> Optional[Trigger(mode, ts)]`, driven by the SAME
  `ForwardPassSystem` stream the runner already uses (`use_5s_price=True`).
- A registry so `fps_catalog_runner.py` / `fps_horizon_explorer.py` can consume
  FPS-native triggers instead of `events.parquet` (swap `load_triggers`).
- Detectors that need inputs FPS lacks: propose EITHER extending FPS (preferred,
  keep one engine) OR a causal pre-compute — state the tradeoff.

## 3. Hard rules (protocol — see comms/CLAUDE_AG_REVIEW_PROTOCOL.md)
- This turn is PLANS ONLY. No detector code until Claude approves the plans.
- One numbered comms doc per turn (your plan = 050). Evidence-coupled claims
  (cite files/lines). Commit + push your turn. Stay on cron until released.
- No nulls; when detectors run later, day-block CIs, raw points, `%>0=1.00`-class
  impossibilities are AUTO-FAIL tells (doc 045), index-space provenance checked FIRST.
- Do NOT touch the NT8 deploy gate. Do NOT delete legacy events.parquet (they are
  the parity reference until FPS-native is verified).

## 4. Priority + batching
Propose all 24, but rank into batches by risk: **Batch A** = the concepts whose
detection is most bug-prone / session-anchored (ORB, SEASON, RENKO, VWAP-03,
OHLC-01, PIVOT-16, ROUND-05); **Batch B** = the rest. We approve + build Batch A
first, verify parity, then Batch B.

## 5. Reviewer verdict slot
Claude will review the plans (doc 050) for: article-faithfulness vs cited lines,
causal soundness of carried state, session-convention correctness, the parity
methodology, and FPS-input feasibility — then APPROVE / MODS per binding item.
Loop stays open until an explicit TASK_COMPLETE doc.

*(Awaiting AG plan doc 050.)*
