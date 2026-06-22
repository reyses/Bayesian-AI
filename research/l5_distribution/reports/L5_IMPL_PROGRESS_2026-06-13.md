# L5 ldist layer — implementation progress (resumable)

> SPEC (authoritative): the **ADDENDUM** at the bottom of
> `comms/HANDOFF_2026-06-13_session_day_nt8.md`. This file = live status so any session/agent can
> resume. Update the checkboxes as you go. Claude is implementing 006 directly (Gemini did 005).

## Status legend: [ ] todo  [~] in progress  [x] done  [!] blocked/flagged-for-user

## BUGS (address first)
- [x] **BUG-LIVE-LOOKAHEAD — FIXED + PARITY-VERIFIED (2026-06-14, user-approved).**
  `core_v2/live_features.py get_v2_vector`: added `anchor_ts = last_5s_open + TF_SECONDS['5s']`
  and `tf_df = tf_df[tf_df['timestamp'] <= anchor_ts - tf_secs]` before each TF's compute -> drops
  the still-forming bar, mirroring `_last_closed_idx`. Validated by `research/test_live_lookahead_parity.py`:
  live get_v2_vector == offline parquet on 27 higher-TF checks (1m/1h/4h × velocity/sigma/z_se × 3
  mid-day 2024 anchors), incl. the slow TFs. Also confirmed pandas-resample bins == offline floor bins.
  (Original note kept below for context.)
- [~] ~~BUG-LIVE-LOOKAHEAD (pre-existing, L1-L4, real-money)~~ — RESOLVED above. `core_v2/live_features.py`
  `get_v2_vector()` (line ~144) resamples the full 5s buffer per TF and takes the LAST row — which for
  any TF>5s is the FORMING (not-yet-closed) bar. Offline `build_dataset` uses `_last_closed_idx`
  (build_dataset.py:98/215) = last CLOSED bar. => live higher-TF L1-L4 features are train/serve-skewed
  vs the training data. Contradicts the file's "100% offline parity" docstring; the historical parity
  test (tools/test_live_v2_parity.py) can't see it (all bars closed in history).
  **EXACT FIX (deferred — changes live decision inputs, needs user OK + parity re-validation):**
  in `get_v2_vector(ts)`, for each `tf != '5s'`, after resample, prune to closed bars before
  compute/last-row: `tf_df = tf_df[tf_df['timestamp'] <= ts - tf_secs]`. (5s bars are already closed via
  on_bar.) This mirrors offline `_last_closed_idx` exactly (a tf bar with open `o` closes at `o+period`;
  usable iff `o+period <= ts` ⇔ `o <= ts - period`). NOT applied this session — sequencing: L5 live
  integration isn't needed until L5 passes the edge-test (Stage B), and this fix should ship as its own
  reviewed change. **ACTION FOR USER: approve fixing the live L1-L4 forming-bar prune.**
- [x] **BUG-STEPFILL-PERIOD (design, in the NEW L5 code).** L5 step-fill MUST use `period=TF_SECONDS[tf]`
  (NOT 5 or 1) on the L5 `bar_ts` array — addressed by building it correct from the start (see L5-2).
- [x] **BUG-RETURN-SHAPE (design).** `compute_L5_ldist` returns ONE ROW PER CLOSED TF BAR, not
  per-1s — built correct from the start (see L5-1).

## OFFLINE L5 (safe, do now)
- [x] **L5-1: SFE `compute_L5_ldist(self, df_1s, tf)`** — DONE + TESTED. Implemented in
  `core_v2/statistical_field_engine.py`: module consts `TF_SECONDS`, `L5_OUTLIER_K=2.0`,
  `L5_MIN_SAMPLES_MOMENTS=4`, `L5_QUANTILE_METHOD='linear'`, `L5_EPS` (after `OU_BOUNDARY`); shared
  per-group `_ldist_group_stats(bar_ts,c,x,tf)` (pure numpy, pinned conventions); method
  `compute_L5_ldist` after `compute_L0`. Groups 1s by `(ts//period)*period` (1D = one-bar-per-session,
  labelled by first ts, to match build_1d_from_1m); returns ONE ROW PER CLOSED TF BAR (`bar_ts` + the
  12 battery cols); n<L5_MIN_SAMPLES_MOMENTS -> NaN skew/kurt/outlier; level = OLS intercept at bar end.
  Used a pure-python group loop (PERF TODO: Numba grouped-quantile kernel for fast TFs over full
  history). Test `research/test_l5_ldist.py` PASS (battery == numpy ref; shape; 1D; small-n guard).
- [x] **L5-2: build_dataset** — DONE + VERIFIED. Added `_load_1s_day` helper and a dedicated **L5 pass**
  (section 6, after the L0 pass): per day, load 1s ONCE, then per TF `compute_L5_ldist` →
  `_align_to_anchor(l5['bar_ts'], l5_feat, anchor_ts, period=TF_SECONDS[tf])` → write
  `FEATURES_5s_v2/L5_{tf}/{day}.parquet`. Loads 1s once/day (not per-TF). AST parses; TF_SECONDS
  imported @ :63; `research/test_l5_causality.py` PASS (no forming bar ever used; warmup NaN; correct
  for partial first bars). [SUPERSEDES the old plan below.]
- [x] ~~L5-2 (old plan)~~ — In `core_v2/build_dataset.py`, per-day loop (~:375-401), AFTER the
  L1-L4 layer loop and BEFORE the `del tf_bars, features` cleanup (~:400): add a per-day 1s load
  (`_load_1s_day(atlas_root, day)` mirroring `_load_anchor_day` @ :145, path DATA/ATLAS/1s/{day}.parquet)
  then per TF:
    `l5 = sfe.compute_L5_ldist(df_1s, tf)`  # one row per closed tf-bar, has 'bar_ts'
    `aligned = _align_to_anchor(l5['bar_ts'].to_numpy(), l5.drop(columns=['bar_ts']), anchor_ts, period=TF_SECONDS[tf])`
    `_write_family(aligned, _family_path(atlas_root, f'L5_{tf}', day), SCHEMA_VERSION)`
  CRITICAL: `period=TF_SECONDS[tf]` (NOT 5/1) and `tf_ts = l5['bar_ts']` (the L5 bar opens). Verify
  `_align_to_anchor`'s exact arg order/signature @ build_dataset.py:206-223 before wiring (the snippet
  above assumes (tf_ts, feature_df, anchor_ts, period) — CONFIRM). Output: FEATURES_5s_v2/L5_{tf}/{day}.parquet.
- [x] **L5-3: features.py STAGE A** — DONE + VERIFIED. Added `_l5_ldist_names(tf)` (12 cols, matches
  compute_L5_ldist emit order, no _{N} suffix) + registered `LAYER_FAMILIES['L5_'+tf]` for all 8 TFs.
  Did NOT touch FEATURE_NAMES / assemble_v2_grid. Verified: `import core_v2.features` OK, N_FEATURES=297
  (assert green), 8 L5 families, L5 NOT in FEATURE_NAMES.
- [x] **L5-4: tests** — DONE. `research/test_l5_ldist.py` PASS (battery vs numpy ref, shape, 1D, small-n
  guard) + `research/test_l5_causality.py` PASS (step-fill causal: no forming bar ever used, warmup NaN).

## GATE (validate before bake) — NEXT, needs the build first
- [ ] **USER: run the V2 feature build** to materialize L5 (heavy, user-runs):
  `python core_v2/build_dataset.py --atlas DATA/ATLAS --fresh` (or the IS range). This writes
  `FEATURES_5s_v2/L5_{tf}/{day}.parquet` alongside L0-L4. Then spot-check a parquet has the 12 L5 cols.
- [ ] **EDGE-TEST**: do any L5 features separate NMP trade outcomes at the |z|>1.85 entry tail (OOS,
  day-block CI)? Reuse the `test_ldist_wedge.py` harness: join L5_{tf}_ldist_* onto the nmp_fade_raw
  trades by entry_ts (last-closed), test per-feature separation. Only survivors advance to Stage B.
  (Write as `research/test_l5_edge.py`, results -> reports/findings/.)

## STAGE B (only after the gate passes)
- [ ] **L5-5**: add `_l5_names` to FEATURE_NAMES loop + fix the magic `37` in the N_FEATURES assert
  (features.py:168) SAME commit + update describe_feature_count() + the stale `139D` docstring.
- [ ] **L5-6 live**: add L5 to `get_v2_vector` via the SAME integer-floor `compute_L5_ldist` on the live
  1s buffer (NOT pandas .resample), last-closed only. Requires BUG-LIVE-LOOKAHEAD fixed first. Parity
  test live==offline byte-identical.

## Battery (exact cols, no _{N} suffix)
`L5_{tf}_ldist_{min,q1,median,q3,max,mean,std,skew,kurtosis,n,level,outlier_pct}` (+ optional
iqr/range/rejection). TFs: 5s,15s,1m,5m,15m,1h,4h,1D.

## Verified code facts (grounding 2026-06-13)
- SFE kernels = Numba `@njit(parallel=True)` CPU (NOT GPU). `_ols_fit_kernel` @ sfe:86, `_rolling_std`
  @ :228. compute_L1 @ :493, L2 @ :566, L3 @ :655, L4 @ :752. `self.windows`=N_BASE @ :467.
- N_BASE = {5s:9,15s:12,1m:15,5m:9,15m:12,1h:12,4h:18,1D:5}. TF_ORDER=[5s,15s,1m,5m,15m,1h,4h,1D].
- build_dataset: `_last_closed_idx` @ :98, `_align_to_anchor` @ :206-223, period set @ :236, per-day
  loop ~:375-401, L5 insertion ~:400, write via `_write_family`/`_family_path`.
- features.py: LAYER_FAMILIES @ :179-207 (add `L5_'+tf`), naming `L{n}_{tf}_{metric}`, N_FEATURES assert
  magic `37` @ :168 (= 8L1+9L2+8L3+12L4; note L4 carries a 12th `vr_proxy` computed OUTSIDE the SFE).
- live: `get_v2_vector` @ live_features.py:120-190 (the buggy resample @ :144).

## OVERNIGHT AUTONOMOUS REBUILD (started 2026-06-13, user asleep)
Found the 2024 ATLAS was CORRUPT: 1s re-ingested today (session boundaries) but MERGED onto old UTC
files (no clean delete) -> ~1,596 duplicate/mislabeled bars/day + overlap across adjacent files; and
5s/1m/all TFs were STALE (UTC-partitioned, not rebuilt from the new 1s). Plan, staged with verify gates:
  1. [DONE] CLEAN: deleted 3,315 2024 TF parquets + 19,998 stale 2024 FEATURE files (2025 preserved).
  2. [DONE] re-ingest 2024 1s (clean) -> build_timeframes. Exit 0; all aggregation validations PASS.
  3. [DONE] VERIFY CLEAN ✓: 1s & 5s same start every sampled day at correct DST session boundary
     (23:00 UTC winter / 22:00 summer); file purity (session_day==filename); ZERO cross-file overlap
     across all 259 2024 1s files (259 not 303 = Sundays correctly merged into Mondays' sessions).
  4. [DONE] build_dataset --start 2024-01-01 --end 2024-12-31 (NOT --fresh; 2025 preserved). Exit 0,
     L5 section ran, no errors. 8 L5 families, 259 files each = matches L1 exactly.
  5. [DONE] L5 BUILD VERIFIED ✓: L5_1m 2024_06_20 rows==L1 rows, timestamp==L1 anchor, 12 cols match
     family, warmup NaN, no infs, sane stats (n 4/44/60 per 1m, std~0.97pt; 1h n~2495 std~8.9pt).
  6. [DONE] PRELIMINARY EDGE first-look (research/test_l5_edge.py -> reports/findings/L5_edge_2024_preliminary.md):
     used the AVAILABLE self-contained test (forward snap-back at |z|>1.8481, day-block CI) instead of the
     blocked NMP-trade test. RESULT (259 day-blocks, 324,479 tail entries): WELL-POWERED NEGATIVE — ALL 12
     L5_1m features Spearman ~0 (|rho|<0.013), every 95% CI includes 0; baseline fade entry itself +0.074pt
     CI[-0.12,+0.28] (not +EV). => On 2024/1m forward proxy, L5 adds NO entry-filter edge. Stays STAGE A
     (materialized, NOT in FEATURE_NAMES). Consistent with the wedge (close≈ldist_level) + NMP-entry-unsolved.
     CAVEAT: forward-proxy not NMP-PnL; 1m only (macro 4h/1h L5 std/skew UNTESTED — natural next probe given
     the "macro vol matters" convergent finding); single-TF.
Each stage gates the next — will NOT build features on unverified data. Results land here + daily journal.

## Log
- 2026-06-13: progress doc created; live lookahead bug verified + fix documented (deferred). Starting L5-1.
- 2026-06-13: L5-1 DONE+TESTED (compute_L5_ldist + _ldist_group_stats + consts in SFE;
  research/test_l5_ldist.py PASS).
- 2026-06-13: L5-2/L5-3/L5-4 DONE+VERIFIED. build_dataset L5 pass (causality-tested), features Stage A
  (import green, 8 families, not in FEATURE_NAMES), both unit+causality tests PASS. **OFFLINE L5 LAYER
  COMPLETE.** NEXT (needs the build): USER runs build_dataset --fresh to materialize L5 parquets, then
  EDGE-TEST (research/test_l5_edge.py vs nmp trades, |z| tail, day-block CI). Stage B (FEATURE_NAMES +
  magic-37 assert bump + live integration) ONLY after the gate passes. BUG-LIVE-LOOKAHEAD still awaits
  user OK. Files touched: core_v2/statistical_field_engine.py, core_v2/build_dataset.py,
  core_v2/features.py; tests research/test_l5_{ldist,causality}.py.
