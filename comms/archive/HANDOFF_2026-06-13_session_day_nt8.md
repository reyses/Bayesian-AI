# HANDOFF → Gemini — 2026-06-13 (session-day + NT8 roll + convert RAW_NT8→parquet)

> Read `comms/CONTEXT_FOR_GEMINI.md` first (disciplines). Claude VERIFIES your output
> (the established safeguard — you've had join bugs before). Full session detail:
> `docs/daily/2026-06-13.md` (Change Reports: volume-roll, session-day, dumper v2.4.x).

## STATE — what changed this session (all tested, committed-pending)
1. **Contract roll fix** — `DATA/pipeline/databento_to_atlas.py`: replaced the calendar-month
   map with a DATE-KEYED `FrontMonthSelector` on the **NT8 roll calendar** (verified against the
   user's NT8 install): roll = **Monday of expiry week = 3rd-Friday − 4 days**. Helpers:
   `_build_roll_calendar(start_yr,end_yr)` → sorted `[(roll_date, from_sym, to_sym)]`;
   `_front_for_day(date, events)` → front symbol. Monotonic, spreads excluded, `_ROLL_OVERRIDES`
   hook (empty — rule matched NT8 4/4 for 2024). Guard test `research/test_front_month_roll.py` PASS.
   Verified 2024 roll-ins: H4 2023-12-11, M4 2024-03-11, U4 2024-06-17, Z4 2024-09-16.
2. **Session-day boundary** — NEW `core_v2/sessions.py`: `session_day(ts)` + vectorized
   `session_day_array(ts)`. DST-aware CME reopen **17:00 America/Chicago** (→ 22:00 UTC summer /
   23:00 UTC winter). A "day" = one continuous session (Sun-eve reopen → next 16:00 CT close,
   labelled by CLOSE date so Mon starts Sun). Maintenance halt at the EDGE, never mid-day. Test
   `research/test_session_day.py` PASS. Wired into databento_to_atlas (partition + roll grouping).
   **Inheritance (verified):** ATLAS session-partitioned → build_timeframes copies filename →
   build_dataset keeps day-stem → run_strategy `day` col → day-block bootstrap. Whole analysis
   layer inherits; only databento_to_atlas needed changing. `L0_time_of_day` stays UTC-clock (correct).
   **METRIC RE-BASELINE**: $/day, Day WR, day-block CIs are now per-session-day — NOT comparable to
   prior UTC-day numbers. Must apply to IS AND OOS identically or they desync.
3. **Dumper** — `docs/nt8/BayesianHistoryDumper.cs` → **v2.4.1** (deployed to NT8 Indicators,
   recompiled by user): session-day partitioning (C# TimeZoneInfo "Central Standard Time"),
   instrument-mismatch safety warning, verbose Output-window diagnostics (per-state + per-series
   first-bar + write try/catch + 0-bars warning), and COMPLETE/summary lines now print the covered
   first→last UTC range per TF. User is re-dumping MNQ_06-26 (06-26 rolls to 09-26 **Mon 2026-06-15**).

## ⭐ PRIMARY TASK — build the maintained RAW_NT8 → ATLAS_NT8 converter
The dumper header points to `tools/sourcing/convert_nt8_csv_to_parquet.py` but it does NOT exist;
the only impl is archived: `archive/DATA/pipeline/convert_nt8_csv_to_parquet.py` (read it — good base,
but: hardcodes CONTRACT='MNQ_06-26', partitions by CSV FILENAME, only does 1m/1s/5s, no roll-stitch).

Build `tools/sourcing/convert_nt8_csv_to_parquet.py` that:
1. **Inputs**: `DATA/RAW_NT8/{contract}/{tf}/*.csv` for ALL contracts present (MNQ_06-26, MNQ_09-26, …).
   CSV cols: `timestamp,open,high,low,close,volume` (UTF-8, may have BOM — strip it).
   Conventions (from the archived file): 1s ts = top-of-second (no shift); 1m ts = top-of-minute
   → **+59** to bar-close; rebins use NT8 close convention `(ts // period) * period + (period-1)`.
2. **SESSION-DAY partitioning (load-bearing)**: do NOT trust the CSV filename. Compute the day with
   `from core_v2.sessions import session_day_array` on each bar's UTC ts, then write one parquet per
   **session-day** → `DATA/ATLAS_NT8/{tf}/{YYYY_MM_DD}.parquet` (flat, no contract subdir). This makes
   OOS match the IS convention regardless of which dumper version wrote the CSVs.
3. **Roll-stitch contracts** on the SAME NT8 calendar as IS: for each session-day, use the FRONT
   contract via `_front_for_day` (import from `DATA.pipeline.databento_to_atlas`, or replicate the
   calendar). So a day uses MNQ_06-26 until the 2026-06-15 roll, MNQ_09-26 after. Where both contracts
   have the same session-day (overlap around the seam), the front contract wins.
4. **Timeframes**: produce every TF the feature build needs — 1s, 5s, 15s, 30s, 1m, 5m, 15m, 30m, 1h,
   4h, 1D (check `core_v2/build_dataset.py` / `core_v2/features.py` for the exact required set). Derive
   the coarser TFs from 1s/1m with the NT8 close-label convention; or — cleaner — emit 1s+1m parquets
   then run the existing `DATA/pipeline/build_timeframes.py` against an ATLAS_NT8 root (confirm it can
   target ATLAS_NT8, not just ATLAS).
5. **Verify** (`--verify`): rebin must be byte-parity with existing `DATA/ATLAS_NT8/5s/2026_03_20.parquet`
   (the archived file's verify mode). ALSO assert your session-day labels equal `core_v2.sessions.session_day`
   on a sample of timestamps (IS/OOS consistency).
6. **Report** to `reports/findings/` (project rule: tools write results to file): per-TF row counts,
   first→last session-day, contract used per day, any gaps.

## DISCIPLINES (mandatory — see CONTEXT_FOR_GEMINI.md)
- **No-lookahead**; session-day must match IS exactly; report N + date ranges.
- **Do NOT** delete/overwrite the user's RAW_NT8 CSVs or the `MNQ_06-26 backup/` folder.
- **Do NOT** run NT8 or training. Python only. Write the tool to `tools/sourcing/`, add to TOOLS_INDEX.
- Leave a mailbox reply (MSG 005) with the RETURN block; Claude will verify before it's trusted.

## SECONDARY FOLLOW-UPS (after the converter)
- **Seam-mask** the feature pipeline: NaN-mask ~N_BASE warmup bars after each `rolled` day (from
  `DATA/ATLAS/roll_manifest.csv`) so no trailing window straddles a contract price jump (contango).
  NO price back-adjustment (short windows are locally roll-invariant; only the seam window is hit).
- **Sweep** standalone research scripts for any that recompute "day" from a raw UTC ts instead of the
  `day` column (those would desync from session-day); most inherit via the column — fix the few that don't.
- After the user re-ingests 2024 (databento_to_atlas + build_timeframes), confirm
  `DATA/ATLAS/roll_manifest.csv` shows `rolled=True` on 2024-03-11/06-17/09-16/12-16, `calendar_fallback=False`.

---

# ADDENDUM — L5 intra-bar distribution (ldist) feature layer

> Spec locked with the user over 2026-06-13; grounded against the REAL code (SFE / build_dataset /
> features / live_features) by a mapping+adversarial workflow. The corrections in §10 OVERRIDE any
> conflicting note you might infer from the archived/reference code — follow this addendum.

## ⚠ 0. NAMING OVERLOAD (read first)
`L5` is ALREADY a load-bearing name in the LIVE stack = the **L5 zigzag decision engine**
(`live/l5_decider.py`: `L5Decider`, `L5Context`, `--engine-mode l5`, `live/state/l5_overlay.txt`,
`L5_STATE`). This new **L5 feature LAYER** does NOT collide with those code *symbols* (different
namespace — our columns are `L5_{tf}_ldist_*`, families `L5_{tf}`, dirs `FEATURES_5s_v2/L5_{tf}/`),
so it is safe to build. But "L5" is now ambiguous in chat/logs/grep ("L5 broke" is unparseable).
DECISION: keep `L5_{tf}_ldist_*` (the `ldist` token disambiguates), and **state the overload in the
SFE + features docstrings**. Do NOT touch any `L5Decider`/engine symbol.

## 1. What it is
`L5 = the descriptive statistics of the WITHIN-BAR 1s-close distribution, computed per TF.` The box
plot + moments of each TF bar's constituent 1-second closes, turned into features. Additive — it does
NOT replace OHLCV or re-ground anything (the wedge test already showed `ldist_level ≈ close`, no
edge as a *grounding*; L5 is the untested *additive* payload).

## 2. Battery — EXACT column names (NO `_{N}` suffix — it's within-bar, no rolling window)
`L5_{tf}_ldist_{min,q1,median,q3,max}`  (5-number box plot)
`L5_{tf}_ldist_{mean,std,skew,kurtosis}`  (moments)
`L5_{tf}_ldist_n`  (count of in-bar 1s sub-bars — the weighting companion; ALWAYS emit)
`L5_{tf}_ldist_level`  (OLS-fitted center at bar end — the test_ldist_wedge intercept)
`L5_{tf}_ldist_outlier_pct`  (% of 1s closes beyond mean ± `L5_OUTLIER_K`·σ)
Optional derivable (materialize per "we won't know if we don't try", or let downstream derive):
`L5_{tf}_ldist_{iqr (=q3−q1), range (=max−min), rejection (=close−level)}`.
≈12 core (×8 TFs = 96 cols), ~15 with derivables (~120). Examples: `L5_1m_ldist_median`,
`L5_5s_ldist_n`, `L5_1D_ldist_outlier_pct`, `L5_4h_ldist_level`.

## 3. The SFE function (the cornerstone — ALL L5 math lives here)
`def compute_L5_ldist(self, df_1s: pd.DataFrame, tf: str) -> pd.DataFrame:`
- Input = RAW **1s** OHLCV df (cols incl. `timestamp` int64, `close`). Derives `period = TF_SECONDS[tf]`
  internally (do NOT pass a raw seconds arg — invites mismatch; no magic 59/60 at call sites).
- Group by integer floor: `bar_ts = (timestamp // period) * period`. Compute the battery over each
  group's 1s **closes**. `level` = OLS intercept evaluated at bar end (x referenced to `bar_ts+period-1`,
  derived from `period`, NOT literal 59).
- **RETURN = ONE ROW PER CLOSED TF BAR** (indexed by/with a `bar_ts` column), length = #tf-bars in the
  input — this DIVERGES from L1–L4's "same length as input" because L1–L4 are fed coarse-TF bars (1-in-
  1-out) while L5 is fed raw 1s and AGGREGATES. (The grounding's "return identical length" note is WRONG
  for L5 — see §10.)
- `n<2` → NaN moments (mirror `_ols_fit_kernel`'s guard); always emit `n`. Do NOT pre-NaN-gate thin
  fast-TF bars beyond the n<2 degeneracy guard.
- **Quantiles ≠ free**: the SFE kernels are **Numba `@njit(parallel=True)` CPU**, NOT GPU — there is no
  CUDA quantile primitive. The 5-number summary needs a sort/partition per group. OFFLINE: write a
  dedicated Numba grouped-quantile kernel (sort within each contiguous `bar_ts` segment) — do NOT use
  `groupby.apply(np.quantile)` over full history (slowest layer by far on 5s/15s). LIVE: O(n) single
  `np.partition` over the current bar's 1s closes — trivial (this is where "trivial" is true).
- Do NOT replicate L4's `vr_proxy` anti-pattern (computed OUTSIDE the SFE — a purity violation). Every
  L5 stat is computed inside `compute_L5_ldist`.

## 4. CAUSALITY / step-fill (THE critical fix — the grounding's first advice was a lookahead bug)
L5 step-fills onto the 5s anchor with **`period = TF_SECONDS[tf]`** and `tf_ts = the L5 bar_ts array`
(the unique `(ts//period)*period` values, ascending) — EXACTLY like L1–L4 via `_align_to_anchor`/
`_last_closed_idx` (build_dataset.py:98, :215, period set at :236). A 1m L5 bar closes at `bar_ts+60`;
at a 5s anchor it is usable only when `bar_ts+60 ≤ anchor_ts`. **Using period=5 or period=1 (as the
build-map suggested) is LOOKAHEAD** — it would surface a 1m within-bar distribution while that 1m bar is
still forming. The 1s granularity is only the SOURCE; the unit that CLOSES (and thus the step-fill
period) is the TF bar. Warmup rows (idx<0) → NaN.

## 5. build_dataset integration
- L5 is **within-bar-only** (no cross-bar trailing window) → **per-day (per-session) 1s load is
  sufficient and correct** (unlike L1–L4 which load all-history for trailing windows). Add
  `_load_1s_day(atlas_root, day)` (mirror `_load_anchor_day`, path `DATA/ATLAS/1s/{day}.parquet`).
- Insertion point ≈ build_dataset.py:400 (inside the per-day loop, after the L1–L4 layer loop): load the
  day's 1s → for each TF `compute_L5_ldist(df_1s, tf)` → step-fill onto that day's 5s anchor (§4) →
  `_write_family(..., L5_{tf}, day)`. Output path: `FEATURES_5s_v2/L5_{tf}/{day}.parquet`.
- Session-day partitioning (the other half of this handoff) makes this clean: each 1s file is ONE
  session, so no TF bar spans a session boundary. (Real 1s day ≈ **56–57k rows**, not 86,400 — overnight
  gaps; the spec's "1D=86400" was an overstatement.)

## 6. Schema registration — STAGED (validate before bake)
**Stage A (materialize only — keeps the import-time assert GREEN):** register
`LAYER_FAMILIES['L5_'+tf]` (per-TF, features=`_l5_names(tf)`) + the build block. Do **NOT** add to
`FEATURE_NAMES` yet, do **NOT** touch `assemble_v2_grid`/`FEATURE_NAMES_V2` (the 25-feature CNN axis is
frozen; CNN is retired — RL consumes the flat grid). This writes the parquets for research without
breaking anything.
**Edge-test (the gate):** does any L5 feature separate the NMP trade outcomes at the |z|>1.85 entry
tail, OOS, day-block CI? Reuse the wedge harness. Only features that clear the bar advance.
**Stage B (bake — only after the gate):** add `_l5_names(tf)` to the `FEATURE_NAMES` build loop AND, in
the SAME commit, fix the magic `37` in the `N_FEATURES` assert (features.py:168 → `37 +
len(_l5_names(tf))` or recompute structurally) or every `import core_v2.features` dies. Also update
`describe_feature_count()` + the stale `139D` docstring.
NOTE: the real flat schema is **297 cols today** (L0 1 + L1 64 + L2 72 + L3 64 + L4 **96** — and FEATURE_NAMES
actually carries a 12th L4 `vr_proxy`/TF that the SFE doesn't emit; recount before asserting any delta).

## 7. Live parity (the SFE makes it structural — but the INPUT slice is the trap)
- Live calls the **identical** `compute_L5_ldist` on the live 1s buffer with the **same integer-floor
  `bar_ts` grouping** — NOT pandas `df.resample()` (its bin edges/origin differ from integer floor →
  non-identical distributions). Recompute-from-scratch each bar close (O(n) partition — trivial).
- Compute over **ONLY the last-closed tf bar's 1s closes** (`bar_ts+period ≤ ts`), never the forming bar.
- ⚠ PRE-EXISTING HOLE TO NOT INHERIT: `live_features.py:~144` resamples the FULL rolling buffer with **no
  `_last_closed_idx` pruning** — it currently includes the forming coarse-TF bar (a latent L1–L4
  live-vs-offline lookahead). L5 must apply last-closed pruning itself. (Worth flagging to the user as a
  separate L1–L4 parity concern.)
- Parity test (mandatory): live last-row L5 vector == offline step-filled L5 row for the same `(day, ts)`,
  byte-identical, on a sample day. Extend `tools/test_live_v2_parity.py`.

## 8. No magic numbers (named config / module constants, each with an origin comment)
`L5_OUTLIER_K` (the ±k·σ band, NOT a bare 2.0), `L5_MIN_SAMPLES_MOMENTS` (n<2 → NaN moments),
`L5_QUANTILE_METHOD = 'linear'` (PIN it — offline & live must match exactly or byte-parity breaks on
tiny fast-TF bars), the skew/kurtosis convention (Fisher vs Pearson + bias correction — pin in the
Numba kernel AND any numpy/scipy reference), and `EPS` for divisions. The `period-1` end-of-bar
x-reference derives from `TF_SECONDS[tf]` — never literal 59/60.

## 9. Verification plan (updated)
1. Unit-test the grouped-quantile + moment kernel vs numpy/scipy on a hand-checked 1s sample (exact match
   under the pinned interpolation + skew/kurtosis convention).
2. Causality test: assert L5 step-fill uses `period=TF_SECONDS[tf]` and that no L5 value at anchor `ts`
   draws on a tf-bar with `bar_ts+period > ts`.
3. Live==offline parity test (byte-identical), §7.
4. Stage-A import test: `import core_v2.features` still passes (assert green) with L5 materialized but not
   in FEATURE_NAMES.
5. Write results to `reports/findings/`; add `compute_L5_ldist` + the build block to TOOLS/feature docs.

## 10. CORRECTIONS to the original L5 proposal / grounding notes (follow THESE)
1. **Step-fill period = `TF_SECONDS[tf]`**, NOT 5 or 1 (the build-map said 5 — that's lookahead). §4.
2. **Return = one row per closed TF bar** (bar_ts-indexed), NOT "same length as the 1s input." §3.
3. **Signature = `compute_L5_ldist(self, df_1s, tf)`** — derive period from tf; drop the redundant
   seconds arg.
4. **No `_{N}` suffix** on L5 columns (within-bar, no window) — L1/L4 unsuffixed style.
5. **Quantiles are Numba-CPU/numpy, not CUDA** — grouped-quantile kernel offline; O(n) partition live.
   "Trivial on CUDA" is false for the offline full-history build.
6. **All math in the SFE** — do NOT copy L4's `vr_proxy` external-compute split.
7. **Materialize (Stage A) before FEATURE_NAMES (Stage B)** — and bump the `N_FEATURES` magic-37 assert
   in the same commit as Stage B, or imports break.
8. **~56k 1s rows/day**, not 86,400.
9. Roll-seam masking (features.py seam-mask) only covers L2/L3/L4 rolling warmup; within-bar L5 has no
   cross-bar warmup → likely EXEMPT, but confirm rather than silently leave unmasked.
