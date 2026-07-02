# SPEC — Regime-adaptive TREND_PTS for the AI auto-labeler (v2)

**Status**: READY TO IMPLEMENT (spec only — no implementation in this session; the executing
agent works on the machine that has `DATA/`, which is gitignored and absent from remote clones).
**Written**: 2026-07-02, from a full read of the labeler, tuner, and recovery_dynamics code.
**Goal**: make the labeler's wiggle filter scale with the local amplitude regime and beat the
fixed-threshold ceiling (pooled F1 0.664) against the 398 human picks — honestly (LODO +
day-block bootstrap CI), not by in-sample sweep luck.

Read first: `CLAUDE.md`, `docs/daily/2026-06-30.md`,
`research/recovery_dynamics/README.md`, `research/ai_auto_labeler/README.md`,
`research/ai_auto_labeler/reports/tune_to_human.md`.

---

## 0. CRITICAL PRE-STEP — rescue `tools/viz/core/cubic_utils.py` (do this FIRST)

`tools/viz/core/cubic_utils.py` is imported by **five committed files**
(`ai_labeler_v2.py`, `tune_to_human.py`, `tools/viz/cusp_marker.py`,
`tools/viz/extract_pick_primitives.py`,
`research/wick_absorption_signal/tools/aperiodic_labeler_sweep.py`) but is **NOT in git**:
no history (`git log --all -- '**/cubic_utils.py'` is empty), not matched by any ignore rule
(`git check-ignore` exit 1). It is an untracked survivor of the 2026-06-30 UTF-16
`.gitignore` incident that the 83-file recovery pass missed. It exists only on the local
machine. **If that machine loses it, the whole labeler stack dies.**

```bash
git add tools/viz/core/cubic_utils.py
git commit -m "rescue(viz): commit cubic_utils.py — untracked but imported by 5 committed files (gitignore-incident survivor)"
```

Also grep for any OTHER untracked-but-imported module before starting
(`git status --porcelain` vs the import graph) — there may be more survivors.

Interface contract used by this spec (verify against the actual file):
`turns, smooth, slope, curv = find_raw_turns(close: np.ndarray, N: int)` where `smooth` is a
same-length centered-cubic smoothing of `close` and `turns` is an index-ordered list of dicts
each having at least `{"index": int}`. `zigzag_turns` uses only `turns[i]["index"]` and `smooth`.

---

## 1. Baseline facts (already measured — do not re-derive)

- Human ground truth: `DATA/cusp_picks/picks_YYYY-MM-DD_multi.json` — **398 picks, 9 session-days**
  (CME session-day: prev 18:00 ET → 17:00 ET). Tuner uses fields `timestamp` (epoch **seconds**),
  `direction` ("LONG"/"SHORT"), `timeframe` (keep only "1m").
- Human swing sizes (|Δclose| between consecutive picks): median 15.8 pt, IQR 7.8–32.2 pt.
- Fixed-threshold sweep (N × T over 9 days, `reports/tune_to_human.md`):
  best cell **N=20, T=3 → F1 0.664** (recall 72%, precision 62%); shipped default
  **N=20, T=4 → F1 0.657** (66%/65%). The knee is FLAT (0.657–0.664) — the fixed-T ceiling.
- Match metric (`score()` in `tune_to_human.py`): pivot↔pick match = |Δt| ≤ TOL(=300 s) AND same
  direction; recall = matched picks / picks; precision = matched pivots / pivots; pooled over days.
  **Do NOT change this metric** — comparability with the 0.657/0.664 baselines is the experiment.
- Amplitude framework (recovery_dynamics, measured on 518–536 days of 2024–2025):
  - Period (first-return time through an every-bar anchor) is a market **constant**
    (mode 2–3 min, 71% < 15 min, 2024≈2025 to <0.3%). ⇒ **CUBIC_N stays fixed** — the
    framework itself predicts the timing scale doesn't move; only the SIZE scale does.
  - Amplitude is the **regime**: typical ~10-min swing breathes ~4× (5.5 pt calm-2024 →
    ~20 pt Apr-2025), slow and persistent (week-scale).
  - Estimators already coded in `research/recovery_dynamics/tools/`:
    `anchor_period.day_periods(close)` → per-anchor `(periods, amps, noreturn, total)`;
    `amplitude_evolution.per_day` → per-day `vol_scale = median(amp/√per)` and
    `ref_amp = median(amp | 8 ≤ per < 15 min)`.
- Data: `DATA/ATLAS/1m/YYYY_MM_DD.parquet` (per session-day; columns include `timestamp`
  epoch-seconds float, `open/high/low/close`), `DATA/ATLAS/1s/` same keying.
- The labeler is HINDSIGHT — centered/non-causal estimates are legal for labels.

## 2. Premise and the counter-hypothesis (the sweep must arbitrate, not assume)

**Premise**: the human's "significant swing" is relative to the current regime, so
`TREND_eff(t) = K_TREND × scale(t)` matches the human better than any fixed T.

**Counter-hypothesis (state it in the report)**: the 8–32 pt spread in human swing sizes is
within-day trend-vs-chop structure, not regime scaling — the human may apply a roughly fixed
smoothed-cubic cutoff, and the flat fixed-T knee (0.657–0.664) hints T is not the binding
constraint at all (F1 varies more across N than across T in the fixed sweep). If so, adaptive-T
gains ~nothing. The Step-3 diagnostics decide which story the data supports BEFORE the sweep
result is interpreted.

Two distinct levers, separated by the window parameter:
- **cross-day** adaptation (day-scalar / multi-day-median scale): only wins if the 9 labeled
  days actually span different regimes;
- **intraday** adaptation (±30–120 min windows): wins on overnight-vs-RTH and hour-scale
  amplitude spread, present within every session-day.

## 3. Step A — diagnostics tool (research before code; ~fast, run it yourself)

New file `research/ai_auto_labeler/tools/diagnose_regime_spread.py` →
writes `research/ai_auto_labeler/reports/diagnose_regime_spread.md`. Contents:

1. **Per-day regime table** for the 9 human days: date, n_picks, day `vol_scale`, day `ref_amp`
   (both via the Step-B module), and per-day best fixed T (mini-sweep T ∈ {2..8} at N=20 on that
   day alone; note per-day F1 on ~44 picks is noisy).
2. **Spearman correlation** between per-day scale and per-day best-T. Positive & material →
   cross-day adaptation has signal.
3. **Intraday spread**: per day, median scale in RTH (09:30–16:00 America/New_York via
   `pd.to_datetime(ts, unit='s', utc=True).tz_convert(...)`; named constants RTH_START/RTH_END)
   vs the rest; report the ratio and the share of human picks falling inside RTH.
4. **The premise's direct test — ratio tightening**: for each consecutive-pick human swing,
   compute `swing / scale(t_pick)` (w60 scale). Report median and IQR/median (relative dispersion)
   for RAW swings vs SCALED ratios. Premise supported iff relative dispersion drops materially
   (≥ ~20%). This is the single most informative number in the whole exercise.

Decision gate: if (2) shows no cross-day signal AND (3) shows ratio < ~1.5× AND (4) shows no
tightening → the adaptive threshold cannot beat fixed on THIS data; still run the sweep (it's
cheap), but a negative outcome is then expected — report it straight, keep fixed T=4, stop.

## 4. Step B — new module `research/ai_auto_labeler/pipeline/amplitude_scale.py`

Single source of truth for the scale; imported by the labeler, the tuner, and the diagnostics.
Re-implements (20 lines) the anchor first-return sampler rather than sys.path-importing across
research projects; cite provenance in the docstring
(`recovery_dynamics/tools/anchor_period.py::day_periods`, `amplitude_evolution.py`).

```python
"""Local amplitude-regime scale for the labeler (hindsight/centered allowed).

Anchor every 1m bar at its close; first return THROUGH that level = one oscillation:
per = first-return time (min), amp = peak |excursion| before return (pt). Regime scale =
windowed median of amp/sqrt(per), rescaled to the 10-min reference period:
    scale(t) = median_{anchors near t}(amp/sqrt(per)) * sqrt(REF_PERIOD_MIN)
Same semantics as amplitude_evolution.py's ref_amp ('typical ~10-min swing'), but computable
at any window. Uses amp/sqrt(per) over all returned anchors (not the 8-15min band) so small
windows keep enough samples; the sqrt rescale removes the period-mix dependence.
Provenance: research/recovery_dynamics/tools/anchor_period.py (day_periods),
amplitude_evolution.py (vol_scale/ref_amp). Hindsight-legal for LABELS only.
"""
AMP_MAXLOOK_MIN     = 360   # forward first-return cap (min) — mirrors anchor_period.MAXLOOK
REF_PERIOD_MIN      = 10.0  # reference period: center of the 8-15min ref_amp band
MIN_WINDOW_SAMPLES  = 30    # fewer returned anchors than this in a window -> widen x2 (median too noisy)
MIN_DAY_SAMPLES     = 50    # amplitude_evolution's day-skip criterion, reused as day-fallback trigger
GLOBAL_FALLBACK_PT  = 8.0   # last-resort scale: pooled 2024-25 8-15min median amp ~8-12pt (anchor_period.md)

def anchor_samples(close):        # -> idx:int[], ratio:float[]  (ratio = amp/sqrt(per), returned anchors only)
def scale_series(close, window_min):  # centered ±window_min median(ratio) * sqrt(REF_PERIOD_MIN) -> float[n]
    # applies the SAME sqrt(REF_PERIOD_MIN) rescale as scale_scalar — every mode must return
    # 'typical ~10-min swing' units, or w-modes and day-modes would sit ~sqrt(10)≈3.16x apart
    # and K_TREND would not be comparable across MODE_GRID.
    # single-day only, no filesystem access. Window widening is its ONLY fallback; if even the
    # whole day is unusable it returns all-NaN and the CALLER resolves (see scale_for_day).
def scale_scalar(close):          # whole-day median(ratio) * sqrt(REF_PERIOD_MIN); np.nan if
    # fewer than MIN_DAY_SAMPLES returned anchors (caller resolves — no fallback logic here).
def scale_for_day(date_key, close, mode, one_m_dir, cache):  # -> float[n]
    # mode ∈ {"w30","w60","w120"}  -> scale_series on this day's close
    # mode ∈ {"day","day_c5","day_c21"} -> per-day scalars; multi-day = CENTERED nanmedian of the
    #   ±(w//2) NEIGHBOR SESSION-DAY scalars taken BY POSITION in the sorted 1m file list
    #   (glob once, sort, index — calendar arithmetic would miss weekends). Shrink at edges.
    #   np.full(n, value) so every mode returns a per-bar array.
    # OWNS all cross-day context and ALL day-level fallbacks: a NaN day (from scale_scalar /
    #   scale_series) resolves to the nearest available neighbor-day scalar by position, else
    #   GLOBAL_FALLBACK_PT. cache is a shared dict holding the glob-sorted 1m file list
    #   (key ("files",)), loaded closes (("close", date_key)) and computed per-day scalars
    #   (("scalar", date_key)), so overlapping neighbor windows across the 9 tuner days never
    #   re-glob, re-read a parquet, or recompute anchor_samples.
```

Implementation notes:
- `anchor_samples` is `day_periods` minus aggregation: for each anchor keep `(a, amp/sqrt(per))`
  for RETURNED anchors; skip flat-next-bar anchors exactly as the original.
- `scale_series`: ratios sorted by anchor idx; for each bar i, `np.searchsorted` the idx array
  for [i−W, i+W] bounds, `np.median` the slice; if slice < MIN_WINDOW_SAMPLES, double W (up to
  whole day); if the whole day still has < MIN_DAY_SAMPLES returned anchors, return all-NaN.
  Neighbor-day / GLOBAL_FALLBACK_PT resolution belongs EXCLUSIVELY to `scale_for_day` (it owns
  `date_key`/`one_m_dir`/`cache`; `scale_series` has no filesystem context by design).
  n≈1380 bars/day → trivial runtime.
- Known small bias: anchors near EOD are censoring-truncated (only short periods return) →
  end-of-day scale biased slightly LOW; centered windows mostly absorb it. Note in docstring;
  acceptable for labels.

## 5. Step C — labeler changes (`research/ai_auto_labeler/pipeline/ai_labeler_v2.py`)

New constants block (values of K_TREND / AMP_MODE come from the Step-D sweep):

```python
# --- regime-adaptive threshold (tuned to the 398 human picks; see reports/tune_to_human.md) ---
ADAPTIVE = True              # False = legacy fixed-TREND_PTS path (A/B + rollback)
K_TREND = <sweep winner>     # TREND_eff(t) = clip(K_TREND * scale(t), TREND_MIN_PTS, TREND_MAX_PTS)
AMP_MODE = "<sweep winner>"  # w30|w60|w120|day|day_c5|day_c21 (see amplitude_scale.scale_for_day)
TREND_MIN_PTS = 2.0          # guardrail floor (8 ticks): smaller smoothed-cubic 'swings' are noise-scale
TREND_MAX_PTS = 15.0         # guardrail cap ≈ human median raw swing (15.8pt): keep real legs in wild regimes
FLAT_BAND_RATIO = FLAT_BAND_PTS / TREND_PTS       # 0.75 — preserve the tuned flat-band:trend coupling
REVERSAL_TOL_RATIO = REVERSAL_TOL_PTS / TREND_PTS # 0.25 — preserve the QC-flag:trend coupling
# The two ratios are derived AT IMPORT from the tuned fixed pair on purpose — one source of
# truth in this constants block. Corollary rule: nothing mutates these module constants at
# runtime. The tuner passes thresholds as EXPLICIT ARGUMENTS to zigzag_turns (it never calls
# process_day), and any future sweep over FLAT_BAND/TREND must parameterize, not monkey-patch.
```

Exact change sites (every current use of the three constants):

1. `zigzag_turns(smooth, turns, thr)` — accept scalar OR per-bar array:
   `thr_arr = np.full(len(smooth), float(thr)) if np.isscalar(thr) else np.asarray(thr, float)`;
   inside the loop compare against `thr_arr[i]` (i = the CONFIRMING turn's index — the regime at
   the moment the reversal is judged; the scale is slow, so the anchoring choice is second-order —
   say so in a comment). Scalar path keeps the existing tuner calls working unchanged.
2. `flat_span(smooth, i, n, band)` — band becomes a parameter (module FLAT_BAND_PTS stays as the
   fixed-path value). Call sites when ADAPTIVE: entry passes `FLAT_BAND_RATIO * thr_arr[i0]`,
   exit passes `FLAT_BAND_RATIO * thr_arr[i1]` — the ratio applies to BOTH (an unscaled
   `thr_arr[i1]` would make the exit zone 1.33× wider than the tuned coupling). Else FLAT_BAND_PTS.
3. `process_day`: after `find_raw_turns`, build
   `thr_arr = np.clip(K_TREND * scale_for_day(...), TREND_MIN_PTS, TREND_MAX_PTS)` when ADAPTIVE
   else `np.full(n, TREND_PTS)`; pass to `zigzag_turns`.
4. Leg-significance filter `if pnl < TREND_PTS:` → `if pnl < thr_arr[i0]:` (a leg is judged by
   the regime at its own start).
5. Flag check `if mae > REVERSAL_TOL_PTS:` → `if mae > REVERSAL_TOL_RATIO * thr_arr[i0]:` when
   ADAPTIVE.
6. CLI: add `--fixed` flag → forces ADAPTIVE=False for A/B regen. Docstring: document both paths.

## 6. Step D — tuner extension (`research/ai_auto_labeler/tools/tune_to_human.py`)

Keep the existing fixed (N,T) sweep verbatim as the baseline section. Add an adaptive section
to the SAME regenerated report `reports/tune_to_human.md`:

- Grids (module constants): `K_GRID = (0.15, 0.21, 0.30, 0.42, 0.60, 0.85, 1.20, 1.70)`
  (geometric ×~1.41 — wide because k's magnitude depends on where the 9 days sit on the 5.5–20 pt
  regime range; the clamps make extreme cells saturate to ≈fixed, which is harmless),
  `MODE_GRID = ("w30","w60","w120","day","day_c5","day_c21")`, N fixed at 20 (the established
  best; the framework says the timing scale is constant — don't sweep N here. Fallback ONLY if
  acceptance fails: repeat at N=15).
- Per day: load close/ts once; `anchor_samples` once; build the 6 scale arrays once; then 48
  cells × `zigzag_turns` with `np.clip(k*scale, TREND_MIN_PTS, TREND_MAX_PTS)`. tqdm over cells.
- **Cache per (cell, day): (r, p, n_piv, n_h)** — recall/precision raw counts from the UNCHANGED
  `score()`. Everything below (pooled F1, LODO, bootstrap) is pure arithmetic on this cache.
- Report per cell: pooled recall/precision/F1 + median/P10/P90 of TREND_eff across all bars +
  **% of bars clamped** (at floor or cap). A champion with >~50% clamped bars is fixed-T in
  disguise — call that out explicitly.
- **Champions table**: best adaptive cell vs best fixed cell (0.664) with per-day breakdown
  (day, n_picks, day scale, fixed F1, adaptive F1, Δ).
- **LODO (the honesty guard)**: for each held-out day d, pick the argmax-pooled-F1 cell on the
  other 8 days (adaptive: over K_GRID×MODE_GRID; fixed: over the existing N×T grid), score d
  with it; pool the 9 held-out (r,p,n_piv,n_h) → LODO-F1(adaptive) vs LODO-F1(fixed).
  In-sample sweep WILL beat 0.664 by luck with 48 cells; LODO is what makes the claim real.
- **Day-block bootstrap CI on ΔF1** (CLAUDE.md operational rule): resample the 9 days with
  replacement, `N_BOOT = 4000`, `BOOT_SEED = 20260702`; recompute pooled F1 for
  champion-adaptive and champion-fixed from the cache; percentile 95% CI of the delta; also
  bootstrap the LODO delta (resample the 9 held-out per-day results). **State significance
  explicitly**; with N=9 days expect a wide CI — "not significant" is the likely honest label
  even for a real win; the claim then rests on direction + LODO consistency + the Step-A
  mechanism evidence, and the report must say exactly that.

## 7. Step E — decision rule (pre-committed, so nobody ships noise)

SHIP adaptive as the default (`ADAPTIVE = True` + winning K_TREND/AMP_MODE) iff ALL of:
1. in-sample champion pooled F1 > 0.664 (the fixed ceiling, not just the 0.657 default);
2. LODO F1(adaptive) ≥ LODO F1(fixed) (directional — significance at N=9 is unrealistic);
3. Step-A mechanism evidence supports it (scale spread exists and/or ratio-tightening holds);
4. champion cell is not clamp-saturated (>50% clamped bars).

Any of these fail → keep `ADAPTIVE = False` (fixed T=4 default), keep the code, and write the
negative result to the report with the same rigor. Either way: report recall/precision/F1 +
CI + explicit significance statement to `research/ai_auto_labeler/reports/tune_to_human.md`.

After shipping: the full 604-day regen (~13 min) is **user-run** — present "ready to regen"
and stop. Then eyeball 2024-03-04 via `tools/viz/cusp_marker.py --load-ai` (reference day:
49 trades, 1 flagged under fixed T; adaptive flat-zone scaling changes entry/exit snapping,
which the F1 metric does NOT see) and watch `flagged/` counts (~2.5/day baseline).

## 8. Risks / what breaks (each with its mitigation)

1. **Overfit on 9 days / 48 cells** → LODO + pre-committed decision rule (Steps D/E).
2. **No regime spread across the 9 labeled days** → cross-day modes can't win by construction;
   intraday modes carry the experiment; Step-A reveals which BEFORE interpretation.
3. **Clamp saturation** masquerading as an adaptive win → per-cell clamp-share reporting (D).
4. **EOD censoring** biases scale low near the close → centered windows absorb; documented bias.
5. **Flat-zone scaling side-effects** (band up to 0.75×15≈11 pt in wild regimes → wide entry/exit
   spans; band floor 1.5 pt in dead regimes) → invisible to F1; caught by the 2024-03-04 eyeball
   and flagged/-count check (Step E).
6. **Degenerate quiet stretches** (holiday half-days → pivot storms at tiny thresholds) →
   TREND_MIN_PTS floor.
7. **`pnl < thr` filter changes trade counts** in wild regimes → expected; compare regen totals
   vs the 30,173-trade fixed baseline before/after.
8. **cubic_utils.py untracked** → Step 0. Non-negotiable, do it first.
9. **Metric drift** → `score()` and TOL stay byte-identical; only new sections are added.

## 9. Conventions checklist (CLAUDE.md — all mandatory)

- No magic numbers: every new value above is a named constant with an origin comment.
- tqdm on the sweep loops. Reports to files, never stdout-only. CUDA note: these are
  numpy/pandas research tools like the existing ones — no CUDA needed; that rule binds the engines.
- Update `research/ai_auto_labeler/README.md` (new module, adaptive logic, how to run each tool).
- Daily journal `docs/daily/<date>.md` change report: what changed / files / what to look for
  next run / expected metric impact.
- Commit sequence: (1) cubic_utils rescue; (2) amplitude_scale.py + diagnostics + its report;
  (3) tuner extension + sweep report; (4) labeler adaptive path + tuned constants; (5) README +
  journal. Research-folder work commits to main per project convention.

## 10. Out of scope (explicitly)

- **Causal/trailing live estimator** of the scale (separate task; the regime is slow, so a
  trailing estimate tracks with low lag — later).
- Full-history regen (user-runs) and downstream cusp-detector re-eval (needs ≥15–20 labeled
  disjoint days first — see `docs/daily/2026-06-30.md` §next).
- 1:1 pivot↔pick matching (stricter metric) — optional robustness appendix only; never the
  headline, it breaks comparability with 0.657/0.664.
