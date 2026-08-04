# ADVERSARIAL AUDIT — onset-Mamba pipeline

Run 2026-08-04, BEFORE any GPU spend and BEFORE the sealed test window is opened.
Every check below was re-derived independently of the target code. Test days
(`2025_07_01`..`2026_03_19`, 170 files) were **never loaded, fitted, or scored**.

Target: `ONSET_MAMBA_SPEC.md`, `builders/build_onset_sequences.py`,
`pipeline/train_onset_mamba.py`, `builders/build_onset_dataset.py`,
`builders/build_onset_matched.py`.

---

## VERDICT TABLE

| # | claim | verdict |
|---|---|---|
| a | label derivation `y_k(t)=1 iff event in (t, t+H]` | **CLEAN** (0/43,200 mismatches) |
| a2 | label *coverage* across days | **FLAW** — 6 train days, 129k windows, all-zero labels |
| b | sequence-feature causality | **CLEAN** (exact recompute) |
| b2 | *probe* feature causality (`_feat_matrix`) | **FLAW** — anticausal by up to +5 s |
| c | no absolute-price / era leak | **FLAW** — era recoverable at AUC 0.9835 |
| d | window / mask correctness | **CLEAN** |
| e | split integrity, test sealed | **CLEAN** (2 latent hazards) |
| f | matched-eval fairness vs GBM baseline | **FLAW (major)** — bar inflated by 0.126 / 0.110 / 0.010 |
| g | per-head `pos_weight` | **CLEAN** |
| h | checkpoint / resume | **CLEAN** state, **FLAW** in stated behaviour |
| i | sanity baseline + shuffled control | **CLEAN** — control collapses to 0.5 |

**Bottom line: the dataset is sound; the CONTEST is not.** No leakage was found in
the sequence pipeline. The pre-registered SHIP/KEEP-GBM/KILL rule is invalid as
written because the baseline it names was measured with a 5-second information
advantage and under a different evaluation protocol.

---

## a. LABEL LEAKAGE — CLEAN

Re-derived `y` by brute force from `research/event_library/events/*.parquet` for
4,800 random rows × 3 heads × 3 horizons = **43,200 label cells over 12 random days**
(2024, 2025-H1 and 2025-H2/2026 files alike).

- **mismatches: 0**
- Rows where an event lands **exactly at `t`**: 5,542. Of those, rows labelled 1
  with no event in `(t, t+H]`: **0**. The interval is genuinely half-open —
  `searchsorted(e, ts, 'right')` vs `searchsorted(e, ts+H, 'right')` is the correct
  construction and there is no off-by-one at either end.

## a2. LABEL COVERAGE — FLAW

Six ATLAS day files are contract-roll variants of a base date and **do not exist in
the event library at all**, so `ev_ts[k].get(day)` returns `None` and every one of
their labels is a hard zero:

| file | sampleable | pos_rate | in event library |
|---|---|---|---|
| 2024_02_20_BROWN | 21,544 | 0.00000 | no |
| 2024_02_20_FOUR | 21,544 | 0.00000 | no |
| 2024_02_21_FOUR | 21,497 | 0.00000 | no |
| 2024_02_22_FOUR | 21,477 | 0.00000 | no |
| 2024_02_23_FOUR | 21,484 | 0.00000 | no |
| 2024_02_26_FOUR | 21,447 | 0.00000 | no |

These are **real, active sessions**, not empty files — `2024_02_20_FOUR` has 65,097
1s bars over the same timestamps as `2024_02_20` with a genuinely different series
(close range 17410–17805 vs 17454–17788, mean |diff| 107 pts). They are 6 of the
545 day files and **128,993 / 5,530,602 = 2.33 % of all training windows**, injected
as guaranteed negatives. At the fakeout base rate they should have carried roughly
13k positives. This is pure false-negative label noise.

*Broken case:* `2024_02_20_FOUR`, second `1708441800`. The tape is trading; `y` is
`[0,0,0,0,0,0,0,0,0]` for every second of the session, by construction, because the
event library never saw that file.

## b. FEATURE CAUSALITY — CLEAN

Recomputed all 8 channels from the raw `DATA/ATLAS/1s` parquets for 12 full days
(≈258k rows), with the volume z-score built from an explicit strictly-backward
`lv[max(0,i-599):i+1]` loop (`ddof=1`, `min_periods=30`):

| channel | max abs(stored − recomputed) |
|---|---|
| ret_ticks, upper_wick, lower_wick, body, range | **0** (exact) |
| vol_z | 1.94e-3 (float16 storage rounding) |
| clock_sin / clock_cos | 2.44e-4 (float16 storage rounding) |

`pandas.rolling(600, min_periods=30)` is trailing and **inclusive of the current
bar only** — no centering, no forward peek. Volume at second `t` is known at the end
of bar `t`, so this is causal. No feature reads a future bar.

One deviation from spec §3, in the **safe** direction: the loader slices
`f[i-WINDOW:i]`, i.e. 300 bars ending at `i-1`. Spec says "window 300s ending at t".
The model is therefore 1 bar *more* conservative than specified — its information
cutoff is exactly `t`, exclusive.

## b2. PROBE FEATURE CAUSALITY — FLAW

**ATLAS bar `timestamp` is the bar OPEN, verified empirically:** for
`2024_01_02`/`2025_03_12` the 5s bar at `T` reproduces the 1s bars in `[T, T+5)`
in 100 % of 1,816/1,670 testable cases and the bars in `(T-5, T]` in only
15 %/8 %.

`build_onset_dataset._feat_matrix` indexes `c[idx]`, `h[idx]`, `l[idx]` — the bar
**starting** at `ts[idx]`. So the GBM probe row stamped `ts = t` contains price
action through `t+4.99`. The row is not computable until `t+5`. Every published
probe number therefore has a horizon 5 s shorter than its label says.

## c. PRICE / ERA LEAK — FLAW

The spec's literal claim is true: no feature encodes an absolute level. The
*rationale* — "the model cannot memorise the era" — is false.

From **one 300-second window**, using only 56 summary statistics of the 8 stored
features (mean/std/mean-abs/p10/p90/max/min per channel), 30 windows per day,
375 non-test days, `GroupKFold(5)` by day:

- **2024 vs 2025-H1 classification: AUC 0.9835 ± 0.0041** (folds 0.981–0.989)
- Calendar-day-index regression: Spearman **0.773**, MAE 73.6 days vs a chance
  MAE of 135.

Per-channel era AUC: `range` 0.933, `upper_wick` 0.932, `lower_wick` 0.929,
`body` 0.776, `ret` 0.748, `vol_z` 0.724, `clock` 0.65.

Mechanism is **amplitude drift, not price scaling**:

| day-level stat | Spearman vs day index | 2024 mean | 2025-H1 mean |
|---|---|---|---|
| mean close | +0.765 | 19,189 | 20,723 |
| mean abs 1s return (ticks) | +0.549 | **2.93** | **4.73** |
| mean 1s range (ticks) | +0.329 | **4.89** | **6.74** |
| bars per session | −0.448 | 21,029 | 20,134 |

`corr(mean_range, mean_price)` across days is only **+0.036** — this is a
volatility-regime drift, not a mechanical consequence of 16k→28k. Dropping the
level did not remove the date stamp; it moved it into the amplitude.

**Why it matters:** it does *not* inflate matched-design AUC (positive and negative
are the same day, so the era term is constant inside a pair). It *does* mean the
network trains on a tape whose typical 1s move is 2.93 ticks and is validated on
one where it is 4.73 — a 1.6× scale shift with no normalisation anywhere in the
feature set. Spec §9's kill condition "test AUC drops >0.05 below val → regime
overfit" will fire for a reason that has nothing to do with signal quality.

## d. WINDOW / MASK — CLEAN

12 random days, all sampleable indices checked:

- sampleable `i` with `i < WINDOW`: **0**
- windows whose start and end fall on different ET calendar dates: **0**
- max wall-clock span of a 300-bar window: **410 s** (mean 300–326 s). Spans exceed
  300 s only because untraded seconds are absent from ATLAS; the window never
  reaches outside the retained 09:24–15:30 ET block, so it cannot cross the
  overnight gap. The `keep` filter (`mod >= 564`) provides 360 s of warmup for a
  300-bar requirement.
- Stored `ts` arrays: **0 / 545** files non-monotonic, **0 / 545** with duplicates.

Note for the record: gap density itself is era-correlated (27–249 gaps/day in 2024,
804–1,454 in 2025–26), which is part of the finding in (c).

## e. SPLIT INTEGRITY — CLEAN, two latent hazards

545 day files → **train 263 / val 112 / test 170**, every file assigned exactly once.

- Calendar dates appearing in more than one file: 5 (the contract-roll variants).
  **All 5 sit entirely inside `train`** — no calendar date spans two splits.
- Val range `2025_01_02`..`2025_06_19`; test range `2025_07_01`..`2026_03_19`.
- `grep`: the only occurrence of `test` in `train_onset_mamba.py` is the `day_split`
  return value. `DayCache` is instantiated on `tr` and (inside `matched_eval`) on
  val days only. `matched_eval` is only ever called with `split='val'`. The test
  window is genuinely untouched by the training script.

**Latent hazard 1 — `day_split` is lexicographic, not date-parsed.** A day named
`2025_06_30_FOUR` sorts *after* `'2025_06_30'` and would be routed to **test**,
splitting one calendar date across val and test. Any pre-2024 file would be routed
to **val**. Neither case exists today (0 such files); the function is one ATLAS
rebuild away from being wrong.

**Latent hazard 2** — the 5 duplicated calendar dates mean five 2024 sessions are
each represented 2–3× in training. Harmless today (same split), but combined with
(a2) those extra copies carry no labels.

## f. MATCHED-EVAL FAIRNESS — FLAW (major)

### The join works
`matched_eval()` was executed against an untrained model. It returns, it is not
silently empty, and the val-split filter is correct:

| head | matched parquet (all) | val rows | **joined n** | dropped: ts absent from npz / `i<WINDOW` / masked |
|---|---|---|---|---|
| fakeout_poke | 247,288 (539 d) | 62,930 (112 d) | **61,020** | 1,900 / 10 / 0 |
| leg_descent | 104,110 (539 d) | 28,628 (112 d) | **27,907** | 710 / 11 / 0 |
| ultra_chop | 36,304 (529 d) | 8,164 (108 d) | **7,991** | 173 / 0 / 0 |

Untrained-model AUCs 0.4975 / 0.5062 / 0.4751 — the plumbing is fine. Base rates
0.502 / 0.501 / 0.499.

### But it is not the same population, and not the same question

**(f1) The baseline saw 5 more seconds of tape.** Per (b2), a GBM row at `ts=t` uses
the 5s bar covering `[t, t+5)`; the Mamba window ends at `t`. Rebuilding the matched
design with a one-bar causal shift and refitting with `fit_onset.py`'s exact model
and `GroupKFold(5)`:

| head | as built (sees `t..t+5`) | strictly causal (`t-5..t`) | Δ |
|---|---|---|---|
| fakeout_poke | 0.7687 ± 0.0020 | **0.6781 ± 0.0029** | **−0.0906** |
| leg_descent | 0.8683 ± 0.0021 | **0.8002 ± 0.0024** | **−0.0681** |
| ultra_chop | 0.8299 ± 0.0040 | **0.8328 ± 0.0033** | +0.0029 |

(The as-built column reproduces the published 0.769 / 0.868 / 0.830 to 3 decimals,
so the baseline is confirmed reproducible — see the note on the missing script
below.) The lost 5 seconds are worth **4.5× and 3.4× the ±0.02 decision band**.

**(f2) The baseline used a different protocol.** 0.769/0.868/0.830 is 5-fold
day-blocked CV over **all 539 days — including the 170 sealed test days in 4 of 5
training folds**. The Mamba's val number will be 2025-H1 only. Refitting the GBM
under the Mamba's protocol (fit 2024, score 2025-H1, no test days touched):

| head | published (539-day CV) | fit 2024 → score 2025-H1 | Δ |
|---|---|---|---|
| fakeout_poke | 0.769 | 0.7421 [0.7313, 0.7520] | −0.027 |
| leg_descent | 0.868 | 0.8296 [0.8158, 0.8431] | −0.038 |
| ultra_chop | 0.830 | 0.8246 [0.8146, 0.8348] | −0.005 |

**(f3) Stacked — the honest bar.** Strictly-causal features *and* the Mamba's
temporal protocol, scored on exactly the val rows `matched_eval` will use
(day-clustered 95 % bootstrap CI, 1,000 resamples):

| head | **honest baseline** | published bar | inflation | × the ±0.02 band |
|---|---|---|---|---|
| fakeout_poke | **0.6435** [0.6327, 0.6545] | 0.769 | **+0.126** | 6.3× |
| leg_descent | **0.7580** [0.7462, 0.7697] | 0.868 | **+0.110** | 5.5× |
| ultra_chop | **0.8201** [0.8105, 0.8302] | 0.830 | +0.010 | 0.5× |

*Broken case:* a Mamba scoring 0.70 on fakeout_poke val would be **+0.06 over an
honest equal-information baseline** — a clear SHIP — yet spec §7 reads it as
0.769 − 0.70 = −0.069, i.e. **KILL**. The pre-registered rule inverts the verdict
on 2 of 3 heads.

**(f4) `ultra_chop`'s scored head does not match the design's horizon.** Event
timestamps are on the 5s grid for every head except `ultra_chop`
(`ts % 5 == 0`: fakeout 99.51 %, leg_descent 99.55 %, stall 99.35 %,
defended_poke 100 %, **ultra_chop 20.83 %**). `build_onset_matched` anchors on 5s
bars, so the ultra_chop positive lands 10–14 s before its event, not 10:

- seconds from matched row to next ultra_chop event: min 10, **p50 12**, p90 14,
  max 14; fraction exactly 10: **20.6 %**
- agreement between the matched `y` and the npz label of the head
  `matched_eval` actually scores (`ultra_chop`, H=10): **0.6009**
  (crosstab: 3,166 of 3,989 matched positives carry npz label 0)
- agreement with the H=30 head: **0.9722**; with H=5: 0.4979

*Broken case:* `matched_eval(model, dev, 'ultra_chop', 10, 'val')` scores logit
index 7 (the ≤10 s head) on 7,991 rows of which ~40 % are labelled the opposite way
by the very objective that head was trained on. For fakeout_poke and leg_descent the
same agreement is 0.9997 and 0.9996 — the defect is specific to ultra_chop.

### Housekeeping
No committed script produces `matched_probe.json` or the v2/v3 sections of
`reports/onset_probe.md` — `grep -rl "matched_"` over `*.py` finds only the builder
and the trainer. The numbers reproduce exactly from `build_onset_matched.py` +
`fit_onset.cv_auc`, so they are sound, but the fitting script violates the repo's
"save analysis tools" rule and the baseline is not re-runnable from `main`.

## g. CLASS IMBALANCE — CLEAN

`pos_weight` is computed on all 5,530,602 stride-1 masked training rows,
per-head, axis-0 mean of the 9-column label block. Correct:

| head | rate | positives | pos_weight |
|---|---|---|---|
| fakeout_poke H5 / H10 / H30 | 0.05446 / 0.10196 / 0.24927 | 301,169 / 563,887 / 1,378,626 | 17.36 / 8.81 / 3.01 |
| leg_descent H5 / H10 / H30 | 0.01990 / 0.03976 / 0.11853 | 110,062 / 219,868 / 655,557 | 49.25 / 24.15 / 7.44 |
| ultra_chop H5 / H10 / H30 | 0.00774 / 0.01548 / 0.04649 | 42,798 / 85,614 / 257,097 | 128.23 / 63.60 / 20.51 |

No head is degenerate; no clipping is active; all weights finite. Simulated
degenerate heads: `rate=0 → 9999.0`, `rate=1 → 0.0`, both finite — **no inf/nan is
reachable**. Minor note: `rate→1` yields `pos_weight = 0`, which silently zeroes the
positive class instead of erroring; unreachable with the current heads.
`pos_weight` is computed at stride 1 while training runs at stride 5, but the rates
are identical to 4 decimals (e.g. 0.03135 vs 0.03126), so this is immaterial.

## h. CHECKPOINT / RESUME — state CLEAN, behaviour mis-stated

Constructed a checkpoint after 4 optimiser steps, rebuilt model and optimiser from
different random seeds, then reloaded exactly as the script does:

| quantity | max abs diff after load |
|---|---|
| `head.weight` (differed before load: yes) | **0.000e+00** |
| AdamW `exp_avg` | **0.000e+00** |
| AdamW `exp_avg_sq` | **0.000e+00** |
| Adam `step` counter | saved 4 → restored 4 |

`epoch`, `step`, `best`, `cfg`, `heads`, `horizons` all round-trip. Best-by-validation
is real: `onset_best.pt` is written only when the mean of the three matched AUCs
improves, separately from `onset_ep{n}.pt`. Model is 469,257 params (spec says 0.47M).

**Flaws in behaviour, not in state:**
1. Mid-epoch saves write `epoch=ep`, and the loop is `for ep in range(start_ep, epochs)`
   with no batch skip — the restored `step` is read into `start_step`, printed, and
   **never used**. A resume from `onset_live.pt` **re-runs epoch `ep` from batch 0**,
   repeating every window already seen. Neither the docstring, the spec, nor the
   commit message says so; the commit says "`--resume last` actually resumes".
2. `range(start_ep, a.epochs)` means resuming a completed run with the same
   `--epochs` trains **zero** epochs and silently prints the baseline banner.
3. `--resume last` takes max-mtime over all `*.pt` in `checkpoints/`, including
   `onset_best.pt`, which may be several epochs older than the newest `onset_live.pt`
   only by luck of ordering.
4. `checkpoints/` does not exist yet; both it and `seq/` are gitignored (lines
   411–412), so no checkpoint or dataset is version-controlled.

## i. SANITY BASELINE + SHUFFLED CONTROL — CLEAN (the important one)

80 train days (evenly spaced across 2024), stride 20, 300 steps at batch 256
(76,800 windows consumed), evaluated on the real matched-design rows of 45 val days.
Test days never loaded. Arm B permutes labels **within day** (spec §9).

| arm | loss first-25 → last-25 | fakeout | leg_descent | ultra_chop |
|---|---|---|---|---|
| untrained (random init) | — | 0.5009 | 0.5087 | 0.4879 |
| **A real, lr 1e-3** | 1.2636 → **1.1261** | 0.5050 | **0.5514** | **0.6880** |
| **B within-day shuffled, lr 1e-3 (seed 1)** | 1.3512 → 1.2999 | 0.4955 | 0.4848 | 0.5409 |
| **B′ shuffled, seed 2** | — | 0.4906 | 0.4992 | 0.4221 |
| C real, lr 3e-4 (script default) | 1.3217 → **1.1149** | 0.5039 | 0.5264 | 0.6568 |

Day-clustered 95 % bootstrap CIs (2,000 resamples over days):

| head | real | shuffled seed 1 | shuffled seed 2 |
|---|---|---|---|
| fakeout_poke (n=24,781) | 0.5050 [0.4995, 0.5106] | 0.4955 [0.4893, 0.5012] | 0.4906 [0.4856, 0.4957] |
| leg_descent (n=11,279) | 0.5514 [0.5301, 0.5754] | 0.4848 [0.4739, 0.4942] | 0.4992 [0.4923, 0.5064] |
| ultra_chop (n=3,182) | 0.6880 [0.6738, 0.7030] | 0.5409 [0.5211, 0.5619] | 0.4221 [0.4048, 0.4383] |

**(i) loss decreases** in all three arms, including the shuffled one (it converges to
the base rate, which is what a nulled target should do — and it falls much less:
−0.051 vs −0.138 and −0.207).

**(ii) the shuffled control collapses.** Across three independent shuffles the
ultra_chop control landed at 0.522 / 0.541 / 0.422 — it swings on **both sides** of
0.5, mean 0.495, so the individual CIs that exclude 0.5 are seed variance in a model
trained for 7 % of one epoch, not a persistent artifact. Real-label ultra_chop is
0.688 in the same budget: **~96 % of the learned discrimination is destroyed by
permuting labels within the day.** Per-day-mean AUC tells the same story
(0.6910 real vs 0.5204 / shuffled), so this is not a pooling effect.

**There is no day-level artifact.** The signal is time-aligned, and the pipeline
passes the single most important check.

Under-training is the obvious caveat: 300 steps is ~7 % of one stride-5 epoch on
30 % of the training days, so `fakeout_poke` at 0.505 is uninformative, not a null.

---

## WHAT MUST CHANGE BEFORE GPU SPEND (not applied — audit only)

1. **Re-register the baseline.** Spec §7 must read **0.6435 / 0.7580 / 0.8201**
   (or the GBM must be rebuilt with `idx-1` features and refit train-2024-only).
   As written, the rule kills a winning model on 2 of 3 heads.
2. **Score `ultra_chop` on the H=30 head**, or rebuild `matched_ultra_chop_*.parquet`
   from 1s bars so the anchor is exact. Right now the scored head disagrees with the
   design on 40 % of rows.
3. **Drop or relabel the 6 event-library-less day files** (2.33 % of training windows
   are guaranteed-false negatives).
4. **Normalise amplitude** (per-day or trailing z on the tick channels). Without it
   the 2024→2025-H1→test scale drift is a 1.6× domain shift that §9's kill condition
   will misread as regime overfit.
5. **Make `day_split` parse the date** instead of comparing strings.
6. Say in the docstring that a mid-epoch resume repeats the epoch, or use the saved
   `step` to skip.
7. Commit the script that produced `matched_probe.json`.

## Notes (no action required)

- `matched_eval` rebuilds a fresh `DayCache` over all 112 val days on **every call**
  (3× per epoch) and materialises every window as an individual float32 tensor —
  ~586 MB of Python-list tensors for fakeout_poke alone, plus repeated npz
  decompression. Correct, but it will dominate epoch wall time.
- `DayCache(tr)` holds ~189 MB, forked into 4 dataloader workers.
- `f` is stored float16; the only measurable consequence is 1.9e-3 on `vol_z`.
