# F-Space Cadence / Flip-Timing Research

**One-line verdict:** real MNQ price has **tight (~10% band), short (~25–38 s), bar-close** regimes that persist
meaningfully longer than a spectrum-matched (Fourier) null — **robust across a week**. The original "continuous
sliding-window from 1s" thesis is the *weaker* representation; **update cadence (bar-close), not continuity or
window length, is the lever.** Structure is real but *descriptive* (in-sample); forward **direction** is untested = next.

---

## Layout
- **`pipeline/`** — the segmenter engine:
  - `stage1_speed_pass.py` — forward-expansion segmenter → PRISTINE / UNPROCESSED_CHAOS. Corrected break rule
    (per-bar tol = max(10%·|Δclose|, 1 tick), break on 5 consecutive off). `--seed_bars N` = regime-length floor.
    Now keeps the full per-segment **R-curve** + direction + flip descriptors.
  - `stage2_parallel_chaos.py` — re-segments chaos → RECOVERED / PURE_CHAOS with **graded tiers** (tier *t* = fits at
    band 10%·*t*; tier = loosening needed to model; 99 = nothing to model).
  - `tiering.py` — tier/`max_consecutive` helpers.
- **`builders/`** — make the F-space datasets + nulls (CPU/numba):
  - `build_run_b2c.py` — **continuous** sliding-window (window=horizon-sec). `build_run_b2t.py` — **tiled** proper-bar
    + causal step-fill (`--full` adds L4-NMP + L5-dist → FEATURES_RUN_B2TF). `build_run_runc.py` — bar-close-sampled
    (Run C). `build_run_c2.py` — continuous **free-N** window. `build_run_base.py` — single-resolution at 5s/15s/1m.
  - `make_brownian_null.py` — null generator: `--null brownian` (iid walk) | `--null fourier` (phase-randomized,
    spectrum-preserving = the strict null).
- **`tools/`** — analysis + orchestration:
  - `build_map.py` (composite survival/explainability map), `week_analysis.py` (3-model day-block-CI contrast),
    `forward_increment_test.py` (micro forward-predict, null-anchored), `make_week_report.py` (plots + GIF),
    `run_week.sh` (sequential week driver — idempotent).
- **`reports/`** — findings + `assets/` (plots, GIF). **START HERE → `WEEK_REPORT_fspace_cadence.md`.**

## Data lives at REPO ROOT (not here — large/gitignored). Run everything FROM repo root.
- `DATA/ATLAS/FEATURES_RUN_*` — F-space parquets: `B2C`, `B2T`, `B2TF`, `C2024`(RunC), `C2_w{60,180,480,960}`,
  `B{5s,15s,1m}`; each with `_BROWN` / `_FOUR` null variants per day.
- `artifacts/stage1_*.json`, `stage2_*.json` — segment outputs (stage-1 writes `artifacts/` relative to CWD).
- `DATA/ATLAS/{1s,5s,15s,1m,...}/YYYY_MM_DD.parquet` — OHLCV.

## Reports (in `reports/`)
- `WEEK_REPORT_fspace_cadence.md` — the headline report (plots + animated GIF).
- `week_3model_contrast_2024_02.md` — B2T/RunC/B2C × 5 days, survival gap + day-block CI.
- `seed_descent_2024_02_20.md` — diminishing-returns seed sweep (functional floor ~25–30 bars).
- `b2c_continuous_2024_02_20.md` — the continuous-vs-tiled A/B + null deflation + corrected-rule flip.
- `map_b2c_b2t_2024_02_20.md` — composite survival/explainability map. `forward_increment_2024_02_20.md` — micro forward test (negative).

## How to run (from repo root)
```
# build features + null for a day
python research/fspace_cadence/builders/build_run_b2t.py --day 2024_02_20
python research/fspace_cadence/builders/make_brownian_null.py --day 2024_02_20 --null fourier
python research/fspace_cadence/builders/build_run_b2t.py --day 2024_02_20_FOUR
# segment
python research/fspace_cadence/pipeline/stage1_speed_pass.py --day 2024_02_20 --tf 1s \
  --run_name R --features_root DATA/ATLAS/FEATURES_RUN_B2T --seed_bars 30
# chaos recovery + graded tiers
python research/fspace_cadence/pipeline/stage2_parallel_chaos.py --day 2024_02_20 --tf 1s \
  --run_name R --features_root DATA/ATLAS/FEATURES_RUN_B2T
# week validation + figures
bash research/fspace_cadence/tools/run_week.sh
python research/fspace_cadence/tools/make_week_report.py
```

## Design spec + open ideas
`docs/JULES_STAGE12_MAP_OVERHAUL.md` — overhaul spec + TODO/PARKED (post-week): rolling-VWAP direction primitive,
anchor→wait-N direction test, RKHS (MMD-null / kernel-ridge), OU fade-regime model, HMM (graveyard-flagged).
**Next build:** the forward **direction** predictor (anchor→wait-N, null-anchored) — the tradeable test + where L4/λ gets exercised.
