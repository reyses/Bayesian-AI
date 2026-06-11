# Regression Segments — DMAIC Project

## Define
Partition each trading day into **regime segments** by how cleanly price-delta can
be fit from the V2 layered feature set. Each segment is labeled:
- `PRISTINE` — a clean polynomial fit found directly (stage 1).
- `RECOVERED` — a clean-enough fit found inside a former chaos gap (stage 2).
- `PURE_CHAOS` / `UNPROCESSED_CHAOS` — no acceptable fit (tier 9).
plus a `volatility_tier` (1 = tightest, 9 = pure chaos).

Goal: a per-day, per-bar regime map used as a **diagnostic** to characterize where
the RL/zigzag engines make or lose money (see `phase_d_mapping.py`).

> ⚠️ **These labels are NON-CAUSAL.** Betas, tiers, and segment boundaries are
> in-sample fits over the whole segment (future bars included); features are
> StandardScaler-fit over the whole day. Valid as a post-hoc diagnostic only —
> never as a live feature or training target (would reintroduce lookahead).

## Measure
- Input: `DATA/ATLAS/5s/<day>.parquet` (OHLCV) + `DATA/ATLAS/FEATURES_5s_v2/<family>/<day>.parquet`.
- Output: `artifacts/stage{1,2}_segments_<day>.json`, aggregated to
  `artifacts/stage2_{week,year}_segments.json`.
- Error band E = `ERROR_BAND_FRACTION` (0.10) × previous segment's price range;
  first segment uses `INITIAL_ERROR_BAND` (1.00). Tier = max-residual vs E with a
  max-consecutive-outlier guard.

## Analyze
Pipeline:
1. **stage1_speed_pass.py** — per day: GroupLasso→ElasticNet feature screen
   (GPU FISTA), GPU batched-OLS forward expansion, emit PRISTINE blocks +
   UNPROCESSED_CHAOS gaps.
2. **stage2_parallel_chaos.py** — re-segment chaos gaps at finer tiers via a
   CPU multiprocessing Pool (CPU sklearn by design — no shared CUDA context
   across workers).
3. **run_week.py / run_year.py** — orchestrate days then aggregate.
4. **mothership_server.py / drone_worker.py** — optional LAN job distribution.
5. **phase_d_mapping.py** — map RL trades onto segment labels; report to
   `reports/findings/phase_d_*.md`.
6. **plot_*.py** — DOE / Minitab-style interaction & Pareto visualizations.

## Improve
Cycle files: `research/Regression segments/cycle_NN.md` (none yet).

## Control
Known issues / decisions (audit 2026-06-08):
- Labels are in-sample/non-causal — banner added to stage1/stage2.
- `run_year.py` Phase-2 numpy import fixed (was a hard NameError crash).
- `mothership_server.py` packaged non-existent `FEATURES_<day>.parquet` under
  L0/L1/L2/L3 — fixed to enumerate real `<family>/<day>.parquet` dirs.
- `phase_d_mapping.py` evaluated a January day against February segments
  (all-UNCLASSIFIED) — fixed to evaluate a day present in the loaded segments.
- Stage1 (GPU FISTA) and Stage2 (CPU sklearn) use different ElasticNet solvers
  by design (multiprocessing) — tiers not strictly comparable at the margin.
- Error band is path-dependent on processing order: partial-day runs
  (`--hours`, `--start-date`) produce different tiers than full-day runs.
- **Convention**: All segment ranges are half-open `[start_idx, end_idx)` and
  `[raw_start_idx, raw_end_idx)`. `raw_end_idx` is the raw index of the first
  bar AFTER the segment.
- Stage 1 `STRIDE` hunt-forward is an optimization. Changing `STRIDE` without
  the exact backward-refine step alters boundaries and tier path-dependence.
