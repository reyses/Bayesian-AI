# Golden vector schema — NT8 native port parity harness (P0, doc 129)

This document specifies **every column** of `research/nt8_port/golden/<day>.parquet`
so the C# side can implement against it **without reading the Python**. Each file is
one reference trading day; **one row per RTH 1-minute bar** (406 rows for a full
session, fewer for a shortened holiday session).

## Causal / timestamp conventions (READ FIRST)
- **Bar stamping**: a bar is stamped at its **OPEN**. A 1-minute bar labelled `bar_ts = T`
  (with `T % 60 == 0`, epoch seconds UTC) covers the 5-second sub-bars in `[T, T+60)`
  and **closes at `T+60`**. The underlying substrate is the 5-second close stream.
- **Decision point**: all decision-bearing columns for row `T` reflect information known
  **by the close of minute `T` (i.e. at `T+60`)**. The C# engine acts on row `T` at the
  boundary `T+60` (= the open of the next minute). This matches the project rule
  "decisions at 1m boundaries; per-TF context is constant until the bar CLOSES".
- **Fire → bar mapping**: a stream fire emitted at 5s timestamp `t` belongs to bar
  `floor(t/60)*60`. Tier/candle streams emit at a bar-close row (`t % 60 == 0`) reflecting
  the bar that just closed; grouping that fire into minute `t` is the acted-upon minute.
- **Zigzag state**: sampled at the **last 5s row of the minute** (`T+55`, known at `T+60`).
- **Units**: price in points (MNQ tick = 0.25, tick value $0.50). `zz_pivot_price` in points.
  `zz_pivot_age_min` and all `*_min` in minutes. Directions: `+1 = long`, `-1 = short`,
  `0 = none/undecided`.

## Reference decider (what produced these values)
- **Entry P** = the pooled **dossier combiner** (`combiner_preview.py` conventions):
  `LogisticRegression` (standardized) over `[pivot_age_min, sig_with_leg, tod, inter,
  consensus, is_<det> one-hots]`, fit on **2024** fires (`y` = direction-agreement with the
  active AI label), test 2025+26. Fit reproduced **exactly** from the current 56-stream pool.
- **Frozen entry threshold** `P >= 0.7339` = the **90th percentile of the 2024-train pooled P**
  (top decile), computed once and frozen. See `golden_manifest.json → frozen_top_decile_threshold`.
- **R-trigger pivots** = a **verbatim port of `training/strategies/zigzag.py::ZigzagStrategy`**
  (the live path): `extreme ± R` flip with `min_bars_5s = 36`, streamed over the continuous
  5s close stream. Per-day `R` (`min_reversal_ticks`) `= max(4, round(ATR(14 1m)×4 / 0.25))`,
  ATR taken **causally at the first RTH 5s bar** (reuses `DayCtx.zz_thr = ATR(14)×4` points).
  *(The archived offline builder used whole-day median-TR; the causal open-anchored ATR is
  used here and flagged for P2 to reconcile — a scalar-R knob, not a state-machine change.)*

## Column dictionary

| column | dtype | units | meaning / causal note |
|---|---|---|---|
| `bar_ts` | int64 | epoch s (UTC) | 1-minute bar **open**; `bar_ts % 60 == 0`. Bar closes at `bar_ts+60`. |
| `date` | str | — | day id `YYYY_MM_DD` (matches the 5s/feature-store file stem). |
| `f_<STREAM>` (×22) | int64 | {−1,0,+1} | Fire state of each **top-K** stream in this bar: `+1` a LONG fire, `−1` a SHORT fire, `0` no fire. If a stream fires both directions in one bar (rare), the **last** fire's direction is recorded. 22 columns, one per top-K stream (list below). |
| `n_fires_topk` | int64 | count | number of top-K stream fires landing in this bar (across all 22 streams). |
| `gov_stream` | str | — | the **governing** stream = the top-K fire with the **highest P** in this bar (`''` if none). |
| `gov_dir` | int64 | {−1,0,+1} | direction of the governing fire. |
| `P_topk` | float64 | prob | **max pooled combiner P over the top-K fires** in this bar (drives `entry`). `NaN` if no top-K fire. |
| `P_any` | float64 | prob | max pooled P over **all reproduced fires** in this bar (top-K ∪ non-top-K reproduced streams). Reference/diagnostic. `NaN` if no fire. |
| `entry` | int64 | {0,1} | `1` iff `P_topk >= 0.7339` (frozen top-decile threshold). |
| `entry_dir` | int64 | {−1,0,+1} | `gov_dir` when `entry==1`, else `0`. The entry decision + side. |
| `zz_leg` | int64 | {−1,0,+1} | R-trigger leg direction as of the bar close (`+1` long leg, `−1` short leg, `0` undecided/warmup). |
| `zz_confirm` | int64 | {−1,0,+1} | `+1`/`−1` if the R-trigger **flipped** (confirmed a pivot) during this minute (new leg dir); `0` otherwise. This is the live reversal-exit / new-entry trigger. |
| `zz_pivot_age_min` | float64 | minutes | minutes since the last confirmed R-trigger pivot, at bar close. |
| `zz_pivot_price` | float64 | points | price of the last confirmed R-trigger pivot (swing high/low). |

**Top-K stream columns (K=22), in weight order:**
`RSI06, MACD07, EXITKMDR, TMPL0, ZIGZAG, ATR09, NMP, DOW19, NMP9RIDEAGAINST, ROUND05,
NMPTFADECALM, RENKO24, ORB02, VWAP03, CTXER, PIVOT16, SAR23, PTRNENGULF, NMP9RIDECALM,
NMPTMTFBRK, TUNNEL20, NMP9FADEAGAINST`

## How K was derived (top-K = 22)
Rank the per-stream one-hot `|coef|` (standardized). `K` = smallest set whose cumulative
`|coef|` ≥ **80% of the stream coefficient mass** (`Σ|coef|` over the `is_<det>` one-hots only
= **3.302**). Cumulative reaches 80.1% at the 22nd stream (`NMP9FADEAGAINST`).

The **grand-total** interpretation (BASE + consensus in the denominator) is **degenerate**:
`stream_mass 3.302 < 0.80 × all_mass 4.321 = 3.457`, i.e. even *all 56 streams* cannot reach
80% of the grand total — so the stream-mass denominator is the only well-defined reading.

**Finding — the combiner's weight is diffuse.** No small subset carries it: the top stream
(RSI06) is only 11% of stream mass; it takes 22 of 56 streams to reach 80%. The port's entry
model is genuinely wide, not a 3-4 signal shortcut. Full ranking in `golden_manifest.json`.

## Streams reproduced vs excluded
Fires are regenerated on the reference days with the **canonical causal generators**
(`dossier_signal_pipeline.GENS`, ~53 streams) **+ TMPL0** (`template_stream_builder`, frozen
2024 K-means codebook `tmpl0_templates_2024.json`, reuses `DayCtx`). Two combiner streams are
**excluded** from the reproduced pool (both rank far below top-K → top-K unaffected):
- `ADX08` (|coef| ≈ 0.010, rank ~42) — a separate ADX tool, not a causal-pipeline primitive.
- `FOOTPRINTIMB` (not in the 55-snapshot) — a **meta** stream over `econ_drift_rows` (second-order
  on the combiner's own P); circular, not a port primitive.
Their combiner one-hots stay `0` on the reproduced fires (exactly as the model sees a non-firing
stream), so the frozen logistic applies unchanged.

## Known caveat — consensus definition
The combiner **fit** computes `consensus` (same-direction co-fires within ±180 s) over the
**label-window-filtered** `signal_rows` pool (`combiner_preview.load_pool`). At **generation**
there are no labels, so `consensus` is computed over **all reproduced fires of the day** (the
live-valid definition; the frozen `mu/sd` are applied unchanged). Standardization is monotone,
so within-day P **ordering** is preserved; absolute P (hence the exact top-decile crossing) may
shift slightly. **P1's compact re-fit should define consensus over its own K streams
consistently (fit == deploy).** The `P_topk`/`P_any` here are the **full-combiner reference**,
not the P1 compact model — the hard bar-by-bar parity target for P2 is the **fire-state and
zigzag columns**; P (P1) is validated against the re-fit compact model.

## Determinism
No randomness anywhere (lbfgs logistic, deterministic; consensus/features/generators/TMPL0
nearest-centroid assign all deterministic). Re-running yields **byte-identical** parquets —
verified via `--verify-determinism` (regenerate one day twice, compare sha256: IDENTICAL).
Per-file `sha256` (first 16 hex) is recorded in `golden_manifest.json → day_sanity` and
`golden_sanity.md`.
