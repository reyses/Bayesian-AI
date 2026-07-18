---
name: distilled-kalman_tuning_eda
description: GA-tuned Kalman CA filter + trailing stop finds macro waves IS but WR decays to ~breakeven OOS-2; trade-path EDA proves the trailing stop itself destroys profit on rollover
metadata: {type: distilled, topic: kalman_tuning_eda, status: concluded}
---
## Verdict
Genetically tuned a Numba Kalman constant-acceleration (position/velocity/
acceleration) filter + scalar trailing-stop exit on 1s MNQ ticks, then ran full
OOS validation and a trade-path EDA. Filter reliably detects ~71pt macro waves,
but the 79.4pt trailing stop that survives entry chop also bleeds out
"Chopped Out" trades after their peak. WR (PF-1) decays 0.305→0.119→0.050
IS→OOS-1→OOS-2. Concluded: abandon scalar trailing stops in favor of a
geometric/causal rollover-detection exit.

## Key numbers (with CIs where they exist)
- Tuned params (H1 2024 IS, GA): Q_JERK=1.81e-09, R_MEAS=12.55, Entry
  Velocity=0.066 pts/sec, Exit=79.4pt trailing stop.
- IS (H1 2024): 630 trades, Net PnL $7,406.50, WR(PF-1)=0.305, avg MFE 70.69 pts.
- OOS-1 (H2 2024): 1,139 trades, Net PnL $4,193.50, WR(PF-1)=0.119, avg MFE 72.02 pts.
- OOS-2 (2025-26): 3,694 trades, Net PnL $478.50, WR(PF-1)=0.050, avg MFE 71.65 pts.
- No CIs reported anywhere in this topic's files (pre-2026-04-22 metric convention).
- Category medians (`reports/eda/category_medians.csv`, n=5,463 total paths):
  Big Winner: duration 4924s, MFE +128.5pt, MAE -16.0pt, time-to-MFE 3572s.
  Chopped Out: duration 2131s, MFE +53.5pt, MAE -30.25pt, time-to-MFE 951s.
  Small Winner: duration 1993.5s, MFE +25.75pt, MAE -14.25pt, time-to-MFE 935.5s.
  Stopped Out: duration 932s, MFE +13.0pt, MAE -50.5pt, time-to-MFE 121.5s.
- Same categories per the prose report table (`kalman_tuning_and_eda_report.md`,
  3-category version, counts given): Big Winner 1,897 (median dur 1h22m, MFE
  +128.5pt, MAE -16.0pt, time-to-MFE 59min); Chopped Out 1,264 (35min, +53.5pt,
  -30.2pt, 15min); Stopped Out 2,218 (15min, +13.0pt, -50.5pt, 2min). Note: these
  3 counts sum to 5,379, not the full 5,463 — the CSV's 4th "Small Winner" bucket
  isn't reconciled in the prose report; not resolved in the source files.

## Graveyard / never-retry (if any)
- Scalar/fixed-distance trailing stop on this Kalman entry: WR collapses
  0.305→0.050 IS→OOS-2, and the 1,264 "Chopped Out" trades hit median +53.5pt
  then bleed out because the 79pt trailing buffer mathematically can't fire
  before a full reversal — the report explicitly flags this as a structural flaw,
  not a tuning issue.

## Reusable assets
- `kalman_genetic_tuner.py` — differential-evolution GA tuner for the Kalman CA
  filter (Q_JERK, R_MEAS, entry velocity, trailing-stop distance).
- `nmp_kalman_oos_validation.py` — runs the tuned filter across all OOS splits,
  produces `reports/findings/kalman_full_trades.csv`.
- `extract_trade_paths.py` — slices tick-by-tick entry→exit paths per trade into
  `reports/findings/trade_paths.parquet` (normalized to entry_price=0.0).
- `trade_path_eda.py` — produces the category-median clustering + lifecycle plots.
- `nmp_kalman_gif_generator.py` — renders the execution overlay GIF.

## Data locations
- Raw ticks: `DATA/ATLAS/1s` (604 trading days, Jan 2024–Mar 2026).
- Trade paths: `research/kalman_tuning_eda/reports/findings/trade_paths.parquet`.
- Full trade list: `research/kalman_tuning_eda/reports/findings/kalman_full_trades.csv`.
- Note: `extract_trade_paths.py` currently reads its input CSV from an external
  path (`C:/Users/reyse/.gemini/antigravity/brain/.../kalman_full_trades.csv`),
  not the repo-local reports folder — a path a future run must fix or re-point.

## Open threads
- The explicit next step stated in the report: replace the scalar trailing
  stop with a geometric/causal rollover detector that cuts near the MFE peak
  before structural collapse — not built in this topic.
- Small Winner vs. reported 3-category count mismatch (5,379 vs 5,463) unresolved.

## Sources
- `research/kalman_tuning_eda/kalman_tuning_and_eda_report.md`
- `research/kalman_tuning_eda/reports/eda/category_medians.csv`
- `research/kalman_tuning_eda/extract_trade_paths.py`
- `research/kalman_tuning_eda/kalman_genetic_tuner.py`
- `research/kalman_tuning_eda/nmp_kalman_oos_validation.py`
- `research/kalman_tuning_eda/trade_path_eda.py`
- `research/kalman_tuning_eda/project.md` (empty DMAIC skeleton)

## Archive recommendation
ARCHIVE — conclusion reached (trailing-stop exit structurally flawed, pivot
recommended), no CI-gated metrics per current convention, and the report's own
next step points at a different exit design (geometric/causal), not further
work in this folder. Keep as lineage evidence for why scalar trailing stops
were rejected.
