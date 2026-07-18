---
name: distilled-kalman_entry
description: GA-Kalman entry+trail (velocity/accel state) tested clean (gap-guarded) — no config beats 0 at 95% CI on any OOS split; graveyard-adjacent.
metadata: {type: distilled, topic: kalman_entry, status: dead}
---
## Verdict
Asked whether a Kalman-filtered velocity/acceleration state (GA-tuned entry
threshold + trailing exit) produces a tradeable entry signal on MNQ. After
gap-guarding a data-contamination bug (worst single loss dropped from -$454
to -$127), the clean OOS $/day for every entry-threshold × exit combination
tested has a 95% CI that includes 0, and most alternate exits (vel_flip,
accel_flip, trail_30) are significantly NEGATIVE. Regret diagnostics show
the entry has near-zero direction edge (51% "wrong side" rate) but IS
systematically LATE (73% of entries fire after price already moved
+26.2pt in the trade's own direction).

## Key numbers (with CIs where they exist)
- Gap-guarded (clean) $/day: IS +36.0 [-20.0,+90.1] incl0; OOS_H2_24 +15.4
  [-61.8,+92.4] incl0; OOS_25_26 -28.4 [-80.8,+22.7] incl0
  (`kalman_gapguard_compare.md`; contaminated pre-fix numbers were +57.9/+32.0/+1.4)
- Best entry×exit combo (0.02, trail_79_GA), OOS_25_26: $/day +10.8
  [-37.0,+58.2] incl0, 166/339 win-days (`kalman_entry_timing_sweep.md`)
- trail_30 exit: OOS_25_26 -165 to -176/day, ALL EXCL0; OOS_H2_24 -68 to
  -77/day (mostly EXCL0) (`kalman_entry_timing_sweep.md`)
- vel_flip / accel_flip exits: -320 to -629/day across all splits, ALL EXCL0
  (`kalman_entry_timing_sweep.md`)
- OOS failure anatomy (N=4833, 465 days): net $/day +10.0 (contaminated);
  big stop-outs (≤-$90) = 1938 trades, sum -$203,825; exit giveback on
  winners = $248,124 (winners kept only 52% of MFE peak); chop entries
  (MFE<10pt) = 855 trades (18%), net -$86,646 (`kalman_failure_diagnosis.md`)
- Entry/direction regret: 51% "wrong side" (>50%=anti-predictive), mean
  MFE edge only +1.1pt; regret v2 (magnitude): only 6% ties, chosen vs
  flip roughly 46/47 split, mean edge +1.7pt (`kalman_regret_analysis.md`,
  `kalman_regret_v2.md`)
- Over-wait test: 73% of entries fire with price already moving the
  trade's way (mean pre-entry move +26.2pt, median +22.0pt); pre-entry MFE
  49.3pt vs forward chosen MFE 39.7pt → entry is right-direction but LATE,
  not directionless (`kalman_regret_v2.md`)
- Macro-alignment filter (1h/4h trend agreement) does not rescue it: best
  filtered subset (1h-R WITH) still -1.07 net$/tr vs -2.00 all, no
  sub-period consistency (2026 split flips sign) (`kalman_macro_rcurve.md`)
- Exit giveback by time-from-peak (OOS, right-direction trades N=4340):
  total giveback $625,198; 5m+ bucket alone $458,906 (3186 trades)
  (`kalman_regret_analysis.md`)

## Graveyard / never-retry (if any)
- vel_flip / accel_flip exits: significantly negative at every entry
  threshold and every split tested (-320 to -629/day, all EXCL0).
- trail_30 exit: significantly negative on OOS_25_26 and mostly on
  OOS_H2_24 (-68 to -176/day).
- Raw (non-gap-guarded) GA-Kalman backtest numbers are CONTAMINATED
  (worst loss -$454 vs -$127 clean) — do not cite pre-gapguard $/day.

## Reusable assets
- `tools/kalman_backtest_gapguarded.py` — gap/price-jump guarded GA-Kalman
  backtest, writes `kalman_clean_trades.csv` (ATLAS 1s source).
- `tools/kalman_entry_timing_sweep.py`, `kalman_clean_entry_sweep.py` —
  entry-threshold × exit sweep harness (trade-level + day-level).
- `tools/kalman_failure_diagnosis.py` — stop-out/giveback/chop leak
  decomposition.
- `tools/kalman_regret_analysis.py`, `kalman_regret_v2.py` — hindsight
  entry-direction and exit-giveback regret diagnostics.
- `tools/kalman_macro_alignment.py`, `kalman_macro_rcurve.py` — 1h/4h
  trend-agreement filter tests.

## Data locations
- Source: `DATA/ATLAS/1s/*.parquet` (per `kalman_backtest_gapguarded.py`).
- Output: `reports/kalman_clean_trades.csv` (clean trade log, gap-guarded).
- Splits referenced throughout: IS (H1-24), OOS_H2_24, OOS_25_26.

## Open threads
- README.md and project.md are stub-only (no Define/Measure/Analyze content
  filled in) — DMAIC was never written up despite the reports existing.
- Late-entry finding (73% over-wait) was never followed into a "shift
  entry earlier" retest within this folder.

## Sources
- research/kalman_entry/reports/kalman_gapguard_compare.md
- research/kalman_entry/reports/kalman_entry_timing_sweep.md
- research/kalman_entry/reports/kalman_clean_entry_sweep.md
- research/kalman_entry/reports/kalman_failure_diagnosis.md
- research/kalman_entry/reports/kalman_regret_analysis.md
- research/kalman_entry/reports/kalman_regret_v2.md
- research/kalman_entry/reports/kalman_macro_alignment.md
- research/kalman_entry/reports/kalman_macro_rcurve.md
- research/kalman_entry/tools/kalman_backtest_gapguarded.py

## Archive recommendation
ARCHIVE (reason: every OOS $/day is CI-includes-0 or significantly negative;
entry has no direction edge and is structurally late; no live consumer
references this folder's outputs; DMAIC/README were never filled in,
suggesting the line of research was already abandoned in place).
