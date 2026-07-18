---
name: distilled-ai_auto_labeler
description: Algorithmic golden-dataset labeler (v1 fixed-window -> v2 regime-adaptive cubic-leg segmentation) that produces DATA/ai_cusp_picks, the ground-truth trade labels consumed across the research program.
metadata: {type: distilled, topic: ai_auto_labeler, status: live}
---
## Verdict
Built to algorithmically generate the "Golden Dataset" of perfect-hindsight trades
mimicking the manual visual labeling workflow. v1 (`ai_labeler.py`) used a fixed
60-bar prominence + fixed 7pt/3pt thresholds. v2 (`ai_labeler_v2.py`) replaced this
with regime-adaptive thresholds (local amplitude scale) and cubic-leg zigzag
segmentation, fixing v1's truncation of slow-forming trends. Adaptive tuning shows
a directional but NOT statistically significant improvement over fixed thresholds
at N=9 days (LODO ΔF1 +0.020, CI includes 0). **DEPENDENCY**: this topic's sole
output, `DATA/ai_cusp_picks`, is grepped as ground-truth labels by 39 files across
exit_dojo, leg_clock, mamba_zigzag_baseline, nt8_catalog, and tools/viz — very
likely THE ground-truth label source for the broader program.

## Key numbers (with CIs where they exist)
- Human label swing size (398 picks, 9 days): median 15.8pt, 25th pct 7.8, 75th pct 32.2; 22% of human swings are below the 7pt threshold `tools/tune_to_human.py` output (reports/tune_to_human.md).
- Best fixed-threshold match: CUBIC_N=20, TREND_PTS=3 -> F1=0.664 (recall 72%, prec 62%).
- Best adaptive match: K_TREND=0.60, AMP_MODE=w60 -> F1=0.691, recall 72%, prec 67%, clamp rate 35.3%.
- LODO (Leave-One-Day-Out): Fixed F1=0.664, Adaptive F1=0.684, delta +0.020.
- Bootstrap CIs: in-sample ΔF1 +0.028 [95% CI -0.003, +0.067] NOT significant; LODO ΔF1 +0.020 [95% CI -0.009, +0.058] NOT significant.
- Cross-day correlation (day scale vs best-T): Spearman 0.184, p=0.635 (weak, not significant).
- Intraday RTH/ON scale ratio: 1.84x-3.84x across 9 days (reports/diagnose_regime_spread.md).
- Swing dispersion tightening: raw swings IQR/median 1.556 -> scaled ratios IQR/median 0.965 (>=20% drop; "premise supported").

## Graveyard / never-retry (if any)
- v1 fixed 60-bar prominence cap: truncated slow-forming trends, mismatched the 7pt-filter/10pt-win criteria — superseded by v2, not deleted from repo but not the active labeler.

## Reusable assets
- `pipeline/ai_labeler_v2.py` — active labeler (regime-adaptive cubic-leg segmentation); run via `--day YYYY_MM_DD` or `--month YYYY_MM`.
- `pipeline/amplitude_scale.py` — local volatility scale used for adaptive thresholds (w30/w60/w120/day modes).
- `pipeline/run_all_months.py` — batch driver across 27 months (2024-01 to 2026-03).
- `tools/tune_to_human.py` — LODO + bootstrap CI tuner against 398 human picks.
- `tools/diagnose_regime_spread.py` — per-day/RTH-vs-ON scale diagnostics.
- `tools/inspect_losses.py` — audits negative-PnL labeler outputs (diagnostic only, no saved report).

## Data locations
- Reads `DATA/ATLAS/1s` and `DATA/ATLAS/1m` (OHLCV parquet).
- Produces `DATA/ai_cusp_picks/ai_picks_<date>_multi.json` (trades), `DATA/ai_cusp_picks/flagged/<date>_flagged.json` (reversal regions).

## Open threads
- Adaptive-vs-fixed win is directional only; N=9 days too small for significance — needs a larger LODO sample to confirm K_TREND=0.60/w60 as a real gain over fixed T=3. 22% of human swings fall below the legacy 7pt threshold — unresolved whether v2's adaptive floor (TREND_MIN_PTS=2.5) fully recovers these.

## Sources
- research/ai_auto_labeler/README.md
- research/ai_auto_labeler/pipeline/ai_labeler_v2.py
- research/ai_auto_labeler/reports/tune_to_human.md
- research/ai_auto_labeler/reports/diagnose_regime_spread.md
- research/ai_auto_labeler/tools/inspect_losses.py
- grep for `ai_cusp_picks` across repo (39 files: research/exit_dojo,
  research/leg_clock, research/mamba_zigzag_baseline, research/nt8_catalog,
  tools/viz/cusp_marker.py, tools/viz/plugins/catalog_overlay.py)

## Archive recommendation
KEEP-LIVE. Not concluded, not dead: the pipeline is the active label generator
and its output (`DATA/ai_cusp_picks`) is a load-bearing dependency for exit_dojo,
mamba_zigzag_baseline, leg_clock, and nt8_catalog tooling. Archiving the folder
would break re-generation of the ground-truth dataset those topics depend on.
