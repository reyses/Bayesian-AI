# LLM News-Intensity Feature for DRS

DMAIC research project. Approved plan: `~/.claude/plans/i-would-like-to-jaunty-spindle.md`.

## D — Define

**Problem.** The Day-Regime Sizer (DRS) canonical GBM (2026-05-18 verdict) lands
at OOS sealed Pearson **+0.139 CI [-0.047, +0.451]** — lower bound crosses zero,
not deployable. Permutation importance shows the binary event flags
(`is_fomc`, `is_cpi`, `is_nfp`, `is_opex`) contribute **$0** of MAE
reduction each. A binary flag tells the model an event happened, not whether
it was hawkish/dovish/on-consensus/shock.

**Hypothesis.** A continuous LLM-scored intensity column (0-10) derived from
the actual press-release text will (a) subsume the dead binary flags and
(b) push OOS sealed Pearson lower CI strictly above zero, making DRS
deployable.

**Scope.** Feature engineering only. The LLM never touches the live decision
path; the live engine still consumes the cross-day parquet via B10 / forward
pass exactly as today. Production cron and live-sizing wiring are explicitly
**out of scope** for this project.

## M — Measure

**Baseline (DRS canonical, 2026-05-18)**:
| Metric                            | Value                                |
| --------------------------------- | ------------------------------------ |
| IS WF Pearson (5-fold, n=217)     | **+0.191** CI [+0.098, +0.405]       |
| OOS sealed Pearson (n=23 NT8)     | **+0.139** CI [-0.047, +0.451]       |
| Naive sizing OOS $/day            | **-$333/day** CI [-$523, -$163]      |
| `is_fomc/cpi/nfp/opex` ΔMAE       | **$0/feature**                       |
| `vix_close_prior` ΔMAE            | +$17/day                             |
| `days_since_fomc` ΔMAE            | +$62/day (strongest event-related)   |

**Targets (Phase A gate)**:
- IS WF Pearson must stay ≥ baseline lower CI (+0.098)
- OOS sealed Pearson lower CI must flip strictly positive (was -0.047)
- `news_intensity_today` ΔMAE must be ≥ +$30/day (beats `vix_close_prior`)

**Targets (Phase B gate, only if A passes)**:
- Bootstrap CI on `delta_Pearson(A+B vs A only)` must exclude 0

## A — Analyze

Binary event flags are uninformative because:
- The model sees the same `is_fomc=1` on a 25bp surprise cut AND on a held-rates-on-consensus meeting.
- Sparse: ~16 FOMC days / 24 CPI days / 24 NFP days in a 293-day window. Low N per category, no within-category variance.

A continuous intensity score derived from the press release text directly
encodes the magnitude / surprise / policy-shift of the event in a single
scalar. The LLM is the cheapest way to produce this signal from unstructured
text without a hand-labeled training set.

**Why local LLM**:
- Reproducible (deterministic gen, seeded)
- Free (no per-token API cost across 80 backfill releases + daily prod use)
- Lives on idle CUDA (live engine uses 0MB GPU)
- No data egress (press releases are public but pipeline data isn't)

**Risk: training-data memorization.** Llama-3.1-8B has training cutoff
~mid-2024 — it may have seen 2024 FOMC statements + actual market reactions.
Mitigation: explicit "score based ONLY on text" prompt + synthetic-statement
anti-cheating spot check + sealed OOS validation on 2026 dates (post-cutoff,
less leakage).

## I — Improve

Iterative cycles, each a separate PDCA file:

- **cycle_01.md — Phase A**: pre-market column (`news_intensity_today`)
  only. Score releases published before 09:30 ET. Validate against
  Phase A gate.
- **cycle_02.md — Phase B** (only if A passes): add EOD-of-yesterday
  column (`news_intensity_prior`). Validate incremental lift on top of A.
- Future cycles only if needed (e.g. swap to larger model if 8B shows
  flat scores; add per-event-type weighting; add Fed speaker speeches).

## C — Control

**Production untouched during dev** (verified via hash check at each phase gate):
- `tools/sourcing/build_cross_day_features.py`
- `tools/sourcing/drs_canonical_gbm.py`
- `DATA/CROSS_DAY/cross_day_features.parquet`
- `DATA/CROSS_DAY/cross_day_features_with_target.parquet`
- `DATA/CROSS_DAY/drs_canonical_gbm.pkl`
- `tools/forward_pass_full_stack.py` and B10 day-multiplier

**Promotion gate** (separate user-approved session):
1. Diff `build_cross_day_features_v2.py` against canonical; apply minimal
   diff to canonical (or to `drs_a_step4_aggregate_day_pnl.py` if simpler)
2. **B10 regression check**: re-run forward pass IS+OOS; $/day delta must
   be within ±$5/day noise floor
3. Refit canonical DRS GBM
4. Git tag + safe branch
5. Journal entry + MEMORY.md update

**Backout cost**: delete `tools/sourcing/llm_news/`, both `_v2` scripts,
`DATA/CROSS_DAY/dev/`, `research/llm_news_intensity/`. Zero production
regression possible because canonical paths are byte-identical until
promotion.

## File map

```
research/llm_news_intensity/
├── project.md                            (this file)
├── cycle_01.md                           Phase A PDCA
├── cycle_02.md                           Phase B PDCA (created when A passes)
└── findings/YYYY-MM-DD_*.md              gate results

tools/sourcing/llm_news/                  self-contained module
├── __init__.py
├── fetch.py                              Fed + BLS scrapers
├── score.py                              Llama-3.1-8B GGUF scorer
├── join.py                               lookahead-safe join helpers
└── cli.py                                python -m ... cli {fetch|score|build|train}

tools/sourcing/build_cross_day_features_v2.py    augmenter (reads canonical, adds LLM cols)
tools/sourcing/drs_canonical_gbm_v2.py           dev DRS trainer (reads dev parquet)

DATA/CROSS_DAY/raw/press_releases/{date}_{event}.txt    raw text archive
DATA/CROSS_DAY/dev/news_scores_v1.parquet               LLM raw output
DATA/CROSS_DAY/dev/cross_day_features_with_target_v2.parquet    augmented
DATA/CROSS_DAY/dev/drs_canonical_gbm_v2.pkl             dev model
```
