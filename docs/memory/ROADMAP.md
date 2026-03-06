# Future Topics Backlog

Items captured during sessions — prioritized by opportunity gap analysis.

## Status Key
- **READY** — scoped, can implement now
- **IDEA** — discussed, needs spec/clarification
- **PARKED** — evaluated, deferred with reason
- **DONE** — completed and committed

## Priority Rationale (2026-02-22 opportunity gap)
```
Ideal:       $3,084,792    Actual: $69,402 (2.25% captured)
#1 leak  skipped signals:   $1,370,418  (44.4%)
#2 leak  exits too early:   $  531,335  (17.2%)
#3 leak  wrong direction:   $   21,283  (0.7%)
```
**Exit quality is the #1 ROI fix.** Too-early and too-late are the same problem:
too-early leaves money on the table, too-late blocks the slot causing skipped signals.
Spectral exit gates (Fourier + Laplace) compress hold to the structural wave duration.
Wrong direction is negligible — the model picks sides fine, it just can't hold.

---

## 1-2. Spectral Entry & Exit Gates (Phases A-B-C)  ★ PRIORITY
**Status:** SPEC WRITTEN — `docs/JULES_SPECTRAL_GATES.md` ready for Jules
**Category:** Spectral Protocol (entry timing + exit probability)
**Attacks:** All three leaks simultaneously:
  - Leak #1 skipped signals ($805K) — kinetic exhaustion frees slot faster
  - Leak #2 exits too early ($278K) — cycle lock prevents premature trail exits
  - Leak #3 entering too late ($33K reversed) — entry phase gate skips exhausted waves
**Spec:** 7 Optuna-tunable parameters per template (no magic numbers):
  - `fft_window_bars` (40-200), `min_spectral_power` (0.1-0.9), `max_entry_phase` (0.1-0.6)
  - `damping_threshold` (0.5-2.0), `velocity_window_bars` (10-60), `min_profit_ticks` (2-20)
  - `exit_prob_floor` (0.15-0.50)
**Architecture:** One FFT per bar (cached), serves both entry gate and exit probability.
  Phase A = `core/spectral.py` (SpectralState + compute functions)
  Phase B = Gate 4 entry phase check in orchestrator
  Phase C = Bayesian P(continuation) → WaveRider exit + trail modulation
**Files:** 6 files, ~550 lines total (new + modified)
**Phase D** (progressive cascade elimination) is a SEPARATE FORK — not in this spec

---

## 3. Pattern Anatomy Visualization
**Status:** IDEA — three phases planned
**Category:** Diagnostics / understanding
**Context:** We have 367 templates but no visual insight into what a pattern
"looks like", where its expected peak is, or how neighbors differ.
**Phases:**
1. **One-off diagnostic** — matplotlib script: pick a top template, plot 16D
   centroid as radar chart, overlay MFE peak as time marker, show 2-3 nearest
   neighbors, plot actual member trade price paths (entry → MFE → exit)
2. **Analytics suite integration** — auto-generate per-run: top 10 pattern
   anatomy charts, depth-level aggregate shapes, peak timing histograms
3. **Permanent feature** — interactive: click a template in the strategy report,
   see its anatomy, member trades, adjacent clusters, exit quality overlay
**Data sources:** `pattern_library.pkl` centroids, `oracle_trade_log.csv` member
trades, `scaler` for feature interpretation, oracle MFE/MAE for peak timing
**Key questions:** Best chart type for 16D (radar vs parallel coords vs heatmap)?
How to show "expected peak" (MFE timing distribution from oracle)?

---

## 4. Volatility-Adaptive Trail Width
**Status:** IDEA — emerged from April analysis
**Attacks:** Leak #2 (exits too early) — specifically high-vol regime bleed
**Context:** April 2025 had 2x MFE/MAE vs average. Trail_stop was 49.8% of exits
at -$9 avg — many were shaken out of correct-direction trades by vol noise.
Static trail width gets whipsawed in high-vol, works fine in low-vol.
**Concept:** Scale trail width by rolling ATR or realized vol. Wider trails in
high-vol regimes (April), tighter in low-vol (July). Could also key off the
Fourier dominant wavelength amplitude.
**Integration point:** `wave_rider.py` → `update_trail()` trail_distance calc.
**Depends on:** Fourier gate provides amplitude; otherwise standalone ATR scaling.

---

## 5. Filter Funnel Diagnostics
**Status:** IDEA
**Attacks:** Leak #1 (skipped signals, $1.37M) — identify which gate blocks most
**Context:** 44.4% of ideal profit is from skipped signals. Need to know which
gate is the biggest bottleneck.
**What needs to happen:**
- Instrument forward pass to count bars failing each gate
- Report a "filter funnel" — resonance, kill zone, velocity, confidence, etc.
- Tune or relax the dominant filter
- Likely candidates: `min_resonance_score` range, kill zone wick conditions

---

## 6. Shape-First Pre-Grouping (cherry-pick from post-snowflake)
**Status:** IDEA
**Branch ref:** `main` commit `2f80cfa` has full implementation (too aggressive)
**Plan:** Cherry-pick only `_shape_label()` as a pre-grouping step before K-means.
Group by `d{depth}|{pattern_type}|{lagrange_zone}|{hurst_cat}` first,
then run existing z-variance recursive split within each group.
Keep snowflake LONG/SHORT split (post-snowflake removed it — bad).
Skip adj-R² stopping criterion (borderline overfitting with 16D/30 samples).
**Files:** `training/fractal_clustering.py`

---

## 7. Live Feedback Loop (data loading overhaul)
**Status:** IDEA
**Context:** Currently data comes from historical OHLCV parquet files. Oracle
peeks at future bars — that's training-only.
**What needs to change:**
- Switch data loading to simulated real-world conditions (paper trading or live feed)
- Oracle anchoring fields (`mean_mfe_ticks` etc.) updated from actual trade outcomes
- `update_from_trade(actual_mfe, actual_mae)` method on PatternTemplate — EMA
- Hook: `oracle_trade_records` in forward pass already captures per-trade MFE/MAE

---

## 8. Direction Signal Quality
**Status:** PARKED — deprioritized by data
**Context:** Was thought to be near coin-flip (46.6% wrong direction). But
opportunity gap shows wrong direction is only $21K / 0.7% of the leak.
The model picks sides fine — it just can't hold. Revisit only if exits
improve and direction becomes the bottleneck.

---

## 9. Sub-Minute Trade Hold Floor
**Status:** PARKED (pending depth isolation results from current run)
**Concept:** `MIN_HOLD_BARS = 4` (60s on 15s bars) to suppress trail_stop exits
within first minute. Hard SL/TP/belief_flip stay active.
**Depends on:** Depth isolation results + parent-based max hold (just implemented)
may solve this without a hold floor.

---

## 10. MIN_TRADE_DEPTH Gate
**Status:** PARKED (pending current run results)
**Concept:** Hard-code `MIN_TRADE_DEPTH = 3` — 1D/4H/1H are context-only,
never fire trades. Currently depth 2 (1H) is kept eligible; next run decides.
**Data needed:** Depth isolation from `--fresh --depth-iso` run.

---

## 11. Geometric Path Templates (True Wave Rider)
**Status:** IDEA — needs full spec, Jules-scale implementation
**Category:** Major architecture evolution
**Attacks:** Half-curve problem (snowflake sees only LONG or SHORT half of cycle),
direction accuracy (51% → near coin-flip), exit timing (no concept of "what comes next")
**Core insight:** Clusters are static snapshots (centroids). Templates should be
**trajectories** — full price cycles through geometric state space:
```
L3_ROCHE (approach) → L2_ROCHE (well) → REVERSAL → L2_ROCHE (other side) → L3_ESCAPE
```
Each template = a mapped path through Lagrange zones, not a point in 16D.
Workers track position along the path (like GPS) and know what comes next.
**Key changes:**
- Templates become sequences of states (ordered zone transitions, not centroids)
- Gate 1 matches against trajectory similarity (DTW or similar), not Euclidean distance
- Workers report "position on path" + "expected next zone" instead of just direction
- Nightmare protocol extends to full state space navigation (not just extremes)
- Entry: "we're at step 3 of a 7-step cycle" → Exit: "step 5 reversal expected"
- No LONG/SHORT split needed — full cycle is one template
**Depends on:** Current run results, spectral gates (Fourier provides cycle structure)
**Branch:** Will fork from `main` when ready to implement
**Previous discussion:** Concept evolved from nightmare protocol mapping + the
half-curve problem observed when snowflake split lost the full cycle view.

---

## 12. Neural Direction Model (trained on I-MR segments)
**Status:** IDEA — strong case, needs spec
**Category:** Direction prediction — replaces physics blend + logistic regression
**Attacks:** Leak #3 (wrong direction) — but also improves entry quality broadly
**Key insight:** I-MR regime segments ARE perfect trades. Each segment has a known
direction + magnitude + the full 192D feature context at its start. This gives
50k-200k labeled samples from ATLAS (not just ~392 actual trades).
**Architecture options:**
- **GBM baseline**: XGBoost/LightGBM on 192D → direction + magnitude (Analysis K got 70.6%)
- **MLP**: 192 → 128 → 64 → 2 (direction class + magnitude regression)
- **1D-CNN on TF grid**: treat 16F × 12TF as a 2D image — conv layers learn
  cross-TF patterns ("4h accel + 15m decel = reversal") that geometric mean misses
**Training data:** Every I-MR segment in ATLAS = one labeled sample. 10 months ×
12 TFs × avg segment length → 50k-200k samples. Enough for neural approaches.
**Integration:** Replaces `_phys_dir` in TBN workers AND/OR replaces the entire
`_determine_direction()` cascade. Serialize trained model, load at startup.
**Depends on:** I-MR segmentation pipeline (waveform analysis), ATLAS data coverage
**Related:** Analysis K (70.6% direction, 1h_hurst #1 feature), JULES_WAVEFORM_SEED_INTEGRATION Part 4

---

## Recently Completed
- `--fresh --depth-iso` combo wired (single command)
- Parent-based max hold (5 × immediate parent TF)
- tqdm progress bar + analytics step prints
- Exit quality reorder (worst→best)
- Semantic playbook names (`generate_semantic_name()`)
- Depth isolation (`--depth-iso`)
- Report reorganization (detail first, summaries last)
- Strategy report sort (PnL desc, Sharpe tiebreaker)
- Exit quality stats fix (exit_reason vs exit_signal_reason)
- Watchdog tuning (TICKS 2->8, MIN_BARS=5)
- Quarterly CSV sharding for large logs (>50k rows)
- Shareable `reports/` directory
- Unicode cp1252 crash fixes (6 files, 14 print-statement emojis)
- Detection funnel counters (bar-level blindness tracking in report)
- Trade visualizer (`tools/trade_visualizer.py`)
- Pattern detection map (`tools/pattern_map.py`)
- MIN_TRADE_DEPTH = 3 (depths 0-2 context-only)
- Archived run data to `runs/2026-02-22_pre-depth-gate/`
