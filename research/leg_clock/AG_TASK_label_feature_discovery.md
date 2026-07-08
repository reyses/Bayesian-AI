# AG TASK — Label feature discovery (entry-filter hunt)

**For: Antigravity.** Self-contained — you have no prior conversation context.
Read this fully before starting. MNQ futures research repo, CUDA/WSL available.

## Goal
We have labeled trading opportunities (hindsight-optimal MNQ trades). Find the
features — BEFORE entry (causal setup) and DURING the trade — that are COMMON to
the labels AND DISTINCT from non-opportunities. This is the search for an ENTRY
FILTER: "what does a real opportunity look like as it forms."

## The labels (the truth)
- `DATA/ai_cusp_picks/ai_picks_YYYY-MM-DD_multi.json`, 576 days.
- Each trade: `{entry_ts, exit_ts, direction (LONG/SHORT), side, entry_price,
  exit_price, pnl_dollars, mae_dollars (max adverse excursion $), is_marginal,
  original_timestamp}`. ts = unix seconds UTC. ~37 trades/day.
- **HINDSIGHT labels** (offline-optimal). This measures the label SIGNATURE and
  feature SEPARABILITY — NOT a tradeable causal system. Never claim a live edge
  from label-fit alone.

## Market data
- `DATA/ATLAS/{5s,1m,5m,15m,1h}/YYYY_MM_DD.parquet` — cols: timestamp(unix s),
  open, high, low, close, volume. 2024 (259 days) + 2025 (277 days).
- MNQ: tick 0.25, $0.50/tick. Train on 2024, test OOS on 2025.

## Tasks
1. **Characterize labels** — distributions (mode + median + tail, NOT just
   mean) of duration, extent(ticks), velocity(ticks/min), MAE(heat), pnl;
   direction balance; entry time-of-day (Central). Starter script already
   written: `research/leg_clock/tools/label_signature.py` (run/extend it).
2. **Causal PRE-ENTRY features** (only data ≤ entry_ts):
   - leg state: current leg dir/extent/velocity (zigzag on 1m — reuse
     `research/level_hold/tools/pivot_level_proximity.zigzag_pivots`).
   - volume as a RATE (contracts/sec) relative to time-of-day normal + short/
     long-window acceleration. Raw volume is NOT comparable across TFs.
   - efficiency ratio (net move / path length over a window) = oscillation(low)
     vs trend(high).
   - band/structure position (reuse `rolling_ols_bands` in
     `research/level_hold/tools/level_hold_study.py`; sigma-relative distances,
     NEVER fixed-tick).
   - candle shape (body/wick fractions).
3. **During-trade features**: realized velocity, extent, MAE, volume profile.
4. **THE test — distinctness, not just commonality**: build a matched NULL =
   random non-entry bars (same day + time-of-day distribution). Compare
   label-entries vs null on every pre-entry feature: mean diff + bootstrap p,
   AND the actionable `P(label | feature-bin)`. A feature is useful only if it
   SEPARATES, not just differs.
5. **Classifier**: logistic + small MLP on pre-entry features, day-disjoint
   (train 2024, test 2025). Report OOS AUC + a shuffle-label AUC floor.

## Honesty rules (mandatory — this repo has been burned by every one of these)
- Causal features only (≤ entry_ts). Any hindsight in a FEATURE invalidates it.
- OOS by YEAR. Report distributions / mode, not point means.
- Null-anchored everywhere: shuffle-label AUC floor; matched non-entry null for
  features; report the GAP vs null.
- **Signal-magnitude bar**: AUC−0.5 gap ≥ 0.10 = REAL; 0.05–0.10 = CONDITIONAL;
  < 0.05 = NOISE. State it explicitly per feature.
- Do NOT lump oscillation and trend — condition on regime (efficiency ratio) or
  the signal dilutes (learned 2026-07-08).
- Labels are hindsight → conclusions are about signature / separability, not a
  live edge.

## Already found — do NOT repeat
Weak/dead: micro-bounce (unconditional ~60%), levels (+1–2pp), touch-count
(dead), bar-to-bar slope persistence (none), pre-trend candle microstructure
(null), volume-buildup filter (+2pp non-monotonic, useless). Robust: leg length
= fat-tail momentum, OOS-stable — but HOLD-only (confirm-then-ride backtest
LOSES −$60..−200/day causally). The ENTRY filter is the unsolved problem — that
is what this task hunts.

## Deliverables
- `research/leg_clock/reports/AG_label_features.md`: the label signature; a
  ranked table of pre-entry features by null-anchored separation (with the
  signal-bar verdict per feature); the OOS classifier AUC vs shuffle null.
- Reusable tools under `research/leg_clock/tools/`.

## Run
`.venv_wsl/bin/python <script>` (WSL). Paths are repo-root-relative. Reports →
`research/leg_clock/reports/`.
