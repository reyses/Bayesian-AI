---
name: V2 Features × Price Descriptive EDA Stack
description: Seven-layer descriptive EDA tools characterizing v2 feature behavior across regimes & price phenomena. No fitting. State-fingerprint findings (4h chord cells 100% pure for FLAT_SMOOTH/CHOPPY).
type: project
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
# V2 Features × Price — Descriptive EDA Stack

**Established:** 2026-05-03.
**Status:** 7 layers built. All run on IS only (208 days, 47k 5m bars). No fitting; pure characterization. Substrate for future regime-conditional strategy work.

## The 7 layers

| Layer | Tool | Question answered |
|---|---|---|
| A1 single, current | `tools/v2_features_regime_eda.py` | Per-feature × per-regime distributions; which features separate which regimes (Cohen-d) |
| A2 pair, current | `tools/v2_features_pairwise_eda.py` | Which pairs interact non-additively; which (X_q, Y_q) cells have extreme price reactions |
| B1 single, lookback | `tools/v2_features_lookback_eda.py` | When feature X does pattern P over N bars (mono/spike/reversal), how does price react |
| B2 pair, lookback | `tools/v2_features_lookback_pair_eda.py` | Joint patterns: when two features both spike up, both fall mono, etc., how does price react |
| Chord (triplet) | `tools/v2_features_chord_eda.py` | Music metaphor: 3-feature combinations whose joint quantile cells encode recognizable state fingerprints |
| Visual overlay | `tools/v2_features_overlay_viz.py` | Per-regime per-day PNGs: price + 12 features + chord stacked for eye-pattern recognition |
| Volume × variation | `tools/v2_features_volume_variation_eda.py` | 4 quadrants of (vol, var) — fakeout (LOW_VOL_HIGH_VAR), compression (HIGH_VOL_LOW_VAR), capitulation, dead zone |

## Headline findings

### Single-feature relationships
- `L2_5m_price_velocity_w` r = +0.86 with past N-bar return — it IS the recent move
- `L2_1h_price_velocity_w` separates DOWN_SMOOTH from UP_SMOOTH at Cohen-d = −1.25 (strongest single regime separator)
- Forward correlations all <0.1 — confirms what Analyses B/M/N showed: bar-to-bar direction is unpredictable from any single feature

### Pairwise current
- `L1_5m_velocity_1b × L1_5m_body` → WR ranges 42-75% across 9 cells via pure quantile binning. Recovers 70% direction accuracy from 2 features, no model fit.
- `L1_15m_velocity_1b × L1_15m_body` → 35pp WR spread (32-67%)
- Pattern: velocity × body at same TF captures "directional bar agreement" — both push same way → continuation; disagree → reversion.

### Lookback single
- `L1_5m_velocity_1b / 6 / RISING_MONO` → 69.6% WR (n=46), +38.1 tick fwd
- `L1_5m_body / 6 / RISING_MONO` → 57.9% WR (n=38), +41.1 tick fwd
- `L1_15m_body / 3 / SPIKE_DOWN` → 59.9% WR, +5 ticks (mean reversion)
- 6-bar monotonic momentum on 5m bar-shape = strong continuation; single-bar 15m spikes = mean revert.

### Lookback pair
- `(L2_1m_velocity_w, L1_5m_velocity_1b) / 6 / REVERSAL_AGREE_DOWN` → 71.0% WR (n=31), +21 tick fwd. Coordinated bearish moves bounce.
- `SPIKE_BOTH_UP at 1h+5m` → +13.7 tick continuation 56% WR (n=200+)

### Chord (3-feature regime fingerprint)
- **`L1_4h_body + L1_4h_velocity_1b + L3_4h_z_low_w`** has cells:
  - cell(1,2,0) = **100% FLAT_SMOOTH** (n=96, fwd −7.5)
  - cell(2,1,1) = **100% FLAT_CHOPPY** (n=96, fwd −22.3)
  - cell(1,0,0) = 88% FLAT_SMOOTH (n=96, fwd +10.8, WR 61%)
- 4h-TF features dominate purity rankings — 4h is the regime-identifying timescale
- The chord IS the regime classifier; deterministic, no fitting needed

### Volume × variation
- LOW_VOL × HIGH_VAR (fakeout territory): 15m_vol_mean × 15m_price_sigma cell n=486, mean_fwd **+28.7 ticks** at 53% WR. Low participation + dispersion → bounce.
- HIGH_VOL × LOW_VAR (compression): 4h_vol_mean × 4h_swing_noise cell n=636, **70% FLAT_CHOPPY purity**. Highest single-cell concentration in the entire stack.
- All 4 corners FLAT_CHOPPY-dominated (regime distribution bias). Differentiation is in forward return: fakeout averages +3.1 fwd, dead zone averages 0.

## Methodology pattern (consistent across layers)

1. **Pruning via shortlist**: Layer 1 produces top-K shortlist (Cohen-d ranking for fingerprint hunting; lookback_corr for momentum). Layers 2/B1/B2/chord use that shortlist.
2. **Quantile binning** preserves rank information without parametric assumptions; 3 quantiles per feature (sometimes 5).
3. **Min cell support** (50 bars typical) filters noise from rare-event combinations.
4. **All layers IS-only** (208 days, 47k bars). Future OOS validation = retest the high-WR/high-purity cells on the 71 OOS days.
5. **Visual overlay tool** complements the math — eye sees co-activation patterns the statistics summarize.

## What this UNLOCKS for next session

Three follow-up directions, in order of value:

1. **OOS validation of EDA findings**. The 70% WR patterns and 100% chord cells need to hold on the OOS 71 days to be tradable. Re-run B1/B2/chord/vol-var with `--split OOS`. Patterns that survive = real; patterns that disappear = sample-specific noise.

2. **State-fingerprint deployment**. Translate the top chord cells into a NinjaTrader rule. Example: when `4h_body in Q1, 4h_velocity_1b in Q2, 4h_z_low_w in Q0` → tag as FLAT_SMOOTH state. Route to FLAT_SMOOTH-appropriate strategy (zigzag bleed-score filter from prior chop-edge work).

3. **Regime-stratified rerun**. Every layer has or can gain a `--by-regime` flag. The 4h chord found cells 100% pure for FLAT_*; what about UP_SMOOTH or DOWN_CHOPPY? Need stratified analysis to find chord cells specific to each regime.

## Anti-patterns ruled out

- 4-feature chords: combinatorial cost too high, sample sizes drop below useful thresholds (n<10).
- Multi-feature lookback (3+ features × 4 windows × 8 patterns each): same problem — would characterize rare events with too few samples for inference.
- Volume features in models without the framing: they don't track price velocity directly. They describe activity intensity. Used as quadrant-binners (LOW_VOL/HIGH_VOL × LOW_VAR/HIGH_VAR), they DO describe state — but as regression inputs to predict price they're useless.
- "Mixing colors" beyond layer 3: each additional dimension narrows the population to noise. Layer 3 (chord) is the natural depth limit.

## Files

- 7 tools under `tools/v2_features_*.py`
- 7 output dirs under `reports/findings/v2_features_*` and `reports/findings/v2_volume_variation/`
- See `docs/daily/2026-05-03.md` for the full session writeup
- Commits: `0a0229aa`, `3645f1ec`, `aa205bd2`, `bccea3c1`, `ba09337c` + (today's vol×var commit)
