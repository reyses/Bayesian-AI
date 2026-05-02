---
name: V2 MA-Alignment Directional System
description: Active research direction — multi-TF MA alignment as a directional filter. Beats fitted composites on lift; deterministic rule, no overfit risk. Pending: tradeable exit rules + joint with L-magnitude.
type: project
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
# V2 MA-Alignment Directional System

**Established:** 2026-05-01.
**Status:** Best directional signal found across 5 composite framings. Operational rule defined. Next session = parallel/joint work.

## The rule

For each of 8 TFs (5s, 15s, 1m, 5m, 15m, 1h, 4h, 1D) at every 5m decision bar:

```
vote_TF = +1 if close_5m > L2_<TF>_vwap_w
         -1 if close_5m < L2_<TF>_vwap_w
          0 if within tolerance (default 0.5 ticks)
```

Sum the votes → `alignment_score` ∈ [-8, +8]. Decision:
- LONG  if `alignment_score >= +7`
- SHORT if `alignment_score <= -7`
- FLAT  otherwise

**Test-set performance** (full year, last 20% as test, 14,827 non-flat 5m bars):
- 70.5% direction accuracy on 20% of bars (+17.6% lift over 52.9% baseline)
- Combined VWAP+PriceMean `|score|>=15`: 70.6% on 19% (+17.8%)

**Why:** ship the deterministic rule. No model file, no train/test risk, walk-forward stable by construction.

## Comparison with the alternatives ruled out

| Approach | Acc | Coverage | Lift |
|---|---:|---:|---:|
| **MA align combined ≥15** | **70.6%** | **19%** | **+17.8%** |
| **MA align vwap ≥7** | **70.5%** | **20%** | **+17.6%** |
| Standalone L \|pred\|>20 | 70.4% | 45% | +10.6% |
| Single-horizon 5-voter \|w\|>5 | 62.6% | 26% | +9.8% |
| 5-voter L-aggregator \|w\|>5 | 59.8% | 13% | +9.6% |
| 2-voter (1m+5m) \|w\|>5 | 58.8% | 1.9% | +8.6% |
| Quantile composite (Q_0.25>0/Q_0.75<0) | n/a | 0% | collapsed |

MA alignment matches L on accuracy at half the coverage **but with 67% more lift over baseline** (+17.6 vs +10.6). The lift gap is the real headline — L's high accuracy is partly inflated baseline (59.8%); MA achieves 70% on a 52.9% balanced baseline.

## Why-this-works (operationalization of the user's distributional intuition)

User framing earlier: "treat price as a probability field — by the time we measure it, the electron is somewhere else". Operational read:
- Bar-to-bar direction (Analysis B) → R²=0.0003. Coin flip.
- Conditional mean of signed MFE (Analysis L) → R²=0.06-0.11, gates to 70-73%. Distributional.
- **MA alignment**: discrete approximation of "is the conditional distribution unambiguously on one side of zero". Each TF's vwap_w is a smoothed-price reference; price-vs-vwap is a sign of recent trend at that timescale. When 7+ of 8 TFs agree, the conditional distribution is concentrated on one side.

The autoregressive features (vwap_w, price_mean_w) carry signal as **direct comparators** (price - vwap_w) but NOT as regression inputs (where they trivially correlate with current price level and don't help direction). The user's counter-proposal correctly identified this distinction.

## TFs that carry the signal

Single-feature accuracy (just one TF's vwap comparison):

| TF | VWAP solo | Lift | Smoothing window |
|---|---:|---:|---|
| 5s | 53.5% | +0.6% | 9 bars (45s) |
| 15s | 53.7% | +0.8% | 12 bars (3 min) |
| 1m | 55.8% | +2.9% | 15 bars (15 min) |
| 5m | 59.0% | +6.2% | 9 bars (45 min) |
| **15m** | **61.7%** | **+8.9%** | **12 bars (3 hours)** |
| **1h** | **61.6%** | **+8.7%** | **12 bars (12 hours)** |
| 4h | 55.2% | +2.3% | 18 bars (3 days) |
| 1D | 54.2% | +1.4% | 5 bars (5 days) |

**The 3-hour-to-12-hour smoothing window is where the signal lives.** Shorter (5s-1m) is too noisy; longer (4h-1D) is too coarse for a 5m decision cadence.

## Where the data lives

- Tool: `tools/v2_composite_ma_alignment.py`
- Outputs: `reports/findings/v2_composite_ma_alignment/`
  - `per_tf_signals.csv` — per-TF accuracy for vwap and mean comparators
  - `vwap_alignment.csv` — score thresholds 1-8 for vwap-only
  - `mean_alignment.csv` — score thresholds 1-8 for price_mean-only
  - `combined_alignment.csv` — score thresholds 1-16 for vwap+mean combined
  - `summary.md` — narrative
- Commit: `7dae2585`
- Source feature schema: `DATA/ATLAS/FEATURES_5s_v2/L2_<TF>/` (vwap_w + price_mean_w columns)

## Next session — parallel and then joint

### Parallel track 1 — MA alignment exit rules
The 70.5% direction accuracy is on the entry signal only. Open questions:
- With this filter, what's the average MFE/MAE of qualifying trades?
- What stop and target sizing makes the alignment-filtered signal tradeable?
- Does the alignment HOLD across the trade window, or does it flip mid-trade?
- If alignment flips → exit signal? Or does flipping during a winning trade mean "let it run"?

Tools to build/extend:
- Modify `v2_composite_ma_alignment.py` to also report MFE/MAE distribution per signal (alignment_score → outcome stats)
- Test alignment-flip-as-exit vs fixed-stop exits

### Parallel track 2 — L-model refinement
Even with MA alignment dominating on lift, the L approach has higher coverage (45% vs 20%). Options to push L further:
- Drop the autoregressive features (vwap_w, price_mean_w) from the regression inputs entirely. They're trivially correlated with price; removing forces the model onto genuinely predictive features (z_se, hurst, swing_noise, vol_accel_w, etc.).
- Feature selection on the 185D v2 schema (Lasso, mutual info, etc.).
- Replace OLS with a richer base learner (GBM, MDN) — but watch for overfit at low sample counts.

### Joint (after both parallel tracks)
Combine MA alignment FILTER (when can we trade?) with a magnitude estimator (how big is the expected move?). Sketch:
- MA alignment says LONG → only consider long trades
- L (or refined model) gives expected MFE magnitude → use for position sizing or skip if magnitude too small
- Pair filter strictness with magnitude threshold for tradeable subset

This converges the two best signals into one system. The MA filter handles direction with high confidence; the magnitude estimator sets risk/sizing.

## Anti-patterns ruled out (do NOT revisit unless new info)

- **Naive cross-TF L voting at common cadence** → fails because horizons don't align (1m predicts 1h ahead, 1h predicts 8h ahead).
- **Apples-to-apples single-horizon refit with handicapped voters (each voter sees only its TF's 23 features)** → still loses to standalone L using all 185 features. Voting handicapped models can't recover what one full model exploits.
- **Strict quantile composite (Q_0.25>0 / Q_0.75<0)** → too stringent under quick-mode GBM (50 trees), collapses to 100% FLAT. Would need either 200+ trees or a softer rule.
- **Stripping autoregressive features from L's inputs** → was Claude's premature suggestion. User's better counter-proposal (use them as alignment signals, not regression inputs) was correct. The features ARE useful when used the right way.

## Single-rule one-liner (for NT8)

```
LONG  if (Close > VWAP_15s_w AND Close > VWAP_1m_w AND ... AND Close > VWAP_1D_w with at least 7 of 8 true)
SHORT if (Close < VWAP_<TF>_w with at least 7 of 8 true)
FLAT  otherwise
```

Approx 20 lines in NT8. Each TF's vwap_w is computed in `core_v2.StatisticalFieldEngine.compute_L2(df, tf)` — the formula is a rolling N-period VWAP per `core_v2.N_BASE` (5s=9, 15s=12, 1m=15, 5m=9, 15m=12, 1h=12, 4h=18, 1D=5).
