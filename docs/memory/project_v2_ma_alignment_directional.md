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

## Trend-direction regression (2026-05-02)

`tools/v2_regress_trend_direction.py` — predicts day net_move from 5m bar features. Day-level 60/20/20 split via `regime_labels_2d.csv`.

| Metric | Value |
|---|---:|
| Test R² | 0.032 |
| Bar-level direction acc | 63.2% (+11.9% lift) |
| Day-level direction acc | **74.6%** (+22.5% lift, 71 days) ← new high water mark |

**Per-regime (validates regime-conditional hypothesis):**

| Regime | Bar acc |
|---|---:|
| UP_SMOOTH | **87.6%** |
| UP_CHOPPY | 74.5% |
| DOWN_SMOOTH | 72.7% |
| FLAT_CHOPPY | 58.1% |
| DOWN_CHOPPY | **43.9%** ← model misses these |
| FLAT_SMOOTH | **42.3%** ← noise floor |

The model implicitly classifies "is this a trending day?" — succeeds 73-88% on trend cells, fails 42-58% on chop/flat. Magnitude is conservatively biased (mean_pred ~ 0.5 × mean_actual on trend cells). This is the empirical proof for regime-conditional strategy selection.

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

### Parallel track 1.5 — 6-class regime classifier (NEW, from net_move result)
The trend-direction regression at 74.6% day-level acc shows the model IS implicitly classifying regimes (87.6% UP_SMOOTH, 42% FLAT_SMOOTH). Make this explicit:
- Train a 6-class classifier on `regime_2d` from bar features
- Compare confusion matrix vs the regression's per-regime accuracy
- Use as the regime gate in the joint router

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

## Connection to prior regime work (2026-05-01 update)

**MA alignment IS a regime classifier.** When `alignment_score >= 7`, the day is by definition in a strong-trend regime (UP_SMOOTH or DOWN_SMOOTH per the 2D taxonomy). When alignment is mixed, you're in chop / transitional.

This connects to two prior threads:
- [feedback_chop_edge_regime_filter.md](feedback_chop_edge_regime_filter.md) — the zigzag counter-trend strategy WINS on chop (+$89/day) and LOSES on trend (-$95/day). Two-feature classifier (`prior_range`, `range_compression`) discriminates with d_OOS=0.77/0.78. Filter rule turns -$552 into +$5,000-6,000.
- `tools/atlas_regime_labeler.py` (2026-04-29) — labels all 348 ATLAS days as UP/DOWN/CHOP/QUIET/TRANSITIONAL. Output: `DATA/ATLAS/regime_labels.csv`.

**The right joint system is regime-conditional strategy selection:**
- HIGH alignment → trend-following (today's MA-align direction)
- LOW alignment + chop conditions → zigzag counter-trend (prior bleed_score filter)
- TRANSITIONAL / mixed → skip

## 2D label system (2026-05-01)

Built `tools/atlas_regime_labeler_2d.py` — extends the existing daily labels with:
- `direction_axis` ∈ {UP, DOWN, FLAT}
- `variation_axis` ∈ {SMOOTH, CHOPPY}
- `regime_2d` = combined (e.g., "UP_SMOOTH")
- `split` ∈ {IS, VAL, OOS} (60/20/20 by date)

Output: `DATA/ATLAS/regime_labels_2d.csv` (348 days). Distribution:

| Regime | Total | IS | VAL | OOS |
|---|---:|---:|---:|---:|
| UP_SMOOTH | 63 | 34 | 17 | 12 |
| UP_CHOPPY | 23 | 15 | 4 | 4 |
| DOWN_SMOOTH | 44 | 25 | 9 | 10 |
| DOWN_CHOPPY | 19 | 12 | 2 | 5 |
| FLAT_SMOOTH | 71 | 51 | 13 | 7 |
| FLAT_CHOPPY | 128 | 71 | 24 | 33 |

UP_CHOPPY and DOWN_CHOPPY have thin OOS samples (4-5 days) — caveat for stat-sig analysis on those cells.

Loader:
```python
from tools.atlas_regime_labeler_2d import load_regime_labels
df = load_regime_labels()
oos_up_smooth = df[(df.split == 'OOS') & (df.regime_2d == 'UP_SMOOTH')]
```

Use this as the substrate for ALL future regime-conditional analysis — MA alignment, L-model, zigzag, any composite all evaluated through the same lens.

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
