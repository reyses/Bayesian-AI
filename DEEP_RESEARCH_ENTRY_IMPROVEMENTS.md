# Deep Research: Entry System Improvements — BayesianBridge

## Current Architecture Summary

The entry system is a **2-phase gate cascade** in `core/execution_engine.py`:

**Phase 1** — Per-candidate screening (all candidates):
- Gate 0: Pattern quality (z-score, pattern_type × regime, physics rules)
- Gate 0.5: Depth filter (min depth, blacklist)
- Gate 1: Template match (centroid distance < gate1_dist)
- Gate 2: Brain profitability (Bayesian should_fire)
- Score competition: best scorer wins

**Phase 2** — Direction + conviction (winner only):
- Direction cascade: PP override → live_momentum → signed_mfe → logistic → brain_dir → template_bias → band_confluence → DMI → velocity
- Gate 3: Belief conviction (TBN is_confident)
- Gate 4: Momentum alignment (F_momentum sign vs trade direction)
- Exit sizing: SL/TP/trail from template stats + OLS + brain offset

**Fallback**: Worker bypass (high conviction, no template match)

---

## IDENTIFIED IMPROVEMENT AREAS

### A. Regime-Aware Gate Architecture

**Problem:** Gate 0 applies universal physics rules regardless of market regime. MOMENTUM_BREAK is blocked when ADX < 25, but BAND_REVERSAL is blocked unconditionally at z < 2.0 (line 657-659). This is exactly backwards — BAND_REVERSAL is the pattern that *should* fire in low-ADX ranging markets.

**Research finding:** Industry consensus classifies markets into trending (ADX > 25, rising slope), ranging (ADX < 20), and transitional (20-25) regimes. The strategy that works in one regime fails in another. Specifically:
- Trend-following patterns (MOMENTUM_BREAK) need ADX > 25 with rising ADX slope
- Mean-reversion patterns (BAND_REVERSAL) need ADX < 20-25 — they *thrive* in ranges
- During regime transitions (ADX 20-25 with rising slope), both can work but with reduced sizing

**Proposed change:** Replace the hardcoded Gate 0 pattern-regime rules with a regime classification function that returns `strong_trend` / `developing` / `exhausting` / `range` / `chop`, then apply a compatibility matrix:

| Regime | MOMENTUM_BREAK | BAND_REVERSAL | TREND_CONTINUATION |
|--------|:---:|:---:|:---:|
| strong_trend | ✅ | ❌ | ✅ |
| developing | ✅ | ⚠️ (reduce size) | ✅ |
| exhausting | ❌ | ✅ | ❌ |
| range | ❌ | ✅ | ❌ |
| chop | ❌ | ❌ | ❌ |

**ADX slope** is the key missing input — a falling ADX at 35 (exhausting) requires different treatment than a rising ADX at 22 (developing). This requires `adx_prev` on MarketState.

### B. Volatility-Normalized Entry Sizing

**Problem:** Current SL/TP sizing (`_compute_sizing`) uses template-trained statistics (p25_mae, mean_mae, p75_mfe, regression_sigma) which are frozen from training data. Live market volatility may be 2-3x different from training conditions. The ATR floor enforcement (line 976-979) only applies in live/replay mode and uses fixed 3x/5x multipliers.

**Research finding:** ATR-based position sizing is the industry standard for volatility adaptation. The core principle is: SL should be proportional to current ATR, not historical template averages. Specifically:
- SL = max(template_SL, ATR × multiplier) where multiplier scales with regime (1.5-2x in trends, 2.5-3x in ranges)
- TP = max(template_TP, ATR × multiplier) where multiplier is 2-5x depending on regime
- Position size = fixed_risk / (SL_ticks × tick_value)

**Proposed change:** Unify OOS and live sizing to use ATR floors in all modes. Compute ATR from the 15s bar data (already available as `swing_noise_ticks` on MarketState which approximates intrabar ATR). Scale multipliers by regime:

```
if regime == 'strong_trend':
    sl_mult = 1.5    # tight in trends (trust the move)
    tp_mult = 5.0    # let winners run
elif regime == 'developing':
    sl_mult = 2.0
    tp_mult = 3.0
elif regime == 'range':
    sl_mult = 2.5    # wider in ranges (more noise)
    tp_mult = 2.0    # shorter targets (mean reversion)
```

### C. Multi-Timeframe Confluence Gate

**Problem:** The TBN has 10 timeframe workers (1s through 4h) but their consensus is only partially used — the conviction gate (Gate 3) checks `is_confident` which is a composite, and the belief direction can override the cascade winner. There's no explicit "how many timeframes agree with this trade?" check.

**Research finding:** Multi-timeframe analysis is universally cited as the highest-impact improvement for entry quality. The principle is simple: trades where multiple timeframes agree on direction have dramatically higher win rates than trades where only the anchor timeframe shows a signal. The recommended pattern is:
1. Higher TF defines trend direction (bias)
2. Medium TF defines trade setup (pattern)
3. Lower TF defines entry timing (trigger)

**Proposed change:** Add a TF confluence score to the gate cascade, between Gate 2 (brain) and Gate 3 (conviction):

```python
# Gate 2.5: Multi-TF Confluence
_align = self.belief_network.get_dmi_alignment()
_tf_agree = _align['aligned_tfs'] / max(1, _align['total_tfs'])
if _tf_agree < 0.4:
    # Majority of timeframes disagree — high risk of whipsaw
    return fail('gate2_5_tf_disagree')
```

Additionally, surface the TBN workers' individual directional bias (z-score sign, DMI direction) as a pre-filter for the direction cascade. If 7/10 workers point short and the signed_mfe regression says long, the regression is probably wrong.

### D. Entry Timing Refinement (Pullback vs Chase)

**Problem:** The current system enters at the close of the 15s bar where the signal fires. There's no concept of "wait for a pullback" or "confirm the breakout." Many entries chase momentum and get immediately stopped out by the first retracement.

**Research finding:** Professional futures traders use two timing approaches:
1. **Pullback entries in trends**: Wait for price to retrace to VWAP or a short-term MA before entering in the trend direction. This improves R:R significantly.
2. **Breakout confirmation**: Wait for a follow-through bar (next bar closes in the signal direction) before committing. This filters false breakouts at the cost of slightly worse entry price.

**Proposed change — two concrete mechanisms:**

**D1. Confirmation bar delay:**
Add an optional 1-bar confirmation delay to BarProcessor. When a signal fires, don't enter immediately — save the signal state and only execute if the next bar's close confirms the direction (higher close for long, lower for short). This can be toggled by regime:
- `strong_trend`: no delay (momentum carries, enter immediately)
- `developing` / `range`: 1-bar confirmation (wait for follow-through)

**D2. Limit entry mode:**
Instead of entering at market on signal bar, queue a limit order at entry_price ± ATR×0.5 (better price). If filled within N bars, trade executes. If not, signal expires. This naturally creates pullback entries. Complexity: requires BarProcessor to support pending orders. May be better suited for live engine only.

### E. Volume/Liquidity Confirmation

**Problem:** The system has no volume awareness. All signals are treated equally regardless of whether the signal bar had 10 contracts or 10,000 traded. Low-volume signals in MNQ are much more likely to be noise.

**Research finding:** Volume confirmation is cited across all sources as one of the most reliable signal filters:
- Breakouts with above-average volume are far more likely to sustain
- Mean-reversion signals at high-volume nodes (VWAP, POC) have higher fill probability
- Low-volume signals during overnight/pre-market sessions produce more false positives

**Proposed change:** Add relative volume (RVOL) to MarketState. RVOL = current bar volume / average volume for this time-of-day over past N days. Gate entry on RVOL > threshold:

```python
# Gate 0 addition: volume confirmation
rvol = getattr(state, 'relative_volume', 1.0)
if rvol < 0.5:
    should_skip = True
    skip_label = 'gate0_low_volume'
```

**Data requirement:** Need volume data in the 15s parquet files. If MNQ volume is available from Databento, compute RVOL in `statistical_field_engine.py`.

### F. Time-of-Day Filter

**Problem:** The system trades uniformly across the entire session. But MNQ has well-known intraday patterns: high-volume moves at 9:30 ET open, 10:00 ET news, 14:00 ET bond close, 15:00 ET MOC. Overnight sessions (18:00-09:30 ET) have lower volume and wider spreads.

**Research finding:** Session-aware filtering is standard in professional futures trading. Entries during the first 30 minutes of cash open and during major economic releases have different statistical profiles than mid-day entries. Some prop firms explicitly restrict trading to cash session hours only.

**Proposed change:** Add session context to Gate 0:

```python
# Compute time-of-day bucket from timestamp
hour_et = _get_et_hour(timestamp)
if hour_et < 9.5 or hour_et > 16:  # overnight
    # Require higher z-score and tighter distance
    if micro_z < 1.5 or dist > gate1_dist * 0.7:
        should_skip = True
        skip_label = 'gate0_session'
```

This is cheap to implement and can be DOE'd across different session windows.

### G. Confidence-Weighted Direction Cascade

**Problem:** The direction cascade (P-1 through P5) returns the first source that fires, with a flat probability estimate. But the reliability of each source varies dramatically — signed_mfe regression (P0.5) trained on 15+ IS samples is far more reliable than velocity fallback (P5). Currently all sources flow to the same entry with the same sizing.

**Research finding:** Multi-source signal confirmation is universally recommended. Rather than a priority waterfall where the first match wins, a scoring/voting approach where multiple agreeing sources produce higher confidence is more robust. Disagreement between sources should reduce position size or skip.

**Proposed change:** Score the direction cascade as votes rather than waterfall:

```python
# Collect all direction opinions with weights
votes = []  # (side, confidence, weight, source)
# P0.5: signed_mfe (high weight — trained)
if smfe_pred: votes.append(('long' if pred > 0 else 'short', abs(pred)*0.1, 3.0, 'smfe'))
# P1: logistic (high weight — trained)
if logistic_prob: votes.append((..., 2.5, 'logistic'))
# P2: template_bias (medium weight)
if template_bias: votes.append((..., 1.5, 'template'))
# P3: band_confluence (medium weight)
# P4: DMI (low weight)
# P5: velocity (lowest weight)

# Weighted vote
long_score = sum(w * c for s, c, w, _ in votes if s == 'long')
short_score = sum(w * c for s, c, w, _ in votes if s == 'short')

# Minimum vote threshold to enter
if max(long_score, short_score) < min_vote_threshold:
    return None, 0.5, 'insufficient_votes'

side = 'long' if long_score > short_score else 'short'
p_long = long_score / (long_score + short_score)
```

This naturally handles disagreement — if signed_mfe says long but DMI, template_bias, and velocity say short, the short case should win despite the trained model disagreeing. Currently the trained model always wins by priority.

### H. Stale Template Detection

**Problem:** Template centroids are frozen from IS training. If market regime shifts significantly, the cluster assignments may become stale — a signal matching cluster #47 today may represent a fundamentally different market state than the IS data that defined cluster #47.

**Research finding:** Overfitting to historical patterns is the #1 failure mode of systematic strategies. Regime changes cause strategies trained on one market condition to fail in another. Adaptive approaches include: rolling recalibration windows, online learning of cluster assignments, or monitoring cluster utilization rates.

**Proposed change — monitoring first, intervention later:**

Add a staleness score per template: track how many OOS trades each template has taken in the last N days. Templates with 0 recent trades may indicate market regime has shifted away from that pattern. Templates with a recent losing streak should trigger brain re-evaluation.

```python
# In brain.should_fire():
if recent_trades < 2 and lib_entry.get('last_trade_days_ago', 0) > 30:
    # Template hasn't fired in a month — it may be stale
    # Require higher conviction to override
    min_prob *= 1.5
```

---

## IMPLEMENTATION PRIORITY

| # | Improvement | Impact | Effort | Risk |
|---|------------|--------|--------|------|
| 1 | **A. Regime classification** | HIGH | Medium | Low — additive, doesn't break existing gates |
| 2 | **C. Multi-TF confluence gate** | HIGH | Low | Low — TBN workers already exist |
| 3 | **B. Volatility-normalized sizing** | HIGH | Medium | Medium — changes SL/TP for all trades |
| 4 | **G. Confidence-weighted direction** | HIGH | Medium | Medium — replaces waterfall with voting |
| 5 | **F. Time-of-day filter** | MEDIUM | Low | Low — simple timestamp check |
| 6 | **D1. Confirmation bar delay** | MEDIUM | Medium | Low — optional, toggled by regime |
| 7 | **E. Volume confirmation** | MEDIUM | Medium | Depends on data availability |
| 8 | **H. Stale template detection** | LOW-MED | Low | Low — monitoring only |
| 9 | **D2. Limit entry mode** | LOW-MED | High | High — requires pending order infra |

**Recommended first batch:** A + C + F (regime gate + TF confluence + session filter). These three together should reduce false signals by filtering out entries in unfavorable regimes, misaligned timeframes, and low-quality sessions — without changing the core matching or sizing logic.

**Second batch:** B + G (volatility sizing + direction voting). These change how trades are sized and directed, requiring more careful backtesting.

---

## KEY DATA REQUIREMENTS

| Field | Source | Used By |
|-------|--------|---------|
| `adx_prev` | statistical_field_engine (shift adx_arr by 1) | Regime classification (ADX slope) |
| `di_plus_prev`, `di_minus_prev` | Same | DI crossover detection |
| `relative_volume` | bar volume / rolling avg by TOD | Volume gate |
| `session_bucket` | timestamp → ET hour | Session filter |

All except `relative_volume` are trivial additions to the existing compute pipeline.
