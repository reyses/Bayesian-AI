# Cause & Effect Matrix (C&E / X-Y Matrix)

## Purpose
Map every measurable parameter (X) to every response variable (Y) with a
theoretical importance rating BEFORE running interaction plots. This prevents
"data dredging" — we state our hypothesis first, then validate with data.

---

## Response Variables (Y's)

| ID | Response (Y) | Definition | Units | Source |
|----|-------------|------------|-------|--------|
| Y1 | **Win Rate** | Trades closed at profit / total trades | % | oracle_trade_log |
| Y2 | **PnL per trade** | Average dollar P&L per trade | $/trade | oracle_trade_log |
| Y3 | **Capture %** | Actual PnL / Oracle MFE (how much of the move we caught) | % | oracle_trade_log |
| Y4 | **Reversal rate** | % of correct-direction trades where market flipped after entry | % | is_report "Reversed" bucket |
| Y5 | **Direction accuracy** | % of trades where entry side matched oracle direction | % | oracle_trade_log |
| Y6 | **Hold efficiency** | PnL per bar held (reward per unit time in trade) | $/bar | oracle_trade_log |
| Y7 | **Hold time** | Average bars held per trade. Shorter = less exposure, faster capital turnover | bars | oracle_trade_log |
| Y8 | **Trade decay** | Rate at which P(profitable) decays over hold time. Fast decay = get out quick. Slow decay = ride it. Half-life of the trade. | bars to 50% | oracle_trade_log |
| Y9 | **Trade count** | Total trades taken (too few = missed opportunity, too many = overtrading) | count | signal_log |
| Y10 | **Golden path capture** | Actual trade PnL / chord length of 1s ticks during the trade. Chord length = `sum(|price[i] - price[i-1]|)` at 1-second resolution = the theoretical MAXIMUM extractable movement. 100% = captured every tick perfectly. Measures how efficiently we harvested available price movement. | % | 1s price data + oracle_trade_log |
| Y11 | **Oracle path efficiency** | Oracle-segmented optimal capture / risk taken. The oracle segments the 1s chord into directional runs, computes optimal entry/exit per segment (minimizing adverse excursion per segment while maximizing captured ticks). `oracle_captured_ticks / oracle_max_MAE_per_segment`. This is the BENCHMARK — the theoretical best risk-adjusted extraction. Our system's Y10/Y3 compared to Y11 tells us how far we are from optimal. | ratio | oracle segmentation of 1s data |
| Y12 | **Risk-adjusted path ratio** | `actual_PnL / max_adverse_excursion` per trade. How many dollars we made per dollar we risked (peak drawdown during the trade). Y10 tells us capture efficiency, Y12 tells us RISK efficiency. A trade that captures 50% of the chord but risked only 2 ticks is better than one that captures 80% but risked 20 ticks. | ratio | oracle_trade_log |

### The Golden Path Hierarchy

```
Level 0: Chord length     = sum(|Δp|) at 1s        ← theoretical max (infinite risk)
Level 1: Oracle segments  = optimal entry/exit       ← best possible risk-adjusted (Y11)
Level 2: Our system       = actual trades            ← what we achieved (Y10, Y12)

Gap: Level 0 → Level 1  = cost of risk management (unavoidable)
Gap: Level 1 → Level 2  = cost of imperfect signals (what we optimize)
```

The oracle (Level 1) answers: "Given the 1s price path, what are the optimal
trade segments that maximize total capture while keeping per-segment MAE below
a threshold?" This is a segmentation problem:

1. Walk the 1s price series
2. Identify directional runs (consecutive ticks in same direction, allowing
   small adverse excursions below a tolerance)
3. For each run: entry = start, exit = end, captured = |Δp|, risk = max MAE within
4. Filter: only keep segments where captured > cost (spread + commission)
5. Sum captured ticks across all valid segments = oracle's optimal extraction
6. Sum max MAE across segments = oracle's total risk

`Y11 = sum(oracle_captured) / sum(oracle_MAE)` = the efficient frontier.
Our job is to approach Y11 with real-time signals.

---

## Input Parameters (X's)

### Category A: Physics / Quantum State (per-bar, continuous)

| ID | Parameter (X) | Field name | What it measures (theory) | Range |
|----|--------------|------------|--------------------------|-------|
| X1 | **Z-score** | `z_score` | Price deviation from regression center in sigma units. High |z| = price stretched far from mean. | -4 to +4 |
| X2 | **Velocity** | `velocity` | `price[i] - price[i-1]` — raw bar-to-bar price change (NOT normalized). Positive = up, negative = down. One bar's delta. | continuous |
| X3 | **F_momentum** | `F_momentum` | `(velocity * volume) / sigma` — volume-weighted velocity normalized by band width. NOT raw momentum — includes volume AND sigma. High volume bar moving fast relative to bands = large F_momentum. | continuous |
| X4 | **Mean reversion force** | `mean_reversion_force` | `-GRAVITY_THETA * z * sigma` = `-0.5 * z * sigma`. Linear pull toward regression center. Scales with BOTH deviation (z) AND volatility (sigma). NOT z²/9 (docstring is wrong). | continuous |
| X5 | **Mom/Rev ratio** | `mom_rev_ratio` | `|F_momentum| / |F_reversion|`. Since F_momentum includes volume/sigma and F_reversion includes z*sigma, this ratio captures: is volume-driven momentum overcoming the gravitational pull? >1 = momentum winning. | 0 to inf |
| X6 | **Net force (F_net)** | `net_force` | `F_gravity + F_momentum + F_repulsion`. Sum of: gravitational pull to center + volume-weighted momentum + band-edge repulsion. The effective acceleration including all three bodies. | continuous |
| X7 | **Hurst exponent** | `hurst_exponent` | R/S method over rolling window (100 bars default). Log-log regression of R/S statistic at 4 sub-scales (w/8, w/4, w/2, w). >0.5 = trending/persistent, <0.5 = mean-reverting/anti-persistent, =0.5 = random walk. Clipped [0,1]. First 100 bars default to 0.5. | 0.0 to 1.0 |
| X8 | **Tunnel (reversion) prob** | `reversion_probability` | `1 - erfi(|z|/sqrt2) / erfi(3/sqrt2)`. OU first-passage probability: P(price hits center before hitting ±3σ). **PURELY a function of |z|** — no other inputs. At z=0: ~1.0, at z=2: ~0.003. Measures distance-based reversion likelihood, NOT market regime. | 0.0 to 1.0 |
| X9 | **Breakout prob** | `breakout_probability` | `erfi(|z|/sqrt2) / erfi(3/sqrt2)`. Complement of tunnel prob. P(price hits ±3σ before center). Also purely a function of |z|. | 0.0 to 1.0 |
| X10 | **Barrier height** | `barrier_height` | `0.025 * (9 - z²)`, clipped to ≥0. OU potential difference between 3σ barrier and current z. Maximum at z=0 (0.225), zero at z=±3. | 0.0 to 0.225 |
| X11 | **Entropy (normalized)** | `entropy_normalized` | `-sum(Pi*log(Pi)) / log(3)` where P0,P1,P2 = softmax of Gaussian distance to center, upper, lower attractors. 1.0 = price equidistant from all 3 (z≈0). Near 0 = price collapsed to one attractor (extreme z). Driven mainly by z-score. | 0.0 to 1.0 |
| X12 | **ADX** | `adx_strength` | Trend strength (directionless). High ADX = strong trend (either direction). Low ADX = no trend, choppy. | 0 to 100 |
| X13 | **DMI diff** | `dmi_plus - dmi_minus` | Directional movement difference. Positive = bullish pressure, negative = bearish pressure. | -100 to +100 |
| X14 | **Sigma** | `regression_sigma` | `sqrt(RSS/(rp-2))` from rolling linear regression. Standard error of regression, NOT Bollinger std. Measures residual scatter around the regression line. High = noisy/volatile around trend. Low = price hugging the regression. | > 0 |
| X15 | **Term PID** | `term_pid` | `Kp*z + Ki*cumsum(z) + Kd*diff(z)` (Kp=0.5, Ki=0.1, Kd=0.2). PID controller output treating z-score as error. Large absolute value = strong corrective force needed. Sign = direction of correction. | continuous |
| X16 | **Oscillation coherence** | `oscillation_entropy_normalized` | `1 / (1 + rolling_std(z, window=5))`. Rolling std of z-score over 5 bars, inverted. 1.0 = z-score perfectly flat (algo-locked). Near 0 = z-score volatile (organic/trending). Measures micro-regime stability. | (0, 1] |
| X31 | **Band-relative speed** | `|velocity| / regression_sigma` | `|price[i]-price[i-1]| / sigma`. How fast price moves relative to band width, WITHOUT volume (unlike F_momentum which includes volume). High = bar covered large fraction of band width. Low = drifting. Note: F_momentum = velocity*volume/sigma, so X31 = F_momentum/volume essentially. | >= 0 |

### Category B: Belief Network (per-signal, aggregated across TFs)

| ID | Parameter (X) | Field name | What it measures (theory) | Range |
|----|--------------|------------|--------------------------|-------|
| X17 | **Conviction** | `belief_conviction` | Path conviction — weighted geometric mean of P(direction) across all active TF workers. High = tree agrees on direction. | 0.0 to 1.0 |
| X18 | **Active levels** | `belief_active_levels` | How many TF workers contributed to the belief. More levels = more confirmation. | 1 to 10 |
| X19 | **Wave maturity** | `wave_maturity` | Weighted avg exhaustion across all TF workers. High = the move is nearly done. | 0.0 to 1.0 |
| X20 | **Decision wave maturity** | `decision_wave_maturity` | Exhaustion at the DECISION TF only (the TF that triggered the signal). More actionable than avg. | 0.0 to 1.0 |
| X21 | **Worker agreement** | (computed) | % of TF workers whose direction matches the trade side. High = consensus. | 0.0 to 1.0 |

### Category C: Signal / Template Context (per-signal, discrete/categorical)

| ID | Parameter (X) | Field name | What it measures (theory) | Range |
|----|--------------|------------|--------------------------|-------|
| X22 | **Depth** | `depth` / `entry_depth` | Which TF level triggered the signal (3=15m ... 12=1s). Deeper = faster TF = smaller moves. | 3 to 12 |
| X23 | **Gate1 distance** | `gate1_dist` | Euclidean distance to nearest template centroid (16D). Lower = better pattern match. | 0 to inf |
| X24 | **Band zone** | `band_zone` | Where price is relative to bands: INNER, UPPER_EXTREME, LOWER_EXTREME. | categorical |
| X25 | **Pattern type** | `pattern_type` | Micro-pattern: COMPRESSION, WEDGE, BREAKDOWN, MOMENTUM_BREAK, BAND_REVERSAL. | categorical |
| X26 | **Template ID** | `template_id` | Which cluster template matched. Some templates are structurally better. | categorical |
| X27 | **Long bias** | `long_bias` | Template's learned directional bias toward LONG. 0.5 = neutral. | 0.0 to 1.0 |

### Category D: Market Context (per-bar, contextual)

| ID | Parameter (X) | Field name | What it measures (theory) | Range |
|----|--------------|------------|--------------------------|-------|
| X28 | **Trend direction (15m)** | `trend_direction_15m` | Macro trend: UP, DOWN, RANGE. Trading with trend should improve WR. | categorical |
| X29 | **Session** | `session` | ASIA, EUROPE, US, OVERLAP. Liquidity and volatility differ by session. | categorical |
| X30 | **Roche snap** | `cascade_detected` | Price at extreme z AND high velocity — violent move in progress. | bool |
| X32 | **Band confluence** | `get_band_confluence()` | Multi-TF Standard Error Band alignment. Aggregates z-score band position (±1σ,±2σ,±3σ) across all active TF workers. IMPLEMENTED in `core/timeframe_belief_network.py` — `BandContext` dataclass per worker + `get_band_confluence()` aggregator. Weighted by TF (higher TFs carry more weight). Outputs direction (long/short/None) + strength + support/resistance scores. Used by exit engine for band-aware trail adjustment. | direction + strength |

### Category F: Level Detection (per-bar, structural) — FUTURE STATE

See `docs/LEVEL_DETECTOR_SPEC.md` for full spec. NOT YET IMPLEMENTED.
Human draws 2 lines (range top/bottom) → system auto-generates Fibonacci levels +
multi-TF swing detection → DBSCAN clustering → scored structural levels.

| ID | Parameter (X) | Field name | What it measures (theory) | Range |
|----|--------------|------------|--------------------------|-------|
| X41 | **Distance to nearest level** | `dist_to_level_ticks` | How far price is from the closest structural level (fib anchor or clustered sub-level) in ticks. Near a level = expect reaction (bounce/break). Far = "empty space" with no structural reference. | 0 to inf ticks |
| X42 | **Level type** | `nearest_level_type` | Is the nearest level `support`, `resistance`, or `pivot`? Determines expected behavior at that price — support = bounce long, resistance = bounce short, pivot = either. | categorical |
| X43 | **Level confidence** | `nearest_level_confidence` | Composite score: cluster density × recency × TF diversity bonus × fib proximity bonus. High confidence = many independent methods agree this level matters. | 0.0 to 1.0 |
| X44 | **Fib reinforced** | `fib_reinforced` | Does the nearest sub-level cluster independently converge near a Fibonacci retracement? True = two independent methods (swing detection + fib math) agree → highest confidence. | bool |
| X45 | **Level TF diversity** | `level_tf_count` | How many timeframes contributed swing points to this level cluster (1-4: daily, 4h, 1h, 15m). More TFs = more structural significance. Bonus: 1→1.0x, 2→1.3x, 3→1.6x, 4→2.0x. | 1 to 4 |
| X46 | **Level recency** | `level_recency_weight` | Exponential decay weight based on age: `exp(-0.693 * age_hours / (half_life * 24))`. Recent levels matter more. Fib anchors NEVER decay (weight=1.0 always). | 0.0 to 1.0 |
| X47 | **Price in zone** | `price_in_level_zone` | Is current price WITHIN a clustered level zone (between price_lower and price_upper)? If yes, we're AT a decision point. If no, we're between levels. | bool |
| X48 | **Fib ratio proximity** | `nearest_fib_ratio` | Which Fibonacci ratio is closest (0.236, 0.382, 0.500, 0.618, 0.764)? The 0.618 and 0.382 levels are historically the strongest reactions. | categorical |

### Category E: Exit Engine Parameters (per-trade, tunable)

IMPLEMENTED in `core/exit_engine.py` — unified ExitEngine class used by both
training and live. 10-level exit cascade: SL → TP → Watchdog → Max hold →
Band urgent → Envelope decay → Trail stop → Breakeven lock → Belief flip → Hold.
See `docs/CLAUDE_CODE_UNIFIED_EXIT_ENGINE.md` for original spec.

| ID | Parameter (X) | Field name | What it measures (theory) | Range |
|----|--------------|------------|--------------------------|-------|
| X33 | **SL distance** | `sl_ticks` | Stop loss width. Priority: `p25_mae * 3.0` → `reg_sigma * 1.1` → `ATR * 2.0`. Floor=8, cap=80 ticks. Wider = more room but bigger losses. Cluster-fitted, NOT hardcoded. | 8 to 80 ticks |
| X34 | **TP distance** | `tp_ticks` | Take profit target. Priority: `network_tp` → `mfe_coeff` → `p75_mfe` → `ATR * 3.0`. Floor=4, cap=200 ticks. | 4 to 200 ticks |
| X35 | **Trail activation** | `trail_activation_ticks` | How much profit before trailing stop engages. `p25_mae * 0.3` or `ATR * 0.6`. Floor=3. Too low = whipsawed out. Too high = never activates. | >= 3 ticks |
| X36 | **Max hold bars** | `max_hold_bars` | Per-template time limit. Forced exit after this many bars regardless of P&L. | varies |
| X37 | **Envelope half-life** | `envelope_half_life_bars` | How fast the profit envelope decays. Currently 40 bars. Modulated by F_net: favorable force slows decay (×1.3), adverse accelerates (×0.7). | bars |
| X38 | **Watchdog threshold** | `watchdog_tick_threshold` | How many adverse ticks before watchdog triggers (currently 8). Combined with bars without MFE progress. | ticks |
| X39 | **Trail tightening rate** | `tightening factor` | `max(0.4, 1.0 - (progress_ratio - 1.0) * 0.15)`. How fast trail tightens as profit grows. Band-aware: ×0.6 at resistance, ×1.4 at support. | 0.4 to 1.0 |
| X40 | **Breakeven lock threshold** | `trail_activation * 0.6` | When to lock trail at breakeven (+2 ticks). Prevents winners from turning into losers. | ticks |

---

## Response Variable Domains

The Y's belong to two fundamentally different decisions:

### Entry Domain: "Should I take this trade?" (X + Y = Z_entry)

| Y | Response | Entry question it answers |
|---|----------|--------------------------|
| Y1 | Win Rate | Will this trade be profitable? |
| Y5 | Direction accuracy | Am I on the right side? |
| Y9 | Trade count | Am I being selective enough? |
| Y10 | Golden path capture | How much of the available 1s chord length did I harvest? |
| Y11 | Oracle path efficiency | What's the BEST POSSIBLE risk-adjusted extraction? (benchmark) |

**X's that score high on ENTRY Y's → should be GATES (binary pass/fail)**
A gate blocks bad trades. If a parameter predicts WR or direction, use it to
decide WHETHER to enter. Gate = "don't trade unless X meets threshold."

### Exit Domain: "How do I manage this trade?" (X + Y = Z_exit)

| Y | Response | Exit question it answers |
|---|----------|--------------------------|
| Y2 | PnL per trade | How much will I make/lose? |
| Y3 | Capture % | How much of the move will I catch? |
| Y4 | Reversal rate | Will the market flip on me? |
| Y6 | Hold efficiency | Am I getting paid for my time? |
| Y7 | Hold time | How long should I stay? |
| Y8 | Trade decay | How fast is my edge evaporating? |
| Y12 | Risk-adjusted path ratio | How much did I make per dollar risked (PnL / max MAE)? |

**X's that score high on EXIT Y's → should be MODULATORS (continuous adjustment)**
A modulator adjusts the exit engine in real-time: tighten/widen stops, accelerate/
slow envelope decay, move TP, trigger urgency exit. Modulator = "adjust behavior
WHILE in the trade based on current X value."

### The Critical Mistake

Using an EXIT parameter as a GATE kills trade count (Y9) and golden path
capture (Y10) without improving WR. Example: Hurst (X7) scores 9 on Y3/Y6/Y8
(exit domain) but only 3 on Y1 (entry domain). Making Hurst a gate filters
out trades that COULD have been profitable if managed differently. Instead,
low Hurst should TIGHTEN the trail and ACCELERATE envelope decay — same trade,
different exit management.

Conversely, using an ENTRY parameter as a modulator wastes it. Conviction (X17)
scores 9 on Y1/Y2 — it predicts WHICH trades win. Using it to adjust trail
width is weak. Use it to decide whether to trade at all.

---

## C&E Matrix: Theoretical Importance Ratings

Scale: 0 = no expected effect, 1 = weak, 3 = moderate, 9 = strong

| X \ Y | Y1 WR | Y2 PnL/t | Y3 Cap% | Y4 Rev | Y5 Dir | Y6 HldEf | Y7 HldTm | Y8 Decay | Y9 Cnt | Y10 Gold |
|-------|-------|---------|---------|--------|--------|---------|---------|---------|--------|---------|
| **X1 z_score** | 3 | 3 | 1 | 9 | 1 | 1 | 3 | 9 | 3 | 3 |
| **X2 velocity** | 3 | 3 | 3 | 3 | 9 | 3 | 3 | 3 | 0 | 0 |
| **X3 F_momentum** | 3 | 3 | 9 | 3 | 9 | 9 | 9 | 9 | 1 | 1 |
| **X4 F_reversion** | 3 | 1 | 1 | 9 | 1 | 1 | 3 | 9 | 3 | 3 |
| **X5 mom/rev ratio** | 9 | 9 | 9 | 9 | 3 | 9 | 9 | 9 | 9 | 9 |
| **X6 F_net (accel)** | 3 | 9 | 9 | 9 | 3 | 9 | 9 | 9 | 0 | 0 |
| **X7 hurst** | 3 | 9 | 9 | 3 | 1 | 9 | 9 | 9 | 9 | 9 |
| **X8 tunnel_prob** | 3 | 3 | 1 | 9 | 1 | 1 | 3 | 9 | 9 | 9 |
| **X9 breakout_prob** | 1 | 3 | 3 | 3 | 1 | 3 | 3 | 3 | 1 | 1 |
| **X10 barrier_hgt** | 1 | 1 | 1 | 3 | 0 | 1 | 1 | 3 | 1 | 1 |
| **X11 entropy** | 1 | 1 | 1 | 1 | 3 | 1 | 1 | 1 | 1 | 1 |
| **X12 ADX** | 3 | 9 | 9 | 3 | 3 | 9 | 9 | 3 | 3 | 3 |
| **X13 DMI diff** | 1 | 3 | 3 | 1 | 9 | 3 | 1 | 1 | 0 | 0 |
| **X14 sigma** | 1 | 9 | 3 | 3 | 0 | 3 | 3 | 3 | 1 | 1 |
| **X15 term_pid** | 3 | 3 | 1 | 3 | 0 | 1 | 3 | 9 | 3 | 3 |
| **X16 osc_coh** | 3 | 3 | 1 | 3 | 0 | 1 | 3 | 9 | 3 | 3 |
| **X17 conviction** | 9 | 9 | 3 | 3 | 9 | 3 | 1 | 1 | 9 | 3 |
| **X18 active_lvl** | 3 | 3 | 3 | 1 | 3 | 1 | 1 | 1 | 3 | 1 |
| **X19 wave_mat** | 3 | 3 | 9 | 9 | 1 | 9 | 9 | 9 | 0 | 0 |
| **X20 dec_wv_mat** | 3 | 3 | 9 | 9 | 1 | 9 | 9 | 9 | 0 | 0 |
| **X21 wkr_agree** | 3 | 3 | 1 | 1 | 9 | 1 | 0 | 0 | 0 | 0 |
| **X22 depth** | 9 | 9 | 9 | 3 | 3 | 9 | 9 | 9 | 9 | 9 |
| **X23 gate1_dist** | 9 | 3 | 1 | 1 | 1 | 1 | 0 | 0 | 9 | 9 |
| **X24 band_zone** | 3 | 3 | 3 | 9 | 3 | 3 | 3 | 3 | 3 | 3 |
| **X25 pattern_type** | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| **X26 template_id** | 3 | 3 | 3 | 3 | 9 | 3 | 3 | 3 | 1 | 1 |
| **X27 long_bias** | 1 | 1 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 |
| **X28 trend_dir** | 3 | 3 | 3 | 3 | 9 | 3 | 3 | 3 | 0 | 0 |
| **X29 session** | 1 | 3 | 1 | 1 | 0 | 3 | 3 | 3 | 1 | 1 |
| **X30 roche_snap** | 1 | 3 | 3 | 9 | 1 | 3 | 1 | 3 | 1 | 1 |
| **X31 band_speed** | 9 | 9 | 9 | 9 | 3 | 9 | 9 | 9 | 3 | 3 |
| **X32 band_confl** | 3 | 3 | 3 | 9 | 9 | 3 | 1 | 1 | 1 | 1 |
| **X33 sl_ticks** | 9 | 9 | 1 | 1 | 0 | 3 | 3 | 3 | 0 | 0 |
| **X34 tp_ticks** | 3 | 9 | 9 | 0 | 0 | 9 | 9 | 3 | 0 | 0 |
| **X35 trail_act** | 3 | 3 | 9 | 1 | 0 | 9 | 9 | 9 | 0 | 0 |
| **X36 max_hold** | 1 | 3 | 9 | 1 | 0 | 9 | 9 | 9 | 0 | 0 |
| **X37 env_halflife** | 1 | 3 | 9 | 1 | 0 | 9 | 9 | 9 | 0 | 0 |
| **X38 watchdog** | 3 | 9 | 1 | 1 | 0 | 3 | 3 | 3 | 0 | 0 |
| **X39 trail_tight** | 1 | 3 | 9 | 1 | 0 | 9 | 9 | 9 | 0 | 0 |
| **X40 be_lock** | 9 | 3 | 1 | 1 | 0 | 3 | 1 | 1 | 0 | 0 |
| **X41 dist_to_lvl** | 9 | 9 | 3 | 9 | 3 | 3 | 3 | 3 | 9 | 3 |
| **X42 level_type** | 3 | 3 | 1 | 3 | 9 | 1 | 1 | 1 | 3 | 3 |
| **X43 level_conf** | 9 | 9 | 3 | 3 | 3 | 3 | 1 | 1 | 9 | 3 |
| **X44 fib_reinf** | 3 | 3 | 1 | 3 | 3 | 1 | 1 | 1 | 3 | 1 |
| **X45 lvl_tf_div** | 3 | 3 | 1 | 3 | 3 | 1 | 1 | 1 | 3 | 1 |
| **X46 lvl_recency** | 3 | 3 | 1 | 3 | 1 | 1 | 1 | 1 | 1 | 1 |
| **X47 in_zone** | 9 | 9 | 3 | 9 | 3 | 3 | 3 | 3 | 9 | 9 |
| **X48 fib_ratio** | 3 | 3 | 1 | 3 | 1 | 1 | 1 | 1 | 1 | 1 |

---

## Parameter Redundancy Map (verified from code)

Parameters that are mathematical transforms of the SAME underlying input
will show identical patterns in interaction plots. Must identify these
BEFORE running analysis to avoid over-counting.

| Group | Root input | Derived parameters | Relationship |
|-------|-----------|-------------------|-------------|
| **Z-group** | X1 z_score | X8 tunnel_prob, X9 breakout_prob, X10 barrier_height, X11 entropy | X8=`1-erfi(|z|/√2)/erfi(3/√2)`, X9=`1-X8`, X10=`0.025*(9-z²)`, X11=`f(softmax(z))`. All purely monotonic functions of |z|. **4 redundant parameters.** |
| **Momentum-group** | X2 velocity, X14 sigma | X3 F_momentum, X31 band_speed | X3=`velocity*volume/sigma`, X31=`|velocity|/sigma`. X3 adds volume weighting. X31 is volumeless. Both include sigma normalization. |
| **Reversion-group** | X1 z_score, X14 sigma | X4 F_reversion | X4=`-0.5*z*sigma`. Product of z and sigma. |
| **PID-group** | X1 z_score (history) | X15 term_pid | X15=`0.5*z + 0.1*cumsum(z) + 0.2*diff(z)`. Linear combination of z, its integral, and its derivative. Adds temporal memory but root signal is still z. |
| **Band-group** | X1 z_score (multi-TF) | X32 band_confluence, X24 band_zone | X32 aggregates z across TFs. X24 categorizes z into zones. Both derived from z but add cross-TF or categorical structure. |
| **Level-group** | X41 dist_to_level | X43 level_conf, X44 fib_reinforced, X45 lvl_tf_div, X46 lvl_recency, X47 in_zone | X43 is composite of density+recency+diversity+fib. X47=binary(dist<zone_width). X44,X45,X46 are sub-components of X43. Root signal: price proximity to structural level. X42 level_type and X48 fib_ratio are independent categorical. |

### Implication: True Independent Parameters

After removing redundancies, the **actually independent** signals are:

| # | Parameter | What it uniquely measures |
|---|-----------|------------------------|
| 1 | **z_score** (X1) | Position in bands (subsumes X8,X9,X10,X11,X4,X15) |
| 2 | **velocity** (X2) | Raw price change direction + magnitude |
| 3 | **volume** (implicit in X3) | Market participation — NOT captured standalone |
| 4 | **sigma** (X14) | Volatility / band width |
| 5 | **hurst** (X7) | Fractal regime (trending vs mean-reverting) |
| 6 | **ADX** (X12) | Trend strength (directionless) |
| 7 | **DMI diff** (X13) | Trend direction (signed) |
| 8 | **osc_coherence** (X16) | Micro-regime stability (5-bar z volatility) |
| 9 | **conviction** (X17) | Cross-TF direction agreement |
| 10 | **wave_maturity** (X19/X20) | Move exhaustion |
| 11 | **depth** (X22) | Timeframe scale |
| 12 | **gate1_dist** (X23) | Pattern match quality |
| 13 | **band_confluence** (X32) | Cross-TF band alignment (IMPLEMENTED in TBN) |
| 14 | **dist_to_level** (X41) | Price proximity to structural level (FUTURE — subsumes X43-X47) |
| 15 | **level_type** (X42) | Support vs resistance vs pivot (FUTURE) |
| 16 | **fib_ratio** (X48) | Which Fibonacci ratio is nearest (FUTURE) |

**16 truly independent signals** out of 48 listed parameters (13 current + 3 future from Level Detector).

### User observations (theory — need empirical validation)

**Observation 1: Standard Error Band confluence (direction)**

The Standard Error Bands across multiple TFs provide structural context that
z_score alone cannot: a single TF's z=+2 means "stretched" but if 4H is at
-2σ and Daily is at -1σ, the structural direction is LONG despite the fast
TF saying "overbought." This cross-TF band profile is what the user trades
manually and is the strongest direction signal available. See
`docs/CLAUDE_CODE_BAND_CONTEXT.md` for implementation spec.

**Observation 2: Physics validity boundary (|z| > 4 = chaos)**

The entire physics model (three-body forces, OU tunneling, regression bands)
assumes price lives WITHIN the band structure. When |z| exceeds ~4σ, the
regression has failed — price has escaped the model's domain of validity.
At that point:

- F_reversion = `-0.5 * z * sigma` grows linearly but price isn't reverting
- Tunnel prob ≈ 0 (erfi saturates), barrier height < 0 (clipped to 0)
- The OU process assumption (mean-reverting) is violated — price is trending
  AWAY from the regression, making all OU-derived quantities meaningless
- F_momentum dominates but it's volume-weighted, so low-volume breakouts
  don't register properly

**Implication: Timeframe escalation rule**

When |z| > threshold at the current TF, the system should NOT trade at that
TF. Instead, escalate to the next higher TF where |z| <= 3σ. At that wider
scale, the regression still contains price and the physics model is valid.

Example: if 15s z=+5.2, the 15s regression is broken. But at 1m, z might be
+2.1 (wider bands). At 5m, z might be +1.3. Trade at 5m — that's where the
model is valid and the bands give meaningful support/resistance.

**What needs proving (empirical validation with oracle data):**
1. Is there a z threshold above which WR drops to ~50% (random)?
   → Sweep |z| buckets vs Y1 WR to find the breakpoint
2. Does the same threshold exist across all TFs or is it TF-dependent?
   → Stratify by depth
3. When |z| > threshold at depth D, does trading at depth D-1 or D-2
   (higher TF) produce better Y1/Y2?
   → Cross-depth analysis needed
4. Is the chaos boundary sharp (cliff) or gradual (slope)?
   → Shape of the WR vs |z| curve tells us gate vs modulator

**Observation 3: Structural levels as trade filter (Level Detector)**

Price doesn't move randomly between arbitrary points — it moves FROM one
structural level TO the next. The Level Detector (`docs/LEVEL_DETECTOR_SPEC.md`)
provides the missing structural context:

- **Entry filter**: Only trade when price is AT or WITHIN a level zone (X47).
  Entries in "empty space" between levels have no structural support and are
  essentially gambling on continuation. At a level, you have a known
  reaction point (bounce or break) that gives the trade a defined setup.

- **Direction from level type**: Support levels → expect LONG bounce.
  Resistance levels → expect SHORT rejection. This is independent of physics
  (z-score can be anything at a support level) and adds a structural direction
  signal (X42) that complements band confluence (X32).

- **Profit target from level spacing**: The NEXT level in the trade direction
  becomes the natural profit target. No need to guess TP with ATR multiples —
  the market structure defines it. This could replace or modulate X34 (tp_ticks).

- **Fib reinforcement as confidence boost**: When a detected sub-level
  independently converges near a Fibonacci retracement, two methods agree.
  This is the highest-confidence setup for a reaction.

**Not yet implemented.** Requires `range_config.json` (2 manual prices) +
pipeline build. See spec for 7-module architecture.

---

## Reading the Matrix

### Top X's per Y (theoretical, sum of row = total influence)

**Y1 WR** — What predicts whether a trade wins?
- X5 mom/rev ratio (9), X17 conviction (9), X22 depth (9), X23 gate1_dist (9),
  X41 dist_to_level (9), X43 level_conf (9), X47 in_zone (9) ← FUTURE

**Y2 PnL/trade** — What predicts dollar magnitude?
- X5 mom/rev (9), X6 F_net (9), X7 hurst (9), X12 ADX (9), X14 sigma (9),
  X17 conviction (9), X22 depth (9)

**Y3 Capture%** — What determines how much of the move we catch?
- X3 F_momentum (9), X5 mom/rev (9), X6 F_net (9), X7 hurst (9),
  X12 ADX (9), X19 wave_maturity (9), X20 dec_wave_mat (9), X22 depth (9)

**Y4 Reversal rate** — What predicts the market flipping on us?
- X1 z_score (9), X4 F_reversion (9), X5 mom/rev (9), X6 F_net (9),
  X8 tunnel_prob (9), X19 wave_maturity (9), X20 dec_wave_mat (9),
  X24 band_zone (9), X30 roche_snap (9)

**Y5 Direction accuracy** — What predicts correct side?
- X2 velocity (9), X3 F_momentum (9), X13 DMI diff (9), X17 conviction (9),
  X21 worker_agree (9), X26 template_id (9), X27 long_bias (9), X28 trend_dir (9),
  X42 level_type (9) ← FUTURE (support=LONG, resistance=SHORT)

**Y6 Hold efficiency** — What predicts efficient time-in-trade?
- X3 F_momentum (9), X5 mom/rev (9), X6 F_net (9), X7 hurst (9),
  X12 ADX (9), X19 wave_maturity (9), X20 dec_wave_mat (9), X22 depth (9)

**Y7 Hold time** — What predicts how long we stay in a trade?
- X3 F_momentum (9), X5 mom/rev (9), X6 F_net (9), X7 hurst (9),
  X12 ADX (9), X19 wave_maturity (9), X20 dec_wave_mat (9), X22 depth (9)
- Note: depth is the strongest driver — depth 3 (15m) holds ~3min avg,
  depth 12 (1s) holds ~1min. This is structural, not tunable.

**Y8 Trade decay** — What predicts how fast a trade's edge evaporates?
- X1 z_score (9) — stretched trades decay faster (reversion pull)
- X4 F_reversion (9) — stronger pull = faster decay
- X5 mom/rev (9) — low ratio = fast decay (reversion winning)
- X6 F_net (9) — adverse acceleration = instant decay
- X7 hurst (9) — low hurst = anti-persistent = move reverses = fast decay
- X8 tunnel_prob (9) — high tunnel = high reversion = fast decay
- X15 term_pid (9) — algo-controlled = predictable decay pattern
- X16 osc_coherence (9) — tight PID = artificial mean reversion = fast decay
- X19 wave_maturity (9) — exhausted wave = edge is gone
- X22 depth (9) — deeper depth = faster TF = faster decay
- KEY: This is the half-life envelope. These parameters should MODULATE
  the decay rate, not gate the entry.

### Key Insight: Parameter Domain Assignment

Each parameter's scores across the Entry Y's vs Exit Y's determines its role.
Sum of entry scores (Y1+Y5+Y9+Y10) vs exit scores (Y2+Y3+Y4+Y6+Y7+Y8):

**ENTRY-DOMAIN parameters → GATES (decide WHETHER to trade)**

| Parameter | Entry sum | Exit sum | Role | Reasoning |
|-----------|-----------|----------|------|-----------|
| X5 mom/rev ratio | 30 | 54 | **Gate + Modulator** | Strong in BOTH — universal predictor. Gate entry, modulate exit. |
| X17 conviction | 30 | 10 | **Gate** | Entry-dominant. Predicts WR (Y1=9) + direction (Y5=9). Weak on exit. |
| X22 depth | 36 | 54 | **Stratification** | Strong everywhere — different depths = different strategies entirely. |
| X23 gate1_dist | 28 | 2 | **Gate** | Entry-only. Pattern match quality predicts WR (Y1=9). Zero exit influence. |
| X41 dist_to_level | 24 | 30 | **Gate** (FUTURE) | Entry-dominant. Suppress entries in "empty space" between levels. |
| X47 in_zone | 30 | 30 | **Gate + Boost** (FUTURE) | Both — at a level = trade, between levels = don't. Also tighten SL at levels. |

**EXIT-DOMAIN parameters → MODULATORS (adjust behavior WHILE in trade)**

| Parameter | Entry sum | Exit sum | Role | Reasoning |
|-----------|-----------|----------|------|-----------|
| X7 hurst | 30 | 54 | **Exit modulator** | Y1=3 (weak gate). Y3/Y6/Y8=9 (strong exit). Low hurst → tighten trail + fast decay. |
| X8 tunnel_prob | 22 | 28 | **Stop modulator** | Y4=9 (reversal). High tunnel → tighten stop (price likely to revert). |
| X19 wave_maturity | 12 | 54 | **Exit timing** | Entry=0 on Y9/Y10. Pure exit signal. High maturity → tighten everything. |
| X12 ADX | 18 | 42 | **Exit modulator** | High ADX → widen trail (ride the trend). Low ADX → tight trail (chop). |
| X6 F_net | 12 | 54 | **Exit urgency** | Zero entry influence. Adverse F_net = accelerate envelope decay. |
| X1 z_score | 12 | 28 | **Exit modulator** | Y4/Y8=9 (reversal/decay). Stretched z → faster decay, tighter stop. |
| X37 env_halflife | 0 | 48 | **Exit knob** | Pure exit. Directly controls envelope decay rate. |
| X39 trail_tight | 0 | 48 | **Exit knob** | Pure exit. Directly controls trail compression. |

**DIRECTION-DOMAIN parameters → DIRECTION CASCADE (decide WHICH side)**

| Parameter | Y5 Dir score | Role | Reasoning |
|-----------|-------------|------|-----------|
| X2 velocity | 9 | **Direction** | Raw price direction. Y5=9, weak elsewhere. |
| X13 DMI diff | 9 | **Direction** | Signed trend. Y5=9, weak elsewhere. |
| X21 worker_agree | 9 | **Direction** | Cross-TF consensus. Y5=9, zero elsewhere. |
| X27 long_bias | 9 | **Direction** | Template directional memory. Y5=9 only. |
| X28 trend_dir | 9 | **Direction** | Macro trend. Y5=9 only. |
| X32 band_confluence | 9 | **Direction** | Cross-TF SE band alignment. Y5=9. Implemented in TBN. |
| X42 level_type | 9 | **Direction** (FUTURE) | Support=LONG, resistance=SHORT. |

---

## Sample Size & Observation Levels

### The N Problem

Each X is measured at a different observation level. Interaction analysis requires
enough observations **per cell** (X_bin × Y) to be statistically reliable.

| Observation level | Examples | Typical N per run | Notes |
|---|---|---|---|
| Per-1s-tick | velocity, z_score, F_net | ~86,400/day | Abundant, but Y's are per-trade |
| Per-bar (15s) | physics state, ADX, hurst | ~5,760/day | Still abundant |
| Per-signal (fired) | conviction, gate1_dist, band_zone | ~100-500/run | Depends on gate strictness |
| Per-trade (taken) | SL/TP/trail, exit reason, PnL | ~30-400/run | **This is the bottleneck** |
| Per-TF-worker | band_context per worker | 10 workers × N bars | Aggregated to 1 value at signal time |
| Per-template | template_id, long_bias | ~5-50 per cluster | **Dangerously small** |
| Per-depth-stratum | depth-stratified anything | ~5-80 per depth | Depth 11-12 may have <10 trades |

### Why This Matters

All Y's except Y9 (trade count) are measured **per-trade**. This means:

- **The sample size ceiling is always the number of trades.** No matter how many
  bars of physics data you have, if you only took 200 trades, that's your N.
- **Binning X's reduces N per cell.** If you bin X17 conviction into 5 levels,
  each level has ~40 trades. If you then cross with X22 depth (10 levels),
  each cell has ~4 trades. That's noise, not signal.
- **Aggregated X's (conviction, band_confluence) are single values per signal.**
  They collapse 10 workers into 1 number. The aggregation is the signal —
  you can't decompose it back into per-worker contributions post-hoc.

### Minimum Sample Size Rules

| Analysis type | Minimum N per cell | Implication |
|---|---|---|
| Simple X sweep (1D) | ≥ 30 trades per bin | Max 5-7 bins with 200 trades |
| X×X interaction (2D) | ≥ 15 trades per cell | Max 3×3 grid with 200 trades |
| Regression (continuous) | ≥ 50 total | OK for most X's with ≥200 trades |
| Per-depth stratification | ≥ 20 per depth | Need ~200 trades per depth → 2000 total |
| Per-template analysis | ≥ 20 per template | Often impossible with <50 per cluster |

### Practical Design for Interaction Tool

Given typical IS run = ~400 trades:

1. **Continuous X's → regression** (no binning needed): z_score, conviction,
   wave_maturity, F_net, hurst, ADX, band_speed, mom_rev_ratio.
   Compute correlation and slope against each Y. N=400 is sufficient.

2. **Categorical X's → group comparison**: depth (3-12), pattern_type (5),
   session (4), band_zone (3). ANOVA or Kruskal-Wallis. Need ≥20 per group.

3. **Interaction plots → 2D heatmap**: Only for top-2 X's per Y.
   Use 3 bins each (low/mid/high) → 9 cells → ~44 trades per cell with N=400.
   This is the maximum feasible resolution.

4. **Per-depth analysis**: Only for depths with ≥30 trades. Typically depths
   3-7 qualify. Depths 8-12 must be pooled into "fast TF" bucket.

5. **Template-level analysis**: Pool templates into groups (e.g., by long_bias
   >0.55 vs <0.45 vs neutral) rather than individual template_id analysis.

### The Optimum Sample Size Question

For each X×Y pair, the required sample size depends on the **effect size** we
want to detect:

```
N_required = (Z_α/2 + Z_β)² × 2σ² / δ²

where:
  Z_α/2  = 1.96 (95% confidence)
  Z_β    = 0.84 (80% power)
  σ      = std dev of Y in population
  δ      = minimum meaningful difference in Y between groups
```

For WR (Y1) with σ≈0.5 (binary outcome):
- Detect 10% WR difference (50% vs 60%): N = ~388 per group
- Detect 15% WR difference (50% vs 65%): N = ~176 per group
- Detect 20% WR difference (50% vs 70%): N = ~99 per group

**Implication**: With 400 trades, we can only reliably detect LARGE effects
(≥15% WR swing). Subtle effects (5% WR improvement from a parameter) require
~1600 trades per group → need 10-month IS run with many trades.

### Aggregation Paradox

Band confluence (X32) and conviction (X17) are AGGREGATED from multiple TF
workers. The aggregation itself is a design choice:

- Geometric mean of P(direction) across workers (conviction)
- Weighted sum of support/resistance scores across workers (band confluence)

The aggregation formula determines the signal quality. But you can't optimize
the aggregation formula with per-trade Y data because N is too small.

**Solution**: Two-stage analysis:
1. Stage 1: Per-worker analysis with per-bar data (huge N) — which workers
   contribute useful signal? Which aggregation formula preserves information?
2. Stage 2: Aggregated signal analysis with per-trade data (small N) —
   does the aggregated signal predict Y1/Y5?

---

## Next Steps

1. **Run IS forward pass** with physics fields in all oracle records
2. **Run `tools/analyze_gates.py --apply`** to get baseline thresholds
3. **Build `tools/gate_interaction_matrix.py`** — validates this matrix with real data
   - Continuous X's: correlation + regression slope against each Y
   - Categorical X's: group means + Kruskal-Wallis test
   - Top-2 interaction: 3×3 heatmap (low/mid/high bins)
   - Report N per cell, flag any cell with <15 trades
   - Power analysis: report detectable effect size given actual N
4. **Reassign parameters** based on data — move from gate to modulator where theory + data agree
5. **Per-depth thresholds** — only for depths with ≥30 trades, pool the rest
