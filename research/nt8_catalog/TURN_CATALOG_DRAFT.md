# TURN CATALOG (DRAFT) — turn/exit timing concepts mined from the NT8 article corpus

> Purpose: find concepts for timing the END of a swing / the reversal moment, to
> feed a turn-detection layer. Motivation: (a) every AI-label turn is simultaneously
> an exit and the next entry (labels chain); (b) the 2026-07-16 `turn_detection_audit.md`
> shows NONE of the 40 entry streams detects turns above chance at ±2m (best dir-recall
> 0.30 = RENKO24, most < 0.10; chance ±2m = 0.43). Target: a turn caught within ~1-2 min.
>
> Corpus scanned: `research/nt8_catalog/raw_articles/*.md` (463 files).
> Method: keyword sweep (exhaustion/climax, reversal, divergence, trailing/SAR, sweep/
> failure-test, delta flip, rotation, momentum stall, time-exit, scale-out) → then
> VERBATIM read of the ~22 highest-signal articles. Every quoted passage below is copied
> verbatim from the cited file (scraper artifacts like `pricefails tomove` preserved as-is).
> Articles scanned (grep-touched): ~120 matched at least one turn keyword; ~22 read in full.
> DATA CONSTRAINT: train year 2024 has 5s OHLCV only — NO tick / order-flow / footprint /
> bid-ask delta. Order-flow-dependent concepts are flagged DATA-BLOCKED for the train year.
>
> Existing league streams referenced for overlap (do NOT double-build): SAR-23, ZIGZAG,
> CURVE, PTRN-ENGULF/PTRN-HAMMER, NMP / NMP-LAMBDA (z-exit), tier-ladder negative-exit,
> RSI06, MACD07, VA13/VP01, SQZ04, FIB17, PIVOT16, HNS22, ROUND05, ORB02, DOW19, SEASON12.

---

## SUMMARY TABLE

| ID | Name | #sources | Portability | Overlap w/ existing stream |
|---|---|---|---|---|
| TURN-01 | Momentum_Stall_Rollover | 2 | EASY | partial: MACD07 (histogram), NMP velocity |
| TURN-02 | Oscillator_Recross_From_Extreme | 3 | EASY | partial: RSI06 (level, not recross) |
| TURN-03 | Price_Oscillator_Divergence | 6 | EASY | none direct (divergence not yet a stream) |
| TURN-04 | Volume_Divergence_OBV | 3 | EASY | none direct (volume-price divergence) |
| TURN-05 | Cumulative_Delta_Divergence_Flip | 3 | **DATA-BLOCKED** | none (order-flow) |
| TURN-06 | Climax_Volume_Exhaustion_Bar | 3 | MODERATE | none direct |
| TURN-07 | Sweep_And_Reclaim (failure test) | 2 | MODERATE | partial: ROUND05, PIVOT16, ORB02 (levels only) |
| TURN-08 | Double_Triple_Top_Bottom | 2 | MODERATE | partial: HNS22, PIVOT16 |
| TURN-09 | HeadShoulders_Neckline_Break | 3 | MODERATE | **HNS22 (already built; ~0 turn recall)** |
| TURN-10 | HeikinAshi_Color_Flip | 1 | EASY | none — distinct transform |
| TURN-11 | Parabolic_SAR_Flip (exit role) | 2 | EASY | **SAR-23 (entry stream; exit role distinct)** |
| TURN-12 | Structural_Swing_Break (trailing) | 3 | MODERATE | **ZIGZAG / SAR-23** |
| TURN-13 | Return_To_Mean_Completion | 3 | EASY | **NMP z-exit (heavy); VA13/VP01, VWAP03** |
| TURN-14 | VolBand_Pierce_Reenter_Exhaustion | 2 | EASY | **SQZ04 (Bollinger); existing APZ concept** |
| TURN-15 | Session_Close_Flatten_Time | 1 | EASY | none (SEASON12 is calendar, not intraday) |
| TURN-16 | Target_Opposing_Structure_Laddering | 3 | MODERATE | partial: FIB17, PIVOT16 |

Portability legend given our data (5s OHLCV, no order flow 2024): EASY = pure OHLCV
derivation; MODERATE = needs pivot/level detection or volume-normalization; DATA-BLOCKED
= requires tick/order-flow we do not have for the train year.

---

## TURN-01 — Momentum_Stall_Rollover
**SOURCE:** `measuring-trend-strength-momentum-indicator.md`; supporting `futures-market-inflection-points-identified.md`
> "Conversely, when the momentum line contracts toward zero, it suggests that the trend is losing speed. This does not always mean reversal, but it can indicate weakening participation."
> "Even when momentum remains above zero, a downward slope can signal deceleration. For example, in a Micro E-mini Nasdaq (MES) uptrend, momentum may stay positive while gradually declining."
> "Spotting trend exhaustion before price reverses. Momentum goes both ways and can help you identify when a trend is running out of energy. This is often visible when the momentum line peaks and begins to move back toward zero, while price continues in the same direction."
> "Because momentum is a lagging indicator, it's most effective when combined with other forms of analysis..."

**MECHANIC:** Momentum = `close[t] − close[t−N]` on 1m (or 5s) bars. Track its 1st derivative (slope). FIRE a leg-ending mark when momentum makes a local peak (in an up-leg) or trough (in a down-leg) and then reverses slope for `k` bars WHILE price is still extending in the trend direction (the peak-and-rollover). Article gives the qualitative rule (peak → back toward zero while price continues) but gives NO period N [UNSPECIFIED — needs declared choice], NO slope-reversal bar count k [UNSPECIFIED], NO magnitude threshold (it explicitly says "no universal threshold... must be interpreted in the context of its own behavior"). Self-notes it is LAGGING.

**DIRECTION SEMANTICS:** Leg ending in the CURRENT trend direction; anticipates a turn to the opposite side. Direction of the new leg = opposite of the peaking momentum.

**PORTABILITY:** EASY — momentum and slope are pure OHLCV.

**OVERLAP:** Partial with MACD07 (the MACD histogram IS a smoothed momentum-rollover) and with any NMP velocity term. Distinct as a raw single-period momentum-slope rollover.

---

## TURN-02 — Oscillator_Recross_From_Extreme
**SOURCE:** `relative-strength-index-in-futures-trading.md`; `understanding-stochastics-in-futures-trading-a-guide-to-momentum-and-market-reversals.md`; `commodity-channel-index-cci-futures.md`
> (RSI) "with a long enough downtrend, the RSI can stay oversold for multiple bars. What is of interest to a trader is when the RSI crosses back above the oversold level, as this may indicate the beginning of an opposite bullish trend."
> (RSI) "an uptrend can stay overbought for multiple bars... what is notable here is when the RSI crosses back below 70, it may indicate the beginning of a bearish trend."
> (Stochastics) "When the %K line crosses above the %D line in the oversold zone, it can signal a buying opportunity. When the %K line crosses below the %D line in the overbought zone, it can signal a selling opportunity."
> (CCI) "Some traders look for the Commodity Channel Index to cross back into the neutral zone (between -100 and +100) from extreme levels as a potential entry signal. Exits may be timed based on returning to neutrality..."

**MECHANIC:** Timing refinement over a bare level: the turn is marked NOT when the oscillator first reaches the extreme, but when it CROSSES BACK. RSI: fire on cross back below 70 (down-turn) / back above 30 (up-turn); default RSI lookback 14 (article: "default parameter of 14... traders are encouraged to experiment"). Stochastics: fire on %K crossing %D inside the >80 / <20 zone; default 14 (5 short-term, 21 long-term given). CCI: fire on re-entry to −100..+100 from beyond; default period 14, but bands are UNBOUNDED and market-specific ("+100 might indicate overbought in ES... +200 or more in NQ") [threshold UNSPECIFIED per instrument].

**DIRECTION SEMANTICS:** Leg ending (the extreme leg) → new leg in the opposite direction (recross down from OB → down-leg begins).

**PORTABILITY:** EASY — RSI/stochastic/CCI are standard OHLCV oscillators.

**OVERLAP:** RSI06 exists but the audit-relevant distinction is the RECROSS event (a timing edge), not the level reading. Three articles independently frame the recross (not the extreme) as the actionable moment.

---

## TURN-03 — Price_Oscillator_Divergence
**SOURCE:** `relative-strength-index-in-futures-trading.md`; `measuring-trend-strength-momentum-indicator.md`; `understanding-stochastics-...reversals.md`; `commodity-channel-index-cci-futures.md`; `spot-market-trends-and-reversals-using-the-macd-indicator.md`; `trade-trend-reversals-futures.md`
> (RSI) "if the price bars are making higher highs and the RSI indicator is making lower highs, this condition is called bearish divergence. Bearish divergence may reflect weakening momentum, where the price action has not yet caught up but may eventually reverse to the downside."
> (Momentum) "Bullish divergence: Price makes a lower low while momentum forms a higher low. Bearish divergence: Price makes a higher high while momentum forms a lower high."
> (CCI) "if a futures market is making higher highs while the CCI is making lower highs, it may suggest weakening momentum and a potential reversal."
> (trade-trend-reversals) "When price makes a new high but RSI or MACD fails to confirm it, the uptrend may be weakening."

**MECHANIC:** Track local price pivots and oscillator pivots (RSI/momentum/CCI/stochastic %K). FIRE bearish-turn when price pivot(t) > price pivot(t−1) AND osc pivot(t) < osc pivot(t−1); mirror for bullish. Needs a pivot detector (swing lookback) and one oscillator — both OHLCV. Oscillator period [UNSPECIFIED — declare, default 14]; pivot lookback [UNSPECIFIED]. Multiple oscillators can vote. NOTE this is a LAGGING confirmation (needs the second pivot to complete) — the turn mark lands AT or slightly AFTER the extreme, so lead time is likely small/negative; test carefully against the ±2m window.

**DIRECTION SEMANTICS:** Leg ending; new leg opposite the price extreme.

**PORTABILITY:** EASY — pivots + oscillator on OHLCV.

**OVERLAP:** No dedicated divergence stream exists yet. The concept is a plumbing primitive shared across TURN-04/05. Distinct from RSI06 (which reads level, not price-vs-oscillator divergence).

---

## TURN-04 — Volume_Divergence_OBV
**SOURCE:** `obv-indicator-forecast-market-moves-and-spot-reversals.md`; supporting `01_Volume_and_OrderFlow.md` (Dow-theory synthesis); `futures-market-inflection-points-identified.md`
> "if price is making new highs while OBV remains flat or declining, this may indicate weakening momentum and a potential reversal."
> "bullish divergence occurs when price is making lower lows, but OBV is making higher lows. This indicates that selling pressure is weakening despite declining prices, and a bullish reversal could be imminent. Conversely, bearish divergence is observed when price is making higher highs, but OBV is making lower highs, suggesting that buying pressure is waning and a downward move may follow."
> (OBV construction) "When the instrument closes higher than the previous close, all the day's volume is considered up-volume... when the instrument closes lower than the previous close, all the day's volume is considered down-volume."

**MECHANIC:** OBV = running sum of `sign(close[t]−close[t−1]) × volume[t]`. Same divergence logic as TURN-03 but with OBV as the confirming series instead of a price-oscillator: fire bearish-turn when price makes a higher pivot-high while OBV makes a lower pivot-high (and mirror). Pivot lookback [UNSPECIFIED]. Article warns "false signals can be common... Sudden spikes in volume caused by external factors, such as news events, can lead to misleading OBV signals."

**DIRECTION SEMANTICS:** Leg ending; new leg opposite the price extreme.

**PORTABILITY:** EASY — OBV needs only close + volume (both in our OHLCV).

**OVERLAP:** None direct (no volume-divergence stream). Distinct signal FAMILY from TURN-03 (volume, not momentum). VWMA10 exists but tests VWMA direction, not OBV divergence.

---

## TURN-05 — Cumulative_Delta_Divergence_Flip  [DATA-BLOCKED for train year]
**SOURCE:** `what-is-cumulative-delta-in-order-flow-trading.md`; `footprint-charts-guide.md`; `how-to-trade-liquidity-traps-in-futures.md`
> (cumulative-delta) "price puts in a new low... the Cumulative Delta is plotted but puts in a higher low relative to the same starting point... This example can potentially be interpreted as a bullish signal..."
> (footprint) "Delta divergence... If price prints a new high with strongly negative delta (meaning sellers were actually more aggressive despite the higher close) the move likely lacks institutional conviction... When they diverge, it's often an early warning before a reversal that hasn't shown up on the candlestick yet."
> (liquidity-traps) "Look for large single-level absorption on the NinjaTrader footprint chart and a delta flip on NinjaTrader cumulative delta."

**MECHANIC:** Would fire a turn when price makes a new swing extreme but cumulative delta (running net of aggressive buy vs sell volume) diverges or flips sign. REQUIRES per-tick bid/ask aggressor classification (footprint / volumetric bars). We do NOT have tick or order-flow data for 2024.

**DIRECTION SEMANTICS:** Leg ending; new leg opposite the price extreme (delta is "the more honest signal").

**PORTABILITY:** DATA-BLOCKED (train year). Could revisit if/when order-flow data is added, OR proxied crudely by TURN-04 (OBV) / TURN-06 (close-position climax) which are the OHLCV-only shadows of this idea. Do NOT claim a delta stream on 2024 data.

**OVERLAP:** None (no order-flow stream exists). Explicitly note: the audit's "no stream detects turns" may partly reflect that the theoretically sharpest turn signal (delta divergence) is unavailable in the train data.

---

## TURN-06 — Climax_Volume_Exhaustion_Bar
**SOURCE:** `futures-market-inflection-points-identified.md`; `footprint-charts-guide.md`; supporting `volume-spread-analysis-and-fomo-of-strong-directional-moves.md`
> (inflection) "Volume surges: One of the most reliable indicators of an inflection point is a sudden increase in trading volume. High volume at a market top or bottom suggests strong trader participation and can precede a reversal in the market trend."
> (footprint) "A single-bar volume spike can reveal trapped traders: participants who entered aggressively on one side and ended up on the wrong side of the move. Classic example: massive ask volume at the high, price immediately reverses, closes near the low. Those buyers are now losing."
> (footprint) "Exhaustion:One sidepushesaggressively and runs out of fuel. Often appears as a sharp delta spike followed by a reversal—one side tried, everyone else noticed, and the move collapsed."
> (VSA) "the trend is experiencing exhaustion and that their participation may be ill-timed."

**MECHANIC:** FIRE a leg-ending mark when a bar (1m or 5s) has a volume spike (volume > `m`× rolling-avg volume) AT a fresh swing extreme AND closes in the far third away from that extreme (push to new high, close near the low ⇒ up-leg exhaustion; mirror for down-leg). Uses volume + close-position — the OHLCV-portable shadow of footprint "trapped traders / exhaustion." The DELTA-based half of the footprint quote is DATA-BLOCKED; only the price/volume half is portable. Volume multiplier m and rolling-avg window [UNSPECIFIED — declare]; close-position fraction [UNSPECIFIED].

**DIRECTION SEMANTICS:** Leg ending; new leg opposite the extreme (climax marks the top/bottom).

**PORTABILITY:** MODERATE — needs volume normalization (session-relative; volume has intraday shape) plus swing-extreme detection.

**OVERLAP:** None direct. Related to PTRN-ENGULF (a bar-shape signal) but the trigger here is volume-spike-at-extreme + close-position, not body engulfment.

---

## TURN-07 — Sweep_And_Reclaim (failure test / liquidity trap)
**SOURCE:** `how-to-trade-liquidity-traps-in-futures.md`; supporting `trade-trend-reversals-futures.md`
> "A liquidity trap in futures trading occurs when price sweeps a key level—such as a session high, swing low, or previous day range boundary—to trigger clustered stop-loss orders before reversing sharply in the opposite direction."
> "Wait for the sweep; a decisive, often rapid move through the level that visibly accelerates as stops are triggered."
> "Wait for the first candle that closes with a clear rejection of the swept level—a strong wick and a close back inside the prior range. This candle is your entry signal."
> (FAQ) "A liquidity trap (fake breakout) shows a rapid spike through a key level, delta divergence..., and a swift rejection candle that closes back inside the prior range."
> (levels) "the most consistently trapped zones are the previous day's high and low, the overnight session high and low (ONH/ONL), and weekly pivot levels."

**MECHANIC:** Pre-mark key levels (prior-day H/L, overnight H/L, session H/L, round numbers, weekly/floor pivots). FIRE a turn when a bar's high/low POKES THROUGH the level (exceeds it) but the bar (or the next) CLOSES BACK INSIDE the prior range with a rejection wick on the swept side. Entirely OHLCV for the price geometry; the article's order-flow confirmation (delta divergence, footprint absorption) is DATA-BLOCKED and would be OMITTED. Wick-size / how-far-through threshold [UNSPECIFIED]; which level set to use [declare]. Article stresses timing: enter AFTER the sweep candle closes, not during.

**DIRECTION SEMANTICS:** Leg ending (the sweep leg fails) → strong reversal leg in the opposite direction; new-leg direction = back inside the range, away from the swept level.

**PORTABILITY:** MODERATE — level construction is deterministic from prior sessions; the poke-and-reclaim is OHLCV. Order-flow confirmation unavailable (train year) but not required for the price-structure trigger.

**OVERLAP:** ROUND05 / PIVOT16 / ORB02 supply candidate LEVELS but none implements the sweep-then-reclaim TIMING; distinct and among the most promising (a genuine turn event, not a level touch).

---

## TURN-08 — Double_Triple_Top_Bottom
**SOURCE:** `traditional-technical-patterns-futures-trading.md`; supporting `trade-trend-reversals-futures.md`
> "A double top... forms when price tests a resistance level twice and fails to break through, often signaling a potential downward reversal. Traders often watch for the price to drop below the lowest point between the two peaks—known as the neckline—as confirmation of a trend change."
> "The inverse, a double bottom, indicates a potential upward reversal. After price tests a support level twice and rebounds, breaking through the neckline can suggest a move higher."
> "a triple top occurs when the price tests resistance three times without success... These patterns are generally considered stronger than their double counterparts due to the additional confirmation..."
> (trade-trend-reversals) "two tests of the same resistance (or support) level that both fail, signaling the market can't continue in the prior direction."

**MECHANIC:** Detect two (or three) swing highs at ~equal price (within tolerance `tol`) separated by an intervening swing low. FIRE the confirmed turn when price CLOSES below that intervening low (the neckline) — mirror for double/triple bottom (close above the intervening high). Price-equality tolerance tol, min bar-separation between the tops, and swing lookback all [UNSPECIFIED — declare]. Confirmation lags to the neckline break (turn already underway), so lead time may be small; test.

**DIRECTION SEMANTICS:** Leg ending (the second/third failed test) → reversal; new-leg direction confirmed at neckline break.

**PORTABILITY:** MODERATE — equal-extreme + neckline logic on pivots (OHLCV).

**OVERLAP:** Partial with HNS22 (both are failed-retest neckline patterns) and PIVOT16. Distinct pattern (2/3 equal tests vs 3-peak head).

---

## TURN-09 — HeadShoulders_Neckline_Break
**SOURCE:** `head-and-shoulders-chart-pattern-spot-potential-market-reversals.md`; `trade-trend-reversals-futures.md`; `futures-market-inflection-points-identified.md`
> "A critical confirmation of this pattern occurs when price breaks below the neckline—a support level connecting the lows between the left shoulder, head, and right shoulder. A break below the neckline on high volume signals a stronger bearish move."
> "Left shoulder: Marks the initial high in price after an uptrend, typically occurring on higher volume. Head: Forms as price moves to a new higher peak, often on lower volume. Right shoulder: Develops when price fails to reach the height of the head and starts declining, often on weaker volume than the left shoulder."
> "The projected move after a breakout is often equal to the distance between the head and neckline."

**MECHANIC:** Identify three pivot-highs where mid (head) > two shoulders and shoulders ~equal; a neckline through the intervening lows. FIRE turn confirmation on a CLOSE below the neckline (mirror for inverse H&S). Volume across shoulders should decline (left > head > right) — the article makes volume divergence a requirement, and we HAVE volume, so it is checkable. Measured-move target = head-to-neckline distance (feeds TURN-16). Tolerances / pivot lookback [UNSPECIFIED].

**DIRECTION SEMANTICS:** Leg ending → reversal at neckline break; new-leg direction = through the neckline.

**PORTABILITY:** MODERATE — pivot + neckline + volume, all OHLCV.

**OVERLAP:** HNS22 ALREADY EXISTS and scores dir-recall@2m ≈ 0.00 / 109 fires in the audit (near-dead on turns). Do NOT rebuild as-is; if pursued, the lever is the neckline-BREAK timing + the volume-divergence gate, not the geometry HNS22 already encodes. Flag: canonical but empirically weak here.

---

## TURN-10 — HeikinAshi_Color_Flip
**SOURCE:** `heikin-ashi-candlestick-charts-explained.md`
> "HA-close: average of the current open, high, low, and close. HA-open: average of the previous HA-open and previous HA-close. HA-high: maximum of the close, HA-open, and HA-close. HA-low: minimum of the low, HA-open, and HA-close."
> "The end of the trend or a trend reversal often displays as a change in color on smaller-bodied candles with wicks on both sides."
> "in a strong uptrend, there are no wicks on the bottom side of the green candle, and these candles tend to be longer-bodied green candles."
> "One trend trading element that can challenge traders is the presence of reversal colored candlestick bars during a strong trend, which can fool traders into exiting a trade prematurely... Heikin Ashi bars take more data points into account and will often recolor a bar based on the current trend."

**MECHANIC:** Recompute HA bars from OHLC via the four formulas quoted verbatim (fully specified — the only concept in this catalog whose formula the article gives completely). FIRE a turn when, after a run of `r` same-color HA bars (green up / red down), an HA bar RECOLORS to the opposite AND is small-bodied with wicks on both sides. Run length r and "small body / two-sided wick" thresholds [UNSPECIFIED — declare]. Caveat from the article: HA "may not reflect the actual bar prices, especially the closing bar price," and single opposite-color bars can be false — hence the small-body + both-wicks gate to reduce whipsaw.

**DIRECTION SEMANTICS:** Leg ending (trend color run breaks) → new leg = the new color.

**PORTABILITY:** EASY — a deterministic OHLC transform; formula given in full.

**OVERLAP:** None. Distinct from PTRN-ENGULF/HAMMER (raw candlestick shapes) — HA is a smoothed recoloring designed specifically to flag trend end. Strong candidate: fully specified, cheap, and no existing stream does it.

---

## TURN-11 — Parabolic_SAR_Flip (exit / trailing role)
**SOURCE:** `use-the-parabolic-sar-to-signal-when-a-trend-may-stop-and-reverse.md`; `trade-trend-reversals-futures.md`
> "When the dots switch from being below the price to above the price (ending a downtrend) or from being above the price to below the price (ending an uptrend), a reversal is signaled."
> "The indicator can also be used to help exit trends early, before a reversal. Some traders also use the distance between the parabolic SAR point and the price to calculate a trailing stop-loss order."
> "Acceleration... set to .02 by default; Acceleration max: user-defined; Acceleration step: Defines the increase each time the most recent extreme (high or low) is achieved; set to .02 by default."
> (weaknesses) "It is considered a lagging indicator and can be slow to react... It can give false signals in choppy markets... in range bound markets."

**MECHANIC:** Compute Parabolic SAR on OHLC (accel 0.02 default, step 0.02 default, max [UNSPECIFIED — user-defined]). FIRE a leg-ending mark when the SAR dot flips side relative to price (below→above = up-leg ends). Also usable as a trailing exit (price crossing the SAR line). Self-described as LAGGING and whipsaw-prone in chop.

**DIRECTION SEMANTICS:** Leg ending → stop-and-reverse; new-leg direction = the side the dots flip to.

**PORTABILITY:** EASY — SAR is standard OHLCV.

**OVERLAP:** SAR-23 already exists (audit: dir-recall@2m 0.13, recall@2m 0.22, 19.5k fires — the 2nd-best turn stream after RENKO24). The distinct angle vs the existing entry stream is using the FLIP purely as an exit/turn mark and/or as the SAR-distance trailing stop. Flag: likely mostly captured by SAR-23; verify SAR-23 already keys on the flip event before rebuilding.

---

## TURN-12 — Structural_Swing_Break (trailing structure)
**SOURCE:** `stop-loss-strategies.md`; `trade-trend-reversals-futures.md`; supporting `04_Mechanics_and_LogicGates.md` (ZigZag structural stop synthesis)
> (stop-loss) "Expert traders commonly place stop-losses beyond the most recent swing high (for short trades) or swing low (for long trades). If price breaks a key structural level, the trade thesis is invalidated—and holding on only increases risk."
> (trade-trend-reversals) "Prior structure breaks: In an uptrend, a reversal typically involves price breaking below a significant prior swing low; pullbacks don't."
> (stop-loss) "As a trade moves in their favor, experienced traders use trailing stops to protect gains without cutting the trade short."

**MECHANIC:** Track the most recent confirmed opposing swing (last higher-low in an up-leg; last lower-high in a down-leg). FIRE a leg-ending mark when price CLOSES beyond that swing (up-leg ends on close below the last swing low). This is the "structure break" definition of a reversal and doubles as a trailing exit that ratchets as new swings form. Swing/pivot detection lookback [UNSPECIFIED — declare]. Article distinguishes this (structural) from arbitrary/round-number stops.

**DIRECTION SEMANTICS:** Leg ending (trend structure broken) → new leg opposite; the break confirms the turn already began at the prior extreme (small/negative lead — test).

**PORTABILITY:** MODERATE — needs a pivot/zigzag detector (OHLCV).

**OVERLAP:** ZIGZAG is exactly a swing-pivot engine and SAR-23 approximates a trailing structure — HEAVY overlap. The audit shows ZIGZAG dir-recall@2m ≈ 0.00 (2586 fires) — a swing-break confirmation lands at/after the turn, so it may be structurally late. Do not rebuild ZIGZAG; if pursued, the question is whether the CLOSE-beyond-swing event adds lead over ZIGZAG's pivot timestamp.

---

## TURN-13 — Return_To_Mean_Completion
**SOURCE:** `mean-reversion-in-futures-trading.md`; `how-to-trade-with-volume-profile-part-1.md`; supporting `01_Volume_and_OrderFlow.md` (VWAP-Z synthesis)
> (mean-reversion) "Set stops and limits: Manage risk by setting a stop-loss and taking profits as price moves back toward the mean." / "Identify the mean: Use a 20-day simple moving average (SMA) as the baseline."
> (volume-profile) "In balanced markets, price tends to rotate around fair value. Your job isn't to predict a breakout—it's to recognize when the market is likely to revert back to where the most business was previously done."
> (volume-profile) "the POC often acts like a magnet, pulling price back toward the area of highest prior agreement." / "Whether price rotates toward the POC or opposite side of the range."

**MECHANIC:** For a snap-back (mean-reversion) leg, the leg COMPLETES when price returns to its reference mean: 20-period SMA (article default), session VWAP, or the volume-profile POC. FIRE a leg-ending mark when price touches / closes through the mean after having been stretched away from it. Which mean to use [declare]; SMA period given as 20 (daily in article — needs re-scoping to intraday) [confirm timeframe]. Direction-conditional: only meaningful for legs that BEGAN as an extension away from the mean.

**DIRECTION SEMANTICS:** Leg ending for a mean-reverting (snap-back) leg; the new leg is the fade completing at fair value. Direction-neutral as a standalone mark (fires wherever price meets the mean).

**PORTABILITY:** EASY (SMA/VWAP) to MODERATE (POC needs intraday volume profile).

**OVERLAP:** HEAVY with NMP z-exit (Z_EXIT=0.4752 = |z|→0 = price back at mean is the SAME event) and with VA13/VP01 (POC) and VWAP03. Likely already covered by NMP's exit threshold; include mainly to document that three independent articles frame "back to the mean" as the fade-leg terminator. Verify against NMP z-exit before building.

---

## TURN-14 — VolBand_Pierce_Reenter_Exhaustion
**SOURCE:** `adaptive-price-zones-indicator.md`; supporting `03_Volatility_and_Momentum.md` (ATR-fade + Bollinger synthesis)
> "Price touches or moves outside the upper band: May suggest a short-term overbought condition. Price touches or moves outside the lower band: Could point to an oversold scenario. Price re-enters the zone: Sometimes seen as a confirmation of a potential reversal or pause in momentum."
> "At its core, APZ applies a double-smoothed exponential moving average (EMA) to price, then builds upper and lower bands based on recent price movement."
> "It doesn't predict reversals; it simply provides a visual representation of volatility-adjusted price movement... False signals can occur, especially in strong trending markets."

**MECHANIC:** Build a volatility envelope — APZ (double-smoothed EMA centerline ± percentage-based bands) or Bollinger — and FIRE a leg-ending mark when price CLOSES OUTSIDE a band and then CLOSES BACK INSIDE (the pierce-and-reenter). Period, deviation/percentage value, EMA smoothing [UNSPECIFIED — article says "Play with the settings"]. Article is explicit it fails in strong trends (band-riding).

**DIRECTION SEMANTICS:** Leg ending (the stretched leg exhausts) → reversal/pause; new leg back toward the centerline.

**PORTABILITY:** EASY — EMA + bands on OHLCV.

**OVERLAP:** SQZ04 (Bollinger) exists but tests the SQUEEZE/breakout, not the pierce-reenter exhaustion; the synthesis already names an `is_apz_exhausted` concept. Overlaps the mean-reversion family (TURN-13) at the exit edge. Include as the volatility-band variant; check it is not a relabel of SQZ04.

---

## TURN-15 — Session_Close_Flatten_Time
**SOURCE:** `how-to-trade-the-close.md`
> "The final hour, sometimes referred to as the 'power hour,' is when institutional traders, retail participants, and algorithmic strategies converge to execute end-of-day orders, adjust positions, take profits, and offset losses."
> "volume is usually highest at the open and close of the daily trading session, with lower volume between—often referred to as the 'volume smile.'"
> "Volume concentration: Volume tends to spike as traders rush to flatten out their positions before the close." / "Trend resolutions or reversals: Intraday trends may either gain momentum or encounter significant resistance as participants react to key levels at the end of the day."

**MECHANIC:** A TIME-of-day prior, not a price trigger: turn probability is elevated in the final hour (flatten window) and around the open (the "volume smile" tails). Use clock/session-time as a conditioning FEATURE that raises the base rate of a turn, to be ANDed with a price trigger (TURN-06/07/10). No price rule by itself; exact power-hour window [UNSPECIFIED — declare, e.g. last 60 min of RTH].

**DIRECTION SEMANTICS:** Direction-neutral timing mark (raised turn hazard), not a directional call.

**PORTABILITY:** EASY — pure timestamp.

**OVERLAP:** None — SEASON12 is calendar seasonality (day-of-year), NOT intraday clock. Weak as a standalone (a soft prior); best as a gate/feature. Include so the reviewer sees the time-of-day hazard axis exists in the corpus.

---

## TURN-16 — Target_Opposing_Structure_Laddering
**SOURCE:** `how-to-trade-liquidity-traps-in-futures.md`; `swing-trading-strategies.md`; `head-and-shoulders-chart-pattern-spot-potential-market-reversals.md`
> (liquidity-traps) "Set targets at the opposing liquidity zone. Target the opposite side of the range. If you entered after a sweep of the overnight low, target the overnight high. Opposing liquidity zones provide natural reference points for profit targets..."
> (swing-trading) target column: "Next higher swing high or a measured move above the line"; "Prior swing high or the next Fibonacci extension."
> (head-and-shoulders) "The projected move after a breakout is often equal to the distance between the head and neckline."

**MECHANIC:** A leg is expected to COMPLETE (turn) at the next opposing structural reference: the opposing session/overnight extreme, the prior swing high/low, a Fibonacci extension, or a measured-move projection (e.g. H&S head-to-neckline distance). FIRE a leg-ending mark as price reaches a pre-computed opposing-structure target. Which reference set / Fib ratios [declare — Fib extension levels UNSPECIFIED here]. This is target-laddering: turns cluster at these levels.

**DIRECTION SEMANTICS:** Leg ending at the target; new leg opposite (or pause). Direction-neutral as a standalone level-arrival mark.

**PORTABILITY:** MODERATE — reference levels are deterministic from prior structure (OHLCV), but require the pivot/level machinery.

**OVERLAP:** FIB17 (Fib levels) and PIVOT16 (structural levels) supply candidate targets; distinct contribution is treating LEVEL ARRIVAL as a turn-timing mark rather than an entry. Overlaps ROUND05 as a "level touch" family.

---

## MOST PORTABLE + DISTINCT (my top 3)

1. **TURN-10 HeikinAshi_Color_Flip** — the ONLY concept whose trigger formula the article
   gives in full (all four HA equations verbatim), a cheap deterministic OHLC transform,
   and no existing stream computes it. Purpose-built to flag trend end (recolor + small
   body + two-sided wick). Best portability-to-novelty ratio.
2. **TURN-07 Sweep_And_Reclaim** — a genuine turn EVENT (poke a mapped level, close back
   inside with a rejection wick), fully OHLCV for the price geometry, distinct from every
   level-touch stream (which mark arrival, not the failure). The order-flow confirmation is
   DATA-BLOCKED but not required for the price trigger. Highest a-priori chance of true lead.
3. **TURN-06 Climax_Volume_Exhaustion_Bar** — uses volume (which we HAVE) as the OHLCV
   shadow of the DATA-BLOCKED delta/footprint exhaustion idea: volume spike at a fresh
   extreme + close in the far third. No existing stream keys on volume-spike-at-extreme +
   close-position; distinct from the raw candle-shape PTRN streams.

(Runner-up worth noting: TURN-02 Oscillator_Recross — EASY and a real timing refinement
over the level-based RSI06, but shares the oscillator family with an existing stream.)

---

## CONSIDERED AND REJECTED (negative space)

- **Candlestick reversal at extreme (hammer / shooting star / bullish-bearish engulfing)** —
  `trade-trend-reversals-futures.md`, `02_Structure_and_PriceAction.md` (Wick_Ratio>0.6).
  REJECT as a NEW stream: already built as PTRN-ENGULF and PTRN-HAMMER. Distinct value is
  only a location gate ("at a key level / after an extended move") = a filter on PTRN, not
  a new detector.
- **Ratchet fixed-tick trailing stop** — `configure-a-custom-trailing-stop.md` (profit
  trigger 3 ticks, trail 7 ticks, frequency 2 ticks). REJECT: GRAVEYARD says fixed-dollar/
  fixed-tick stops are measured net losers (per-trade fixed stop ≈ −$31/day; §4). The
  defensible trailing variants (SAR-distance, swing-structure) are already TURN-11/12.
- **VWMA↔SMA convergence (volume velocity)** — `01_Volume_and_OrderFlow.md`,
  `what-is-a-volume-weighted-moving-average-vwma.md`. Real exhaustion idea (shrinking VWMA−SMA
  spread = volume dropping out) but VWMA10 already exists and it is a SLOW warning, not
  ±1-2m precision. REJECT (lagging; overlaps VWMA10).
- **Golden / Death cross (50/200 SMA)** — `03_Volatility_and_Momentum.md`,
  `spot-market-trends-...macd`. REJECT: a macro regime flag on the order of days; useless at
  ±2m turn resolution.
- **ADX peak / falling as trend-end** — `03_Volatility_and_Momentum.md`,
  `directional-movement-index-explained.md`. REJECT: ADX08 exists and scored dir-recall@2m
  ≈ 0.00 (671 fires) in the audit; ADX is a lagging strength gauge, not a turn timer.
- **Bollinger squeeze breakout (as a turn)** — `03_Volatility_and_Momentum.md`, SQZ04.
  REJECT: the squeeze predicts EXPANSION/breakout, not a swing turn. The turn-relevant
  half (band pierce-and-reenter) is kept as TURN-14.
- **Value-area "80% rotation" probability** — synthesis `01_Volume_and_OrderFlow.md`.
  REJECT the numeric claim: `comms/001` audit found the 80% figure is IMPORTED / not in the
  source article. The article-faithful rotation idea (POC magnet, rotate toward opposite
  value) is kept, un-numbered, inside TURN-13/16.
- **Fibonacci retracement stall (38.2 / 50 / 61.8%)** — `02_Structure_and_PriceAction.md`,
  `fibonacci-trading-...`. REJECT as a turn: this times the END of a PULLBACK (continuation),
  i.e. a re-entry into the prior trend, not a swing reversal; and FIB17 exists. (The Fib
  EXTENSION as a target IS kept in TURN-16.)
- **Cup-and-handle / flags-and-pennants** — `traditional-technical-patterns-futures-trading.md`.
  REJECT: multi-bar/multi-day continuation formations; wrong horizon for a ±2m turn.
- **Standalone footprint absorption at a level** — `footprint-charts-guide.md`,
  `ninjatrader-order-flow.md`. REJECT for train year: DATA-BLOCKED (needs bid/ask volume per
  price level). Its OHLCV shadows are TURN-05 (blocked) and TURN-06 (portable).
- **VSA no-demand / no-supply / shakeout** — `volume-spread-analysis-and-fomo-of-strong-directional-moves.md`.
  REJECT: the article is a livestream teaser with no concrete, quantified rule to port
  (only qualitative "exhaustion / shakeout" language); would require inventing thresholds.
