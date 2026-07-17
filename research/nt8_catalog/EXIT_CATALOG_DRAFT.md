# EXIT CATALOG (DRAFT) — exit / trade-management mechanics mined from the exit-strategy corpus

> Purpose: mine the NEW exit-strategy corpus for causal EXIT and turn-timing mechanics
> portable to 5s/1m OHLCV on MNQ — trailing/stop rules WITH parameters, break-even moves,
> scale-out ladders, time-based exits, momentum-stall exits, and regime/chop filters used as
> exit context. Companion to `TURN_CATALOG_DRAFT.md` (16 turn concepts, TURN-01..16); this
> file catalogs the EXIT-side and does NOT re-derive turn concepts already there (overlap
> flagged per entry).
>
> Corpus scanned: `research/nt8_catalog/raw_articles_exits/*.md` (per `_INDEX.md`: 57 OK +
> 16 THIN + failed). Method: read `_INDEX.md` for per-file status → VERBATIM read of the 23
> highest-signal OK files (skipped FAILED/THIN/0-word rows and the OFF-TOPIC options pages
> `advanced/basic-options-strategies`, `three-types-options`). Files read in full or in
> load-bearing part: `wpi_trading-system-development-s7526d00k` (thesis, incl. its
> NinjaScript/EasyLanguage code appendices), `stop-losses-complete-guide` (buildalpha),
> `trade-exit-strategy` (tradesviz), `trading-exit-strategies` (axi), `protective-strategies`
> (atas), `optimizing-exit-conditions-project-sim-yee-kai` + `backtesting-trading` (quantinsti),
> `epat-project-...-candlestick-machine-learning...`, `futures-trading-strategies` (quantvps),
> `mustknow-simple-effective-exit-trading-strategies` + `trailingstop` (investopedia),
> `trading-the-average-true-range` (topstep), `end-of-day-trading-strategies` +
> `day-traders-guide-mastering-trade-exits` + `stop-trading-pl` (optimusfutures),
> `scaling-in-out-trading-strategies` (traderspost), `scaling-in-and-out-of-trades`
> (tradethepool), `earn2trade-trailing-stop-order` + `when-to-call-it-a-day` +
> `cutting-your-losses` (earn2trade), `short-term-reversal-with-futures` (quantpedia),
> `reddit_futurestrading_1jbgf9z...`, `github_ayb-ninjatrader-automated-trading-strategy`.
> Every quoted passage below is copied verbatim from the cited file.
>
> DATA CONSTRAINT (same as turn catalog): train year has 5s OHLCV only — NO tick / order-flow /
> footprint / bid-ask delta, and NO external symbols ($VIX, breadth). Concepts needing those are
> flagged DATA-BLOCKED.
>
> GRAVEYARD ANCHORS (read before trusting any fixed-exit concept):
> - `comms/091_2026-07-16_BRACKET_GRID_OLD_SCHOOL_VERDICT.md`: on OUR calibrated entries a sealed
>   SL/TP grid does NOT add expectancy — "A bracket mean +2.06 < plain 5m hold +2.87 ... B bracket
>   +0.83 < hold +1.23", verdict "fixed stops/targets subtract vs short holds here too. Old-school
>   lane: measured, closed." Brackets only RESHAPE (consistency at lower EV), they do not add edge.
> - MEMORY §4 GRAVEYARD: 7 fixed-exit families already measured as net losers (per-trade fixed stop
>   ≈ −$31/day; session-P&L stop −$79/day sig; vol-adaptive exit thresholds OOS −$112/day; etc.).
>   R-trigger reversal is STRUCTURALLY OPTIMAL for binary exit; only CONTINUOUS SIZING (B9) wins.
> Any concept that is a pure fixed-stop / fixed-target variant is tagged **GRAVEYARD-ADJACENT**
> with a one-line note on what (if anything) distinguishes it; if nothing does, it is in Rejected.
>
> Existing league streams referenced for overlap (do NOT double-build): SAR-23, ZIGZAG, CURVE,
> NMP / NMP-LAMBDA (z-exit; Z_EXIT=0.4752), tier-ladder negative-exit, SQZ04 (Bollinger/Keltner
> squeeze), VWMA10, RSI06, MACD07, plus the bracket sweep (doc 091).

---

## SUMMARY TABLE

| ID | Name | #src | Role | Portability | Overlap w/ existing |
|---|---|---|---|---|---|
| EXIT-01 | ATR_Trailing_Stop (Chandelier) | 4 | exit-only (trail) | EASY | GRAVEYARD-ADJACENT; distinct from SAR-23 (ATR vs accel) |
| EXIT-02 | Breakeven_Move | 3 | exit-only (risk-off) | EASY | GRAVEYARD-ADJACENT; not in any stream |
| EXIT-03 | ScaleOut_Ladder | 5 | exit-only (partial) | EASY→MODERATE | **tier-ladder (profit side); B9 continuous sizing** |
| EXIT-04 | Time_Stop / MFE_Duration | 2 | exit-only (clock) | EASY | partial TURN-15 (EOD); MFE-window time-stop is NEW |
| EXIT-05 | Keltner_Momentum_Decel_Reversal | 1 | turn (stop&reverse) | EASY→MODERATE | SQZ04 (bands), TURN-01/14; **full params** |
| EXIT-06 | Efficiency_Ratio_Chop_Filter | 1 | regime-context | EASY | none direct (no efficiency-ratio gate) |
| EXIT-07 | Squeeze / VIX_Agreement_Chop | 1 | regime-context | SQZ04 / DATA-BLOCKED | **SQZ04**; VIX = external data |
| EXIT-08 | Structural_Swing_Trail + ATR_buffer | 3 | exit-only (trail) | MODERATE | **ZIGZAG / SAR-23 / TURN-12**; buffer is the delta |
| EXIT-09 | MA_Trail / Ribbon_Flatten | 3 | exit-only (trail) | EASY | partial VWMA10, TURN-01 |
| EXIT-10 | ATR_Overshoot_MeanRevert_Exit | 2 | turn / profit-take | EASY | **NMP z-exit**, TURN-13 (ATR analog of |z|) |
| EXIT-11 | Opposite_Signal_Exit | 2 | turn (symmetric) | EASY | SAR-23 flip / TURN-11; label-chain premise |
| EXIT-12 | RR_Ratchet_Stop | 1 | exit-only (dynamic) | EASY | none direct |

Portability legend (our data, 5s OHLCV, no order flow / no external symbols): EASY = pure OHLCV
derivation; MODERATE = needs pivot/level detection or volume/vol normalization; DATA-BLOCKED =
requires tick/order-flow or an external symbol we do not have.

Role legend: exit-only = closes/reduces an open position, no directional call; turn = simultaneously
ends one leg and implies the opposite (exit AND next entry); regime-context = a gate/feature that
conditions WHEN an exit should fire, not a trigger by itself.

---

## EXIT-01 — ATR_Trailing_Stop (Chandelier-style volatility trail)
**SOURCE:** `trading-the-average-true-range.md` (topstep); `stop-losses-complete-guide.md` (buildalpha); `trailingstop.md` (investopedia); `github_ayb-ninjatrader-automated-trading-strategy.md`
> (topstep) "a daily stop-loss may be set at 1.5X or 2X the ATR. This gives an asset price freedom to vary naturally during a trading day, but still sets a reasonable exit position."
> (topstep) "The Average True Range is a moving average (typically 14-days) of the true ranges."
> (buildalpha) "ATR based Stop – risking a multiple of ATR units away from the entry price. This is dynamic as the stop will widen or narrow as ATR fluctuates. For example, two ATR units below the long entry. If ATR is 0.44 then the stop would be 0.88 below the entry."
> (buildalpha) "ATR based Trailing Stop – risking a multiple of ATR units but as a dynamic trailing stop."
> (investopedia) "The average true range over n number of days multiplied by y number of days. For example, you can set a stop equal to five days away from the most recent close, with each day represented by the average true range over the past 14 days. In a trend-following strategy, this is a useful way of letting the market run without getting stopped out by normal price swings that don't amount to a change in the trend."
> (github) "it will send an order to open a position and immediately send a target limit and stop order to close the position based on Average True Range. Additionally, the stop order will move as price moves in your direction."

**MECHANIC:** Trail a stop from the trade's high-watermark (long) / low-watermark (short) at `m × ATR(p)`. FIRE the exit when price retraces `m×ATR` off the extreme. Parameters GIVEN: ATR period `p` = 14 (topstep, investopedia "past 14 days"); multiplier `m` ∈ {1.5, 2} (topstep) or 2 (buildalpha example) or the "n-days-of-ATR" form (investopedia: `m` expressed as days-away, e.g. 5 × ATR(14)). Rescale `p` to intraday bars for us [declare — 14 daily bars ≠ 14 5s bars]. Recomputes each bar as ATR fluctuates. Buildalpha's own test: an ATR trail was neither uniformly best nor worst — "There is no best and everything is relative to the strategy and symbols traded."

**DIRECTION/ROLE SEMANTICS:** Exit-only. Trails behind an existing position; direction-neutral (mirror logic for shorts).

**PORTABILITY:** EASY — ATR + running extreme, pure OHLCV.

**OVERLAP:** SAR-23 is a trailing stop-and-reverse but keyed on the parabolic accel factor (0.02), NOT ATR — a distinct trail geometry (ATR widens in vol; SAR tightens with time-in-trend). **GRAVEYARD-ADJACENT**: it is still a stop overlay, and MEMORY §4 shows vol-adaptive exit thresholds lost OOS −$112/day ("fat-tailed peaks overshoot mean-based formulas"). Distinguisher vs the graveyard: this is a TRAILING ratchet from the high-watermark (protects open profit), not a fixed from-entry stop — but it must still beat the plain-hold baseline (doc 091) before it earns a slot.

---

## EXIT-02 — Breakeven_Move (risk-off after a trigger distance)
**SOURCE:** `protective-strategies.md` (atas); `trading-exit-strategies.md` (axi, strategy #8); `mustknow-simple-effective-exit-trading-strategies.md` (investopedia)
> (atas) "The Breakeven parameter sets a trigger for moving the stop-loss to a specified level. It can be set in ticks or percentages. By default, the value is set to 5 ticks. This means that if the price moves by 5 ticks in the desired direction, the stop-loss will be moved."
> (atas) "If Offset = 0, the stop will be moved precisely to the opening price. If Offset = 1 Tick (by default), then the stop will be moved 1 tick above the opening price for long positions or 1 tick below the opening price for short positions."
> (atas) "Cons: it may lead to prematurely closing trades that could have potentially been profitable."
> (axi) "Some look at their initial risk (1R) and move their stop to break even once the position is 1R in profit. So, if your initial risk is $500 (1R) and your position is now $500 (1R) in profit, you would move your stop to break even."
> (investopedia) "raise your stop to break even as soon as a new trade moves into a profit. This can build confidence because you now have a free trade."

**MECHANIC:** After price advances `trigger` in favor, move the stop to entry `± offset`. Parameters GIVEN: trigger = 5 ticks (atas default) OR +1R (axi); offset = 0 or 1 tick (atas default). Two-state: {risk-on} → (trigger met) → {breakeven, "free trade"}. Everything is specified; the only choice is trigger units.

**DIRECTION/ROLE SEMANTICS:** Exit-only. Removes downside risk on the runner; no directional call.

**PORTABILITY:** EASY — a distance counter + one stop move.

**OVERLAP:** Not implemented in any current stream. **GRAVEYARD-ADJACENT**: it is a stop mechanic and atas itself warns it "may lead to prematurely closing trades." Distinguisher: unlike a fixed from-entry stop it does NOT cap upside (the runner is free) and does NOT subtract from winners — it only converts losers-that-turned-green into scratches. That asymmetry is what the pure-stop graveyard families lacked; testable as a cheap overlay on B9-sized legs, but expect it to interact with R-trigger (which already recovers ~1R off the low) — it may just pre-empt the R-trigger and shave the runner.

---

## EXIT-03 — ScaleOut_Ladder (fractional profit-taking + runner)
**SOURCE:** `scaling-in-out-trading-strategies.md` (traderspost); `mustknow-simple-effective-exit-trading-strategies.md` (investopedia); `protective-strategies.md` (atas); `trade-exit-strategy.md` (tradesviz); `scaling-in-and-out-of-trades.md` (tradethepool)
> (traderspost) "A typical approach might involve taking 25% profits at a 2:1 risk-reward ratio, another 25% at 3:1, and letting the remainder run with a trailing stop."
> (investopedia) "Larger positions benefit from a tiered exit strategy, exiting one-third at 75% of the distance between risk and reward targets and the second third at the target. Place a trailing stop behind the third piece after it exceeds the target..."
> (atas) "closing of 50% of the position at the TP1 level, and the remaining 50% at the TP2 level."
> (tradesviz) "50% off at target + 50% trailing to EOD beats both all-in and all-out ... The optimal split depends on YOUR data."
> (tradesviz) "Taking 100% at target beats holding 100% to EOD (because losers get worse)."

**MECHANIC:** Split the position into `k` tranches, each closed at its own trigger, last tranche on a trail. Parameter SETS given (three distinct ladders): (a) 25% @ 2R, 25% @ 3R, 50% trail (traderspost); (b) 1/3 @ 75%-of-R-distance, 1/3 @ 1R-target, 1/3 trail (investopedia); (c) 50% @ TP1, 50% @ TP2 (atas / tradesviz). Triggers can be R-multiples or structural targets. tradesviz is explicit the split is data-fit, not universal.

**DIRECTION/ROLE SEMANTICS:** Exit-only. Reduces size as the leg matures; keeps a runner for the right tail.

**PORTABILITY:** EASY (R-multiple triggers) → MODERATE (structural-target triggers need level machinery).

**OVERLAP:** HEAVY with our tier-ladder negative-exit and, more importantly, with **B9 continuous remaining-amplitude sizing** — B9 already IS the optimal continuous version of "take some off as the move matures." MEMORY §4: "The only rewarded lever is the ENTRY filter" and pyramid-attenuation (C15) LOST at every recall budget despite AUC 0.883. A DISCRETE 25/25/50 ladder is the coarse version of what B9 does continuously; do not expect the discrete ladder to beat B9. Catalog it as the human-legible baseline B9 should dominate; if it does NOT dominate on a population, that is signal.

---

## EXIT-04 — Time_Stop / MFE_Duration (statistical-window exit)
**SOURCE:** `trading-exit-strategies.md` (axi, strategy #6); `trade-exit-strategy.md` (tradesviz)
> (axi) "System traders often test the strength of various entry techniques using a time stop. i.e. Exit after X number of bars from entry."
> (axi) "many traders implement a time stop that closes their position if there has been consolidation in a tight range over X number of sessions."
> (tradesviz) "time-stops - closing a position after a predetermined duration because the statistical window for profitability has expired. Time-stops are an institutional concept that almost nobody in retail trading talks about."
> (tradesviz) "If your data shows that winning setups typically hit peak profitability within 2-3 minutes, then holding those positions for 2 hours is exposing capital to reversion risk for no reason."
> (tradesviz) "Data shows: MFE duration peaks at ~20 min for your setups. Rule: Implement a time-stop. If the trade hasn't hit peak momentum within 25 min, close it — the statistical window is expired."

**MECHANIC:** Two variants. (a) Hard time-stop: exit `X` bars after entry regardless (axi). (b) MFE-duration time-stop: measure, per setup, the typical time-to-Maximum-Favorable-Excursion; if the trade has not made meaningful progress by ~1.25× that window, close it (tradesviz's "peak within 20 min → cut by 25 min"). Also the "tight-range consolidation for X sessions → exit" dead-trade variant. Parameter `X` / the MFE window is [UNSPECIFIED per instrument — MUST be measured on our own trade population first; the 20/25-min numbers are illustrative].

**DIRECTION/ROLE SEMANTICS:** Exit-only, clock-driven. No directional call; frees capital from stalled legs.

**PORTABILITY:** EASY — a bar counter (+ optional range test). We have `bars_held` (MINUTES) and K-horizon bar units already; NOTE the MEMORY §6 warning: bars_held is minutes, K-horizons are 5s units — a time-stop MUST be defined in the correct unit or it fires 12× too late (the B9-horizon bug).

**OVERLAP:** Partial TURN-15 (session-close flatten is a special-case time-stop at the EOD boundary). The MFE-DURATION time-stop is NEW — no existing stream cuts on "time since entry vs the setup's typical time-to-peak." MEMORY §4 note: day-level/holding-time fixes have mostly failed, BUT those were session-P&L and hour-of-day skips, not an MFE-window time-stop keyed to the leg's own excursion clock. Worth a measured pass; cheap.

---

## EXIT-05 — Keltner_Momentum_Decel_Reversal (stall-at-band stop-and-reverse)
**SOURCE:** `wpi_trading-system-development-s7526d00k.md` (§4.4 Forex Stop and Reverse + Appendix D code — the ONLY fully-parameterized turn mechanic in this corpus)
> (§4.4.3) "This system will place its first trade at the top or bottom of an optimized keltner channel; long at the bottom or short at the top. After that, it remains in the market, and when the price reaches the top or bottom of a channel again, it will decide whether to maintain its position based on momentum. If momentum is bullish at the top, or bearish at the bottom, i.e. the price is accelerating at the edge of a channel, the system will maintain the current position. If the price is decelerating based on momentum at the top or bottom of the channel, it will reverse the position, going short at the top and long at the bottom."
> (Appendix D code, Keltner Floors) "LengthKeltner( 28) ... NumATRs( 1.5) ... LengthMom( 6) ... Avg = AverageFC( Price, LengthKeltner ); Shift = NumATRs * AvgTrueRange( LengthKeltner ); LowerBand = Avg - Shift; Mom = Momentum( Price, LengthMom ); Accel = Momentum( Mom, 1 ); { 1 bar acceleration } ... if Setup and Mom < 0 and Accel < 0 then begin Sell Short..."
> (Appendix D code, Keltner Ceilings) "LengthKeltner( 22) ... NumATRs( 1.5) ... LengthMom( 11) ..."

**MECHANIC:** Build a Keltner channel = `EMA(price, L)` ± `1.5 × ATR(L)`. When price reaches the far band, read momentum `Mom = Momentum(price, Lmom)` and its 1-bar acceleration `Accel = Momentum(Mom, 1)`: if BOTH are decelerating against the leg (`Mom<0 ∧ Accel<0` at the low band), REVERSE; if accelerating, HOLD. Parameters FULLY GIVEN: L = 28 (floors) / 22 (ceilings), NumATRs = 1.5, Lmom = 6 (floors) / 11 (ceilings), Accel = 1-bar momentum-of-momentum. This is a momentum-second-derivative gate on a volatility band — the exit fires only when the push into the band is EXHAUSTING, not merely touching it.

**DIRECTION/ROLE SEMANTICS:** Turn (stop-and-reverse). Ends the current leg AND opens the opposite; new-leg direction = away from the pierced band.

**PORTABILITY:** EASY→MODERATE — EMA, ATR, and two nested momentums, all pure OHLCV. The two-sided asymmetric params (28/6 down vs 22/11 up) are a tell that they were optimized on EURUSD 60-min and will NOT transfer to MNQ 5s/1m — re-optimize or treat as structure, not constants.

**OVERLAP:** SQZ04 already computes Bollinger/Keltner; TURN-14 is the band pierce-and-reenter; TURN-01 is momentum rollover. This concept's DISTINCT contribution is the combination — band-touch AND momentum-deceleration (2nd-derivative) as the gate, with a fully specified parameterization to anchor an implementation (kills [UNSPECIFIED] holes that plague TURN-01/14). Best portability-to-specificity ratio of the turn-type exits.

---

## EXIT-06 — Efficiency_Ratio_Chop_Filter (trend-quality regime gate)
**SOURCE:** `wpi_trading-system-development-s7526d00k.md` (§2.9.2 + §4.3 Kaufman Efficiency Day Trader)
> "The Kaufman Efficiency Ratio is a measure of efficiency. It is calculated by dividing the price change by the absolute sum of the price movement. The result is a number between 0 and 1, with high values indicating a more efficient trending market. A higher efficiency means that during a time period, the price moves mostly in one direction."
> (§4.3.1) "A higher efficiency might indicate that a trend is stronger, and is more likely to continue in its current direction. A lower efficiency would indicate less confidence, and there would be a lower chance that the trend will continue."

**MECHANIC:** `ER = |close[t] − close[t−N]| / Σ|close[i] − close[i−1]|` over the last `N` bars (0..1). Use as EXIT CONTEXT: when ER is LOW the tape is chop (price oscillating, sum-of-moves ≫ net-move) → the "ride" thesis is weak → exit earlier / tighten / suppress re-entry; when ER is HIGH let the leg run. Lookback `N` [UNSPECIFIED — declare; Kaufman's canonical KAMA uses 10]. This is the direct, computable answer to the reddit practitioner's exact unsolved problem (EXIT-07) and to our own residual "irreducible chop cost" (MEMORY §4 day-level fixes).

**DIRECTION/ROLE SEMANTICS:** Regime-context (direction-neutral). A gate/feature ANDed with a price trigger; raises/lowers the exit hazard, does not fire alone.

**PORTABILITY:** EASY — two rolling sums over close, pure OHLCV, causal, cheap.

**OVERLAP:** None direct — no current stream is an efficiency-ratio regime gate. Related in spirit to Hurst (MEMORY §3: causal but LAGGING, warmup bug) and to variance_ratio (the DROPPED de-facto λ gate, MEMORY §0), but ER is a cleaner, bounded, single-window trend-quality scalar. Strong candidate as a CHOP GATE for the exit/re-entry decision rather than a standalone trigger; test it as a conditioner on B7/B9 actions.

---

## EXIT-07 — Squeeze / VIX_Agreement (practitioner chop detection)
**SOURCE:** `reddit_futurestrading_1jbgf9z_weekly-results-fully-automated.md` (a real automated-NQ trader thread; the highest-value tip is at comment depth 2)
> "One of the most frustrating things for me was trying to figure out what days were going to be choppy. If I could do that I could avoid a lot of my losses."
> "There's a couple ways to determine chop. Tried the Bollinger-Keltner combo? Also called the squeeze"
> "I used to have it in my strategy to check for chop but at least for my entries it was always squeezing too late and would squeeze for too long (aka lagging indicator) maybe I need to play with the setting some more."
> "I'd suggest charting the $VIX (tradestation symbol). Choppy days in my opinion are when your indicators are not in agreement. MA's, MACD, Bollinger Bands etc whatever you use. When I stay on the same side of the VIX I do well ... when I go agaist it ie up when VIX says down I suffer."

**MECHANIC:** Two practitioner chop signals. (a) Squeeze = Bollinger bands inside Keltner channels → low-vol compression → chop/pre-expansion; suppress trend-exit logic while squeezed. (b) VIX-agreement = trade only on the side the VIX confirms; "indicators not in agreement" = chop. Both are regime CONTEXT for whether to hold or bail. NO parameters given; the trader flags the squeeze as **LAGGING** ("squeezing too late, too long") — a self-reported failure mode.

**DIRECTION/ROLE SEMANTICS:** Regime-context, direction-neutral (VIX-agreement adds a directional veto).

**PORTABILITY:** Squeeze = overlaps SQZ04 (already built) and is self-described lagging. VIX-agreement = **DATA-BLOCKED** ($VIX is an external symbol; not in our 5s OHLCV; would need a VIX feed aligned to bar time). 

**OVERLAP:** SQZ04 IS the squeeze — do not rebuild; the value here is the corroboration that a live NQ bot operator found the squeeze too lagging for entry-timing chop detection (matches SQZ04's role as expansion predictor, TURN rejected note). Included mainly as evidence for EXIT-06 (a NON-lagging chop gate) being the better lane, and to log the VIX-agreement axis as DATA-BLOCKED-but-known.

---

## EXIT-08 — Structural_Swing_Trail + ATR buffer (anti-stop-hunt structure trail)
**SOURCE:** `stop-trading-pl.md` (optimusfutures); `mustknow-simple-effective-exit-trading-strategies.md` (investopedia); `futures-trading-strategies.md` (quantvps, trend-following)
> (stop-trading-pl) "A futures trader who follows this approach looks for swing highs during a downtrend, and as long as the price keeps making lower highs, the trend is still functioning. When price breaks a previous swing high during a downtrend and suddenly starts making higher highs, a change in trend direction is likely, and a trade exit should be considered."
> (stop-trading-pl) "a trader never puts his stop directly at the moving average or swing point, but always adds some extra space depending on market fluctuations (some futures traders use ATR). This allows the futures trader to avoid volatility spikes and stop 'hunting.'"
> (investopedia) "Algorithms now routinely target common stop-loss levels ... As a general rule, an additional 10 to 15 cents should work on a low-volatility trade, while a momentum play may require an additional 50 to 75 cents."
> (quantvps) "Exits occur when momentum weakens or the trend structure fails."

**MECHANIC:** Trail behind the last confirmed opposing swing (last higher-low in an up-leg; last lower-high in a down-leg); FIRE the exit on a close beyond that swing — BUT place the actual trigger a buffer `b` BEYOND the swing, where `b` scales with volatility (ATR) to dodge stop-runs. Buffer sizes GIVEN as instrument-scale hints (investopedia: +10–15¢ low-vol, +50–75¢ momentum) → the portable form is `b = c × ATR` [c UNSPECIFIED — declare]. Swing-detection lookback [UNSPECIFIED].

**DIRECTION/ROLE SEMANTICS:** Exit-only trail that ratchets with structure; the swing-break also confirms a turn (small/negative lead — same lag caveat as TURN-12).

**PORTABILITY:** MODERATE — needs a pivot/zigzag detector + ATR, all OHLCV.

**OVERLAP:** HEAVY with ZIGZAG (a swing engine), SAR-23 (a trailing structure), and **TURN-12 (Structural_Swing_Break)** — TURN-12 already encodes the close-beyond-swing exit. The DISTINCT delta here is the ATR BUFFER placed beyond the structural level to survive volatility spikes / stop-hunts — TURN-12 does not specify a buffer and the audit showed ZIGZAG dir-recall@2m ≈ 0.00 (lands at/after the turn). If pursued, the buffer is the only new lever; the structure timing is already covered.

---

## EXIT-09 — MA_Trail / Ribbon_Flatten (moving-average trailing exit)
**SOURCE:** `stop-trading-pl.md` (optimusfutures); `trailingstop.md` (investopedia); `futures-trading-strategies.md` (quantvps, MA Ribbon)
> (stop-trading-pl) "If you are a day trader, you should use the 20-day moving average as one of your tools for price context. Short-term moving averages (usually below the 20-period setting) are more 'erratic' and susceptible to noise, but they will also get you out of trades faster."
> (investopedia) "A close above or below a specific moving average, such as the eight-, 20-, or 50-day trailing stops."
> (quantvps) "When the ribbon expands and aligns in one direction, it signals strong trend structure. When it compresses or flattens, it often signals weakening momentum or a potential transition phase. ... exits are taken when the ribbon begins to contract or flatten, signaling a loss of momentum."

**MECHANIC:** Two forms. (a) Single-MA trail: exit on a CLOSE beyond a trailing MA (periods GIVEN: 8, 20, 50; 20 recommended for day-traders; <20 = faster/noisier exits). (b) MA-ribbon: plot a fan of MAs; exit when the fan CONTRACTS/FLATTENS (spread → 0), i.e. the multi-timeframe agreement collapses = momentum stall. Ribbon spacing/periods [UNSPECIFIED — declare the fan].

**DIRECTION/ROLE SEMANTICS:** Exit-only trail; the ribbon-flatten variant doubles as a momentum-stall turn read (a leg-end mark).

**PORTABILITY:** EASY — MAs on close, pure OHLCV.

**OVERLAP:** Partial with VWMA10 (tests VWMA direction, not a close-vs-MA trailing exit) and with TURN-01 (the ribbon-flatten IS a momentum rollover, expressed as MA-fan compression). Distinct as an explicit MA-CLOSE trailing exit and as the fan-spread stall metric. The single-MA close-trail is the cheapest exit in the corpus; worth a baseline pass vs the ATR trail (EXIT-01) and R-trigger.

---

## EXIT-10 — ATR_Overshoot_MeanRevert_Exit (excursion-based profit-take)
**SOURCE:** `trading-exit-strategies.md` (axi, strategy #5 "Large daily move"); `trading-the-average-true-range.md` (topstep)
> (axi) "One technical indicator that many professional traders love to use is the Average True Range (ATR). ... When one of your trades moves in your favour outside of this, you need to proactively manage the trade, so you don't give back too much open profit. Depending on your trading timeframe, if there's a large daily move of, say, 300 pips, you could look to sell as the market has most likely overshot."
> (topstep) "Depending on a trader's timeframe, a move beyond current ATR levels would indicate a change in market trend."
> (topstep) "if the historic ATR contracts while prices are trending upwards, then this might indicate that market sentiment may turn."

**MECHANIC:** When the favorable excursion of the open leg exceeds `m × ATR` (the move has "overshot" its normal range), take profit / exit into the stretch rather than trailing — a fade-the-overextension exit. Also topstep's contracting-ATR-into-trend as an exhaustion tell. Multiplier `m` [UNSPECIFIED — declare; the "300 pips" is instrument-specific, not portable].

**DIRECTION/ROLE SEMANTICS:** Turn / profit-take. Exits a stretched leg anticipating mean-reversion; direction-conditional (only meaningful once the leg is extended away from its mean).

**PORTABILITY:** EASY — ATR + excursion, pure OHLCV.

**OVERLAP:** This is the ATR analog of **NMP z-exit** — |z| measures excursion in std-of-residual units; `move/ATR` measures it in ATR units. Z_EXIT=0.4752 (back-to-mean) and this "moved > m×ATR → overshot" are near-duplicates from different normalizers. HEAVY overlap with NMP and TURN-13 (return-to-mean). Include to document that the corpus independently frames "moved too far, fade it" as an exit; verify it adds nothing over the already-calibrated NMP z-exit before building.

---

## EXIT-11 — Opposite_Signal_Exit (symmetric counter-signal close)
**SOURCE:** `epat-project-oil-commodity-futures-candlestick-machine-learning-strategy-chytil-mario.md`; `wpi_trading-system-development-s7526d00k.md` (§4.2.3 Whole Breakouts)
> (epat) "In the strategy, the counter signal of the same pattern might also be used for exiting the position (supposing that predicts a trend-reversal against the current trend)."
> (wpi) "Finally, it uses a Parabolic SAR rule, combined with simple moving average. If the two indicators align to signal an exit, then the system exits the trade, regardless of profit or loss."

**MECHANIC:** Exit when the entry logic fires in the OPPOSITE direction (the same detector's mirror signal). No separate exit rule — the exit IS the next entry's trigger. WPI's variant requires TWO indicators (SAR flip AND SMA) to AGREE before exiting "regardless of profit or loss." This is exactly the label-chaining premise stated in the turn catalog intro ("every AI-label turn is simultaneously an exit and the next entry").

**DIRECTION/ROLE SEMANTICS:** Turn (symmetric). The exit and the opposite entry are the same event.

**PORTABILITY:** EASY — reuses whatever detector produced the entry; no new machinery.

**OVERLAP:** SAR-23 flip (TURN-11) is precisely this for the SAR detector. Generalizes to any stream: "hold until the mirror fires." Distinct as a PRINCIPLE (agreement-gated symmetric exit) rather than a new detector; the WPI "two-indicator agreement before flip" is the useful refinement (reduces whipsaw vs a single-detector flip). Overlaps every reversal stream by construction.

---

## EXIT-12 — RR_Ratchet_Stop (maintain minimum reward:risk)
**SOURCE:** `trading-exit-strategies.md` (axi, strategy #12)
> "With a risk/reward stop, you adjust your stop loss to maintain a minimum risk/reward of 1:1 at all times. This powerful approach helps you to maintain your profits if your trade gets close to your profit target, but does not touch it, then reverses."

**MECHANIC:** As open profit grows, ratchet the stop so the remaining risk never exceeds a fixed fraction of captured profit — hold R:R ≥ 1:1 (or a chosen ratio) at all times. FIRE the exit when a pullback would breach that ratio. Target ratio [1:1 given; generalizable]. A profit-scaled trail (distance = f(open profit)) rather than a price- or ATR-scaled trail.

**DIRECTION/ROLE SEMANTICS:** Exit-only, dynamic trail keyed to accumulated profit.

**PORTABILITY:** EASY — open-PnL and a ratio, pure bookkeeping.

**OVERLAP:** None direct. Conceptually adjacent to the tier-ladder and to B9 (both protect captured amplitude), but the trigger is a fixed R:R invariant, not a horizon or a size taper. Cheap to test; likely dominated by B9/R-trigger but distinct enough to log. Note MEMORY §4: "cut-and-bank a winner LOSES — hold−cut EV positive at every level" — an R:R ratchet is a soft cut-and-bank, so expect it to underperform holding; measure against R-trigger.

---

## METHODOLOGY NOTES (how to TEST exits — from the WPI thesis + quant sources)
> Max 10; these are evaluation methods, not exit rules. Several directly corroborate our graveyard.

1. **Exit efficiency, risk-constrained** (`trade-exit-strategy.md`): `Exit Efficiency = Actual PnL / Best Possible PnL × 100`, where "best" = the best price reachable AFTER the last execution *without ever exceeding the maximum drawdown (MAE) you actually experienced* — NOT the day's high (that is cherry-picking). Constraining best-exit by realized MAE is the honest ceiling. "Most traders sit between 35-55%."
2. **MFE/MAE duration → time-stops** (same): measure time-elapsed from entry to Maximum-Favorable-Excursion per setup; if winners peak by ~X min, a trade not moving by ~1.25X min is dead capital. This is the measured basis for EXIT-04 — derive X from OUR trades, do not import the 20/25-min figures.
3. **EOD-hold benchmark** (same): always compare active exit management against the dumbest alternative — "hold every trade until close." "If EOD beats you, your 'active management' is just active destruction of edge." A free null model for any exit rule.
4. **Optimizing TP/SL overfits** (`optimizing-exit-conditions-project-sim-yee-kai.md`): 70:30 train/test, variable TP (1.01→1.10) × SL (0.99→0.89); result — "The supposedly 'optimized' exit conditions appear to underperform as compared to the 'control' on multiple occasions ... the idea of an optimal TP and SL combination fails to work in this instance." Independent confirmation of doc 091's bracket verdict.
5. **Stops can be net-negative; strategy-type-dependent** (`stop-losses-complete-guide.md`): on a 2-period-RSI mean-reversion strategy "the version without a stop produced 2x the net profit in backtests"; "Mean reversion strategies are often harmed by stop losses where trend following strategies are often improved. ... There is no best stop loss strategy. Each strategy requires individualized testing." Our zigzag/NMP is mean-reversion-flavored — this predicts stops hurt, matching MEMORY §4.
6. **WPI Kaufman stop-loss failure** (`wpi_...`, §4.3.4): "the stop losses would always trigger at inopportune times. The problem was that adjusting a stop loss would just shift the problem from one spot to another. With a less sensitive stop loss, it would stay in trades too long, and with an over sensitive stop loss, it would exit too fast ... so the system was tested without any stop losses." A second independent replication of the fixed-stop graveyard.
7. **Walk-forward analysis** (`wpi_...` §2.9.4; `backtesting-trading.md`): optimize on an in-sample window, validate on the NEXT out-of-sample step, roll forward ("optimized over 7 weeks, then tested on the most-recent week ... each week the time frame stepping forward"). The correct anti-overfit protocol for any exit-parameter search.
8. **Backtest bias checklist** (`backtesting-trading.md`): guard look-ahead bias, survivorship bias, overfitting/optimization bias, and ignoring trading costs; intraday strategies need "a backtesting period of 3-4 years." (Look-ahead is OUR central scar — MEMORY §3.)
9. **Expectancy / R-multiple / system quality** (`wpi_...` §2.9.5): `R-multiple = trade PnL / risk`; `Expectancy = mean(R-multiples)`; use average-loss OR max-loss as "risk"; expectunity = annualized. A clean cross-exit-rule comparison metric (pairs with our PF-based Trade WR).
10. **Lagged-signal confirmation** (`epat-...`): wait one bar after a pattern before acting, to let volume/direction confirm — reduces false positives. Relevant to the turn-timing lead/lag tradeoff (a confirmed turn lands later; quantify the lead lost vs whipsaw avoided).

---

## MOST PORTABLE + DISTINCT (my top 3, with parameters)

1. **EXIT-05 Keltner_Momentum_Decel_Reversal** — the ONLY turn/exit mechanic in the corpus with a
   COMPLETE parameterization (verbatim NinjaScript-style code): band = `EMA(price, 28↓/22↑) ±
   1.5×ATR(28↓/22↑)`; `Mom = Momentum(price, 6↓/11↑)`; `Accel = Momentum(Mom, 1)`; reverse when
   `Mom<0 ∧ Accel<0` at the low band (mirror at high). It gates the exit on momentum DECELERATION
   (2nd derivative) at a volatility band — fires on exhaustion, not mere touch. Fills the [UNSPECIFIED]
   holes that weaken TURN-01/TURN-14. EASY→MODERATE, pure OHLCV. (Re-optimize the params for MNQ.)
2. **EXIT-06 Efficiency_Ratio_Chop_Filter** — `ER = |close[t]−close[t−N]| / Σ|Δclose|` over N,
   bounded 0..1, high=trend / low=chop. A cheap, causal, NON-lagging trend-quality gate — the direct
   answer to the reddit operator's unsolved "which days are choppy" problem AND to our residual chop
   cost. EASY, pure OHLCV. Distinct: no existing stream is an efficiency-ratio regime gate. Use as an
   AND-gate/feature on B7/B9 exit actions, not a standalone trigger. (N: declare; KAMA canonical = 10.)
3. **EXIT-04 Time_Stop / MFE_Duration** — cut any trade that has not reached its setup's typical
   time-to-peak window (institutional time-stop). EASY (a bar counter), no existing stream does it, and
   it targets dead-capital legs that trailing/price stops ignore. Parameter MUST be measured on our own
   MFE-duration distribution (illustrative 20-min-peak → 25-min-cut). CAUTION: define the window in the
   correct time unit (minutes vs 5s K-units) or it fires 12× too late (the B9-horizon bug, MEMORY §6).

(Runner-up: EXIT-02 Breakeven_Move — fully specified (atas 5-tick trigger, 1-tick offset), EASY, and
uniquely does NOT cap the runner; but it likely just pre-empts the R-trigger, so measure the interaction.)

---

## CONSIDERED AND REJECTED (negative space)

- **Fixed dollar / fixed-percentage stop-loss** — `stop-losses-complete-guide.md` ($100/$200, 1%/2%),
  `cutting-your-losses.md` ("the 2% rule ... risk no more than 2% of your account equity"),
  `protective-strategies.md` (atas math stop "0.1% of the asset's price"). REJECT: pure fixed stop; doc
  091 + MEMORY §4 (per-trade fixed stop ≈ −$31/day) + WPI Kaufman + buildalpha's own mean-reversion test
  all measure it as EV-subtractive on populations like ours. No distinguisher.
- **Fixed take-profit / bracket (SL+TP) as an edge** — `optimizing-exit-conditions-...`, WPI Whole
  Breakouts profit target, axi #9. REJECT as additive: doc 091 measured brackets on OUR entries — "A
  bracket mean +2.06 < plain 5m hold +2.87 ... fixed stops/targets subtract vs short holds here too."
  Brackets only RESHAPE to consistency at LOWER EV; kept as a reward-design note, not an edge.
- **Session-P&L / daily-loss-limit stop ("call it a day")** — `when-to-call-it-a-day.md`
  ("stringent stop-loss limits", flat 3:10–5 PM CST), `stop-trading-pl.md`, `cutting-your-losses.md`
  ("no more than 5% across all positions"). REJECT: MEMORY §4 measured the intraday session-P&L stop at
  −$79/day CI[−154,−22] sig LOSS (81% of stopped OOS days recover). Also mostly discipline prose.
- **Short-term weekly reversal (volume/open-interest conditioned)** — `short-term-reversal-with-futures.md`
  ("high-volume low-open interest ... Period of Rebalancing: Weekly"). REJECT: wrong horizon (Wed–Wed
  weekly, not ±1–2 min) and open-interest is not in our intraday OHLCV (DATA-BLOCKED). The one portable
  idea — reversal magnitude scales with volume/overreaction — is already implied by TURN-06 (climax).
- **End-of-day / one-trade-a-day swing hold** — `end-of-day-trading-strategies.md` (Larry Williams;
  "three up closes → go short ... close at end of day"). REJECT for horizon (daily swing). The EOD-FLATTEN
  time boundary is retained inside EXIT-04 / TURN-15, but the EOD ENTRY strategies are out of scope.
- **Gap-midpoint exit** — `trading-exit-strategies.md` (axi #7, "put your exit at the midpoint of that
  session's candle"). REJECT: gap-specific, daily-candle framing; niche, not a recurring intraday trigger.
- **Fundamental / news exit** — axi #11 ("exit a position following negative news"). REJECT: not a
  causal-OHLCV mechanic (external/discretionary; LLM-as-FEATURE only per MEMORY §4).
- **Account-target / month-goal exit** — axi #13 ("close all positions once this target is achieved").
  REJECT: account-level bookkeeping, not a per-trade exit signal.
- **Hedging / spread / calendar exits** — `futures-trading-strategies.md` (quantvps #13/#15). REJECT:
  portfolio risk-offset and relative-value, not an intraday exit-timing mechanic for a single MNQ leg.
- **Position sizing (Van Tharp / Optimal F / fixed-fractional)** — `stop-losses-complete-guide.md`,
  `end-of-day-trading-strategies.md`, WPI. REJECT: sizing, not exit timing (out of scope; and B9/B10
  already own the sizing lever).
- **Rigid-exit psychology / disposition-effect prose** — `day-traders-guide-mastering-trade-exits.md`,
  `trade-exit-strategy.md` (disposition effect), `when-to-call-it-a-day.md`, `what-is-revenge-trading`
  (per index). REJECT per scope: discipline/psychology, no quantified causal rule. (Valuable framing:
  "most traders exit winners too early and hold losers too long" — but it is a bias, not a detector.)
- **Displaced / MA-cross exit variants** — axi (#1 displaced MA), `backtesting-trading.md` (golden/death
  cross). REJECT/fold: displaced-MA is a variant of EXIT-09; the 50/200 cross is a multi-day macro flag,
  useless at ±2m resolution (same as the turn catalog's golden/death-cross rejection).
