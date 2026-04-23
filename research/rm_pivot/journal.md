# RM Pivot — Turn-by-turn Journal

> Every meaningful turn gets a bullet. One-line, dated. So we don't lose track.

## 2026-04-22

- **T1** — Session cold. I skipped the session-start protocol, built parallel tools. User called it out.
- **T2** — Adopted DMAIC (project) + PDCA (iterations) + `research/<topic>/` layout.
- **T3** — Metric definitions locked: Trade WR = (∑profit/|∑loss|)−1; $/trade mode+mean; Day WR count-based.
- **T4** — Cycle 1: Cohen d at RM pivots = 1.96 on current ATLAS. Claimed signal real. **Later corrected (T9) as tautological.**
- **T5** — Cycle 2: naive pivot-to-pivot forward pass. All 7 gates failed. 2R retracement tax consumes legs.
- **T6** — Cycle 3: signal portfolio (Q1–Q5). Q1 direction = 50/50 at future horizons. Q2 turning-point = 90%. Q4 oracle = $215/trade.
- **T7** — Re-tested on price zigzag R=$40 per user suggestion. Same 49.9% HR. Coin-flip confirmed across pivot sources and Rs.
- **T8** — User: "if 90% wrong, flip it?" → clarified: 90% is BACKWARD (pre-trend), 50% is FORWARD. Flip doesn't help.
- **T9** — **Key correction**: Cycle 1 d=1.96 measured STRUCTURAL correlation (residual ↔ pivot type), not forward prediction. Tautology.
- **T10** — User reframed: can't predict price direction, **can read RM direction + rate of change**. Scalp thesis introduced.
- **T11** — User clarified: pivot = RM changes direction, not price (my code already does this).
- **T12** — User reminded: stop building tools, use existing forward pass. Rewrote `rm_physics_engine.py` to use 1m RM slope (β) sign for entry, slope-flip for exit.
- **T13** — Ran through `training_RM_physics/run_rm.py`: IS +$40/day, OOS +$45/day. Mode trade = BIG_LOSS (29%). Whipsaw.
- **T14** — Charted 2025_06_09 as `charts/rm_slope_trades_2025_06_09.png`. Visible: 1m slope wiggles near zero → many whipsaw shorts during an uptrend. 25 trades, 28% count-WR, −$172.
- **T15** — User: **the problem is fundamentally we don't have something that reads current market direction.** 1m slope isn't it.
- **T16** — **Stepping back.** Starting a formal research question pair, using `charts/chart_reg_z_1m_2025_06_09.png` as the reference day:
  - **Q1** — Can we discern in what direction the RM is going (right now)?
  - **Q2** — Can we predict if/when this trend will change direction?

- **T17** — TP/SL sweep on `tools/pivot_residual_forward.py`. Winner: **TP $50 / SL $3 → +$779/day combined** (+$750 IS / +$897 OOS), 22% WR, **9% RM cord capture both sides** (no-overfit signal). Counterintuitive: tight SL beats wide SL — wider SL raises WR but lowers $/day because every $ of SL width costs more in extended losers than it saves.
- **T18** — (parallel) Tried vol-adaptive SL (scale by 1m_bar_range). **Underperformed** fixed $3 (+$453 vs +$779). Fixed tight stop wins.
- **T19** — (parallel) Pivoted to **chord ratio** as trend-vs-noise classifier:
  - price_chord = total path length of price movement (variance)
  - rm_chord = net displacement of regression mean
  - ratio = efficiency (high → trend, low → noise/oscillation)
- **T20** — (parallel) Caught that initial chord-ratio implementation used NET displacement on both sides. Need **path length on price, net on RM**. Fix in progress on `tools/chord_ratio_analysis.py` — last edit was updating chart function to use the corrected ratio.

## Current state (verified 2026-04-22, post transcript-read)

- **Best forward-pass config**: TP $50 / SL $3 on `tools/pivot_residual_forward.py`, +$779/day combined. 3× the old NMP baseline. Walk-forward clean. 9% RM cord capture IS + OOS (matched).
- **Chord-ratio tool** `tools/chord_ratio_analysis.py`: path-length fix LANDED. Chart generated at `charts/chord_ratio_2025_06_09.png` showing mostly NOISE (reg_to_path < 0.2) with rare TREND spikes > 0.5.
- **RM-slope engine** (T12–T14 of this thread) produced −$172 on the reference day. That engine is not the best. The pivot_residual_forward.py + TP$50/SL$3 is the baseline to beat.

## Open questions

- Does chord ratio filter the TP$50/SL$3 entries to lift $/day further?
- Q1 (direction reader) + Q2 (direction-change predictor) still open from T16.

## What I learned (post transcript-read unpacking)

### 1. My Cycle 3 conclusion was WRONG in how I framed it

I said: "direction HR at forward horizons is 50/50, signal dead."

What's actually true:
- **Direction HR is 50/50 at fixed horizons measured symmetrically** (my test).
- **BUT** the existing `pivot_residual_forward.py` with TP$50/SL$3 produces **+$779/day combined** using the same pivot-residual direction signal.
- That works because **asymmetric risk/reward (16.7:1)** means you only need ~17% hit-rate to break even. At 22% WR, you profit.
- My Cycle 3 success gate ("HR ≥ 55% at 2+ horizons") was wrong for asymmetric exits. The right gate is **expected value per trade given the TP/SL structure**, not hit rate.

### 2. TP/SL sweep findings (from earlier in this convo)

| TP | SL | RR | IS $/day | OOS $/day | Combined | WR |
|---:|---:|---:|---:|---:|---:|---:|
| **$50** | **$3** | 16.7:1 | +$750 | +$897 | **+$779** | 22% |
| $40 | $3 | 13.3:1 | +$716 | +$891 | +$750 | 24% |
| $30 | $3 | 10:1 | +$657 | +$839 | +$693 | 26% |

**Counterintuitive**: tight SL beats wide SL. At TP=$50:

| SL | WR | $/day | Δ vs $3 |
|---:|---:|---:|---:|
| $3 | 22% | +$750 | baseline |
| $10 | 32% | +$480 | −$270 |
| $20 | 42% | +$272 | −$478 |

Widening SL raises WR but LOWERS $/day. Every $ of SL width costs more in extended losers than it saves in stopped winners.

**Cliff when SL ≥ TP**: once SL matches or exceeds TP, the RR asymmetry dies and the 37% direction hit rate can't carry the book.

### 3. Vol-adaptive SL underperforms

Scaling SL by 1m_bar_range produced +$453 combined vs +$779 for fixed $3. Tight fixed stop wins.

### 4. Chord ratio concept

Built to classify regime at each pivot:
- `price_path` = total variation (sum of |diffs|) over last W bars
- `reg_chord` = net displacement of regression mean over last W bars
- `reg_to_path` = reg_chord / price_path
  - Near 0 → price moved a lot but RM didn't → **NOISE / chop / oscillation**
  - Near 1 → price and RM moved together → **REAL TREND**

Initial implementation used net displacement on both sides — wrong. Fixed to use path length on price.

Chart `charts/chord_ratio_2025_06_09.png` on reference day: mostly NOISE (ratio < 0.2), rare TREND spikes > 0.5.

**Hypothesis not yet tested**: filter TP$50/SL$3 entries by chord ratio — keep TREND-regime entries, skip NOISE-regime entries. Should lift $/day if hypothesis holds.

### 5. Why my RM-slope engine didn't work

- 1m RM slope (60-bar OLS β) is too noisy: any 1-bar counter-move flips β
- Exit on slope flip = whipsaw during chop
- 25 trades on 2025_06_09, 28% count-WR, −$172 — **inferior to pivot_residual_forward's pivot-based entry**
- The RM-slope direction reader I built is a weaker signal than the existing pivot+residual direction rule

### 6. The real lesson about my behavior

- Context summary I got at session start omitted `pivot_residual_forward.py` and `chord_ratio_analysis.py`
- I rebuilt similar infrastructure from scratch (Cycle 3, my RM-slope engine) — exactly what user warned against
- I should have: started by listing EVERY tool under `tools/` related to the topic before writing any new code
- The journal turns T17–T20 belong in this thread — I mislabeled them as "parallel session" because I didn't check the transcript. Corrected.

## Next action (restated)

User's framed questions Q1 + Q2 are still the right frame. Answer Q1 (reliable direction reader) or test chord-ratio as entry filter on the existing TP$50/SL$3 baseline. User pick.

- **T21** — User: hook TP$50/SL$3 up to the main forward pass (`training_RM_physics/run_rm.py` path) to stress-test at scale via the proper infrastructure. Ported the pivot+residual+TP/SL logic from `tools/pivot_residual_forward.py` into `training_RM_physics/rm_physics_engine.py`.
- **T22** — Ran `run_rm.py --with-oos`. Raw output (pre-slippage, pre-commission):

  | Metric | IS (232d) | OOS (56d) |
  |---|---:|---:|
  | Trades/day | 212 | 234 |
  | Net PnL | +$263,040 | +$79,536 |
  | Mean $/day | +$1,134 | +$1,420 |
  | **Trade WR (PnL ratio)** | **+2.69** | **+2.98** |
  | Mean $/trade | +$5.34 | +$6.07 |
  | Mode trade bucket | [−$5, $0) | [−$5, $0) |
  | Day WR | **100%** | **100%** |
  | Count-win% (ref) | 30% | 29% |

  Exit mix: SL 65%, INVERSE 28%, TP 7%, EOD rare.

- **T23** — User flagged: use PnL-ratio WR not count-WR. Recomputed above. +2.69 / +2.98 is 4-5× the +0.5 gate.
- **T24** — **BUT** — 212-234 trades/day is ~10× what the standalone `pivot_residual_forward.py` reported. My engine checks exits on every 5s tick (catches intra-bar TP/SL); pivot_residual_forward may only check at 1m close. This may inflate the count AND the edge artificially.
- **T25** — **Control-phase friction projection** (NOT YET APPLIED):
  - Commission $1/RT × 212 = **−$212/day**
  - Slippage 1 tick × 2 sides × 212 × $0.50 = **−$212/day**
  - Net after frictions ≈ +$1,134 − $424 = **~$700/day** (matches pivot_residual_forward baseline)
- **T26** — 100% winning days is structural, not a bug: 212 × +$5 edge / day, variance averages away. NOT lookahead — SL always −$3, TP always +$50 exactly, INVERSE bounded.

## Concerns to vet before trusting numbers

1. **Engine vs pivot_residual_forward trade-count mismatch**. Need to reconcile. Either my engine is more realistic (intra-bar exits) or there's a bug firing extra trades.
2. **100% day-WR under $3 tight SL seems too clean**. Worth spot-checking specific days where price gaps hard.
3. **Friction sensitivity** not yet tested — the real number post-slippage / commissions is the one that matters.

- **T27** — User: build 1s ticker. Architecture: 1s price drives exits, 5s features read by nearest-past lookup, 1m close triggers entry decision. Also: noticed latent lookahead in current `FeatureTicker`: on 1m boundary crossing, engine reads `bar_data['close']` which is the about-to-complete bar's future close. Fix: 1s ticker will expose MOST RECENTLY COMPLETED 1m bar as `bar_data`, so close is always historical.
- **T28** — Also adding high/low to state so engine's TP/SL check uses 1s bar extremes (realistic intra-second fill).
- **T29** — Built `training_RM_physics/ticker_1s.py`. Updated engine `_check_exits` to use `state['high']`/`state['low']` for precise TP/SL. Fixed 1m-lookahead by using MOST-RECENTLY-COMPLETED 1m bar as bar_data. Ran pipeline.
- **T30** — **RESULT: −$595/day IS, −$707/day OOS, 1% / 0% day-WR, trade WR −0.60 / −0.62.** Complete collapse vs 5s engine's +$1,134/+$1,420.

### Why the collapse (root cause analysis)

- 5s engine's `state['price']` was the enclosing **1m bar's close** (constant for all 12 ticks within a minute). SL checks compared entry vs 1m close, not vs any intra-minute price.
- If price dipped −$5 mid-minute and recovered to close flat, the 5s engine **never saw the stop**.
- 1s engine with high/low: TP checked against 1s HIGH, SL against 1s LOW → catches every intra-second fill.
- Result: SL fires **6×** more often, TP fires **5×** less often. The "clean" TP paths in the 5s result were actually getting stopped out and flipped first.

### Corrected baselines

- **5s engine TP$50/SL$3 at 1m-close precision**: +$1,134/day (ARTIFACT — lookahead)
- **1s engine TP$50/SL$3 at 1s high/low precision**: **−$595/day IS, −$707/day OOS** (HONEST)

This matches the 2026-04-21 journal's prior finding: *"1m-close sims overstated by ~3× vs 1s intra-bar reality"* — except here it's worse than 3×; it's a sign flip (positive to negative).

### Consequences

- The +$779/day `pivot_residual_forward.py` baseline (in the "parallel" TP/SL sweep) likely has the **same lookahead bug**. We need to re-verify with 1s high/low.
- **$3 SL is physically untradeable on MNQ at 1s precision.** Normal 1s bar high-low range is 2-10 ticks ($1-$5), which alone crosses the $3 stop.
- **Tight stops die at 1s precision.** Any strategy using SL tighter than typical 1s bar range will lose to intra-second whipsaws.

### What survives the 1s reality check

- Nothing at the TP$50/SL$3 geometry.
- Need to find a strategy whose SL is wider than 1s bar volatility, OR whose entries are rare enough that the low hit-rate is offset by larger R:R, OR uses non-price exits (inverse, time-based, mean-cross).

- **T31** — User: try SL=$10. Ran. **IS −$966/day, OOS −$1,251/day, 0% day WR, Trade WR −0.56.** Widening SL makes it WORSE. Wider SL → bigger losses per trade, TP still rare, inverse exits now range −$5 to −$9 instead of being capped at −$3.

### Combined 1s result

| Config | IS $/day | Day WR | Mean $/trade | Trade WR |
|---|---:|---:|---:|---:|
| TP$50/SL$3 | −$595 | 1% | −$2.62 | −0.60 |
| TP$50/SL$10 | −$966 | 0% | −$5.13 | −0.56 |

**Neither geometry works** at 1s precision. The pivot-residual direction signal is **dead** when measured honestly:
- Cycle 3 confirmed direction HR at future horizons = 50/50
- Asymmetric TP/SL cannot rescue 50/50 — TP is too far away to hit often; any SL bleeds
- Our "baseline" wins (5s engine +$1,134, pivot_residual_forward +$779) were **lookahead artifacts** from the 1m-close exit-check approximation

### Status of the RM pivot research

Direction signal at pivots: **does not exist** when measured without lookahead. The research premise from 2026-04-21 journal ("Cohen d=−2.46, 86% oracle accuracy, but requires pivot lookahead") is literally correct — the "but requires pivot lookahead" was the operative caveat. The journal warned us.

- **T32** — User: add random variation based on 1s range to fills (adverse slippage up to 50% of 1s bar range). "That's what we will experience during live" — it's the real-world friction, not a stress test.
- **T33** — Implemented adverse slippage in engine (`_check_exits`, seeded random). Ran TP$50/SL$10 at 1s precision with slippage.

### Live-realistic baseline (TP$50/SL$10, 1s, adverse slippage)

| Metric | IS | OOS |
|---|---:|---:|
| Trades | 43,691 | 11,736 |
| Net PnL | −$255,400 | −$79,583 |
| **$/day** | **−$1,101** | **−$1,421** |
| **Trade WR** | **−0.60** | **−0.63** |
| **Day WR** | **0%** | **0%** |
| $/trade | −$5.85 | −$6.78 |

Slippage adds ~$135/day drag over unsimulated fills. **No day is positive.** TP$50/SL$X family is dead.

### What the whole 1s-realistic journey shows

| Config | IS $/day | Trade WR | Day WR | Notes |
|---|---:|---:|---:|---|
| 5s engine (1m-close lookahead) TP$50/SL$3 | **+$1,134** | +2.69 | 100% | ARTIFACT |
| 1s engine TP$50/SL$3 (no slippage) | −$595 | −0.60 | 1% | SL fails fast |
| 1s engine TP$50/SL$10 (no slippage) | −$966 | −0.56 | 0% | SL bleeds slower |
| **1s engine TP$50/SL$10 + slippage** | **−$1,101** | **−0.60** | **0%** | LIVE REALITY |

- **T34** — User: "if wrong 80%, guess inverse?" Ran with direction flipped. **IS −$1,550/day, Trade WR −0.72.** Slightly more TP hits (2,154 vs 1,301 — mild trend-continuation signal) but INVERSE exit logic breaks under flipped direction so total collapsed.
- **T35** — User: "what signal can flip losers to winners? How fast till we realize we entered wrong?" Built `tools/time_to_wrong.py`, wrote `findings/2026-04-22_time_to_wrong.md`. Finding: **losers hit −$5 at median 14s, winners at median 66s** — ~5× asymmetry. Looked like a strong early-detection signal.
- **T36** — Implemented flip rule: if trade is adverse by $5 within 15s of entry, close + open opposite direction. **Result: IS −$1,327/day, worse than baseline.** 14,078 flips fired but each is $5 cost + flipped trade faces same 50/50 direction problem + double friction.

### Why flip didn't rescue

- Asymmetry at −$5 was real (16:1 losers:winners in first 15s), but:
  1. Some winners dip to −$5 briefly before recovering → flipped into losers
  2. Flipped position has the same 50/50 underlying direction → no magic alignment
  3. 2× trade count = 2× friction (slippage + commission)
- Math: −$5 flip cost + flipped trade ≈ −$5 expected = **−$10 per flipped trade, same as riding to SL**. Plus friction.

### Big-picture status (provisional pre-T37)

Direction at RM pivots is truly 50/50. No geometry tried (TP$50/SL$3, TP$50/SL$10, SL$20, flipped, flip-rescue) generates edge at 1s precision with slippage. **The pivot-residual direction signal does not yield tradeable edge when measured honestly.**

- **T37** — User: "wait 1 or 2 bars to see if the entry is correct." Implemented **phantom entry**: at pivot confirm, register signal; wait `phantom_wait_bars × 60s`; commit only if price moved favorably by `phantom_min_favorable_pts`. First trial: wait=1 bar, min_favorable=0 (any favorable move).
- **T38** — Ran phantom-entry. **IS −$571/day, OOS −$748/day. Loss HALVED vs baseline.**
  - Trade count dropped 188 → 110/day (42% of entries filtered out as immediate adverse)
  - Direction signal exists *weakly*: fast-adverse trades are disproportionately losers, and filtering them saves money

| Config | IS $/day | Trades/day | Trade WR |
|---|---:|---:|---:|
| Baseline (no filter) | −$1,101 | 188 | −0.60 |
| Flip rule | −$1,327 | 241 | −0.64 |
| **Phantom 1-bar** | **−$571** | **110** | **−0.63** |

### Updated status

Phantom-entry (1-bar confirmation) is the best honest result so far but still loses $571/day on IS. Direction signal is weak-but-real, just not strong enough to overcome the friction at this trade frequency. Tuning knobs remaining: `phantom_wait_bars` (try 2), `phantom_min_favorable_pts` (require larger confirm), combining with chord ratio as regime filter.

- **T39** — Ran `tools/sweep_phantom.py`. Phantom-wait sweep:
  - C1 w=1: −$571/d
  - C2 w=2: −$402/d
  - C3 w=3: −$272/d  ← prev winner
  - Longer wait monotonically better; min-favorable-pts threshold didn't help; inverse direction worse.

- **T40** — User: "at SL level we can't reliably predict what 1s will do." Seed variance check via `tools/seed_variance_check.py`:
  - C3 run with seeds 1–5. Range: **$5/day. Std $2.** Slippage model seed-noise is negligible.
  - Deeper point stands: at SL $5–$10, individual TRADE outcomes are dominated by 1s intra-bar randomness. Aggregate stable, per-trade a coin flip at these magnitudes.

- **T41** — Tested SL=$30 (well above 1s range). **IS −$515/day** (worse than C3 −$272). But Trade WR and Day WR both improved (−0.46 and 7%). Character shifted: fewer SL fires, more INVERSE exits, net worse.

### Conclusion for Control phase

The direction signal is the fundamental cap, not the stop width. Rule-based approaches can't rescue it. Best honest result at 1s precision + slippage is **C3: −$272/day IS, 3% day WR**.

- **T42** — User: "the last thing we can do is use NN to try and solve it." Moving research to an NN approach — can a neural network find a direction signal that rules can't?
- **T43** — Built `tools/train_pivot_direction_nn.py`. Small CNN on 91D → P(win). 6×15 grid (12 core × 6 TF + 3 helper × 6 TF). Trained 25 epochs with walk-forward split (70/15/15 of IS days).
  - Best val AUC: 0.6226 (epoch 3, then overfits)
  - Test AUC: **0.6344**
  - Test accuracy: 58.4%
  - Calibration is CLEAN: P(win) buckets [0–0.3]=12.5% actual, [0.3–0.4]=31.2%, [0.5–0.6]=51.7%, **[0.6–0.7]=63.4%** actual.
- **T44** — Built `tools/apply_pivot_nn_filter.py`. Apply filter to trades:
  - TAKE if P(win) ≥ 0.55 (high edge — keep)
  - FLIP if P(win) ≤ 0.45 (trade opposite direction)
  - SKIP if 0.45 < P(win) < 0.55 (noise)

### RESULT: NN filter rescues direction signal

| Cohort | Variant | Trades | $/day | Trade WR | Day WR |
|---|---|---:|---:|---:|---:|
| Test IS (last 35d) | original | 3,457 | −$664 | −0.49 | 6% |
| Test IS (last 35d) | **nn-filtered** | 2,374 | **+$502** | **+1.08** | **94%** |
| **OOS 2026 (56d)** | original | 5,670 | −$715 | −0.49 | 2% |
| **OOS 2026 (56d)** | **nn-filtered** | **3,684** | **+$478** | **+0.97** | **93%** |

Daily PnL swing: **+$1,166/day (test IS)**, **+$1,193/day (OOS)**. Signal transfers IS→OOS.

### Interpretation

- CNN learned a direction signal from 91D features that we couldn't extract with rules alone.
- Direction at RM pivots is NOT purely random when conditioned on the full feature vector — residual sign alone was the coin-flip signal.
- The **FLIP regime** contributes most of the lift (~$500/day on OOS). Low P(win) trades reliably go the opposite way.
- Day WR leaps from 2-6% to 93-94%. Trade WR from −0.49 to +0.97.

### Caveats / concerns before celebrating

1. **Model peaked at epoch 3 then overfit** on val — small effective sample. Need larger training data or regularization before live.
2. **4-month effective sample** (3 train + 1 val/test months IS + 2 month OOS). Regime stability longer-term unknown.
3. **Commission not yet applied**: 3,684 OOS trades / 56d = 66/day × $1 RT = $66/day drag. Net still ~+$412/day.
4. **Slippage already in the model** (1s ticker random slippage). Covered.
5. **Post-hoc filter** — needs integration into the engine for a clean forward-pass test.

### T45 — Next: wire NN into engine, run full pipeline, validate the filtered $/day

Current raw RM engine (no NN): −$272/day IS (C3 phantom). NN-filtered: +$478/day OOS (post-hoc). True end-to-end integration should confirm or shave this number. Small chance of deviation because post-hoc uses trades that the engine already produced, while a live engine would decide entry based on NN prediction at the pivot moment (same features, same timing — should match).
