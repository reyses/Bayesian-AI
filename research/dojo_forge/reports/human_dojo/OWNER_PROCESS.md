# Owner's discretionary turn-calling process (verbatim capture 2026-07-28, via TG)

## The process (owner's words, structured)
1. **1m timeframe** — locate the day's cusps (pivots).
2. **Horizontal line through the last 3 cusps** — recent pivot levels as support/resistance.
3. See where price is **currently located** relative to those levels.
4. **Switch to 1h** — observe how the 1m levels correlate with the 1h structure (multi-TF confluence).
5. **Wait until it "makes sense"** (confluence + patience).
6. **Cusp/leg-onset tell**: if closes have been **flat + alternating** (small, choppy, sign-flipping) → you're in / near the cusp → **a leg is coming**.
7. **NMP = ENERGY gauge** (owner metaphor): where the leg is in its energy cycle — building vs exhausting.
8. **Legs ALTERNATE** (the oscillation) → energy state + alternation time the next (opposite) leg.
9. **Cubic regression (orange line)** — added to 'better see the STATE of things' (trend/turn/energy at a glance).

## Extracted testable signals (never tested mechanically before)
- (a) recent-3-cusp horizontal LEVELS (S/R from last 3 pivots)
- (b) 1m↔1h multi-TF confluence
- (c) flat+alternating closes as the leg PRECURSOR (volatility compression / chop → leg)

## Test results
- **(c) naive, short-term (leg ≥M within 10 bars from a flat bar): FALSE / opposite.**
  Consolidation (range≤p30 & ER≤0.35) → P(leg≥20pt)=7.8% vs 39.6% non-flat (0.20×); mean move 9.8 vs 22.3pt.
  = "chop begets chop" (matches recovery_dynamics). AUC of low-range/low-ER/high-alt ~0.18/0.48/0.52.
- **Caveat**: this mis-tests the owner's actual signal — (1) the leg comes when the flat RESOLVES (breakout), not during it; (2) flat-alone ignores the cusp-LEVEL + 1h confluence that carry the edge.
- **Right tests (pending)**: compression→EXPANSION breakout leg; flat AT a recent-3-cusp level vs flat-anywhere.

## Data being collected
- Chart replay recorder (tools/viz/plugins/chart_replay_recorder.py) → replay_<day>.<who>.jsonl (owner's clicked turns).

## Refined entry/exit rule (owner demo 2026-07-28, on 2025_01_21 02:50)
- Don't hold the short. WAIT until price is on the LOWER side of the cubic AND the current bottom ALIGNS with the previous bottom (a prev-pivot level). At that touch: EXIT short + ENTER long (flip).
- Harvests BOTH legs of the oscillation (legs alternate). Demo: SHORT 21591→21576.5 (+14.5pt) + LONG 21576.5→21603.5 (+27pt) = +41.5pt vs short-hold +3.2pt.
- Causal signal = "price hits prev-bottom level on the lower-cubic side → reverse." Confluence: level-alignment + cubic-side.

## The "music"/theme — overall trend bias (owner 2026-07-28)
- The OVERALL THEME (big-scale trend / 1h) = the "music". It biases leg-play:
  - DOWNTREND theme: shorts are WITH-trend (ride, bigger targets); longs are COUNTER-trend bounces (smaller — exit at the "other cusp" = a LOWER HIGH, then flip back short).
- Complete cycle: theme(down) → SHORT cusp (with-trend) → reverse LONG at aligned bottom (prev-bottom lvl, lower-cubic side) → exit LONG at other cusp (lower high) → re-SHORT. Alternating oscillation legs, biased by the big-trend music; conviction/size scaled by with- vs counter-trend.
- => Complete codifiable method: triggers = prev-pivot LEVEL touch + matching CUBIC SIDE + alignment; bias = big-scale trend. Judgment piece ("until it makes sense") to proxy with big-scale cubic/regime.

## Exit-long via 5s rejection at a major level (owner demo 2026-07-28, 03:06)
- Counter-trend long (against down theme) = lower conviction -> drop to 5s, be vigilant for a fake retracement.
- Exit signal: the bounce runs into a MAJOR level (close proximity) AND the 5s shows REJECTION (rolls over / momentum turns) -> exit long; failed bounce in a down theme sets up the re-short.
- Demo: bounce spiked to the 21608 major, rejected, dropped to ~21593 on the 5s. Owner's "fake retracement" read confirmed. Honest caveat: owner would HESITATE live — a judgment call.

## Theme-read is the load-bearing judgment (demo 2026-07-28, re-short from 21607)
- Re-short from the top got a SMALL down leg (21607->21591 = +16pt at the aligned bottom), then price BROKE UP through the 21620 major to new highs (21625). The stated "down theme" FAILED.
- Whole window 02:30-03:20 = rising lows/highs = TRUE theme UP. So the LONG (with true trend) was the big winner (+30.8); shorts (counter) were small/failed.
- LESSON: the theme/"music" read flips which legs are big vs small; mis-reading it (down when up) makes the with-"theme" legs the losers. Take each leg at its cusp; don't marry a theme the price action argues against. This is the key discretionary risk to model/measure.

## Adaptive scratch-and-flip (owner honesty 2026-07-28)
- Owner would NOT hold a losing thesis. Re-short from the top: he'd hesitate, scratch ~break-even, then flip LONG with the actual up-move. Adaptive correction prevents the wrong-theme loss (mechanical hold lost -18pt).
- The scratch-and-flip trigger = the CUBIC HARSH TURN against the position (systematizable version of "it's not working"). 
- Whole method reduces to 2 measurable rules: (1) confluence ENTRY (prev-pivot level + matching cubic side + alignment), (2) SCRATCH-AND-FLIP on a cubic harsh-turn against you. Model must capture the exit/scratch discipline, not just entries.

## Day-level theme map (owner technique 2026-07-28)
- Owner used to go to DAY level and color-code the different THEMES. Built: full day segmented by big-leg direction (big-R=55 zigzag) — green=up theme, red=down theme + 30m cubic overlay = the theme curve.
- Validates today's error: at 02:50-03:00 the map is transitioning RED→GREEN (down->UP theme). That's why the LONG worked (+30) and re-short failed — it was an up-theme. The day-map would have flagged "favor longs here."
- => The THEME (the load-bearing judgment) is DEFINABLE + systematizable: regime = big-leg direction / big-cubic slope. Proposed: add a theme strip to the recorder (current regime always visible). Render: scratchpad/daytheme.png; tool logic inline (big-R zigzag color-segmentation).

## Top-down TF cascade = the systematizable THEME engine (owner 2026-07-28)
- Owner's process: day → 4h → 1h → 1m, testing/adjusting the read at each scale.
- Systematized: regime (rolling-slope sign, deadband) at 1h/4h/day. ALIGNED colors = strong theme (trade with it, big); mixed/gray = no theme/transition (don't force direction).
- Settles today: at 02:50-03:00 ALL scales were CHOP → no theme → the forced re-short failed. 09:00+ all GREEN = strong up theme. The cascade would have said "no theme @03:00, don't short."
- => THREE codifiable pieces complete: THEME (cascade alignment) + ENTRY (prev-pivot level + cubic side + alignment) + SCRATCH-FLIP (cubic harsh turn). Enough for a v1 causal backtest. Render: scratchpad/cascade.png.

## Level coordinate system = codified (owner 2026-07-28, "touch as many highs/lows as possible... a region expressed as a line")
- Owner's placement rule is a 1D clustering objective: lines at DENSITY PEAKS of the pivot-price distribution; each line a REGION (band ±τ), strong = revisited at temporally-SEPARATED times.
- CRITICAL: must be a **TF TELESCOPE** (day→4h→1h), NOT one flat R. Single-R over 3 days = 25% coverage (fails). Per-scale (coarse R/long window→few big lines; fine R/recent window→micro lines) + dedup (coarse owns its region) reproduces the hand-drawn nested set.
- Tool: `research/dojo_forge/tools/level_coordinate_system.py` — greedy density-peak + NMS per scale; emits per-bar coordinate FEATURES (norm_pos between bracketing lines, signed dist to nearest up/down). Render: reports/human_dojo/levels_<day>.png.
- Validated on 2026_07_14..16: day 29458(9t) wall, 4h 29223 support-turned-resistance, 1h 29160(6t) live pivot — matches owner's eye. The levels are the COORDINATE FRAME (not a predictor); cubic/theme/NMP/leg-position are read against it. Ties to old z_high/z_low + "we need a new feature" thread.
- NEXT: wire into chart_replay_recorder (replace recency CUSP_R/MAJOR_R with this telescope); optional N→N+1 stability test.

## What the dojo actually measures (owner insight, 2026-07-29)
"Since it is not real money, no overprocessing on my side — my brain is patterning without risk." The dojo isolates the PATTERN ENGINE from the fear system. This is the correct target: the hesitation/fear layer (owner's own words: "in truth I would hesitate... the re-short would have been a hesitation and probably break even") is the one component the machine should NOT inherit — discipline is free for silicon. Expect dojo P&L > live P&L; that gap = the fear tax = the quantified automation prize. Dojo-you is the distillation target; live-you is dojo-you minus the tax.

## GRADING RUBRIC — quality of read, NOT P&L (owner 2026-07-29)
"We should not grade on PnL, it should be quality — for instance we should have noticed that we captured a large down leg, then the theme actually turned into the up leg. That's the stuff we need to learn to identify."
Verdicts grade the READ, on these dimensions (from the captured process):
1. **Leg position** — did the student know where it was in the current leg (early body / late / at the cusp)? Did it notice a leg COMPLETING?
2. **Theme transition** — did it catch the flip (large leg captured -> expect alternation; cascade turning)? Concrete miss tonight: big down leg banked, theme flipped up, Claude shorted into the new up-leg twice (legs 1&3) — the "legs alternate" rule under-weighted.
3. **Reference use** — right level for the context (not just nearest), room respected, region-not-hairline thinking.
4. **Timing tell** — wick/deceleration read at the decision moment (bar107-style), not just slope sign.
5. **Restraint** — passed when there was nothing (no theme / no room / mid-cell), didn't chase.
P&L stays tracked separately as the EDGE ledger (ride-edge gate owns go/no-go) but is NOT the teaching signal — at leg-level it grades variance, not judgment. A losing trade with a correct read grades AGREE; a winning trade with a wrong read grades DISAGREE.
Future tooling (offered): post-hoc auto-quality per leg once the fog lifts — leg-capture fraction, exit-vs-actual-pivot lag, theme-flip detection lag. Objective complements to the verdicts.

## "LOOK LEFT" — the oscillation-cycle context check (owner post-mortem, 2026-07-29)
Miss identified at bar ~108 (2025_12_19): flatted at the LOW of a clearly visible oscillation cycle (trough 24980 -> crest 25011 -> trough 24990, ~25pt amplitude, ~10-12 bar half-period) and then passively "waited for the next cusp" — failing to consult the left of the chart, where the cycle said the base case was an up-swing to the 25010 band (bar 110 delivered exactly that). The same logic ("look at the previous oscillations") had been the owner's OWN entry reason 10 bars earlier, then got dropped.
RULE: at every decision point, LOOK LEFT — read the prior cycle (amplitude, period, which phase we're in). Within a range regime, the oscillation continuing is the base case; position in the CYCLE (not just distance to a line) is part of the coordinate system. Caveat honestly: the cycle read earns the swing to the opposite band; range EXPANSION beyond it (25042 here) is bonus, not predicted.
=> Rubric dimension 6: **Cycle context** — did the student consult the left (prior oscillation amplitude/period/phase) before deciding?

## THESIS-BEFORE-REVEAL protocol (owner 2026-07-29, "this will sharpen us both")
When the owner asks "your read?": Claude consults priors (dojo corpus slices via SQL, OWNER_PROCESS rubric, AND the research program's stats — flat ~10%/bar leg hazard/memorylessness, leg alternation, amplitude/length distributions, cascade theme) and states a STRUCTURED THESIS **before** the next bar is revealed: priors cited -> directional thesis + horizon -> explicit falsifier. Owner states theirs; discussion happens PRE-reveal; then step; then both theses grade against the bar. Pre-commitment kills hindsight bias by construction; the (thesis, outcome) pairs from BOTH sides build a calibration ledger (who reads what well, where each is blind). Disagreement slices are the highest-value corpus. Log theses as notes with explicit who= (per-event override now supported). This is also a live rehearsal of the glass-cockpit interpreter role from the north star.

## TWO-WAY QUESTIONING (owner 2026-07-29, "so we get more rich narration")
Claude may/should QUESTION the owner before the reveal — Socratic elicitation of tacit knowledge free narration misses. Question types (rubric-tied): falsifier ("what kills this thesis?"), reconciliation ("theme says X but indicator says Y — which wins?" — surfaces the owner's PRIORITY ORDERING, the least-documented layer), selection ("why this level and not that one?"), dog-that-didn't-bark ("you didn't mention the wick — irrelevant or missed?"), look-left ("where in the cycle are we?").
DISCIPLINE: max ~one sharp question per slice — casual is the methodology; an interrogation wakes the fear system and contaminates the pattern-engine signal. "Later" is a valid owner answer and is logged too. Answers land in the corpus as owner narration at the exact slice.

## "IT JUST SOUNDS RIGHT" is a protected answer (owner 2026-07-29)
Many answers will be "I don't know, it just sounds right" — that is DATA, not failure: the pattern engine operating below verbal access (the "music" layer, always known to exist). RULES:
1. Log it VERBATIM. Never press for a reason past it — pressed humans confabulate, and a confabulated rule in the corpus is worse than none (qwen would learn the fake reason).
2. Allowed light probes only: "what would make it stop sounding right?" (falsifier often accessible when reason isn't) and "where are you looking?" (attention reportable when computation isn't).
3. Corpus handles the split by design: articulable calls -> rules (direct distillation); "sounds-right" calls -> (chart_state -> action) pairs learned by imitation — this is WHY the target is 60 LEGS, not 60 rules.
Hypothesis to check at corpus scale: entries skew tacit, VETOES skew articulable — and the vetoes may be where the transferable edge concentrates.

## SPITBALLED REASONS = HYPOTHESES, deferred verdict (owner 2026-07-29)
Owner: "sometimes when asked why I will fabricate something right out of my ass — but it will later become true." Mechanism: the fabrication is generated by the SAME pattern engine that made the call (self-description with partial verbal access), so it's a noisy-but-correlated sample of the real computation — not plausibility-anchored confabulation. THIRD answer class (between articulated and sounds-right): SPITBALLED -> log tagged as HYPOTHESIS, verdict deferred, then TEST against corpus/historical data ("brain proposes, data disposes"). Verified -> promoted to rule with receipts; refuted -> dies without polluting the curriculum. Don't over-pressure (sometimes-not-always, owner's words): 1-question cap stands, "no idea" ends any thread.

## F-SPACE TRIANGULATION (owner 2026-07-29: "my brain says the answer is in F-space, I just don't know how to access it yet")
Access mechanism = triangulation, not introspection: every dojo slice now silently carries the causal combiner snapshot (P_topk/P_any, gov_stream+dir, n_fires, zz_leg/confirm/age, top-3 firing streams) from research/nt8_port/atlas_backtest. The owner's calls (incl. tacit "sounds right" ones) get PAIRED with F-space states in the corpus -> at corpus scale, correlate call-slices vs pass-slices in F-space to find WHICH features the pattern engine is actually reading. `fspace` command = conscious on-demand look. Coverage: RTH window only (14:30-21:15 UTC, 1min), 64/584 days empty — the liquidity-gated session starts naturally land inside coverage.

## RULE: wait for the high-energy state to RELEASE before entering (owner post-mortem, 2025_08_24 session)
Owner's articulated lesson after a mistimed short entry (-6.6pt, entered during high-energy chop that later crashed as predicted): don't enter INTO high energy/volatility, even with the right directional read. Sequence should be: (1) identify real resistance forming (not just a level touch — the point where price actually fails to continue the leg), (2) let the high-energy state exhaust/release its momentum first, (3) THEN enter. Direction was right both times tonight; the miss was entering mid-energy instead of after the release. Ties to dimension 4 (timing tell) + dimension 5 (restraint) — this is the specific, actionable form of "restraint": not just "don't trade in energy," but "wait for the energy to finish, then trade the aftermath."

## Refinement: HIGH CONVICTION at the exact top, tell = buyers struggling (owner, same session)
Sharpens the release-rule above: optimal isn't passive waiting, it's HIGH-CONVICTION entry right AT the highest point, sized/committed once the specific tell fires: the green (buyer) bars visibly struggling to hold on (shrinking bodies, failing to make new highs cleanly, upper wicks growing) — that's the observable signature of the energy exhausting itself. Then let it crash under its own weight rather than needing to push it. So the full sequence: high energy up -> watch for buyers struggling (the tell) -> enter WITH CONVICTION at that exact point -> gravity does the rest.

## Volume confirms the "buyers struggling" tell independently (owner asked, 2026-07-29)
Checked volume through the energy episode (2025_08_24, bars 94-108, median vol ~250): bar94 rally-start spiked to 1190 (4.8x, real conviction). Bars95-98 stayed elevated. Then bars99-106 (the grind at the highs, owner's "buyers struggling") THINNED to 163-414 — near/below median, i.e. no fresh volume confirming the new highs. Bar107 (the crash) spiked back to 715 (2.9x) on the -10.25 down bar.
This INDEPENDENTLY confirms the price-action "struggling to hold on" tell with a second, unrelated signal (classic volume-divergence-at-highs). => candidate COMPUTABLE feature: volume-fade ratio (recent-N vol / prior-N vol) during a push into a high, as a companion trigger alongside the candle-shape tell.

## RULE: avoid the "no-no zone" — last ~20min before a session/day boundary (owner, 2025_08_24 session)
Owner names it explicitly: don't trade in the last ~20 minutes before a session/day-boundary (whether that boundary is a real halt or, as discovered this session, just a data-file split). Receipt: this exact session's final ~20min produced 3 whipsaw trades (-6.6pt, +5.11pt, -2.39pt forced-EOD) = -3.92pt net, in a tight, low-conviction, back-and-forth range — the SAME choppy signature as the earlier sparse-liquidity overnight sessions, even though volume wasn't thin here. Candidate mechanism: behavior/participation genuinely changes near any known boundary (session end, day rollover), independent of raw volume. Ties to the liquidity-floor design principle (avoid structurally unreliable windows) — this may need a SEPARATE guard from the volume floor, since this window wasn't low-volume, just low-conviction/choppy.

## Mechanism for the no-no zone: NY close / session rollover (owner)
Owner identifies WHY the no-no zone is choppy, not just that it is: it's around the NY close -- a real participant-rollover event (NY session traders closing/unwinding, handoff to the next session's participant pool), not an arbitrary data-file artifact. This explains the whipsaw mechanically: liquidity/direction gets contested between outgoing and incoming flow. Owner's nuance: "there's a chance to get good money but it's hard" -- higher variance, not simply avoid-always; read carefully or skip, don't trade it casually/on autopilot.

## Regime-dependence finding: the deceleration-short tell needs a theme check first (Claude self-test, 2025_06_05 trend day)
The confirmed-reversal + volume-divergence short setup (validated on 2025_08_24's range-bound session) FAILED TWICE on a trend day (2025_06_05), both counter-trend against a persistent 1h UP regime. The tell (deceleration after a volume-confirmed push) still fired correctly as a LOCAL signal, but it was insufficient against a stronger higher-timeframe trend. LESSON: the deceleration/volume-divergence tell should be gated by theme/regime alignment (per the earlier cascade work) -- strong in range-bound/chop regimes, needs a much higher bar (or should be flipped to a WITH-trend entry trigger instead) when a clear directional regime is active. Don't pattern-match a tell validated in one regime onto a different regime without re-checking fit.

## Structural gap: level-anchored room-to-target doesn't work on trend/breakout days (Claude self-test finding)
On a strong trend day (2025_06_05), a genuine with-trend setup (theme aligned UP across all 3 scales, pullback resolving back up) could NOT be traded under the room-to-target rule because no reference level exists ABOVE price when it's making fresh highs -- only trailing support levels exist. The level-coordinate-system is built from PAST pivots, so it structurally can't provide a forward target on a breakout. NEEDED FOR TREND DAYS: a different room/target concept -- e.g. measured-move projection (prior leg length applied forward), or ATR-multiple target, rather than requiring a pre-existing telescope line. This is a real gap between the range-bound-day toolkit (validated well tonight) and trend-day trading, not a minor edge case -- flag before assuming the captured process generalizes across regimes.

## Self-critique: regime-alignment is not validated edge (Claude self-test, biggest loss of the session)
The with-trend long (entered specifically because I'd "learned" to avoid counter-trend after 2 losses) lost -20.1pt -- more than either counter-trend loss. This is a genuine check on the regime-adjustment reasoning from earlier in this session: "align with the higher-timeframe trend" is folk wisdom that FEELS like risk management, but this project's own prior research (leg-reversal memorylessness, the N=160-163 owner_process_v1 backtest showing the bare mechanized skeleton is coin) already found that momentum/trend alignment does not carry proven forward predictive power here. I reached for an intuitive fix without checking it against established findings in the same codebase. LESSON FOR FUTURE SESSIONS (owner and student both): a new in-session heuristic needs to be checked against existing validated research before being traded with conviction -- "it sounds reasonable" is not the same as "it's been shown to work," and this project has been burned by that gap before (the detrend "breakthrough" retraction earlier in the program is the same class of error).
