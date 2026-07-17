# Exit Dojo — pilot synthesis (10 episodes, Sonnet pilots)
2026-07-17 (night). Companion to `pilot_scorecard.md` (numbers) and
`reports/decisions/ep_NN.txt` (raw transcripts). Design: Moises — "a testing
grounds... sonnets try to solve the puzzle and record findings... like a
slideshow."

**Leakage caveat (governs everything below)**: single-prompt play — attention
CAN see future frames despite the sequential-commitment contract. Scores are
OPTIMISTIC; every rule here is a HYPOTHESIS until codified causally and passed
through the sealed 2024-tune / 2025-26-test harness. Nothing here is a result.

## 1. Scores (see pilot_scorecard.md for the full table)
- **7/10 episodes beat the fixed-5m hold.** Mean captured **+11.5 pts/trade vs
  −2.5** for fixed-5m (delta ≈ +14 pts/trade on this sample); median +7.75.
- Mean oracle (label-end) ref +29.1 pts → pilots reached ~40% of oracle on
  average; **median capture ratio 0.475** (ep_10's 7.25 is a degenerate
  |oracle|=1pt denominator — ignore it). Program context: the static R-trigger
  system's oracle-normalized ratio is 0.23 — the pilots roughly DOUBLED it,
  under leakage, at N=10. Direction of evidence, not a claim.
- Losses vs fixed-5m: ep_05 (−26.75 — confirmation tax after a t=5 peak),
  ep_07 (−10.75 — wrong-side bail 4-5 min late), ep_09 (−2.0 — noise).

## 2. The emergent grammar — 10/10 independent convergence
Every pilot, in its own words, landed on the same master rule:

> **Exit only on a CONFLUENCE of 2-3 independent signal families arriving in
> the same 1-3 minute window. Every lone signal is a false positive.**

Receipts: "confluence, not any single stat" (ep_02); "every signal lined up
at once" (ep_03); "triple stack" (ep_04); "stacked reversal signature"
(ep_05); "unanimous" (ep_06); "stacked confirmation" (ep_08); "confluence,
not any single signal" (ep_09); ep_10 held 45 min BECAUSE the 2-family bar
was never met.

### The confirmed component set, ranked by citation frequency
1. **ER10 regime change — cited by all 10.** Collapse from a trend peak
   (0.8 → 0.25) = with-trend thesis death; a NEW FLOOR matters more than a low
   level (ep_02); ER10 ≥ 0.5 = trend intact, hold through giveback (ep_10).
   Inverted on the wrong side: RISING adverse ER = adverse trend
   re-accelerating (ep_07).
2. **Multi-family fresh against-fires** (KMDR / HA / PROPP / CLIMAX), 2+
   families within ~1-3 min. Staleness matters — a fire's age resets its
   evidentiary value (ep_10 demanded "fresh within 1m").
3. **Giveback DYNAMICS, never level.** What fired: giveback VELOCITY crossing
   ~50% of the largest leg in one bar (ep_05); giveback breaking out of a
   sustained near-zero band (ep_08); giveback ERASED to 0% by a fresh adverse
   extreme (ep_07, wrong-side). What was explicitly rejected: giveback % spikes
   alone (ep_02, ep_03, ep_04 all held 20-35% givebacks that recovered).
4. **Bar anatomy**: close pinned at the bar's extreme / outsized single-bar
   range with no intrabar recovery (ep_02, ep_03, ep_06, ep_08). A violent
   rejection bar is a family of its own.
5. **Vol(5m) direction on the adverse move** (ep_03): vol breaking OUT on the
   down-move = real; calm/contracting vol on a wick = noise.

### Three regimes, three rule sets
- **WITH-TREND (winner/midflip)**: hold through everything until ER10-collapse
  + ≥2 fresh against-fires + a giveback-dynamics event stack in one window.
  Craft touch (ep_01): after thesis death, don't sell the low — exit into the
  next favorable bounce (recovered +24.75 vs the 5m ref on that episode).
- **WRONG-SIDE (instantfail)**: bail on {giveback erased + fresh adverse
  extreme + RISING adverse ER}. Both wrong-side pilots under-performed on
  execution: ep_07 stated the rule but acted at t=7 (−65.25, worse than the
  5m hold); ep_08 never applied it, rode a −84pt MAE and got bailed out by a
  bounce (+9.75) — **survivorship, not skill; label ended at t=0. The
  wrong-side rule needs to fire in the first 2-3 minutes or it's worthless.**
- **CHOP**: the grammar goes silent (ep_10 — no 2-family bar ever met, held
  45 min through a +15.25 peak for +7.25 luck). Chop needs its own overlay:
  time-stop / take-what's-given, gated by the CTX-ER chop state. ep_10's own
  fix: lower the bar to 1 family + rising ER10 when already past a peak.

### Anti-patterns (independently confirmed across pilots)
- Lone fires of ANY family (all 10). Specifically: KMDR fired against the
  whole trend and was wrong the whole way (ep_02, ep_04).
- Drawdown DEPTH alone (ep_07, ep_08) and wick depth alone (ep_03).
- Giveback % thresholds alone (ep_02/03/04) — the static PROP-TURN kill
  (comms 093) rediscovered from the other direction.

## 3. The latency finding (ties to the program's central result)
The confirmation stack pays a **2-6 minute tax**: ep_05's peak was t=5, full
stack confirmed t=7 (gave back 27 of 38 pts); ep_06's own summary: "by t=13
the trade had already surrendered most of its open profit" — its fix is a
TIGHTEN-FIRST tier (the t=10-12 single-family buildup = scale down / tighten,
unanimity = exit). This is the same lag family as P_hold (+3 min) and every
static detector (±2m bar unreachable). The pilots didn't beat the lag — they
managed AROUND it (tighten tiers, exit-into-bounce). Consistent with the LAW:
turns live in paths, not snapshots; only a path-integrator with per-bar state
can price the stack continuously instead of re-deciding per frame.

## 4. Actionable outputs
1. **Mamba state vector additions** (on top of the doc-095 leg-geometry
   block): ER10 + its 1m delta and rolling floor/peak; per-family fresh-fire
   ages (not just counts); giveback velocity + band-breakout flag (not level);
   bar-anatomy pair (close-position-in-range, single-bar range z); vol(5m)
   delta signed by adverse/favorable direction.
2. **Mamba reward hints**: penalize exits coincident with a lone fresh fire;
   credit wrong-side bails scaled by earliness (minutes from entry);
   chop-regime episodes get a time-stop credit instead of a capture target.
3. **EXIT-GRAMMAR-01 (graduation candidate)**: causal rule — ER10 crossing
   below a rolling floor AND ≥2 distinct families fresh-against within 3 min
   AND one giveback-dynamics event (velocity / band-breakout / erased) →
   tighten at 1 condition, exit at 2-3. Parameterize on 2024, seal, test
   2025-26 vs the bracket-grid and 5m-hold refs. Build NOT started — morning
   scope decision alongside the full-run budget (100-300 episodes).

## 5. Method notes for the full run (if funded)
- One distinct day per episode held; stratification worked (all 4 types
  produced distinct lessons). Keep 30/30/20/20 winner/midflip/instantfail/chop.
- The decision contract parsed cleanly (after scorer accepted the "t=Nm"
  suffix); keep the format.
- Add to the packet: vol(5m) delta + per-family fire AGE (pilots wanted both;
  ep_03 and ep_10 derived them by hand from the slideshow).
- Wrong-side episodes need a harder question: the packet shows drawdown
  immediately; the skill being tested is speed. Consider scoring wrong-side
  eps on exit-minute percentile, not just points.
